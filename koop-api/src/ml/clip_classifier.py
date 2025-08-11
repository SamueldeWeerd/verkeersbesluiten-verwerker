import torch
import clip
from PIL import Image
from io import BytesIO
import logging

class ImageClassifier:
    """
    CLIP-based image classifier to determine if an image contains 
    maps, satellite images, or aerial images.
    """
    
    def __init__(self, settings=None, device=None):
        """
        Initialize the CLIP model for image classification.
        
        Args:
            settings: Application settings
            device: torch device to use. If None, auto-detects CUDA availability.
        """
        from src.config.settings import get_settings
        self._settings = settings or get_settings()
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
            logging.info(f"✅ CLIP model loaded successfully on {self.device}")
        except Exception as e:
            logging.error(f"❌ Failed to load CLIP model: {e}")
            raise
        
        # Define classification prompts
        self.classification_prompts = [
            "a map, a schematic map, a city map, a road map, a topographic map",
            "a satellite image, an aerial photograph, an aerial view, a bird's eye view",
            "anything else, miscellaneous content, text, documents, signs, other content"
        ]
        
        # Minimum confidence threshold for maps/aerial/satellite images
        self.confidence_threshold = 0.4
    
    def classify_image_from_bytes(self, image_bytes):
        """
        Classify an image from bytes data.
        
        Args:
            image_bytes: Raw image data as bytes
            
        Returns:
            dict: Classification results with 'is_map_or_aerial', 'confidence', 'probabilities'
        """
        try:
            # Load image from bytes
            image = Image.open(BytesIO(image_bytes))
            return self.classify_image(image)
        except Exception as e:
            logging.error(f"❌ Error processing image from bytes: {e}")
            return {
                'is_map_or_aerial': False,
                'confidence': 0.0,
                'probabilities': {
                    'maps': 0.0,
                    'aerial_satellite': 0.0,
                    'miscellaneous': 0.0
                },
                'error': str(e)
            }
    
    def classify_image_from_path(self, image_path):
        """
        Classify an image from file path.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            dict: Classification results with 'is_map_or_aerial', 'confidence', 'probabilities'
        """
        try:
            # Load image from file
            image = Image.open(image_path)
            return self.classify_image(image)
        except Exception as e:
            logging.error(f"❌ Error processing image from path {image_path}: {e}")
            return {
                'is_map_or_aerial': False,
                'confidence': 0.0,
                'probabilities': {
                    'maps': 0.0,
                    'aerial_satellite': 0.0,
                    'miscellaneous': 0.0
                },
                'error': str(e)
            }
    
    def classify_image(self, pil_image):
        """
        Classify a PIL Image object.
        
        Args:
            pil_image: PIL Image object
            
        Returns:
            dict: Classification results with 'is_map_or_aerial', 'confidence', 'probabilities'
        """
        try:
            # Convert to RGB if necessary
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # Preprocess image
            image_tensor = self.preprocess(pil_image).unsqueeze(0).to(self.device)
            
            # Tokenize text prompts
            text_inputs = clip.tokenize(self.classification_prompts).to(self.device)
            
            # Get predictions
            with torch.no_grad():
                image_features = self.model.encode_image(image_tensor)
                text_features = self.model.encode_text(text_inputs)
                
                # Compute similarity
                logits_per_image, _ = self.model(image_tensor, text_inputs)
                probabilities = logits_per_image.softmax(dim=-1).cpu().numpy()[0]
            
            # Determine if this is a map/aerial/satellite image
            # Prompts: 0 = maps, 1 = satellite/aerial, 2 = miscellaneous/other
            map_confidence = probabilities[0]
            aerial_confidence = probabilities[1]
            misc_confidence = probabilities[2]
            
            # Consider it a map or aerial image if either maps or aerial confidence is high
            # and miscellaneous confidence is relatively low
            is_map_or_aerial = (
                (map_confidence > self.confidence_threshold or 
                 aerial_confidence > self.confidence_threshold) and
                misc_confidence < 0.6  # Not primarily miscellaneous content
            )
            
            # Overall confidence is the max of map and aerial confidence
            overall_confidence = max(map_confidence, aerial_confidence)
            
            result = {
                'is_map_or_aerial': bool(is_map_or_aerial),  # Convert numpy bool to Python bool
                'confidence': float(overall_confidence),
                'probabilities': {
                    'maps': float(map_confidence),
                    'aerial_satellite': float(aerial_confidence),
                    'miscellaneous': float(misc_confidence)
                },
                'classification': self._get_classification_label(probabilities)
            }
            
            logging.debug(f"🔍 Image classification: {result}")
            return result
            
        except Exception as e:
            logging.error(f"❌ Error during image classification: {e}")
            return {
                'is_map_or_aerial': False,
                'confidence': 0.0,
                'probabilities': {
                    'maps': 0.0,
                    'aerial_satellite': 0.0,
                    'miscellaneous': 0.0
                },
                'error': str(e)
            }
    
    def _get_classification_label(self, probabilities):
        """Get the most likely classification label."""
        labels = ['maps', 'aerial_satellite', 'miscellaneous']
        max_idx = int(probabilities.argmax())  # Convert numpy int to Python int
        return labels[max_idx]
    
    def should_download_image(self, image_data):
        """
        Convenience method to determine if an image should be downloaded.
        
        Args:
            image_data: Either bytes data or file path
            
        Returns:
            bool: True if the image should be downloaded (is map/aerial), False otherwise
        """
        if isinstance(image_data, (str, bytes)):
            if isinstance(image_data, str):
                result = self.classify_image_from_path(image_data)
            else:
                result = self.classify_image_from_bytes(image_data)
        else:
            # Assume PIL Image
            result = self.classify_image(image_data)
        
        return result.get('is_map_or_aerial', False)


# Global classifier instance (initialized when needed)
_classifier_instance = None

def get_classifier():
    """Get a singleton instance of the ImageClassifier."""
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = ImageClassifier()
    return _classifier_instance

def classify_image_bytes(image_bytes):
    """
    Convenience function to classify image from bytes.
    
    Args:
        image_bytes: Raw image data as bytes
        
    Returns:
        dict: Classification results
    """
    classifier = get_classifier()
    return classifier.classify_image_from_bytes(image_bytes)

def should_download_image(image_bytes):
    """
    Convenience function to determine if an image should be downloaded.
    
    Args:
        image_bytes: Raw image data as bytes
        
    Returns:
        bool: True if the image should be downloaded
    """
    classifier = get_classifier()
    return classifier.should_download_image(image_bytes) 