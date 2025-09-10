"""
Minimal GIM(Roma) feature matcher that returns keypoints between two images.
Just run: matcher = RomaMatcher(); result = matcher.match_images(image1, image2)
"""

import sys
import logging
from pathlib import Path
import subprocess
import torch
from torch import nn
from PIL import Image
import numpy as np
from huggingface_hub import hf_hub_download
import cv2
import os
import gc
import psutil

# Setup logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s %(name)s %(levelname)s] %(message)s')
logger = logging.getLogger("roma_matcher")

# Import configuration for CPU optimization
sys.path.append(str(Path(__file__).parent.parent))
from memory_config import config

# Setup device and optimization based on configuration
if config.roma_force_cpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable CUDA
    DEVICE = torch.device("cpu")
    
    # Setup multi-threading for CPU performance
    threads_used = config.setup_torch_threading()
    logger.info(f"🖥️  ROMA CPU optimization: {threads_used} threads on {config.cpu_cores} cores")
    
    # Use float32 for CPU compatibility
    os.environ["TORCH_DTYPE"] = "float32"
else:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if DEVICE.type == "cuda":
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
        logger.info(f"🚀 ROMA GPU acceleration: {gpu_count}x {gpu_name}")
        
        # Use mixed precision for GPU efficiency
        os.environ["TORCH_DTYPE"] = "float16"
    else:
        logger.info("⚠️  No GPU detected, falling back to CPU mode")
        # Setup CPU threading as fallback
        threads_used = config.setup_torch_threading()
        logger.info(f"🖥️  ROMA CPU fallback: {threads_used} threads")
        os.environ["TORCH_DTYPE"] = "float32"
MODEL_REPO_ID = "Realcat/imcui_checkpoints"

# Local paths for caching - use absolute paths for Docker compatibility
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent  # Go up one level from third_party to project root
CHECKPOINTS_DIR = PROJECT_ROOT / "ROMA_checkpoints"
THIRD_PARTY_DIR = SCRIPT_DIR  # We're already in third_party

# RANSAC parameters
RANSAC_REPROJ_THRESHOLD = 3.0
RANSAC_CONFIDENCE = 0.999

def setup_roma():
    """Set up the Roma environment."""
    
    roma_dir = THIRD_PARTY_DIR / "RoMa"
    
    if not roma_dir.exists():
        print("Cloning RoMa repository...")
        roma_dir.parent.mkdir(exist_ok=True)
        subprocess.run(
            ["git", "clone", "https://github.com/Vincentqyw/RoMa.git", str(roma_dir)],
            check=True
        )
    
    sys.path.append(str(roma_dir))

def download_or_load_checkpoint(filename):
    """
    Download a checkpoint if not in cache, otherwise load from cache.
    
    Args:
        filename: Name of the checkpoint file
        
    Returns:
        Path to the checkpoint file
    """
    # Create checkpoints directory if it doesn't exist
    CHECKPOINTS_DIR.mkdir(exist_ok=True)
    
    # Check if file exists in cache
    cached_file = CHECKPOINTS_DIR / filename
    if cached_file.exists():
        logger.info(f"Loading cached checkpoint: {filename}")
        return cached_file
    
    # Download if not in cache
    logger.info(f"Downloading checkpoint: {filename}")
    
    # Use cache_dir instead of local_dir to avoid nested directory structure
    import tempfile
    with tempfile.TemporaryDirectory() as temp_dir:
        downloaded_file = hf_hub_download(
            repo_type="model",
            repo_id=MODEL_REPO_ID,
            filename=f"roma/{filename}",
            cache_dir=temp_dir,
            local_files_only=False
        )
        # Move the downloaded file to our desired location
        target_path = CHECKPOINTS_DIR / filename
        target_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Copy the file from the temp cache to our persistent location
        import shutil
        shutil.copy2(downloaded_file, target_path)
        logger.info(f"Cached checkpoint to: {target_path}")
    
    return target_path

def filter_matches_with_ransac(keypoints0, keypoints1, confidence):
    """Filter matches using RANSAC homography estimation."""
    kpts0 = keypoints0.astype(np.float32)
    kpts1 = keypoints1.astype(np.float32)
    
    H, mask = cv2.findHomography(
        kpts0, kpts1,
        method=cv2.RANSAC,
        ransacReprojThreshold=RANSAC_REPROJ_THRESHOLD,
        confidence=RANSAC_CONFIDENCE,
    )
    
    mask = mask.ravel().astype(bool)
    return kpts0[mask], kpts1[mask], confidence[mask], mask

class RomaMatcher(nn.Module):
    """GIM(Roma) matcher for feature matching between two images."""
    
    def __init__(self, memory_efficient=None):
        super().__init__()
        
        # Use configuration setting if not explicitly provided
        if memory_efficient is None:
            memory_efficient = config.roma_memory_efficient
        
        # CPU-optimized configuration with multi-threading
        if memory_efficient:
            self.conf = {
                "model_name": "gim_roma_100h.ckpt",
                "model_utils_name": "dinov2_vitl14_pretrain.pth",
                "max_keypoints": config.roma_max_keypoints,
                "coarse_res": config.roma_coarse_res,
                "upsample_res": config.roma_upsample_res,
                "max_image_size": config.roma_max_image_size,
            }
            logger.info(f"🔧 ROMA multi-core CPU mode: {config.roma_max_image_size}px, {config.roma_max_keypoints} keypoints")
        else:
            # High-performance mode
            self.conf = {
                "model_name": "gim_roma_100h.ckpt",
                "model_utils_name": "dinov2_vitl14_pretrain.pth",
                "max_keypoints": 3000,
                "coarse_res": (560, 560),
                "upsample_res": (864, 1152),
                "max_image_size": None,  # No limit
            }
            logger.info("🚀 ROMA high-performance mode: unlimited image size")
        self._init()
        
    def _init(self):
        setup_roma()
        try:
            from third_party.RoMa.romatch.models.model_zoo import roma_model
        except ImportError as e:
            raise RuntimeError(f"Failed to import roma_model: {e}")

        # Load or download model weights
        model_path = download_or_load_checkpoint(self.conf["model_name"])
        dinov2_weights_path = download_or_load_checkpoint(self.conf["model_utils_name"])

        logger.info("Loading GIM(Roma) model")
        
        # Load weights
        weights = torch.load(model_path, map_location="cpu")
        if "state_dict" in weights.keys():
            weights = weights["state_dict"]
        for k in list(weights.keys()):
            if k.startswith("model."):
                weights[k.replace("model.", "", 1)] = weights.pop(k)

        dinov2_weights = torch.load(dinov2_weights_path, map_location="cpu")

        # Set dtype based on device - always use float32 for CPU
        amp_dtype = torch.float32
        logger.info(f"Using device: {DEVICE}, dtype: {amp_dtype}")

        # Initialize Roma model
        self.net = roma_model(
            resolution=self.conf["coarse_res"],
            upsample_preds=True,
            weights=weights,
            dinov2_weights=dinov2_weights,
            device=DEVICE,
            amp_dtype=amp_dtype,
        )
        self.net.upsample_res = self.conf["upsample_res"]
        
        # Ensure model is on CPU and uses float32
        self.net = self.net.to(device=DEVICE, dtype=torch.float32)
        self.net.eval()  # Set to evaluation mode
        
        logger.info("GIM(Roma) model loaded successfully")
        logger.info(f"Model device: {next(self.net.parameters()).device}")
        logger.info(f"Model dtype: {next(self.net.parameters()).dtype}")
        
        # Log memory usage
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        logger.info(f"Memory usage after model loading: {memory_mb:.1f} MB")
    
    def _resize_image_if_needed(self, image, max_size):
        """Resize image if it's larger than max_size while preserving aspect ratio.
        Ensures dimensions are multiples of 14 (ROMA patch size requirement)."""
        W, H = image.size
        max_dim = max(W, H)
        
        if max_dim > max_size:
            scale = max_size / max_dim
            new_W = int(W * scale)
            new_H = int(H * scale)
            
            # Ensure dimensions are multiples of 14 (ROMA patch size)
            new_W = ((new_W + 6) // 14) * 14  # Round to nearest multiple of 14
            new_H = ((new_H + 6) // 14) * 14  # Round to nearest multiple of 14
            
            # Ensure minimum size
            if new_W < 14:
                new_W = 14
            if new_H < 14:
                new_H = 14
            
            logger.info(f"Resizing image from {W}x{H} to {new_W}x{new_H} for memory efficiency (14-aligned)")
            
            # Create resized image and immediately cleanup original reference
            resized = image.resize((new_W, new_H), Image.Resampling.LANCZOS)
            del image  # Free original image memory
            gc.collect()
            return resized
        
        return image

    def match_images(self, image0, image1):
        """
        Match features between two images.
        
        Args:
            image0: First image (PIL Image, path, or numpy array)
            image1: Second image (PIL Image, path, or numpy array)
            
        Returns:
            dict containing:
                keypoints0: Nx2 array of keypoints in first image
                keypoints1: Nx2 array of corresponding keypoints in second image
                confidence: N-length array of match confidence scores
                inlier_ratio: Ratio of matches that passed RANSAC filtering
        """
        # Handle different input types
        if isinstance(image0, str):
            image0 = Image.open(image0).convert('RGB')
        elif isinstance(image0, np.ndarray):
            image0 = Image.fromarray(image0)
            
        if isinstance(image1, str):
            image1 = Image.open(image1).convert('RGB')
        elif isinstance(image1, np.ndarray):
            image1 = Image.fromarray(image1)
        
        # Store original sizes before resizing
        original_W_A, original_H_A = image0.size
        original_W_B, original_H_B = image1.size
        
        # Memory optimization: limit image size
        scale_factor_A = 1.0
        scale_factor_B = 1.0
        
        if self.conf.get("max_image_size"):
            # Check if resizing is needed and calculate scale factors
            max_size = self.conf["max_image_size"]
            max_dim_A = max(original_W_A, original_H_A)
            max_dim_B = max(original_W_B, original_H_B)
            
            if max_dim_A > max_size:
                scale_factor_A = max_size / max_dim_A
            if max_dim_B > max_size:
                scale_factor_B = max_size / max_dim_B
                
            image0 = self._resize_image_if_needed(image0, max_size)
            image1 = self._resize_image_if_needed(image1, max_size)
            
        W_A, H_A = image0.size
        W_B, H_B = image1.size
        
        logger.info(f"Scale factors: A={scale_factor_A:.3f}, B={scale_factor_B:.3f}")
        
        # Validate image dimensions are multiples of 14 (ROMA requirement)
        # Track additional scaling from 14-alignment
        align_scale_A = 1.0
        align_scale_B = 1.0
        
        if W_A % 14 != 0 or H_A % 14 != 0:
            logger.warning(f"Image0 dimensions ({W_A}x{H_A}) not multiple of 14, adjusting...")
            # Adjust to nearest multiple of 14
            new_W_A = ((W_A + 6) // 14) * 14
            new_H_A = ((H_A + 6) // 14) * 14
            align_scale_A = new_W_A / W_A  # Track the additional scaling
            image0 = image0.resize((new_W_A, new_H_A), Image.Resampling.LANCZOS)
            W_A, H_A = new_W_A, new_H_A
            
        if W_B % 14 != 0 or H_B % 14 != 0:
            logger.warning(f"Image1 dimensions ({W_B}x{H_B}) not multiple of 14, adjusting...")
            # Adjust to nearest multiple of 14
            new_W_B = ((W_B + 6) // 14) * 14
            new_H_B = ((H_B + 6) // 14) * 14
            align_scale_B = new_W_B / W_B  # Track the additional scaling
            image1 = image1.resize((new_W_B, new_H_B), Image.Resampling.LANCZOS)
            W_B, H_B = new_W_B, new_H_B
        
        logger.info(f"Final image dimensions: {W_A}x{H_A} and {W_B}x{H_B}")

        # Match with memory optimization
        with torch.no_grad():
            # Aggressive memory cleanup before inference
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Log memory before inference
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            logger.info(f"Memory before ROMA inference: {memory_mb:.1f} MB")
            
            # Check if we have enough memory (fail fast if not)
            if memory_mb > 6000:  # 6GB limit
                raise RuntimeError(f"Memory usage too high before inference: {memory_mb:.1f} MB")
            
            # Ensure we're using CPU and float32 with multi-threading
            cpu_threads = torch.get_num_threads()
            logger.info(f"🚀 Running ROMA inference on {DEVICE} using {cpu_threads} threads")
            
            warp, certainty = self.net.match(image0, image1, device=DEVICE)
            matches, certainty = self.net.sample(
                warp, certainty, num=self.conf["max_keypoints"]
            )
            kpts1, kpts2 = self.net.to_pixel_coordinates(matches, H_A, W_A, H_B, W_B)

        # Convert to numpy
        kpts0_np = kpts1.cpu().numpy()
        kpts1_np = kpts2.cpu().numpy()
        conf_np = certainty.cpu().numpy()
        
        # Filter matches with RANSAC
        kpts0_filtered, kpts1_filtered, conf_filtered, mask = filter_matches_with_ransac(
            kpts0_np, kpts1_np, conf_np
        )
        
        inlier_ratio = mask.sum() / len(mask)
        logger.info(f"RANSAC filtering: kept {mask.sum()}/{len(mask)} matches ({inlier_ratio:.1%} inliers)")

        # Scale keypoints back to original image coordinates
        # Account for both memory-saving resize and 14-alignment resize
        total_scale_A = scale_factor_A * align_scale_A
        total_scale_B = scale_factor_B * align_scale_B
        
        if total_scale_A != 1.0:
            logger.info(f"Scaling image A keypoints by {1/total_scale_A:.3f} (resize: {1/scale_factor_A:.3f}, align: {1/align_scale_A:.3f})")
            kpts0_filtered = kpts0_filtered / total_scale_A
            
        if total_scale_B != 1.0:
            logger.info(f"Scaling image B keypoints by {1/total_scale_B:.3f} (resize: {1/scale_factor_B:.3f}, align: {1/align_scale_B:.3f})")
            kpts1_filtered = kpts1_filtered / total_scale_B

        print(f"ROMA SUCCESS result: {len(kpts0_filtered)} matches at original scale")

        # Aggressive memory cleanup after inference
        del warp, certainty, matches, kpts1, kpts2  # Clear intermediate tensors
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        # Log memory usage after inference
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        logger.info(f"Memory usage after inference and cleanup: {memory_mb:.1f} MB")
        
        return {
            "keypoints0": kpts0_filtered,
            "keypoints1": kpts1_filtered,
            "confidence": conf_filtered,
            "inlier_ratio": inlier_ratio
        }
    
    def cleanup(self):
        """Clean up model and free memory."""
        try:
            if hasattr(self, 'net') and self.net is not None:
                # Move model to CPU and clear
                self.net = self.net.cpu()
                del self.net
                
            # Clear all cached data
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            logger.info("🧹 ROMA model cleaned up successfully")
        except Exception as e:
            logger.warning(f"⚠️ Error during ROMA model cleanup: {e}")
    
    def __del__(self):
        """Ensure cleanup when object is destroyed."""
        self.cleanup()

if __name__ == "__main__":
    # Simple usage example
    matcher = RomaMatcher(memory_efficient=True)  # Use memory-efficient mode
    result = matcher.match_images("source_image_luchtfoto.png", "dest_image_luchtfoto_2022.png")
    print(f"Found {len(result['keypoints0'])} matches with {result['inlier_ratio']:.1%} inlier ratio") 