"""
Aerial Imagery Matcher - ROMA-based matcher for aerial and satellite imagery
"""
import cv2
import numpy as np
import logging
from typing import Tuple, List

from memory_config import config

logger = logging.getLogger(__name__)


class AerialImageryMatcher:
    """ROMA-only matcher for aerial and satellite imagery."""
    
    def __init__(self):
        self.roma_model = None
        self.roma_available = False
        self.min_match_count = 4
        self._load_roma_model()
    
    def _load_roma_model(self):
        """Load the ROMA model during initialization."""
        try:
            logger.info("Loading ROMA model with multi-core CPU optimization...")
            from third_party.roma_minimal import RomaMatcher
            
            # Use configuration for optimal CPU performance
            self.roma_model = RomaMatcher()  # Will use config automatically
            self.roma_available = True
            
            # Log multi-core configuration
            mode = "multi-core CPU" if config.roma_memory_efficient else "high-performance"
            logger.info(f"✅ ROMA model loaded successfully ({mode} mode)")
            logger.info(f"🔧 CPU cores: {config.cpu_cores}, PyTorch threads: {config.torch_threads}")
            
        except Exception as e:
            logger.error(f"❌ Error setting up ROMA: {e}")
            self.roma_available = False
            self.roma_model = None
    
    def roma_match_both_images(self, src_img: np.ndarray, dst_img: np.ndarray) -> Tuple[List[cv2.KeyPoint], List[cv2.KeyPoint], List[cv2.DMatch], str]:
        """Use ROMA model for satellite imagery matching."""
        try:
            if not self.roma_available:
                logger.error("ROMA not available")
                return [], [], [], "ROMA_FAILED"
            
            logger.info("Using ROMA for satellite imagery matching...")
            
            # Call ROMA matcher
            result = self.roma_model.match_images(src_img, dst_img)
            
            if result and len(result['keypoints0']) >= 4:
                # Convert ROMA keypoints to cv2.KeyPoint objects
                kp1_coords = result['keypoints0']
                kp2_coords = result['keypoints1']
                
                kp1 = []
                kp2 = []
                matches = []
                
                for i, (src_coords, dst_coords) in enumerate(zip(kp1_coords, kp2_coords)):
                    # Create cv2.KeyPoint objects
                    src_kp = cv2.KeyPoint(x=float(src_coords[0]), y=float(src_coords[1]), size=10.0)
                    dst_kp = cv2.KeyPoint(x=float(dst_coords[0]), y=float(dst_coords[1]), size=10.0)
                    
                    kp1.append(src_kp)
                    kp2.append(dst_kp)
                    
                    # Create cv2.DMatch object - ROMA keypoints are already matched
                    match = cv2.DMatch(_queryIdx=i, _trainIdx=i, _distance=0.0)
                    matches.append(match)
                
                logger.info(f"ROMA matching successful! {len(matches)} matches")
                return kp1, kp2, matches, "ROMA"
            else:
                logger.error("ROMA matching failed - insufficient matches")
                return [], [], [], "ROMA_FAILED"
                
        except Exception as e:
            logger.error(f"ROMA matching error: {e}")
            return [], [], [], "ROMA_FAILED"
