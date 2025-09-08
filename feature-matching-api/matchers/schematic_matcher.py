"""
Schematic Map Matcher - KAZE/ORB/SIFT matcher optimized for schematic maps
"""
import cv2
import numpy as np
import logging
from typing import Tuple, List

from .base_matcher import BaseFeatureMatcher
from utils.memory_utils import MemoryManager

logger = logging.getLogger(__name__)


class SchematicMapMatcher(BaseFeatureMatcher):
    """KAZE Feature Matcher optimized for schematic maps."""
    
    def __init__(self):
        super().__init__()
        # Lazy loading for memory efficiency
        self.kaze = None
        self.orb = None
        self.sift = None
    
    def _get_kaze_detector(self):
        """Lazy load KAZE detector."""
        if self.kaze is None:
            logger.info("Loading KAZE detector...")
            MemoryManager.log_memory_usage("Before KAZE loading")
            self.kaze = cv2.KAZE_create(
                extended=True,
                upright=True,
                threshold=0.0001,
                nOctaves=6,
                nOctaveLayers=4,
                diffusivity=cv2.KAZE_DIFF_CHARBONNIER
            )
            MemoryManager.log_memory_usage("After KAZE loading")
        return self.kaze
    
    def _get_orb_detector(self):
        """Lazy load ORB detector."""
        if self.orb is None:
            logger.info("Loading ORB detector...")
            self.orb = cv2.ORB_create(
                nfeatures=1000,
                scaleFactor=1.2,
                nlevels=8,
                edgeThreshold=31,
                firstLevel=0,
                WTA_K=2,
                scoreType=cv2.ORB_HARRIS_SCORE,
                patchSize=31,
                fastThreshold=20
            )
        return self.orb
    
    def _get_sift_detector(self):
        """Lazy load SIFT detector."""
        if self.sift is None:
            try:
                logger.info("Loading SIFT detector...")
                self.sift = cv2.SIFT_create(
                    nfeatures=1000,
                    nOctaveLayers=3,
                    contrastThreshold=0.03,
                    edgeThreshold=5,
                    sigma=1.6
                )
            except AttributeError:
                self.sift = None
        return self.sift
    
    def detect_and_compute_multi(self, image: np.ndarray) -> Tuple[List[cv2.KeyPoint], np.ndarray, str]:
        """Try multiple detectors and return the best result."""
        # Enhance edges for better feature detection
        enhanced_img = self.image_processor.enhance_edges(image)
        
        # Try KAZE first (best for schematic maps)
        kaze = self._get_kaze_detector()
        kp, desc = kaze.detectAndCompute(enhanced_img, None)
        if desc is not None and len(kp) >= self.min_match_count:
            logger.info(f"KAZE: {len(kp)} keypoints detected")
            return kp, desc, "KAZE"
        
        # Try ORB as backup
        orb = self._get_orb_detector()
        kp, desc = orb.detectAndCompute(enhanced_img, None)
        if desc is not None and len(kp) >= self.min_match_count:
            logger.info(f"ORB: {len(kp)} keypoints detected")
            return kp, desc, "ORB"
        
        # Try SIFT if available
        sift = self._get_sift_detector()
        if sift is not None:
            kp, desc = sift.detectAndCompute(enhanced_img, None)
            if desc is not None and len(kp) >= self.min_match_count:
                logger.info(f"SIFT: {len(kp)} keypoints detected")
                return kp, desc, "SIFT"
        
        # If all failed, return empty result
        return [], None, "NONE"
