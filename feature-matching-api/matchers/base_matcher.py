"""
Base Feature Matcher - Common functionality for all feature matchers
"""
import cv2
import numpy as np
import logging
from typing import Tuple, List, Optional

from services.image_processing_service import ImageProcessingService
from services.visualization_service import VisualizationService

logger = logging.getLogger(__name__)


class BaseFeatureMatcher:
    """Base class for feature matchers with common functionality."""
    
    def __init__(self):
        self.ratio_threshold = 0.75
        self.ransac_threshold = 5.0
        self.min_match_count = 10
        self.image_processor = ImageProcessingService()
        self.visualization_service = VisualizationService()
    
    def match_features_adaptive(self, desc1: np.ndarray, desc2: np.ndarray, 
                               detector1: str, detector2: str) -> List[cv2.DMatch]:
        """Match features using the appropriate matcher for the detector types."""
        if desc1 is None or desc2 is None:
            return []
        
        # Use appropriate matcher based on descriptor types
        if detector1 in ["KAZE", "SIFT"] and detector2 in ["KAZE", "SIFT"]:
            matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        elif detector1 == "ORB" and detector2 == "ORB":
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        else:
            return []  # Mixed descriptors not supported
        
        # Perform matching with ratio test
        matches = matcher.knnMatch(desc1, desc2, k=2)
        
        # Apply Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < self.ratio_threshold * n.distance:
                    good_matches.append(m)
        
        return good_matches
    
    def estimate_homography(self, kp1: List[cv2.KeyPoint], kp2: List[cv2.KeyPoint], 
                           matches: List[cv2.DMatch]) -> Tuple[Optional[np.ndarray], List[cv2.DMatch], List[cv2.DMatch]]:
        """Estimate homography using RANSAC and return inlier/outlier matches."""
        if len(matches) < self.min_match_count:
            return None, [], matches
        
        # Extract matched points
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        # Estimate homography using RANSAC
        homography, mask = cv2.findHomography(
            src_pts, dst_pts, 
            cv2.RANSAC, 
            self.ransac_threshold,
            maxIters=2000,
            confidence=0.995
        )
        
        if homography is None:
            return None, [], matches
        
        # Separate inliers and outliers
        inlier_matches = []
        outlier_matches = []
        
        for i, match in enumerate(matches):
            if mask[i]:
                inlier_matches.append(match)
            else:
                outlier_matches.append(match)
        
        return homography, inlier_matches, outlier_matches
