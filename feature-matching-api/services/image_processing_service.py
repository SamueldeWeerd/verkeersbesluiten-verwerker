"""
Image Processing Service - Handles image preprocessing and transformations
"""
import cv2
import numpy as np
import logging
from typing import Tuple

logger = logging.getLogger(__name__)


class ImageProcessingService:
    """Service for image preprocessing and transformations"""
    
    def __init__(self):
        pass
    
    def limit_image_size(self, image: np.ndarray, max_dimension: int = 2048) -> np.ndarray:
        """
        Limit image size to a maximum dimension while preserving aspect ratio.
        
        Args:
            image: Input image (BGR format)
            max_dimension: Maximum width or height allowed
            
        Returns:
            Resized image if needed, or original image if already within limits
        """
        height, width = image.shape[:2]
        max_current = max(height, width)
        
        if max_current <= max_dimension:
            logger.info(f"Image size OK: {width}x{height} (≤ {max_dimension})")
            return image.copy()
        
        # Calculate scale factor to fit within max_dimension
        scale_factor = max_dimension / max_current
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        
        logger.info(f"Limiting source size: {width}x{height} → {new_width}x{new_height}")
        logger.info(f"Scale factor: {scale_factor:.3f} (memory safety)")
        
        # Resize image using high-quality interpolation
        resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        return resized_image
    
    def resize_destination_to_preserve_source_quality(
        self, 
        src_img: np.ndarray, 
        dst_img: np.ndarray, 
        buffer_factor: float = 1.5
    ) -> Tuple[np.ndarray, float]:
        """
        Resize destination image to better accommodate source image warping.
        
        Args:
            src_img: Source image that will be warped
            dst_img: Destination image to be resized
            buffer_factor: Factor to apply as buffer (1.5 = 50% larger)
            
        Returns:
            Tuple of (resized_destination, resize_scale_factor)
        """
        src_height, src_width = src_img.shape[:2]
        dst_height, dst_width = dst_img.shape[:2]
        
        logger.info(f"Destination resize analysis:")
        logger.info(f"  Source: {src_width}x{src_height}")
        logger.info(f"  Original destination: {dst_width}x{dst_height}")
        
        # Calculate target size based on source dimensions with buffer
        min_source_dim = min(src_width, src_height)
        max_source_dim = max(src_width, src_height)
        target_min_dim = int(min_source_dim * buffer_factor)
        target_max_dim = int(max_source_dim * buffer_factor)
        
        # Determine if resizing is needed
        current_min_dim = min(dst_width, dst_height)
        current_max_dim = max(dst_width, dst_height)
        
        if current_min_dim >= target_min_dim and current_max_dim >= target_max_dim:
            logger.info("Destination is already large enough - no resize needed")
            return dst_img.copy(), 1.0
        
        # Calculate scale factor needed
        scale_factor_for_min = target_min_dim / current_min_dim
        scale_factor_for_max = target_max_dim / current_max_dim
        scale_factor = max(scale_factor_for_min, scale_factor_for_max)
        
        # Memory safety limits
        max_safe_scale_factor = 3.0
        max_safe_pixels = 2048 * 2048
        
        if scale_factor > max_safe_scale_factor:
            logger.warning(f"Scale factor {scale_factor:.1f}x too large, limiting to {max_safe_scale_factor:.1f}x")
            scale_factor = max_safe_scale_factor
        
        # Calculate new dimensions
        new_width = int(dst_width * scale_factor)
        new_height = int(dst_height * scale_factor)
        new_pixels = new_width * new_height
        
        # Check pixel limit
        if new_pixels > max_safe_pixels:
            pixel_scale_factor = (max_safe_pixels / new_pixels) ** 0.5
            scale_factor *= pixel_scale_factor
            new_width = int(dst_width * scale_factor)
            new_height = int(dst_height * scale_factor)
            logger.warning(f"Limiting size to {max_safe_pixels/1000000:.1f}M pixels for memory safety")
        
        logger.info(f"Final scale factor: {scale_factor:.3f}")
        logger.info(f"New destination size: {new_width}x{new_height}")
        
        # Resize destination image
        resized_dst = cv2.resize(dst_img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        return resized_dst, scale_factor
    
    def enhance_edges(self, image: np.ndarray) -> np.ndarray:
        """Enhance edges in the image for better feature detection."""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Histogram equalization for better contrast
        equalized = cv2.equalizeHist(gray)
        
        # Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(equalized, (3, 3), 0)
        
        # Unsharp mask for edge enhancement
        gaussian = cv2.GaussianBlur(blurred, (0, 0), 2.0)
        
        # Ensure both images have the same shape
        if blurred.shape != gaussian.shape:
            gaussian = cv2.resize(gaussian, (blurred.shape[1], blurred.shape[0]))
        
        unsharp_mask = cv2.addWeighted(blurred, 1.5, gaussian, -0.5, 0)
        
        # Edge enhancement using morphological operations
        kernel = np.ones((3, 3), np.uint8)
        enhanced = cv2.morphologyEx(unsharp_mask, cv2.MORPH_CLOSE, kernel)
        
        return enhanced
    
    def scale_keypoints(self, keypoints: list, scale_factor: float) -> list:
        """Scale keypoints back to original image coordinates."""
        if scale_factor == 1.0:
            return keypoints
        
        scaled_keypoints = []
        for kp in keypoints:
            scaled_kp = cv2.KeyPoint(
                x=kp.pt[0] / scale_factor,
                y=kp.pt[1] / scale_factor,
                size=kp.size / scale_factor,
                angle=kp.angle,
                response=kp.response,
                octave=kp.octave,
                class_id=kp.class_id
            )
            scaled_keypoints.append(scaled_kp)
        
        return scaled_keypoints