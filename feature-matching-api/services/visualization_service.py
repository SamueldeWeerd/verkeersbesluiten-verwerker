"""
Visualization Service - Handles creation of feature matching visualizations and analysis
"""
import cv2
import numpy as np
import os
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class VisualizationService:
    """Service for creating feature matching visualizations and analysis"""
    
    def __init__(self):
        pass
    
    def create_all_visualizations(
        self,
        src_img: np.ndarray,
        dst_img: np.ndarray,
        kp1: List[cv2.KeyPoint],
        kp2: List[cv2.KeyPoint],
        matches: List[cv2.DMatch],
        inlier_matches: List[cv2.DMatch],
        outlier_matches: List[cv2.DMatch],
        output_dir: str
    ) -> Dict[str, Any]:
        """
        Create all visualization files for feature matching results.
        
        Args:
            src_img: Source image
            dst_img: Destination image
            kp1: Source keypoints
            kp2: Destination keypoints
            matches: All matches
            inlier_matches: Good matches (inliers)
            outlier_matches: Rejected matches (outliers)
            output_dir: Directory to save visualizations
            
        Returns:
            Dictionary with creation results
        """
        try:
            logger.info(f"Creating visualization files in: {output_dir}")
            logger.info(f"Input validation:")
            logger.info(f"  Source image: {src_img.shape if src_img is not None else 'None'}")
            logger.info(f"  Dest image: {dst_img.shape if dst_img is not None else 'None'}")
            logger.info(f"  Total matches: {len(matches)}")
            logger.info(f"  Inlier matches: {len(inlier_matches)}")
            logger.info(f"  Outlier matches: {len(outlier_matches)}")
            
            results = {}
            
            # Create all matches visualization
            results["all_matches"] = self._create_all_matches_visualization(
                src_img, dst_img, kp1, kp2, matches, output_dir
            )
            
            # Create inlier matches visualization
            results["inlier_matches"] = self._create_inlier_matches_visualization(
                src_img, dst_img, kp1, kp2, inlier_matches, output_dir
            )
            
            # Create outlier matches visualization
            results["outlier_matches"] = self._create_outlier_matches_visualization(
                src_img, dst_img, kp1, kp2, outlier_matches, output_dir
            )
            
            # Create analysis summary
            results["analysis_summary"] = self._create_analysis_summary(
                src_img, dst_img, len(matches), len(inlier_matches), 
                len(outlier_matches), output_dir
            )
            
            # Overall success if at least one visualization was created
            success = any(result.get("success", False) for result in results.values())
            
            return {
                "success": success,
                "visualizations": results,
                "files_created": sum(1 for result in results.values() if result.get("success", False))
            }
            
        except Exception as e:
            logger.error(f"Error creating visualizations: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def _create_all_matches_visualization(
        self,
        src_img: np.ndarray,
        dst_img: np.ndarray,
        kp1: List[cv2.KeyPoint],
        kp2: List[cv2.KeyPoint],
        matches: List[cv2.DMatch],
        output_dir: str
    ) -> Dict[str, Any]:
        """Create visualization showing all feature matches."""
        try:
            all_matches_img = cv2.drawMatches(
                src_img, kp1, dst_img, kp2, matches,
                None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )
            
            all_matches_path = os.path.join(output_dir, "all_feature_matches.png")
            success = cv2.imwrite(all_matches_path, all_matches_img)
            
            if success:
                logger.info(f"All matches visualization saved: {all_matches_path}")
                return {"success": True, "path": all_matches_path}
            else:
                return {"success": False, "error": "Failed to save image"}
                
        except Exception as e:
            logger.error(f"Error creating all matches visualization: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def _create_inlier_matches_visualization(
        self,
        src_img: np.ndarray,
        dst_img: np.ndarray,
        kp1: List[cv2.KeyPoint],
        kp2: List[cv2.KeyPoint],
        inlier_matches: List[cv2.DMatch],
        output_dir: str
    ) -> Dict[str, Any]:
        """Create visualization showing only inlier matches."""
        try:
            if not inlier_matches:
                return {"success": False, "error": "No inlier matches to visualize"}
            
            inlier_matches_img = cv2.drawMatches(
                src_img, kp1, dst_img, kp2, inlier_matches,
                None, matchColor=(0, 255, 0), flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )
            
            inlier_matches_path = os.path.join(output_dir, "inlier_matches.png")
            success = cv2.imwrite(inlier_matches_path, inlier_matches_img)
            
            if success:
                logger.info(f"Inlier matches visualization saved: {inlier_matches_path}")
                return {"success": True, "path": inlier_matches_path}
            else:
                return {"success": False, "error": "Failed to save image"}
                
        except Exception as e:
            logger.error(f"Error creating inlier matches visualization: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def _create_outlier_matches_visualization(
        self,
        src_img: np.ndarray,
        dst_img: np.ndarray,
        kp1: List[cv2.KeyPoint],
        kp2: List[cv2.KeyPoint],
        outlier_matches: List[cv2.DMatch],
        output_dir: str
    ) -> Dict[str, Any]:
        """Create visualization showing only outlier matches."""
        try:
            if not outlier_matches:
                return {"success": False, "error": "No outlier matches to visualize"}
            
            outlier_matches_img = cv2.drawMatches(
                src_img, kp1, dst_img, kp2, outlier_matches,
                None, matchColor=(0, 0, 255), flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )
            
            outlier_matches_path = os.path.join(output_dir, "outlier_matches.png")
            success = cv2.imwrite(outlier_matches_path, outlier_matches_img)
            
            if success:
                logger.info(f"Outlier matches visualization saved: {outlier_matches_path}")
                return {"success": True, "path": outlier_matches_path}
            else:
                return {"success": False, "error": "Failed to save image"}
                
        except Exception as e:
            logger.error(f"Error creating outlier matches visualization: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def _create_analysis_summary(
        self,
        src_img: np.ndarray,
        dst_img: np.ndarray,
        total_matches: int,
        inlier_count: int,
        outlier_count: int,
        output_dir: str
    ) -> Dict[str, Any]:
        """Create a summary image with matching statistics."""
        try:
            # Calculate inlier ratio
            inlier_ratio = inlier_count / total_matches if total_matches > 0 else 0
            
            # Create white background for text
            analysis_height = 400
            analysis_width = 600
            analysis_img = np.ones((analysis_height, analysis_width, 3), dtype=np.uint8) * 255
            
            # Add title
            cv2.putText(analysis_img, "FEATURE MATCHING ANALYSIS", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            
            # Add statistics
            stats_text = [
                f"Source Image: {src_img.shape[1]}x{src_img.shape[0]}",
                f"Destination Image: {dst_img.shape[1]}x{dst_img.shape[0]}",
                f"Total Matches Found: {total_matches}",
                f"Inlier Matches: {inlier_count}",
                f"Outlier Matches: {outlier_count}",
                f"Inlier Ratio: {inlier_ratio:.3f}",
                f"Match Quality: {self._assess_match_quality(inlier_ratio)}"
            ]
            
            y_offset = 100
            for i, text in enumerate(stats_text):
                cv2.putText(analysis_img, text, (50, y_offset + i * 35), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            
            # Add legend
            cv2.putText(analysis_img, "Visualization Legend:", (50, y_offset + 280), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            cv2.putText(analysis_img, "Green lines: Inlier matches (good)", (70, y_offset + 310), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 150, 0), 1)
            cv2.putText(analysis_img, "Red lines: Outlier matches (rejected)", (70, y_offset + 330), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 150), 1)
            
            # Save analysis image
            analysis_path = os.path.join(output_dir, "feature_matching_analysis.png")
            success = cv2.imwrite(analysis_path, analysis_img)
            
            if success:
                logger.info(f"Analysis summary saved: {analysis_path}")
                return {
                    "success": True, 
                    "path": analysis_path,
                    "stats": {
                        "total_matches": total_matches,
                        "inlier_count": inlier_count,
                        "outlier_count": outlier_count,
                        "inlier_ratio": inlier_ratio,
                        "quality": self._assess_match_quality(inlier_ratio)
                    }
                }
            else:
                return {"success": False, "error": "Failed to save analysis image"}
                
        except Exception as e:
            logger.error(f"Error creating analysis summary: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def _assess_match_quality(self, inlier_ratio: float) -> str:
        """Assess the quality of matching based on inlier ratio."""
        if inlier_ratio >= 0.5:
            return "Excellent"
        elif inlier_ratio >= 0.3:
            return "Good"
        elif inlier_ratio >= 0.2:
            return "Fair"
        else:
            return "Poor"
    
    def create_overlay_visualization(
        self,
        src_img: np.ndarray,
        dst_img: np.ndarray,
        homography: np.ndarray,
        alpha: float = 0.6,
        scale_factor: float = 1.0
    ) -> Dict[str, Any]:
        """
        Create warped overlay visualization.
        
        Args:
            src_img: Source image
            dst_img: Destination image
            homography: Homography matrix
            alpha: Transparency factor
            scale_factor: Scale factor for canvas sizing
            
        Returns:
            Dictionary with overlay results
        """
        try:
            height, width = dst_img.shape[:2]
            
            # Create canvas with padding
            padding = 800
            canvas_width = width + 2 * padding
            canvas_height = height + 2 * padding
            
            # Calculate destination offset in canvas
            dst_x_offset = padding
            dst_y_offset = padding
            canvas_to_dest_offset = (dst_x_offset, dst_y_offset)
            
            # Create translation matrix
            translation = np.array([[1, 0, dst_x_offset],
                                   [0, 1, dst_y_offset],
                                   [0, 0, 1]], dtype=np.float32)
            
            # Combine translation with homography
            adjusted_homography = translation @ homography
            
            # Create warped image
            warped_canvas = cv2.warpPerspective(
                src_img,
                adjusted_homography,
                (canvas_width, canvas_height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )
            
            # Create overlay result
            overlay_result = dst_img.copy()
            
            # Extract destination region from warped canvas
            dest_region = warped_canvas[dst_y_offset:dst_y_offset+height, 
                                       dst_x_offset:dst_x_offset+width]
            
            # Create mask for valid pixels
            mask = cv2.cvtColor(dest_region, cv2.COLOR_BGR2GRAY)
            mask = (mask > 0).astype(np.uint8) * 255
            
            # Apply overlay where we have warped content
            if dest_region.shape[:2] == overlay_result.shape[:2]:
                mask_indices = mask > 0
                if np.any(mask_indices):
                    dst_pixels = overlay_result[mask_indices].astype(np.float32)
                    src_pixels = dest_region[mask_indices].astype(np.float32)
                    
                    if dst_pixels.shape == src_pixels.shape:
                        blended_pixels = (dst_pixels * (1 - alpha) + src_pixels * alpha).astype(np.uint8)
                        overlay_result[mask_indices] = blended_pixels
            
            return {
                "success": True,
                "overlay_result": overlay_result,
                "warped_canvas": warped_canvas,
                "canvas_to_dest_offset": canvas_to_dest_offset
            }
            
        except Exception as e:
            logger.error(f"Error creating overlay visualization: {str(e)}")
            return {"success": False, "error": str(e)}