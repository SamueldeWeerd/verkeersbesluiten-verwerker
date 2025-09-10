"""
Feature Matching Service - Core feature detection and matching logic
"""
import cv2
import numpy as np
import os
import shutil
import logging
from typing import Tuple, List, Optional, Dict, Any

# Import our utilities and services
from services.image_processing_service import ImageProcessingService
from services.georeferencing_service import GeoreferencingService
from services.visualization_service import VisualizationService
from utils.memory_utils import MemoryManager
from memory_config import config
from models.response_models import FeatureMatchingResult
from matchers import BaseFeatureMatcher, SchematicMapMatcher, AerialImageryMatcher

logger = logging.getLogger(__name__)


class FeatureMatchingService:
    """Main service that orchestrates the feature matching process."""
    
    def __init__(self):
        self.image_processor = ImageProcessingService()
        self.georeferencing_service = GeoreferencingService()
        self.visualization_service = VisualizationService()
        self.memory_manager = MemoryManager()
        
        # Log configuration on first initialization
        if not hasattr(FeatureMatchingService, '_config_logged'):
            config.log_config(logger)
            FeatureMatchingService._config_logged = True
    
    def select_matcher_for_map_type(self, map_type: Optional[str]) -> str:
        """Select the appropriate matcher based on map type."""
        aerial_types = {'luchtfoto', 'luchtfoto-2022', 'satellite', 'satellite-prev'}
        
        if map_type and map_type.lower() in aerial_types:
            return 'aerial'
        else:
            return 'schematic'
    
    def match_images(
        self,
        source_image_path: str,
        destination_image_path: str,
        output_dir: Optional[str] = None,
        overlay_transparency: float = 0.6,
        map_type: Optional[str] = None,
        reuse_matcher: Optional[object] = None
    ) -> FeatureMatchingResult:
        """
        Main feature matching function with modular architecture.
        
        Args:
            source_image_path: Path to source image
            destination_image_path: Path to destination image
            output_dir: Optional output directory
            overlay_transparency: Overlay transparency
            map_type: Map type for coordinate system determination
            reuse_matcher: Optional pre-created matcher to reuse (for performance)
            
        Returns:
            FeatureMatchingResult object
        """
        try:
            # Validate and load images
            if not os.path.exists(source_image_path):
                return FeatureMatchingResult(success=False, error_message=f"Source image not found: {source_image_path}")
            
            if not os.path.exists(destination_image_path):
                return FeatureMatchingResult(success=False, error_message=f"Destination image not found: {destination_image_path}")
            
            # Load images
            logger.info(f"Loading images: {source_image_path}, {destination_image_path}")
            source_img = cv2.imread(source_image_path)
            original_destination_img = cv2.imread(destination_image_path)
            
            if source_img is None or original_destination_img is None:
                return FeatureMatchingResult(success=False, error_message="Failed to load images")
            
            # Limit source image size for memory safety
            source_img = self.image_processor.limit_image_size(source_img, max_dimension=2048)
            
            # Resize destination to preserve source quality
            destination_img, dest_resize_factor = self.image_processor.resize_destination_to_preserve_source_quality(
                source_img, original_destination_img, buffer_factor=1.5
            )
            
            # Import coordinate service (needed for debug logging)
            from services.coordinate_transformation_service import CoordinateTransformationService
            
            # Select and use appropriate matcher (reuse if provided)
            matcher_type = self.select_matcher_for_map_type(map_type)
            should_cleanup_matcher = False  # Track if we need to cleanup
            
            if reuse_matcher is not None:
                # Use the provided matcher (for performance optimization)
                matcher = reuse_matcher
                logger.info(f"♻️ Reusing existing {matcher_type} matcher")
            elif matcher_type == 'aerial':
                # Create new aerial matcher
                logger.info("🆕 Creating new aerial matcher")
                should_cleanup_matcher = True
                
                # DEBUG: Check coordinate transformations BEFORE ROMA
                coord_service_before = CoordinateTransformationService()
                logger.info(f"DEBUG: Coordinate transformations available BEFORE ROMA: {coord_service_before.is_available()}")
                
                matcher = AerialImageryMatcher()
                # Log ROMA matcher status
                if matcher.roma_available:
                    logger.info("✅ ROMA matcher initialized successfully")
                else:
                    logger.error("❌ ROMA matcher initialization failed")
                    return FeatureMatchingResult(success=False, error_message="ROMA matcher not available for aerial imagery")
            else:
                # Create new schematic matcher
                logger.info("🆕 Creating new schematic matcher")
                should_cleanup_matcher = True
                matcher = SchematicMapMatcher()

            # Perform matching based on matcher type
            try:
                if matcher_type == 'aerial':
                    kp1, kp2, matches, detector_name = matcher.roma_match_both_images(source_img, destination_img)
                    
                    # DEBUG: Check coordinate transformations AFTER ROMA
                    coord_service_after = CoordinateTransformationService()
                    logger.info(f"DEBUG: Coordinate transformations available AFTER ROMA: {coord_service_after.is_available()}")
                    
                    # DEBUG: Test actual transformation
                    try:
                        test_lat, test_lon = coord_service_after.rd_to_latlon(155000, 463000)
                        logger.info(f"DEBUG: Test transformation successful: RD(155000,463000) -> WGS84({test_lat:.6f},{test_lon:.6f})")
                    except Exception as e:
                        logger.error(f"DEBUG: Test transformation FAILED after ROMA: {e}")
                        
                else:
                    # Traditional feature detection and matching
                    kp1, desc1, detector1 = matcher.detect_and_compute_multi(source_img)
                    kp2, desc2, detector2 = matcher.detect_and_compute_multi(destination_img)
                    
                    if desc1 is None or desc2 is None:
                        return FeatureMatchingResult(success=False, error_message="Feature detection failed")
                    
                    matches = matcher.match_features_adaptive(desc1, desc2, detector1, detector2)
                        
            finally:
                # Only cleanup if we created the matcher (not if it was reused)
                if should_cleanup_matcher and matcher_type == 'aerial':
                    matcher.cleanup()
                    logger.info("🧹 Aerial matcher cleaned up after use")
            
            if len(matches) < 10:  # Use base matcher's min count
                return FeatureMatchingResult(success=False, error_message=f"Insufficient matches: {len(matches)}")
            
            # Estimate homography
            base_matcher = BaseFeatureMatcher()
            homography, inlier_matches, outlier_matches = base_matcher.estimate_homography(kp1, kp2, matches)
            
            if homography is None:
                return FeatureMatchingResult(success=False, error_message="Failed to estimate homography")
            
            # Calculate inlier ratio and quality check
            inlier_ratio = len(inlier_matches) / len(matches) if len(matches) > 0 else 0
            if inlier_ratio < 0.1:
                return FeatureMatchingResult(success=False, error_message=f"Poor match quality: {inlier_ratio:.3f}")
            
            # Create overlay
            alpha = 1.0 - overlay_transparency
            overlay_result = self.visualization_service.create_overlay_visualization(
                source_img, destination_img, homography, alpha=alpha, scale_factor=dest_resize_factor
            )
            
            if not overlay_result["success"]:
                return FeatureMatchingResult(success=False, error_message="Failed to create overlay")
            
            # Save results if output directory provided
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                
                # Save main output images
                cv2.imwrite(os.path.join(output_dir, "warped_overlay_result.png"), overlay_result["overlay_result"])
                
                # Create visualizations
                self.visualization_service.create_all_visualizations(
                    source_img, destination_img, kp1, kp2, matches, inlier_matches, outlier_matches, output_dir
                )
                
                # Handle georeferencing if PGW file exists
                dest_base = os.path.splitext(destination_image_path)[0]
                dest_pgw_path = f"{dest_base}.pgw"
                
                if os.path.exists(dest_pgw_path):
                    logger.info("PGW file found, creating georeferenced GeoTIFF")
                    try:
                        # Extract the canvas offset information from the overlay result
                        canvas_to_dest_offset = overlay_result.get("canvas_to_dest_offset", (800, 800))
                        
                        # Load destination image to get dimensions
                        dest_img = cv2.imread(destination_image_path)
                        dest_height, dest_width = dest_img.shape[:2]
                        
                        # Extract only the valid warped region (crop out black borders)
                        warped_canvas = overlay_result["warped_canvas"]
                        canvas_x_offset, canvas_y_offset = canvas_to_dest_offset
                        
                        # Crop the warped canvas to remove black padding and get only the transformed source
                        # Find the bounding box of non-black pixels in the warped canvas
                        gray_canvas = cv2.cvtColor(warped_canvas, cv2.COLOR_BGR2GRAY)
                        non_zero_coords = cv2.findNonZero(gray_canvas)
                        
                        if non_zero_coords is not None:
                            # Get bounding rectangle of non-zero pixels
                            x, y, w, h = cv2.boundingRect(non_zero_coords)
                            cropped_warped_image = warped_canvas[y:y+h, x:x+w]
                            crop_offset_in_canvas = (x, y)
                            
                            logger.info(f"Cropped warped image to {w}x{h} from {x},{y} in canvas")
                        else:
                            # Fallback: use destination region if no valid pixels found
                            logger.warning("No valid pixels found in warped canvas, using destination region")
                            cropped_warped_image = warped_canvas[canvas_y_offset:canvas_y_offset+dest_height, 
                                                               canvas_x_offset:canvas_x_offset+dest_width]
                            crop_offset_in_canvas = (canvas_x_offset, canvas_y_offset)
                        
                        # Create georeferenced outputs using the georeferencing service
                        georef_result = self.georeferencing_service.create_georeferenced_outputs(
                            warped_image=cropped_warped_image,
                            warped_full_canvas=overlay_result["warped_canvas"], 
                            output_dir=output_dir,
                            dest_image_path=destination_image_path,
                            dest_pgw_path=dest_pgw_path,
                            homography=homography,
                            crop_offset=crop_offset_in_canvas,  # Actual crop position in canvas
                            canvas_to_dest_offset=canvas_to_dest_offset,  # Canvas offset relative to destination
                            dest_resize_factor=dest_resize_factor,
                            map_type=map_type
                        )
                        
                        if georef_result.get("success"):
                            logger.info("✅ GeoTIFF created successfully")
                        else:
                            logger.warning(f"⚠️ GeoTIFF creation failed: {georef_result.get('error', 'Unknown error')}")
                            
                    except Exception as e:
                        logger.error(f"❌ Error creating GeoTIFF: {str(e)}")
                        # Continue execution even if GeoTIFF creation fails
            
            # Clean up memory
            self.memory_manager.memory_cleanup()
            
            return FeatureMatchingResult(
                success=True,
                warped_overlay=overlay_result["overlay_result"],
                warped_source=overlay_result["warped_canvas"],
                homography=homography,
                matches_count=len(matches),
                inlier_ratio=inlier_ratio
            )
            
        except Exception as e:
            logger.error(f"Feature matching failed: {str(e)}")
            self.memory_manager.memory_cleanup()
            return FeatureMatchingResult(success=False, error_message=f"Feature matching failed: {str(e)}")


    def perform_feature_matching(
        self,
        source_image_path: str,
        destination_image_path: str,
        output_dir: str,
        overlay_transparency: float = 0.6,
        map_type: Optional[str] = None,
        reuse_matcher: Optional[object] = None
    ) -> Dict[str, Any]:
        """
        Perform feature matching between two images
        
        Args:
            source_image_path: Path to source image
            destination_image_path: Path to destination image  
            output_dir: Directory to save outputs
            overlay_transparency: Transparency for overlay (0.0-1.0)
            map_type: Optional map type for coordinate system determination
            
        Returns:
            Dictionary containing matching results and metadata
        """
        logger.info(f"Starting feature matching: {os.path.basename(source_image_path)} -> {os.path.basename(destination_image_path)}")
        
        # Perform feature matching using the refactored service
        result = self.match_images(
            source_image_path=source_image_path,
            destination_image_path=destination_image_path,
            output_dir=output_dir,
            overlay_transparency=overlay_transparency,
            map_type=map_type,
            reuse_matcher=reuse_matcher
        )
        
        if not result.success:
            return {
                "success": False,
                "error_message": result.error_message,
                "matches_count": 0,
                "inlier_ratio": 0.0
            }
        
        # Check for georeferenced files
        georeferenced_files = self._check_georeferenced_files(output_dir)
        
        return {
            "success": True,
            "matches_count": result.matches_count,
            "inlier_ratio": result.inlier_ratio,
            "georeferenced": len(georeferenced_files) > 0,
            "georeferenced_files": georeferenced_files,
            "quality_status": self._assess_quality(result.inlier_ratio)
        }
    
    def test_multiple_buffers(
        self,
        source_image_path: str,
        geometry_input: Any,
        map_type: str,
        test_buffer_sizes: list,
        overlay_transparency: float,
        session_output_dir: str,
        map_cutting_service
    ) -> Dict[str, Any]:
        """
        Test multiple buffer sizes and find the best one for feature matching
        
        Args:
            source_image_path: Path to source image
            geometry_input: Geometry for map cutting
            map_type: Type of map to cut
            test_buffer_sizes: List of buffer sizes to test
            overlay_transparency: Transparency for overlay
            session_output_dir: Session output directory
            map_cutting_service: Instance of MapCuttingService
            
        Returns:
            Dictionary containing best result and all test results
        """
        logger.info(f"Testing buffer sizes: {test_buffer_sizes}")
        
        # Create matcher once for reuse across all buffer tests (performance optimization)
        matcher_type = self.select_matcher_for_map_type(map_type)
        reusable_matcher = None
        
        if matcher_type == 'aerial':
            logger.info("🚀 Creating reusable ROMA matcher for all buffer tests")
            reusable_matcher = AerialImageryMatcher()
            if not reusable_matcher.roma_available:
                logger.error("❌ ROMA matcher initialization failed")
                return {
                    "success": False,
                    "error_message": "ROMA matcher not available for aerial imagery"
                }
        elif matcher_type == 'schematic':
            logger.info("🚀 Creating reusable schematic matcher for all buffer tests")
            reusable_matcher = SchematicMapMatcher()
        
        best_result = None
        best_buffer = None
        best_inlier_count = 0
        buffer_results = []
        
        for buffer_meters in test_buffer_sizes:
            logger.info(f"\n--- Testing buffer size: {buffer_meters}m ---")
            
            # Determine map types to test
            map_types_to_test = self._get_map_types_to_test(map_type)
            
            for test_map_type in map_types_to_test:
                logger.info(f"\n--- Testing {test_map_type} with buffer size: {buffer_meters}m ---")
                
                # Create test directory
                buffer_test_dir = self._create_buffer_test_dir(
                    session_output_dir, buffer_meters, test_map_type, len(map_types_to_test) > 1
                )
                
                try:
                    # Cut map with this buffer
                    cut_result = map_cutting_service.cut_map_for_matching(
                        geometry_input, test_map_type, buffer_meters, buffer_test_dir
                    )
                    
                    if not cut_result["success"]:
                        logger.warning(f"Map cutting failed for {buffer_meters}m {test_map_type} buffer: {cut_result['error_message']}")
                        continue
                    
                    # Perform feature matching with reusable matcher
                    match_result = self.perform_feature_matching(
                        source_image_path=source_image_path,
                        destination_image_path=cut_result["destination_path"],
                        output_dir=buffer_test_dir,
                        overlay_transparency=overlay_transparency,
                        map_type=test_map_type,
                        reuse_matcher=reusable_matcher
                    )
                    
                    if match_result["success"]:
                        inlier_count = int(match_result["matches_count"] * match_result["inlier_ratio"])
                        logger.info(f"Buffer {buffer_meters}m {test_map_type}: {match_result['matches_count']} matches, {inlier_count} inliers (ratio: {match_result['inlier_ratio']:.3f})")
                        
                        # Store result
                        buffer_result = {
                            "buffer_meters": buffer_meters,
                            "map_type": test_map_type,
                            "matches_count": match_result["matches_count"],
                            "inlier_count": inlier_count,
                            "inlier_ratio": match_result["inlier_ratio"],
                            "cut_result": cut_result,
                            "match_result": match_result,
                            "test_dir": buffer_test_dir
                        }
                        buffer_results.append(buffer_result)
                        
                        # Check if this is the best result
                        if inlier_count > best_inlier_count:
                            best_inlier_count = inlier_count
                            best_result = buffer_result
                            best_buffer = buffer_meters
                            logger.info(f"🏆 New best buffer: {buffer_meters}m {test_map_type} with {inlier_count} inliers")
                    else:
                        logger.warning(f"Feature matching failed for {buffer_meters}m {test_map_type} buffer: {match_result.get('error_message', 'Unknown error')}")
                        
                except Exception as e:
                    logger.warning(f"Error testing {buffer_meters}m {test_map_type} buffer: {str(e)}")
                    continue
        
        # Cleanup the reusable matcher after all buffer tests
        try:
            if reusable_matcher is not None and matcher_type == 'aerial':
                reusable_matcher.cleanup()
                logger.info("🧹 Reusable ROMA matcher cleaned up after all buffer tests")
        except Exception as e:
            logger.warning(f"⚠️ Error cleaning up reusable matcher: {e}")
        
        if best_result is None:
            return {
                "success": False,
                "error_message": "Feature matching failed for all tested buffer sizes"
            }
        
        logger.info(f"\n🎯 Selected best buffer: {best_buffer}m with {best_inlier_count} inliers")
        
        return {
            "success": True,
            "best_result": best_result,
            "best_buffer": best_buffer,
            "buffer_results": buffer_results
        }
    
    def copy_best_results(self, best_result: Dict[str, Any], session_output_dir: str) -> None:
        """Copy the best results to the final output directory"""
        logger.info("Copying best results to output directory...")
        best_test_dir = best_result["test_dir"]
        
        # Copy all files from the best test directory to the output directory
        for filename in os.listdir(best_test_dir):
            src_file = os.path.join(best_test_dir, filename)
            dst_file = os.path.join(session_output_dir, filename)
            if os.path.isfile(src_file):
                shutil.copy2(src_file, dst_file)
    
    def _check_georeferenced_files(self, output_dir: str) -> list:
        """Check for georeferenced files in output directory"""
        georeferenced_files = []
        
        # Check for common georeferenced file types
        file_extensions = ['.tif', '.tiff', '.pgw', '.prj']
        
        for filename in os.listdir(output_dir):
            file_path = os.path.join(output_dir, filename)
            if os.path.isfile(file_path):
                for ext in file_extensions:
                    if filename.lower().endswith(ext):
                        georeferenced_files.append(filename)
                        break
        
        return georeferenced_files
    
    def _assess_quality(self, inlier_ratio: float) -> str:
        """Assess the quality of feature matching based on inlier ratio"""
        if inlier_ratio >= 0.7:
            return "excellent"
        elif inlier_ratio >= 0.5:
            return "good"
        elif inlier_ratio >= 0.3:
            return "fair"
        else:
            return "poor"
    
    def _get_map_types_to_test(self, map_type: str) -> list:
        """Get map types to test based on the requested map type"""
        #TODO: Uncomment this in case we want to test both luchtfoto and luchtfoto-2022 for better accuracy
        # if map_type == "luchtfoto":
        #     return ["luchtfoto", "luchtfoto-2022"]
        if map_type == "luchtfoto":
            return ["luchtfoto"]
        else:
            return [map_type]
    
    def _create_buffer_test_dir(self, session_output_dir: str, buffer_meters: int,
                              test_map_type: str, multiple_types: bool) -> str:
        """Create directory for buffer test results"""
        if multiple_types:
            # Include map type in directory name when testing multiple
            buffer_test_dir = os.path.join(session_output_dir, f"buffer_{buffer_meters}m_{test_map_type}_cutout")
        else:
            buffer_test_dir = os.path.join(session_output_dir, f"buffer_{buffer_meters}m_cutout")
        
        os.makedirs(buffer_test_dir, exist_ok=True)
        logger.info(f"📁 Saving cutout for {buffer_meters}m {test_map_type} buffer to: {buffer_test_dir}")
        
        return buffer_test_dir


# For backward compatibility
def match_schematic_maps(
    source_image_path: str,
    destination_image_path: str,
    output_dir: Optional[str] = None,
    overlay_transparency: float = 0.6,
    map_type: Optional[str] = None
) -> FeatureMatchingResult:
    """Backward compatibility function."""
    service = FeatureMatchingService()
    return service.match_images(source_image_path, destination_image_path, output_dir, overlay_transparency, map_type)