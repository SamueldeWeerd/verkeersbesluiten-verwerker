"""
Matching Service - Handles feature matching operations
"""
import os
import shutil
import logging
from typing import Dict, Any, Optional
from matcher import match_schematic_maps

logger = logging.getLogger(__name__)


class MatchingService:
    """Service class for handling feature matching operations"""
    
    def __init__(self):
        pass
    
    def perform_feature_matching(
        self,
        source_image_path: str,
        destination_image_path: str,
        output_dir: str,
        overlay_transparency: float = 0.6,
        map_type: Optional[str] = None
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
        
        # Perform feature matching
        result = match_schematic_maps(
            source_image_path=source_image_path,
            destination_image_path=destination_image_path,
            output_dir=output_dir,
            overlay_transparency=overlay_transparency,
            map_type=map_type
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
                    
                    # Perform feature matching
                    match_result = self.perform_feature_matching(
                        source_image_path=source_image_path,
                        destination_image_path=cut_result["destination_path"],
                        output_dir=buffer_test_dir,
                        overlay_transparency=overlay_transparency,
                        map_type=test_map_type
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
    
    def _check_georeferenced_files(self, output_dir: str) -> list:
        """Check for created georeferenced files"""
        georeferenced_files = []
        
        tif_file = os.path.join(output_dir, "warped_source.tif")
        pgw_file = os.path.join(output_dir, "warped_source.pgw")
        
        if os.path.exists(tif_file):
            georeferenced_files.append(("GeoTIFF", "warped_source.tif"))
        
        if os.path.exists(pgw_file):
            georeferenced_files.append(("World File", "warped_source.pgw"))
        
        return georeferenced_files
    
    def _assess_quality(self, inlier_ratio: float) -> str:
        """Assess the quality of matching based on inlier ratio"""
        if inlier_ratio >= 0.3:
            return "excellent"
        elif inlier_ratio >= 0.2:
            return "good"
        elif inlier_ratio >= 0.1:
            return "fair"
        else:
            return "poor"
    
    def _get_map_types_to_test(self, map_type: str) -> list:
        """Determine which map types to test based on the requested map type"""
        #TODO: Uncomment this in case we want to test both luchtfoto and luchtfoto-2022 for better accuracy
        # if map_type == "luchtfoto":
        #     return ["luchtfoto", "luchtfoto-2022"]
        if map_type == "luchtfoto":
            return ["luchtfoto"]
        else:
            return [map_type]
    
    def _create_buffer_test_dir(self, session_output_dir: str, buffer_meters: int, 
                              test_map_type: str, include_map_type: bool) -> str:
        """Create directory for buffer test results"""
        if include_map_type:
            buffer_test_dir = os.path.join(session_output_dir, f"buffer_{buffer_meters}m_{test_map_type}_cutout")
        else:
            buffer_test_dir = os.path.join(session_output_dir, f"buffer_{buffer_meters}m_cutout")
        
        os.makedirs(buffer_test_dir, exist_ok=True)
        logger.info(f"📁 Saving cutout for {buffer_meters}m {test_map_type} buffer to: {buffer_test_dir}")
        
        return buffer_test_dir
    
    def copy_best_results(self, best_result: Dict[str, Any], session_output_dir: str) -> None:
        """Copy the best test results to the final output directory"""
        logger.info("Copying best results to output directory...")
        best_test_dir = best_result["test_dir"]
        
        # Copy all files from the best test directory to the output directory
        for filename in os.listdir(best_test_dir):
            src_file = os.path.join(best_test_dir, filename)
            dst_file = os.path.join(session_output_dir, filename)
            if os.path.isfile(src_file):
                shutil.copy2(src_file, dst_file)