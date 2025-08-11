"""
Feature Matching Service - Core feature detection and matching logic
"""
import cv2
import numpy as np
import os
import shutil
import logging
from typing import Tuple, List, Optional, Dict, Any
from dataclasses import dataclass

# Import our utilities and services
from services.image_processing_service import ImageProcessingService
from services.georeferencing_service import GeoreferencingService
from services.visualization_service import VisualizationService
from utils.memory_utils import MemoryManager
from memory_config import config

logger = logging.getLogger(__name__)


@dataclass
class FeatureMatchingResult:
    """Container for feature matching results."""
    success: bool
    warped_overlay: Optional[np.ndarray] = None
    warped_source: Optional[np.ndarray] = None
    homography: Optional[np.ndarray] = None
    matches_count: int = 0
    inlier_ratio: float = 0.0
    error_message: str = ""


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
        map_type: Optional[str] = None
    ) -> FeatureMatchingResult:
        """
        Main feature matching function with modular architecture.
        
        Args:
            source_image_path: Path to source image
            destination_image_path: Path to destination image
            output_dir: Optional output directory
            overlay_transparency: Overlay transparency
            map_type: Map type for coordinate system determination
            
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
            
            # Select and use appropriate matcher
            matcher_type = self.select_matcher_for_map_type(map_type)
            logger.info(f"Selected matcher type: {matcher_type}")
            
            if matcher_type == 'aerial':
                # DEBUG: Check coordinate transformations BEFORE ROMA
                from services.coordinate_transformation_service import CoordinateTransformationService
                coord_service_before = CoordinateTransformationService()
                logger.info(f"DEBUG: Coordinate transformations available BEFORE ROMA: {coord_service_before.is_available()}")
                
                matcher = AerialImageryMatcher()
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
                matcher = SchematicMapMatcher()
                # Traditional feature detection and matching
                kp1, desc1, detector1 = matcher.detect_and_compute_multi(source_img)
                kp2, desc2, detector2 = matcher.detect_and_compute_multi(destination_img)
                
                if desc1 is None or desc2 is None:
                    return FeatureMatchingResult(success=False, error_message="Feature detection failed")
                
                matches = matcher.match_features_adaptive(desc1, desc2, detector1, detector2)
            
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
        
        # Perform feature matching using the refactored service
        result = self.match_images(
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