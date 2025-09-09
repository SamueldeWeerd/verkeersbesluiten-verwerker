"""
Georeferencing Service - Handles coordinate system transformations and georeferencing
"""
import os
import cv2
import numpy as np
import logging
from typing import Tuple, Optional, Dict, Any

logger = logging.getLogger(__name__)

# Optional rasterio for GeoTIFF creation
try:
    import rasterio
    from rasterio.transform import Affine
    from rasterio.crs import CRS
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False


class GeoreferencingService:
    """Service for handling georeferencing and coordinate transformations"""
    
    def __init__(self):
        pass
    
    def create_georeferenced_outputs(
        self,
        warped_image: np.ndarray,
        warped_full_canvas: np.ndarray,
        output_dir: str,
        dest_image_path: str,
        dest_pgw_path: str,
        homography: np.ndarray,
        crop_offset: Tuple[int, int],
        canvas_to_dest_offset: Tuple[int, int],
        dest_resize_factor: float,
        map_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create georeferenced PGW and optionally GeoTIFF outputs.
        
        Args:
            warped_image: The warped source image (cropped)
            warped_full_canvas: The full warped canvas
            output_dir: Output directory
            dest_image_path: Path to destination image
            dest_pgw_path: Path to destination PGW file
            homography: Homography matrix
            crop_offset: Offset of crop within canvas
            canvas_to_dest_offset: Offset of canvas relative to destination
            dest_resize_factor: Factor by which destination was resized
            map_type: Map type for coordinate system determination
            
        Returns:
            Dictionary with georeferencing results
        """
        try:
            logger.info("Starting georeferencing process")
            logger.info(f"Destination resize factor: {dest_resize_factor:.3f}")
        
            
            # Load original destination image dimensions
            original_dest_img = cv2.imread(dest_image_path)
            if original_dest_img is None:
                return {"success": False, "error": "Failed to load destination image"}
            
            original_dest_height, original_dest_width = original_dest_img.shape[:2]
            
            # Read and parse destination PGW file
            pgw_params = self._read_pgw_file(dest_pgw_path)
            if not pgw_params["success"]:
                return pgw_params
            
            # Calculate adjusted PGW parameters for resized destination
            adjusted_params = self._adjust_pgw_for_resize(pgw_params, dest_resize_factor)
            
            # Calculate world coordinates for warped image corners
            world_coords_result = self._calculate_world_coordinates(
                warped_image, warped_full_canvas, homography,
                crop_offset, canvas_to_dest_offset,
                adjusted_params, original_dest_width, original_dest_height
            )
            
            if not world_coords_result["success"]:
                return world_coords_result
            
            # Create output PGW file
            pgw_result = self._create_pgw_file(
                output_dir, world_coords_result["world_corners"],
                warped_image.shape, adjusted_params["rotation_x"], adjusted_params["rotation_y"]
            )
            
            # Create GeoTIFF if available
            geotiff_created = False
            if RASTERIO_AVAILABLE:
                geotiff_result = self._create_geotiff(
                    warped_image, output_dir, pgw_result["pixel_x_size"],
                    pgw_result["pixel_y_size"], pgw_result["top_left_x"],
                    pgw_result["top_left_y"], adjusted_params["rotation_x"],
                    adjusted_params["rotation_y"], map_type
                )
                geotiff_created = geotiff_result["success"]
            
            return {
                "success": True,
                "pgw_created": pgw_result["success"],
                "geotiff_created": geotiff_created,
                "world_bounds": world_coords_result["world_corners"],
                "pixel_size": {
                    "x": pgw_result["pixel_x_size"],
                    "y": pgw_result["pixel_y_size"]
                }
            }
            
        except Exception as e:
            logger.error(f"Georeferencing failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def _read_pgw_file(self, pgw_path: str) -> Dict[str, Any]:
        """Read and parse PGW file parameters."""
        try:
            with open(pgw_path, 'r') as f:
                lines = f.readlines()
            
            if len(lines) < 6:
                return {"success": False, "error": "Invalid PGW file format"}
            
            return {
                "success": True,
                "pixel_x_size": float(lines[0].strip()),
                "rotation_y": float(lines[1].strip()),
                "rotation_x": float(lines[2].strip()),
                "pixel_y_size": float(lines[3].strip()),
                "top_left_x": float(lines[4].strip()),
                "top_left_y": float(lines[5].strip())
            }
            
        except Exception as e:
            return {"success": False, "error": f"Failed to read PGW file: {str(e)}"}
    
    def _adjust_pgw_for_resize(self, pgw_params: Dict[str, Any], resize_factor: float) -> Dict[str, Any]:
        """Adjust PGW parameters for resized destination image."""
        return {
            "pixel_x_size": pgw_params["pixel_x_size"] / resize_factor,
            "pixel_y_size": pgw_params["pixel_y_size"] / resize_factor,
            "rotation_x": pgw_params["rotation_x"],
            "rotation_y": pgw_params["rotation_y"],
            "top_left_x": pgw_params["top_left_x"],
            "top_left_y": pgw_params["top_left_y"]
        }
    
    def _calculate_world_coordinates(
        self,
        warped_image: np.ndarray,
        warped_full_canvas: np.ndarray,
        homography: np.ndarray,
        crop_offset: Tuple[int, int],
        canvas_to_dest_offset: Tuple[int, int],
        adjusted_params: Dict[str, Any],
        original_dest_width: int,
        original_dest_height: int
    ) -> Dict[str, Any]:
        """Calculate world coordinates for warped image corners."""
        try:
            warped_height, warped_width = warped_image.shape[:2]
            crop_x, crop_y = crop_offset
            canvas_x_offset, canvas_y_offset = canvas_to_dest_offset
            
            # Create translation matrix
            translation = np.array([[1, 0, canvas_x_offset],
                                   [0, 1, canvas_y_offset],
                                   [0, 0, 1]], dtype=np.float32)
            
            adjusted_homography = translation @ homography
            
            # Define corners of cropped warped image in canvas coordinates
            warped_corners_canvas = np.array([
                [crop_x, crop_y, 1],
                [crop_x + warped_width, crop_y, 1],
                [crop_x, crop_y + warped_height, 1],
                [crop_x + warped_width, crop_y + warped_height, 1]
            ], dtype=np.float32).T
            
            # Find source coordinates by inverting the adjusted homography
            adjusted_homography_inv = np.linalg.inv(adjusted_homography)
            source_corners_homogeneous = adjusted_homography_inv @ warped_corners_canvas
            
            # Convert from homogeneous to regular coordinates
            source_corners = []
            for i in range(4):
                x = source_corners_homogeneous[0, i] / source_corners_homogeneous[2, i]
                y = source_corners_homogeneous[1, i] / source_corners_homogeneous[2, i]
                source_corners.append((x, y))
            
            # Map source corners to destination coordinates using original homography
            source_corners_homogeneous = np.array([
                [source_corners[0][0], source_corners[0][1], 1],
                [source_corners[1][0], source_corners[1][1], 1],
                [source_corners[2][0], source_corners[2][1], 1],
                [source_corners[3][0], source_corners[3][1], 1]
            ], dtype=np.float32).T
            
            dest_corners_homogeneous = homography @ source_corners_homogeneous
            
            # Convert to regular coordinates (resized destination coordinate system)
            dest_corners_resized = []
            for i in range(4):
                x = dest_corners_homogeneous[0, i] / dest_corners_homogeneous[2, i]
                y = dest_corners_homogeneous[1, i] / dest_corners_homogeneous[2, i]
                dest_corners_resized.append((x, y))
            
            # Convert destination coordinates to world coordinates
            world_corners = []
            for dest_x, dest_y in dest_corners_resized:
                world_x = (adjusted_params["top_left_x"] + 
                          (dest_x * adjusted_params["pixel_x_size"]) + 
                          (dest_y * adjusted_params["rotation_x"]))
                world_y = (adjusted_params["top_left_y"] + 
                          (dest_x * adjusted_params["rotation_y"]) + 
                          (dest_y * adjusted_params["pixel_y_size"]))
                world_corners.append((world_x, world_y))
            
            return {
                "success": True,
                "world_corners": world_corners,
                "source_corners": source_corners,
                "dest_corners": dest_corners_resized
            }
            
        except np.linalg.LinAlgError:
            return {"success": False, "error": "Could not invert homography matrix"}
        except Exception as e:
            return {"success": False, "error": f"Coordinate calculation failed: {str(e)}"}
    
    def _create_pgw_file(
        self, 
        output_dir: str, 
        world_corners: list, 
        image_shape: tuple, 
        rotation_x: float, 
        rotation_y: float
    ) -> Dict[str, Any]:
        """Create PGW world file for the warped image."""
        try:
            height, width = image_shape[:2]
            
            # Calculate pixel sizes
            world_width = world_corners[1][0] - world_corners[0][0]  # Top-right X - Top-left X
            world_height = world_corners[2][1] - world_corners[0][1]  # Bottom-left Y - Top-left Y
            
            pixel_x_size = world_width / width
            pixel_y_size = world_height / height
            
            # Use top-left world coordinates as the origin
            top_left_x = world_corners[0][0]
            top_left_y = world_corners[0][1]
            
            # Create output PGW file
            output_pgw_path = os.path.join(output_dir, "warped_source.pgw")
            
            with open(output_pgw_path, 'w') as f:
                f.write(f"{pixel_x_size}\n")
                f.write(f"{rotation_y}\n")
                f.write(f"{rotation_x}\n")
                f.write(f"{pixel_y_size}\n")
                f.write(f"{top_left_x}\n")
                f.write(f"{top_left_y}\n")
            
            logger.info(f"Created PGW file: {output_pgw_path}")
            
            return {
                "success": True,
                "pixel_x_size": pixel_x_size,
                "pixel_y_size": pixel_y_size,
                "top_left_x": top_left_x,
                "top_left_y": top_left_y
            }
            
        except Exception as e:
            logger.error(f"Failed to create PGW file: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def _create_geotiff(
        self,
        image: np.ndarray,
        output_dir: str,
        pixel_x_size: float,
        pixel_y_size: float,
        top_left_x: float,
        top_left_y: float,
        rotation_x: float,
        rotation_y: float,
        map_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create a GeoTIFF file with embedded georeferencing."""
        try:
            output_tiff_path = os.path.join(output_dir, "warped_source.tif")
            
            # Create affine transformation
            transform = Affine(pixel_x_size, rotation_x, top_left_x,
                              rotation_y, pixel_y_size, top_left_y)
            
            # Determine coordinate system
            if map_type == "osm":
                crs_code = 'EPSG:3857'  # Web Mercator for OSM
                logger.info("Using Web Mercator (EPSG:3857) for OSM map")
            else:
                crs_code = 'EPSG:28992'  # RD New for Dutch PDOK services
                logger.info(f"Using RD New (EPSG:28992) for {map_type or 'unknown'} map")
            
            # Convert BGR to RGB for proper color interpretation
            if len(image.shape) == 3:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                height, width, channels = rgb_image.shape
            else:
                rgb_image = image
                height, width = rgb_image.shape
                channels = 1
            
            # Write GeoTIFF
            with rasterio.open(
                output_tiff_path,
                'w',
                driver='GTiff',
                height=height,
                width=width,
                count=channels,
                dtype=rgb_image.dtype,
                crs=crs_code,
                transform=transform
            ) as dst:
                if channels == 3:
                    for i in range(channels):
                        dst.write(rgb_image[:, :, i], i + 1)
                else:
                    dst.write(rgb_image, 1)
            
            logger.info(f"Created GeoTIFF: {output_tiff_path}")
            return {"success": True, "path": output_tiff_path}
            
        except Exception as e:
            logger.error(f"Failed to create GeoTIFF: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def determine_coordinate_system(self, map_type: Optional[str]) -> str:
        """Determine the appropriate coordinate system based on map type."""
        if map_type == "osm":
            return 'EPSG:3857'  # Web Mercator
        else:
            return 'EPSG:28992'  # RD New for Dutch services