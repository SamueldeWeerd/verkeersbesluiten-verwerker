"""
Map Cutting Service - Main orchestration service for cutting map sections
"""
import cv2
import os
import shutil
import logging
from typing import Dict, Any, Union, Optional
from dataclasses import dataclass

# Import our services and utilities
from services.coordinate_transformation_service import CoordinateTransformationService
from services.tile_service import TileService
from utils.geometry_utils import GeometryUtils
from utils.coordinate_utils import CoordinateUtils
from utils.worldfile_utils import WorldFileUtils
from utils.tile_server_utils import TileServerUtils

logger = logging.getLogger(__name__)


@dataclass
class MapCutterResult:
    """Container for map cutting results."""
    success: bool
    image: Optional[object] = None  # cv2 image
    world_file_content: Optional[str] = None
    bounds: Optional[Dict[str, float]] = None
    output_name: str = ""
    error_message: str = ""


class MapCuttingService:
    """Main service for orchestrating map cutting operations."""
    
    def __init__(self):
        """Initialize the map cutting service."""
        self.coordinate_service = CoordinateTransformationService()
        self.tile_service = TileService()
        self.geometry_utils = GeometryUtils()
        self.coordinate_utils = CoordinateUtils()
        self.worldfile_utils = WorldFileUtils()
        self.tile_server_utils = TileServerUtils()
    
    def cut_map(
        self,
        geometry_input: Union[str, dict, list, tuple],
        map_type: str,
        buffer_meters: float = 800,
        target_width: int = 2048,
        output_dir: Optional[str] = None,
        output_name: str = "map_cutout"
    ) -> MapCutterResult:
        """
        Cut out a map section from various map sources based on geometry input.
        
        Args:
            geometry_input: Geometry in various formats (GeoJSON, WKT, coordinates)
            map_type: Type of map to cut
            buffer_meters: Buffer around geometry in meters
            target_width: Target image width in pixels
            output_dir: Optional output directory
            output_name: Base name for output files
            
        Returns:
            MapCutterResult with success status and results
        """
        try:
            # Validate map type
            if not self.tile_server_utils.is_supported_map_type(map_type):
                return MapCutterResult(
                    success=False,
                    error_message=f"Map type '{map_type}' not supported. Available types: {', '.join(self.tile_server_utils.SUPPORTED_MAP_TYPES)}"
                )
            
            # Check coordinate transformation availability
            if not self.coordinate_service.is_available():
                return MapCutterResult(
                    success=False,
                    error_message="Coordinate transformation not available. Please install pyproj: pip install pyproj"
                )
            
            logger.info(f"Starting map cutting: {map_type} with {buffer_meters}m buffer")
            
            # Process geometry and calculate bounds
            bounds_result = self._process_geometry_and_bounds(geometry_input, buffer_meters)
            if not bounds_result["success"]:
                return MapCutterResult(
                    success=False,
                    error_message=bounds_result["error"]
                )
            
            bounds_rd = bounds_result["bounds_rd"]
            
            # Convert bounds to lat/lon for tile calculations
            latlon_bounds = self.coordinate_service.transform_bounds_rd_to_latlon(bounds_rd)
            
            logger.info(f"RD bounds: X=[{bounds_rd['min_x']:.0f}, {bounds_rd['max_x']:.0f}], Y=[{bounds_rd['min_y']:.0f}, {bounds_rd['max_y']:.0f}]")
            logger.info(f"Lat/Lon bounds: Lat=[{latlon_bounds['min_lat']:.6f}, {latlon_bounds['max_lat']:.6f}], Lon=[{latlon_bounds['min_lon']:.6f}, {latlon_bounds['max_lon']:.6f}]")
            
            # Calculate optimal zoom level
            zoom = self.coordinate_utils.calculate_optimal_zoom(
                (latlon_bounds['min_lat'], latlon_bounds['max_lat']),
                (latlon_bounds['min_lon'], latlon_bounds['max_lon']),
                target_width, map_type, buffer_meters
            )
            
            logger.info(f"Using zoom level: {zoom}")
            
            # Validate zoom level for map type
            zoom_validation = self.coordinate_utils.validate_zoom_for_map_type(zoom, map_type, buffer_meters)
            if not zoom_validation["valid"]:
                return MapCutterResult(
                    success=False,
                    error_message=zoom_validation["error"]
                )
            
            # Calculate tile bounds
            min_tile_x, max_tile_y = self.coordinate_utils.deg2num(latlon_bounds['min_lat'], latlon_bounds['min_lon'], zoom)
            max_tile_x, min_tile_y = self.coordinate_utils.deg2num(latlon_bounds['max_lat'], latlon_bounds['max_lon'], zoom)
            
            logger.info(f"Tile bounds: X=[{min_tile_x}, {max_tile_x}], Y=[{min_tile_y}, {max_tile_y}]")
            
            # Calculate actual coordinate bounds based on tiles
            actual_bounds = self.coordinate_utils.calculate_tile_bounds(min_tile_x, min_tile_y, max_tile_x, max_tile_y, zoom)
            
            # Transform actual bounds to RD and Web Mercator
            actual_bounds_rd = self.coordinate_service.transform_bounds_latlon_to_rd({
                'min_lat': actual_bounds['lat_lon']['bottom_right'][0],
                'max_lat': actual_bounds['lat_lon']['top_left'][0],
                'min_lon': actual_bounds['lat_lon']['top_left'][1],
                'max_lon': actual_bounds['lat_lon']['bottom_right'][1]
            })
            
            # Determine output coordinate system
            crs_info = self.worldfile_utils.determine_coordinate_system(map_type)
            if crs_info["crs"] == "EPSG:3857":
                output_bounds = actual_bounds["web_mercator"]
                logger.info(f"Using Web Mercator (EPSG:3857) - native for OSM tiles")
            else:
                output_bounds = actual_bounds_rd
                logger.info(f"Using RD New (EPSG:28992) - native for Dutch PDOK services")
            
            # Download and stitch tiles
            tile_result = self.tile_service.download_and_stitch_tiles(
                min_tile_x, min_tile_y, max_tile_x, max_tile_y, zoom, map_type
            )
            
            if not tile_result["success"]:
                return MapCutterResult(
                    success=False,
                    error_message=tile_result["error"]
                )
            
            stitched_image = tile_result["image"]
            image_width = tile_result["image_width"]
            image_height = tile_result["image_height"]
            
            logger.info(f"Successfully created {image_width}×{image_height} image with {tile_result['downloaded_tiles']}/{tile_result['total_tiles']} tiles")
            
            # Generate world file content
            world_file_content = self.worldfile_utils.create_world_file_content(
                output_bounds, image_width, image_height
            )
            
            # Generate descriptive output name
            descriptive_name = self.worldfile_utils.generate_descriptive_filename(map_type, bounds_rd)
            
            # Save files if output directory specified
            if output_dir:
                save_result = self._save_outputs(
                    output_dir, descriptive_name, stitched_image, 
                    world_file_content, crs_info, output_bounds, 
                    image_width, image_height, map_type
                )
                if not save_result["success"]:
                    logger.warning(f"Failed to save some outputs: {save_result['error']}")
            
            return MapCutterResult(
                success=True,
                image=stitched_image,
                world_file_content=world_file_content,
                bounds=output_bounds,
                output_name=descriptive_name
            )
            
        except Exception as e:
            logger.error(f"Map cutting failed: {str(e)}")
            return MapCutterResult(
                success=False,
                error_message=f"Map cutting failed: {str(e)}"
            )
    
    def _process_geometry_and_bounds(self, geometry_input: Union[str, dict, list, tuple], buffer_meters: float) -> Dict[str, Any]:
        """Process geometry input and calculate buffered bounds."""
        try:
            logger.info("Processing geometry input...")
            
            if self.geometry_utils.is_shapely_available():
                # Use Shapely for precise geometric operations
                buffered_bounds = self.geometry_utils.buffer_geometry(geometry_input, buffer_meters)
                if buffered_bounds is None:
                    return {
                        "success": False,
                        "error": "Failed to parse geometry input"
                    }
                
                # Get original bounds for logging
                original_bounds = self.geometry_utils.get_geometry_bounds(geometry_input)
                if original_bounds:
                    logger.info(f"Original geometry bounds (RD): X=[{original_bounds['min_x']:.0f}, {original_bounds['max_x']:.0f}], Y=[{original_bounds['min_y']:.0f}, {original_bounds['max_y']:.0f}]")
                
                logger.info(f"With {buffer_meters}m geometric buffer: X=[{buffered_bounds['min_x']:.0f}, {buffered_bounds['max_x']:.0f}], Y=[{buffered_bounds['min_y']:.0f}, {buffered_bounds['max_y']:.0f}]")
                
                return {
                    "success": True,
                    "bounds_rd": buffered_bounds
                }
            
            else:
                # Fallback method without Shapely
                coordinates = self.geometry_utils.extract_coordinates_from_input(geometry_input)
                if not coordinates:
                    return {
                        "success": False,
                        "error": "Failed to extract coordinates from input"
                    }
                
                # Calculate bounding box with rectangular buffer
                geom_bounds = self.geometry_utils.calculate_bounds_simple(coordinates)
                bounds_rd = {
                    'min_x': geom_bounds['min_x'] - buffer_meters,
                    'max_x': geom_bounds['max_x'] + buffer_meters,
                    'min_y': geom_bounds['min_y'] - buffer_meters,
                    'max_y': geom_bounds['max_y'] + buffer_meters
                }
                
                logger.info(f"Original geometry bounds (RD): X=[{geom_bounds['min_x']:.0f}, {geom_bounds['max_x']:.0f}], Y=[{geom_bounds['min_y']:.0f}, {geom_bounds['max_y']:.0f}]")
                logger.info(f"With {buffer_meters}m rectangular buffer: X=[{bounds_rd['min_x']:.0f}, {bounds_rd['max_x']:.0f}], Y=[{bounds_rd['min_y']:.0f}, {bounds_rd['max_y']:.0f}]")
                logger.warning("Using rectangular buffer fallback. Install Shapely for proper geometric buffering.")
                
                return {
                    "success": True,
                    "bounds_rd": bounds_rd
                }
                
        except Exception as e:
            logger.error(f"Error processing geometry: {str(e)}")
            return {
                "success": False,
                "error": f"Geometry processing failed: {str(e)}"
            }
    
    def _save_outputs(
        self,
        output_dir: str,
        output_name: str,
        image: object,
        world_file_content: str,
        crs_info: Dict[str, Any],
        bounds: Dict[str, float],
        image_width: int,
        image_height: int,
        map_type: str
    ) -> Dict[str, Any]:
        """Save all output files."""
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # Save image
            image_path = os.path.join(output_dir, f"{output_name}.png")
            cv2.imwrite(image_path, image)
            logger.info(f"Saved image: {image_path}")
            
            # Save georeferencing files
            georef_result = self.worldfile_utils.save_georeferencing_files(
                output_dir, output_name, world_file_content, 
                crs_info["prj_content"], crs_info["crs"]
            )
            
            # Create alternative coordinate system files for comparison
            alt_result = self.worldfile_utils.create_georeferencing_package(
                output_dir, output_name, bounds, image_width, image_height, map_type, create_alternative=True
            )
            
            return {
                "success": True,
                "files": {
                    "image": image_path,
                    "primary_crs": georef_result,
                    "alternative_crs": alt_result.get("alternative")
                }
            }
            
        except Exception as e:
            logger.error(f"Error saving outputs: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def cut_georeferenced_map(
        self,
        geometry_input: Union[str, dict, list],
        map_type: str,
        buffer_meters: float,
        output_dir: str,
        target_width: int = 2048,
        output_name: str = "temp"
    ) -> Dict[str, Any]:
        """
        Cut out a georeferenced map section based on geometry input
        
        Args:
            geometry_input: Geometry as GeoJSON, WKT, or coordinate list
            map_type: Type of map to cut (osm, luchtfoto, etc.)
            buffer_meters: Buffer distance in meters around geometry
            output_dir: Directory to save output files
            target_width: Target width for the output image
            output_name: Base name for output files
            
        Returns:
            Dictionary containing cutting results and metadata
        """
        logger.info(f"Cutting {map_type} map with {buffer_meters}m buffer")
        
        result = self.cut_map(
            geometry_input=geometry_input,
            map_type=map_type,
            buffer_meters=buffer_meters,
            target_width=target_width,
            output_dir=output_dir,
            output_name=output_name
        )
        
        if not result.success:
            return {
                "success": False,
                "error_message": result.error_message
            }
        
        # Prepare response data
        response_data = {
            "success": True,
            "output_name": result.output_name,
            "bounds": result.bounds,
            "image_shape": {
                "width": result.image.shape[1],
                "height": result.image.shape[0]
            },
            "files": {
                "map_image": f"{result.output_name}.png",
                "world_file": f"{result.output_name}.pgw"
            }
        }
        
        # Add geometry-specific information
        geometry_info = self._analyze_geometry_input(geometry_input)
        response_data.update(geometry_info)
        
        return response_data
    
    def cut_map_for_matching(
        self,
        geometry_input: Union[str, dict, list],
        map_type: str,
        buffer_meters: float,
        output_dir: str,
        target_width: int = 2048
    ) -> Dict[str, Any]:
        """
        Cut map specifically for feature matching purposes
        
        This method handles the specific file naming and path management
        needed for feature matching operations.
        
        Args:
            geometry_input: Geometry for cutting
            map_type: Type of map
            buffer_meters: Buffer size
            output_dir: Output directory
            target_width: Target image width
            
        Returns:
            Dictionary with cutting result and prepared paths for matching
        """
        logger.info(f"Cutting {map_type} map with {buffer_meters}m buffer for matching...")
        
        # Cut the map
        cut_result = self.cut_map(
            geometry_input=geometry_input,
            map_type=map_type,
            buffer_meters=buffer_meters,
            target_width=target_width,
            output_dir=output_dir,
            output_name="destination_cutout"
        )
        
        if not cut_result.success:
            return {
                "success": False,
                "error_message": cut_result.error_message
            }
        
        # Prepare file paths for matching
        actual_destination_path = os.path.join(output_dir, f"{cut_result.output_name}.png")
        actual_pgw_path = os.path.join(output_dir, f"{cut_result.output_name}.pgw")
        
        destination_path = os.path.join(output_dir, "destination_cutout.png")
        destination_pgw_path = os.path.join(output_dir, "destination_cutout.pgw")
        
        # Copy files to expected names for matching
        if os.path.exists(actual_destination_path):
            shutil.copy2(actual_destination_path, destination_path)
        else:
            return {
                "success": False,
                "error_message": f"Generated map image not found for {buffer_meters}m {map_type} buffer"
            }
        
        if os.path.exists(actual_pgw_path):
            shutil.copy2(actual_pgw_path, destination_pgw_path)
        
        logger.info(f"Map cutting successful for {map_type}. Image size: {cut_result.image.shape}")
        
        return {
            "success": True,
            "destination_path": destination_path,
            "destination_pgw_path": destination_pgw_path,
            "cut_result": cut_result,
            "image_shape": cut_result.image.shape,
            "bounds": cut_result.bounds
        }
    
    def get_buffer_sizes_for_map_type(self, map_type: str) -> list:
        """
        Get appropriate buffer sizes to test based on map type limitations
        
        Args:
            map_type: Type of map
            
        Returns:
            List of buffer sizes to test
        """
        if map_type.startswith("bgt-") or map_type in ["bag"]:
            # BGT and detailed map types are only available at zoom 13+, limit buffer sizes
            return [10, 100, 190]  # Smaller buffers for detailed tile services
        else:
            # OSM and other services support wider zoom ranges
            return [20, 800, 5000]  # Full range for OSM, luchtfoto and BRTA
    
    def _analyze_geometry_input(self, geometry_input: Union[str, dict, list]) -> Dict[str, Any]:
        """Analyze the geometry input and return metadata"""
        geometry_info = {}
        
        if isinstance(geometry_input, dict):
            geometry_info["input_type"] = "geometry"
            geometry_info["geometry_type"] = geometry_input.get("type", "unknown")
        elif isinstance(geometry_input, str):
            geometry_info["input_type"] = "geometry"
            geometry_info["geometry_format"] = "WKT"
        elif isinstance(geometry_input, list):
            geometry_info["input_type"] = "geometry"
            geometry_info["geometry_format"] = "coordinate_list"
            geometry_info["point_count"] = len(geometry_input)
        
        return geometry_info

    def get_service_info(self) -> Dict[str, Any]:
        """Get information about the map cutting service capabilities."""
        coordinate_info = self.coordinate_service.get_transformation_info()
        tile_info = self.tile_service.get_service_info()
        
        return {
            "coordinate_transformations": coordinate_info,
            "tile_services": tile_info,
            "geometry_processing": {
                "shapely_available": self.geometry_utils.is_shapely_available(),
                "supported_formats": [
                    "GeoJSON (dict or string)",
                    "WKT string", 
                    "Coordinate lists [[x,y], [x,y], ...]",
                    "Single point (x, y)",
                    "Shapely geometry objects"
                ]
            },
            "output_formats": {
                "image": "PNG",
                "georeferencing": ["PGW (world file)", "PRJ (projection file)"],
                "coordinate_systems": ["RD New (EPSG:28992)", "Web Mercator (EPSG:3857)"]
            }
        }


# For backward compatibility
def cut_map(
    geometry_input: Union[str, dict, list, tuple],
    map_type: str,
    buffer_meters: float = 800,
    target_width: int = 2048,
    output_dir: Optional[str] = None,
    output_name: str = "map_cutout"
) -> MapCutterResult:
    """Backward compatibility function."""
    service = MapCuttingService()
    return service.cut_map(geometry_input, map_type, buffer_meters, target_width, output_dir, output_name)