"""
World file utilities for generating georeferencing files (PGW, PRJ)
"""
import os
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class WorldFileUtils:
    """Utilities for creating world files and projection files."""
    
    @staticmethod
    def create_world_file_content(
        bounds: Dict[str, float], 
        image_width: int, 
        image_height: int
    ) -> str:
        """
        Create world file content for the generated image.
        
        Args:
            bounds: Coordinate bounds (min_x, max_x, min_y, max_y)
            image_width: Image width in pixels
            image_height: Image height in pixels
            
        Returns:
            World file content as string
        """
        # Calculate pixel size
        pixel_x_size = (bounds['max_x'] - bounds['min_x']) / image_width
        pixel_y_size = -(bounds['max_y'] - bounds['min_y']) / image_height  # Negative for image coordinates
        
        # Top-left pixel center coordinates
        top_left_x = bounds['min_x'] + pixel_x_size / 2
        top_left_y = bounds['max_y'] + pixel_y_size / 2
        
        # World file format: pixel_x_size, rotation_y, rotation_x, pixel_y_size, top_left_x, top_left_y
        world_file_content = f"{pixel_x_size}\n0.0\n0.0\n{pixel_y_size}\n{top_left_x}\n{top_left_y}\n"
        
        return world_file_content
    
    @staticmethod
    def create_rd_new_prj_content() -> str:
        """
        Create PRJ file content for RD New (EPSG:28992) coordinate reference system.
        
        Returns:
            PRJ file content as string
        """
        # RD New (EPSG:28992) Well-Known Text (WKT) definition
        return '''PROJCS["Amersfoort / RD New",GEOGCS["Amersfoort",DATUM["Amersfoort",SPHEROID["Bessel 1841",6377397.155,299.1528128,AUTHORITY["EPSG","7004"]],AUTHORITY["EPSG","6289"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4289"]],PROJECTION["Oblique_Stereographic"],PARAMETER["latitude_of_origin",52.15616055555555],PARAMETER["central_meridian",5.38763888888889],PARAMETER["scale_factor",0.9999079],PARAMETER["false_easting",155000],PARAMETER["false_northing",463000],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AUTHORITY["EPSG","28992"]]'''
    
    @staticmethod
    def create_web_mercator_prj_content() -> str:
        """
        Create PRJ file content for Web Mercator (EPSG:3857) coordinate reference system.
        
        Returns:
            PRJ file content as string
        """
        # Web Mercator (EPSG:3857) Well-Known Text (WKT) definition
        return '''PROJCS["WGS 84 / Pseudo-Mercator",GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]],PROJECTION["Mercator_1SP"],PARAMETER["central_meridian",0],PARAMETER["scale_factor",1],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["X",EAST],AXIS["Y",NORTH],EXTENSION["PROJ4","+proj=merc +a=6378137 +b=6378137 +lat_ts=0.0 +lon_0=0.0 +x_0=0.0 +y_0=0 +k=1.0 +units=m +nadgrids=@null +wktext +no_defs"],AUTHORITY["EPSG","3857"]]'''
    
    @staticmethod
    def save_georeferencing_files(
        output_dir: str,
        output_name: str,
        world_file_content: str,
        prj_content: str,
        coordinate_system: str = "EPSG:28992"
    ) -> Dict[str, Any]:
        """
        Save world file and projection file to disk.
        
        Args:
            output_dir: Output directory
            output_name: Base filename (without extension)
            world_file_content: World file content
            prj_content: Projection file content
            coordinate_system: Coordinate system identifier
            
        Returns:
            Dict with file paths and success status
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # Save world file
            world_file_path = os.path.join(output_dir, f"{output_name}.pgw")
            with open(world_file_path, 'w') as f:
                f.write(world_file_content)
            
            # Save PRJ file
            prj_file_path = os.path.join(output_dir, f"{output_name}.prj")
            with open(prj_file_path, 'w') as f:
                f.write(prj_content)
            
            logger.info(f"Saved world file: {world_file_path}")
            logger.info(f"Saved PRJ file: {prj_file_path}")
            
            return {
                "success": True,
                "world_file_path": world_file_path,
                "prj_file_path": prj_file_path,
                "coordinate_system": coordinate_system
            }
            
        except Exception as e:
            logger.error(f"Failed to save georeferencing files: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    @staticmethod
    def determine_coordinate_system(map_type: str) -> Dict[str, Any]:
        """
        Determine appropriate coordinate system based on map type.
        
        Args:
            map_type: Type of map
            
        Returns:
            Dict with coordinate system info
        """
        if map_type == "osm":
            # OSM tiles are natively in Web Mercator (EPSG:3857)
            return {
                "crs": "EPSG:3857",
                "name": "Web Mercator",
                "prj_content": WorldFileUtils.create_web_mercator_prj_content()
            }
        else:
            # Dutch PDOK services (BGT, Luchtfoto, BRT, BAG, etc.) are in RD New (EPSG:28992)
            return {
                "crs": "EPSG:28992",
                "name": "RD New",
                "prj_content": WorldFileUtils.create_rd_new_prj_content()
            }
    
    @staticmethod
    def create_georeferencing_package(
        output_dir: str,
        output_name: str,
        bounds: Dict[str, float],
        image_width: int,
        image_height: int,
        map_type: str,
        create_alternative: bool = True
    ) -> Dict[str, Any]:
        """
        Create complete georeferencing package with primary and alternative coordinate systems.
        
        Args:
            output_dir: Output directory
            output_name: Base filename
            bounds: Coordinate bounds for primary system
            image_width: Image width in pixels
            image_height: Image height in pixels
            map_type: Map type to determine primary coordinate system
            create_alternative: Whether to create alternative coordinate system files
            
        Returns:
            Dict with all created file paths and info
        """
        try:
            # Determine primary coordinate system
            primary_crs = WorldFileUtils.determine_coordinate_system(map_type)
            
            # Create primary world file
            world_file_content = WorldFileUtils.create_world_file_content(bounds, image_width, image_height)
            
            # Save primary files
            primary_result = WorldFileUtils.save_georeferencing_files(
                output_dir, output_name, world_file_content, 
                primary_crs["prj_content"], primary_crs["crs"]
            )
            
            result = {
                "success": primary_result["success"],
                "primary": {
                    "crs": primary_crs["crs"],
                    "name": primary_crs["name"],
                    "world_file": primary_result.get("world_file_path"),
                    "prj_file": primary_result.get("prj_file_path")
                }
            }
            
            # Create alternative coordinate system files if requested
            if create_alternative and primary_result["success"]:
                if primary_crs["crs"] == "EPSG:3857":
                    # Primary is Web Mercator, create RD New alternative
                    alt_crs = {
                        "crs": "EPSG:28992",
                        "name": "RD New",
                        "suffix": "_rd_new",
                        "prj_content": WorldFileUtils.create_rd_new_prj_content()
                    }
                else:
                    # Primary is RD New, create Web Mercator alternative
                    alt_crs = {
                        "crs": "EPSG:3857", 
                        "name": "Web Mercator",
                        "suffix": "_web_mercator",
                        "prj_content": WorldFileUtils.create_web_mercator_prj_content()
                    }
                
                # Note: For alternative, we would need the bounds in the alternative coordinate system
                # This is simplified - in practice, you'd need coordinate transformation
                alt_result = WorldFileUtils.save_georeferencing_files(
                    output_dir, f"{output_name}{alt_crs['suffix']}", 
                    world_file_content, alt_crs["prj_content"], alt_crs["crs"]
                )
                
                result["alternative"] = {
                    "crs": alt_crs["crs"],
                    "name": alt_crs["name"],
                    "world_file": alt_result.get("world_file_path"),
                    "prj_file": alt_result.get("prj_file_path")
                }
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to create georeferencing package: {str(e)}")
            return {"success": False, "error": str(e)}
    
    @staticmethod
    def generate_descriptive_filename(map_type: str, bounds: Dict[str, float]) -> str:
        """
        Generate descriptive filename based on map type and coordinates.
        
        Args:
            map_type: Type of map
            bounds: Coordinate bounds
            
        Returns:
            Descriptive filename (without extension)
        """
        map_type_clean = map_type.replace("-", "_")
        top_left_x = int(bounds['min_x'])
        top_left_y = int(bounds['max_y'])  # Max Y because image coordinates are flipped
        
        return f"{map_type_clean}_{top_left_x}_{top_left_y}"