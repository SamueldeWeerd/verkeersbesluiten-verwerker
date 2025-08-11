"""
Coordinate utilities for basic coordinate transformations and calculations
"""
import math
import logging
from typing import Tuple, Dict, Any

logger = logging.getLogger(__name__)


class CoordinateUtils:
    """Utilities for coordinate calculations and transformations."""
    
    @staticmethod
    def deg2num(lat_deg: float, lon_deg: float, zoom: int) -> Tuple[int, int]:
        """
        Convert lat/lon coordinates to OSM tile numbers.
        
        Args:
            lat_deg: Latitude in degrees
            lon_deg: Longitude in degrees  
            zoom: Zoom level
            
        Returns:
            Tuple of (tile_x, tile_y)
        """
        lat_rad = math.radians(lat_deg)
        n = 2.0 ** zoom
        xtile = int((lon_deg + 180.0) / 360.0 * n)
        ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
        return xtile, ytile
    
    @staticmethod
    def num2deg(xtile: int, ytile: int, zoom: int) -> Tuple[float, float]:
        """
        Convert OSM tile numbers to lat/lon coordinates (top-left corner).
        
        Args:
            xtile: Tile X coordinate
            ytile: Tile Y coordinate
            zoom: Zoom level
            
        Returns:
            Tuple of (latitude, longitude) for top-left corner
        """
        n = 2.0 ** zoom
        lon_deg = xtile / n * 360.0 - 180.0
        lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
        lat_deg = math.degrees(lat_rad)
        return lat_deg, lon_deg
    
    @staticmethod
    def deg_to_web_mercator(lon: float, lat: float) -> Tuple[float, float]:
        """
        Convert lat/lon to Web Mercator (EPSG:3857) coordinates.
        
        Args:
            lon: Longitude in degrees
            lat: Latitude in degrees
            
        Returns:
            Tuple of (x, y) in Web Mercator
        """
        x = lon * 20037508.34 / 180.0
        y = math.log(math.tan((90 + lat) * math.pi / 360.0)) / (math.pi / 180.0)
        y = y * 20037508.34 / 180.0
        return x, y
    
    @staticmethod
    def web_mercator_to_deg(x: float, y: float) -> Tuple[float, float]:
        """
        Convert Web Mercator coordinates to lat/lon.
        
        Args:
            x: Web Mercator X coordinate
            y: Web Mercator Y coordinate
            
        Returns:
            Tuple of (latitude, longitude)
        """
        lon = x * 180.0 / 20037508.34
        lat = math.atan(math.exp(y * math.pi / 20037508.34)) * 360.0 / math.pi - 90.0
        return lat, lon
    
    @staticmethod
    def tile_to_bbox(x: int, y: int, zoom: int) -> Dict[str, float]:
        """
        Convert tile coordinates to Web Mercator bounding box.
        
        Args:
            x: Tile X coordinate
            y: Tile Y coordinate
            zoom: Zoom level
            
        Returns:
            Dict with xmin, ymin, xmax, ymax in EPSG:3857
        """
        # Convert tile to lat/lon
        n = 2.0 ** zoom
        lon_min = x / n * 360.0 - 180.0
        lat_max = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
        
        lon_max = (x + 1) / n * 360.0 - 180.0
        lat_min = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n))))
        
        # Convert lat/lon to Web Mercator
        xmin, ymin = CoordinateUtils.deg_to_web_mercator(lon_min, lat_min)
        xmax, ymax = CoordinateUtils.deg_to_web_mercator(lon_max, lat_max)
        
        return {
            'xmin': xmin,
            'ymin': ymin, 
            'xmax': xmax,
            'ymax': ymax
        }
    
    @staticmethod
    def calculate_optimal_zoom(
        bounds_lat: Tuple[float, float], 
        bounds_lon: Tuple[float, float], 
        target_width: int = 2048, 
        map_type: str = "osm",
        buffer_meters: float = 800,
        tile_size: int = 256
    ) -> int:
        """
        Calculate optimal zoom level for given bounds and target image width.
        
        Args:
            bounds_lat: (min_lat, max_lat)
            bounds_lon: (min_lon, max_lon)
            target_width: Target image width in pixels
            map_type: Type of map service (affects minimum useful zoom level)
            buffer_meters: Buffer size in meters (affects zoom selection)
            tile_size: Tile size in pixels
            
        Returns:
            Optimal zoom level
        """
        # Different map types have different optimal zoom ranges
        if map_type.startswith("bgt-"):
            min_useful_zoom = 13  # BGT tiles are visible from zoom 13+
            max_service_zoom = 18  # BGT service limit
            max_multiplier = 3.0  # Allow larger images to stay within zoom 13+ range
        elif map_type in ["luchtfoto", "luchtfoto-2022"]:
            min_useful_zoom = 13  # High zoom for aerial photos
            max_service_zoom = 18  # Luchtfoto WMTS service limit
            max_multiplier = 3.0  # Allow larger images for aerial photos
        elif map_type == "bag":
            min_useful_zoom = 13  # High zoom for BAG buildings
            max_service_zoom = 18  # BAG service limit
            max_multiplier = 3.0  # Allow larger images for building details
        elif map_type in ["brta", "brta-omtrek", "top10"]:
            min_useful_zoom = 13  # Medium zoom for BRT maps
            max_service_zoom = 18  # BRT service limit
            max_multiplier = 3.0  # Allow larger images for topographic maps
        else:
            min_useful_zoom = 1
            max_service_zoom = 18
            max_multiplier = 1.3  # Standard multiplier for OSM
        
        # Adjust max zoom based on buffer size
        if buffer_meters >= 1000:
            max_service_zoom = min(max_service_zoom, 13)  
        elif buffer_meters >= 600:
            max_service_zoom = min(max_service_zoom, 14)  
        elif buffer_meters >= 400:
            max_service_zoom = min(max_service_zoom, 15)
        elif buffer_meters >= 200:
            max_service_zoom = min(max_service_zoom, 18)   
        
        # Start from adjusted max zoom and work down
        for zoom in range(max_service_zoom, min_useful_zoom - 1, -1):
            # Calculate tile coverage
            min_tile_x, max_tile_y = CoordinateUtils.deg2num(bounds_lat[0], bounds_lon[0], zoom)
            max_tile_x, min_tile_y = CoordinateUtils.deg2num(bounds_lat[1], bounds_lon[1], zoom)
            
            # Calculate image dimensions
            width_tiles = max_tile_x - min_tile_x + 1
            image_width = width_tiles * tile_size
            
            if image_width <= target_width * max_multiplier:
                return zoom
        
        return min_useful_zoom
    
    @staticmethod
    def validate_zoom_for_map_type(zoom: int, map_type: str, buffer_meters: float) -> Dict[str, Any]:
        """
        Validate if zoom level is appropriate for the map type.
        
        Args:
            zoom: Calculated zoom level
            map_type: Map type
            buffer_meters: Buffer size used
            
        Returns:
            Dict with validation result
        """
        # Check if the zoom level is too low for this map type
        if map_type.startswith("bgt-") and zoom < 13:
            return {
                "valid": False,
                "error": f"Buffer size {buffer_meters}m is too large for {map_type} tiles. BGT tiles are only available at zoom level 13+ (approx. 500m detail level). Try using a smaller buffer or switch to OSM map type for large areas."
            }
        elif map_type in ["luchtfoto", "luchtfoto-2022", "bag", "brta", "brta-omtrek", "top10"] and zoom < 13:
            return {
                "valid": False,
                "error": f"Buffer size {buffer_meters}m is too large for {map_type} tiles. This map type is only available at zoom level 13+ (approx. 500m detail level). Try using a smaller buffer or switch to OSM map type for large areas."
            }
        
        return {"valid": True}
    
    @staticmethod
    def calculate_tile_bounds(min_tile_x: int, min_tile_y: int, max_tile_x: int, max_tile_y: int, zoom: int) -> Dict[str, Any]:
        """
        Calculate the actual coordinate bounds of a tile grid.
        
        Args:
            min_tile_x: Minimum tile X coordinate
            min_tile_y: Minimum tile Y coordinate
            max_tile_x: Maximum tile X coordinate
            max_tile_y: Maximum tile Y coordinate
            zoom: Zoom level
            
        Returns:
            Dict with actual bounds in various coordinate systems
        """
        # Get the actual lat/lon bounds of the tile grid
        actual_top_left_lat, actual_top_left_lon = CoordinateUtils.num2deg(min_tile_x, min_tile_y, zoom)
        actual_bottom_right_lat, actual_bottom_right_lon = CoordinateUtils.num2deg(max_tile_x + 1, max_tile_y + 1, zoom)
        
        # Convert to Web Mercator
        web_merc_min_x, web_merc_max_y = CoordinateUtils.deg_to_web_mercator(actual_top_left_lon, actual_top_left_lat)
        web_merc_max_x, web_merc_min_y = CoordinateUtils.deg_to_web_mercator(actual_bottom_right_lon, actual_bottom_right_lat)
        
        return {
            "lat_lon": {
                "top_left": (actual_top_left_lat, actual_top_left_lon),
                "bottom_right": (actual_bottom_right_lat, actual_bottom_right_lon)
            },
            "web_mercator": {
                'min_x': web_merc_min_x,
                'max_x': web_merc_max_x,
                'min_y': web_merc_min_y,
                'max_y': web_merc_max_y
            }
        }