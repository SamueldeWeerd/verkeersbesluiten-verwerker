"""
Coordinate Transformation Service - Handles transformations between coordinate systems
"""
import logging
from typing import Tuple, Optional, Dict, Any

logger = logging.getLogger(__name__)

# Optional coordinate transformation support
try:
    import pyproj
    PYPROJ_AVAILABLE = True
except ImportError:
    PYPROJ_AVAILABLE = False


class CoordinateTransformationService:
    """Service for handling coordinate transformations between different coordinate systems."""
    
    def __init__(self):
        """Initialize coordinate transformation service."""
        self.transformers_available = False
        self.rd_to_wgs84 = None
        self.wgs84_to_rd = None
        
        if PYPROJ_AVAILABLE:
            try:
                # RD New (Netherlands) to WGS84 transformer
                self.rd_to_wgs84 = pyproj.Transformer.from_crs(
                    "EPSG:28992",  # RD New
                    "EPSG:4326",   # WGS84
                    always_xy=True
                )
                # WGS84 to RD New transformer  
                self.wgs84_to_rd = pyproj.Transformer.from_crs(
                    "EPSG:4326",   # WGS84
                    "EPSG:28992",  # RD New
                    always_xy=True
                )
                self.transformers_available = True
                logger.info("Coordinate transformers initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize coordinate transformers: {e}")
                self.transformers_available = False
        else:
            logger.warning("pyproj not available. Coordinate transformations disabled.")
    
    def is_available(self) -> bool:
        """Check if coordinate transformation is available."""
        return self.transformers_available
    
    def rd_to_latlon(self, x: float, y: float) -> Tuple[float, float]:
        """
        Convert RD New coordinates to WGS84 lat/lon.
        
        Args:
            x: RD X coordinate (Easting)
            y: RD Y coordinate (Northing)
            
        Returns:
            Tuple of (latitude, longitude)
            
        Raises:
            ValueError: If coordinate transformation not available
        """
        if not self.transformers_available:
            raise ValueError("Coordinate transformation not available. Install pyproj.")
        
        try:
            lon, lat = self.rd_to_wgs84.transform(x, y)
            return lat, lon
        except Exception as e:
            logger.error(f"Error transforming RD to lat/lon: {e}")
            raise
    
    def latlon_to_rd(self, lat: float, lon: float) -> Tuple[float, float]:
        """
        Convert WGS84 lat/lon to RD New coordinates.
        
        Args:
            lat: Latitude
            lon: Longitude
            
        Returns:
            Tuple of (x, y) in RD coordinates
            
        Raises:
            ValueError: If coordinate transformation not available
        """
        if not self.transformers_available:
            raise ValueError("Coordinate transformation not available. Install pyproj.")
        
        try:
            x, y = self.wgs84_to_rd.transform(lon, lat)
            return x, y
        except Exception as e:
            logger.error(f"Error transforming lat/lon to RD: {e}")
            raise
    
    def transform_bounds_rd_to_latlon(self, bounds_rd: Dict[str, float]) -> Dict[str, float]:
        """
        Transform RD bounds to lat/lon bounds.
        
        Args:
            bounds_rd: RD coordinate bounds (min_x, max_x, min_y, max_y)
            
        Returns:
            Dict with lat/lon bounds (min_lat, max_lat, min_lon, max_lon)
        """
        if not self.transformers_available:
            raise ValueError("Coordinate transformation not available. Install pyproj.")
        
        try:
            # Transform corner points
            min_lat, min_lon = self.rd_to_latlon(bounds_rd['min_x'], bounds_rd['min_y'])
            max_lat, max_lon = self.rd_to_latlon(bounds_rd['max_x'], bounds_rd['max_y'])
            
            return {
                'min_lat': min_lat,
                'max_lat': max_lat,
                'min_lon': min_lon,
                'max_lon': max_lon
            }
        except Exception as e:
            logger.error(f"Error transforming RD bounds to lat/lon: {e}")
            raise
    
    def transform_bounds_latlon_to_rd(self, bounds_latlon: Dict[str, float]) -> Dict[str, float]:
        """
        Transform lat/lon bounds to RD bounds.
        
        Args:
            bounds_latlon: Lat/lon bounds (min_lat, max_lat, min_lon, max_lon)
            
        Returns:
            Dict with RD bounds (min_x, max_x, min_y, max_y)
        """
        if not self.transformers_available:
            raise ValueError("Coordinate transformation not available. Install pyproj.")
        
        try:
            # Transform corner points
            min_x, min_y = self.latlon_to_rd(bounds_latlon['min_lat'], bounds_latlon['min_lon'])
            max_x, max_y = self.latlon_to_rd(bounds_latlon['max_lat'], bounds_latlon['max_lon'])
            
            return {
                'min_x': min_x,
                'max_x': max_x,
                'min_y': min_y,
                'max_y': max_y
            }
        except Exception as e:
            logger.error(f"Error transforming lat/lon bounds to RD: {e}")
            raise
    
    def transform_bounds_rd_to_web_mercator(self, bounds_rd: Dict[str, float]) -> Dict[str, float]:
        """
        Transform RD bounds to Web Mercator bounds.
        
        Args:
            bounds_rd: RD coordinate bounds
            
        Returns:
            Dict with Web Mercator bounds
        """
        try:
            # First transform to lat/lon
            latlon_bounds = self.transform_bounds_rd_to_latlon(bounds_rd)
            
            # Then transform to Web Mercator
            from utils.coordinate_utils import CoordinateUtils
            
            min_x, min_y = CoordinateUtils.deg_to_web_mercator(
                latlon_bounds['min_lon'], latlon_bounds['min_lat']
            )
            max_x, max_y = CoordinateUtils.deg_to_web_mercator(
                latlon_bounds['max_lon'], latlon_bounds['max_lat']
            )
            
            return {
                'min_x': min_x,
                'max_x': max_x,
                'min_y': min_y,
                'max_y': max_y
            }
        except Exception as e:
            logger.error(f"Error transforming RD bounds to Web Mercator: {e}")
            raise
    
    def get_transformation_info(self) -> Dict[str, Any]:
        """
        Get information about available transformations.
        
        Returns:
            Dict with transformation info
        """
        return {
            "available": self.transformers_available,
            "pyproj_available": PYPROJ_AVAILABLE,
            "supported_transformations": [
                "RD New (EPSG:28992) ↔ WGS84 (EPSG:4326)",
                "RD New (EPSG:28992) → Web Mercator (EPSG:3857)",
                "WGS84 (EPSG:4326) → Web Mercator (EPSG:3857)"
            ] if self.transformers_available else [],
            "requirements": "pip install pyproj" if not PYPROJ_AVAILABLE else None
        }
    
    def validate_rd_coordinates(self, x: float, y: float) -> bool:
        """
        Validate if coordinates are within reasonable RD New bounds.
        
        Args:
            x: RD X coordinate
            y: RD Y coordinate
            
        Returns:
            True if coordinates are within reasonable bounds
        """
        # Approximate bounds for Netherlands in RD New
        return (10000 <= x <= 280000) and (300000 <= y <= 620000)
    
    def validate_latlon_coordinates(self, lat: float, lon: float) -> bool:
        """
        Validate if lat/lon coordinates are within reasonable bounds for Netherlands.
        
        Args:
            lat: Latitude
            lon: Longitude
            
        Returns:
            True if coordinates are within reasonable bounds
        """
        # Approximate bounds for Netherlands
        return (50.5 <= lat <= 53.7) and (3.2 <= lon <= 7.3)