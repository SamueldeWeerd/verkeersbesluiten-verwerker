"""
Geometry utilities for parsing and processing various geometry input formats
"""
import json
import logging
from typing import List, Tuple, Dict, Any, Union, Optional

logger = logging.getLogger(__name__)

# Optional geometry support
try:
    from shapely.geometry import Point, LineString, Polygon, MultiPoint, MultiLineString, MultiPolygon
    from shapely.geometry import shape
    from shapely import wkt
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False


class GeometryUtils:
    """Utilities for handling various geometry input formats."""
    
    @staticmethod
    def parse_geometry(geometry_input: Union[str, dict, list, tuple]) -> Optional[object]:
        """
        Parse geometry input into a Shapely geometry object.
        
        Args:
            geometry_input: Can be:
                - GeoJSON dict/string
                - WKT string  
                - List of coordinates [[x,y], [x,y], ...]
                - Tuple of (x,y) for single point
                - Shapely geometry object
                
        Returns:
            Shapely geometry object or None if parsing failed
        """
        if not SHAPELY_AVAILABLE:
            logger.warning("Shapely not available - geometry parsing limited")
            return None
        
        try:
            # Already a Shapely geometry
            if hasattr(geometry_input, 'bounds'):
                return geometry_input
            
            # Handle string inputs
            if isinstance(geometry_input, str):
                # Try parsing as GeoJSON
                try:
                    geojson = json.loads(geometry_input)
                    return shape(geojson)
                except (json.JSONDecodeError, ValueError):
                    pass
                
                # Try parsing as WKT
                try:
                    return wkt.loads(geometry_input)
                except Exception:
                    pass
                    
                return None
            
            # Handle dict (GeoJSON)
            if isinstance(geometry_input, dict):
                return shape(geometry_input)
            
            # Handle coordinate lists/tuples
            if isinstance(geometry_input, (list, tuple)):
                return GeometryUtils._parse_coordinate_list(geometry_input)
            
            return None
            
        except Exception as e:
            logger.error(f"Error parsing geometry: {e}")
            return None
    
    @staticmethod
    def _parse_coordinate_list(coords: Union[list, tuple]) -> Optional[object]:
        """Parse coordinate list into appropriate geometry."""
        if not SHAPELY_AVAILABLE:
            return None
            
        # Single point: (x, y) or [x, y]
        if len(coords) == 2 and all(isinstance(coord, (int, float)) for coord in coords):
            return Point(coords)
        
        # Multiple points: [[x,y], [x,y], ...]
        if all(isinstance(item, (list, tuple)) and len(item) == 2 for item in coords):
            if len(coords) == 1:
                return Point(coords[0])
            elif len(coords) == 2:
                return LineString(coords)
            else:
                return MultiPoint(coords)
        
        return None
    
    @staticmethod
    def calculate_bounds_simple(coordinates: List[Tuple[float, float]]) -> Dict[str, float]:
        """
        Calculate bounding box from coordinates (fallback when Shapely not available).
        
        Args:
            coordinates: List of (x, y) coordinate tuples
            
        Returns:
            Dict with min_x, max_x, min_y, max_y
        """
        if not coordinates:
            raise ValueError("No coordinates provided")
        
        x_coords = [coord[0] for coord in coordinates]
        y_coords = [coord[1] for coord in coordinates]
        
        return {
            'min_x': min(x_coords),
            'max_x': max(x_coords),
            'min_y': min(y_coords),
            'max_y': max(y_coords)
        }
    
    @staticmethod
    def extract_coordinates_from_input(geometry_input: Union[str, dict, list, tuple]) -> List[Tuple[float, float]]:
        """
        Extract all coordinates from various input formats (fallback method).
        
        Args:
            geometry_input: Various geometry input formats
            
        Returns:
            List of (x, y) coordinate tuples
        """
        coordinates = []
        
        try:
            # Handle string inputs
            if isinstance(geometry_input, str):
                # Try parsing as GeoJSON
                try:
                    geojson = json.loads(geometry_input)
                    coordinates.extend(GeometryUtils._extract_coords_from_geojson(geojson))
                except (json.JSONDecodeError, ValueError):
                    pass
                return coordinates
            
            # Handle dict (GeoJSON)
            if isinstance(geometry_input, dict):
                coordinates.extend(GeometryUtils._extract_coords_from_geojson(geometry_input))
                return coordinates
            
            # Handle coordinate lists/tuples
            if isinstance(geometry_input, (list, tuple)):
                # Single point: (x, y) or [x, y]
                if len(geometry_input) == 2 and all(isinstance(coord, (int, float)) for coord in geometry_input):
                    coordinates.append(tuple(geometry_input))
                
                # Multiple points: [[x,y], [x,y], ...]
                elif all(isinstance(item, (list, tuple)) and len(item) == 2 for item in geometry_input):
                    coordinates.extend([tuple(item) for item in geometry_input])
            
            return coordinates
            
        except Exception as e:
            logger.error(f"Error extracting coordinates: {e}")
            return []
    
    @staticmethod
    def _extract_coords_from_geojson(geojson: dict) -> List[Tuple[float, float]]:
        """Extract coordinates from GeoJSON geometry."""
        coordinates = []
        geometry_type = geojson.get('type', '').lower()
        coords = geojson.get('coordinates', [])
        
        if geometry_type == 'point':
            coordinates.append(tuple(coords))
        elif geometry_type in ['linestring', 'multipoint']:
            coordinates.extend([tuple(coord) for coord in coords])
        elif geometry_type == 'polygon':
            # For polygon, extract all rings
            for ring in coords:
                coordinates.extend([tuple(coord) for coord in ring])
        elif geometry_type == 'multilinestring':
            for line in coords:
                coordinates.extend([tuple(coord) for coord in line])
        elif geometry_type == 'multipolygon':
            for polygon in coords:
                for ring in polygon:
                    coordinates.extend([tuple(coord) for coord in ring])
        
        return coordinates
    
    @staticmethod
    def is_shapely_available() -> bool:
        """Check if Shapely is available for advanced geometry operations."""
        return SHAPELY_AVAILABLE
    
    @staticmethod
    def get_geometry_bounds(geometry_input: Union[str, dict, list, tuple]) -> Optional[Dict[str, float]]:
        """
        Get bounds from geometry input using best available method.
        
        Args:
            geometry_input: Various geometry input formats
            
        Returns:
            Dict with min_x, max_x, min_y, max_y or None if failed
        """
        try:
            if SHAPELY_AVAILABLE:
                geometry = GeometryUtils.parse_geometry(geometry_input)
                if geometry is not None:
                    minx, miny, maxx, maxy = geometry.bounds
                    return {
                        'min_x': minx,
                        'max_x': maxx,
                        'min_y': miny,
                        'max_y': maxy
                    }
            
            # Fallback method
            coordinates = GeometryUtils.extract_coordinates_from_input(geometry_input)
            if coordinates:
                return GeometryUtils.calculate_bounds_simple(coordinates)
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting geometry bounds: {e}")
            return None
    
    @staticmethod
    def buffer_geometry(geometry_input: Union[str, dict, list, tuple], buffer_meters: float) -> Optional[Dict[str, float]]:
        """
        Apply buffer to geometry and return buffered bounds.
        
        Args:
            geometry_input: Various geometry input formats
            buffer_meters: Buffer distance in meters
            
        Returns:
            Dict with buffered bounds or None if failed
        """
        try:
            if SHAPELY_AVAILABLE:
                geometry = GeometryUtils.parse_geometry(geometry_input)
                if geometry is not None:
                    buffered_geometry = geometry.buffer(buffer_meters)
                    minx, miny, maxx, maxy = buffered_geometry.bounds
                    return {
                        'min_x': minx,
                        'max_x': maxx,
                        'min_y': miny,
                        'max_y': maxy
                    }
            
            # Fallback: rectangular buffer
            bounds = GeometryUtils.get_geometry_bounds(geometry_input)
            if bounds:
                return {
                    'min_x': bounds['min_x'] - buffer_meters,
                    'max_x': bounds['max_x'] + buffer_meters,
                    'min_y': bounds['min_y'] - buffer_meters,
                    'max_y': bounds['max_y'] + buffer_meters
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error buffering geometry: {e}")
            return None