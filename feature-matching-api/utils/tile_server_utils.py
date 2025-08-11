"""
Tile server utilities for managing tile server configurations and URL generation
"""
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class TileServerUtils:
    """Utilities for managing tile server configurations and URL generation."""
    
    # Tile server configurations
    TILE_SERVERS = {
        "osm": [
            "https://tile.openstreetmap.org",
            "https://a.tile.openstreetmap.org", 
            "https://b.tile.openstreetmap.org",
            "https://c.tile.openstreetmap.org"
        ],
        "bgt-achtergrond": [
            "https://service.pdok.nl/lv/bgt/wmts/v1_0"
        ],
        "bgt-omtrek": [
            "https://service.pdok.nl/lv/bgt/wmts/v1_0"
        ],
        "bgt-standaard": [
            "https://service.pdok.nl/lv/bgt/wmts/v1_0"
        ],
        "luchtfoto": [
            "https://service.pdok.nl/hwh/luchtfotorgb/wmts/v1_0"
        ],
        "brta": [
            "https://service.pdok.nl/brt/achtergrondkaart/wmts/v2_0"
        ],
        "brta-omtrek": [
            "https://service.pdok.nl/brt/achtergrondkaart/wmts/v2_0"
        ],
        "top10": [
            "https://service.pdok.nl/brt/achtergrondkaart/wmts/v2_0"
        ],
        "bag": [
            "https://service.pdok.nl/lv/bag/wms/v2_0"
        ]
    }
    
    # Map type configurations
    MAP_TYPE_CONFIGS = {
        "osm": {
            "format": "png",
            "tile_pattern": "{server}/{zoom}/{x}/{y}.png",
            "service_type": "xyz"
        },
        "bgt-achtergrond": {
            "format": "png",
            "layer": "achtergrondvisualisatie",
            "tilematrixset": "EPSG:3857",
            "tile_pattern": "{server}/{layer}/{tilematrixset}/{zoom:02d}/{x}/{y}.png",
            "service_type": "wmts"
        },
        "bgt-omtrek": {
            "format": "png",
            "layer": "omtrekgerichtevisualisatie",
            "tilematrixset": "EPSG:3857",
            "tile_pattern": "{server}/{layer}/{tilematrixset}/{zoom:02d}/{x}/{y}.png",
            "service_type": "wmts"
        },
        "bgt-standaard": {
            "format": "png",
            "layer": "standaardvisualisatie",
            "tilematrixset": "EPSG:3857",
            "tile_pattern": "{server}/{layer}/{tilematrixset}/{zoom:02d}/{x}/{y}.png",
            "service_type": "wmts"
        },
        "luchtfoto": {
            "format": "jpeg",
            "layer": "Actueel_orthoHR",
            "tilematrixset": "EPSG:3857",
            "tile_pattern": "{server}/{layer}/{tilematrixset}/{zoom:02d}/{x}/{y}.jpeg",
            "service_type": "wmts"
        },
        "luchtfoto-2022": {
            "format": "jpeg",
            "layer": "2022_orthoHR",
            "tilematrixset": "EPSG:3857",
            "tile_pattern": "{server}/{layer}/{tilematrixset}/{zoom:02d}/{x}/{y}.jpeg",
            "service_type": "wmts"
        },
        "brta": {
            "format": "png",
            "layer": "standaard",
            "tilematrixset": "EPSG:3857",
            "tile_pattern": "{server}/{layer}/{tilematrixset}/{zoom:02d}/{x}/{y}.png",
            "service_type": "wmts"
        },
        "brta-omtrek": {
            "format": "png",
            "layer": "grijs",
            "tilematrixset": "EPSG:3857",
            "tile_pattern": "{server}/{layer}/{tilematrixset}/{zoom:02d}/{x}/{y}.png",
            "service_type": "wmts"
        },
        "top10": {
            "format": "png",
            "layer": "standaard",
            "tilematrixset": "EPSG:3857",
            "tile_pattern": "{server}/{layer}/{tilematrixset}/{zoom:02d}/{x}/{y}.png",
            "service_type": "wmts"
        },
        "bag": {
            "format": "png",
            "service_type": "wms",
            "layers": "pand",
            "crs": "EPSG:3857"
        }
    }
    
    # Supported map types list
    SUPPORTED_MAP_TYPES = [
        "osm",                    # OpenStreetMap
        "bgt-achtergrond",       # BGT background
        "bgt-omtrek",            # BGT outline
        "bgt-standaard",         # BGT standard
        "luchtfoto",             # Aerial photography
        "luchtfoto-2022",        # Winter aerial photography
        "brta",                  # BRT-A standard
        "brta-omtrek",           # BRT-A outline
        "top10",                 # BRT-TOP10 NL
        "bag",                   # BAG buildings
        "bgt-bg-bag",            # BGT background + BAG overlay
        "bgt-bg-omtrek",         # BGT background + outline overlay
        "brta-bag",              # BRT-A + BAG overlay
        "brta-omtrek"            # BRT-A + BGT outline overlay
    ]
    
    def __init__(self, user_agent: str = "MapAI-FeatureMatching/1.0"):
        """
        Initialize tile server utilities.
        
        Args:
            user_agent: User agent string for requests
        """
        self.user_agent = user_agent
        self.current_server_index = 0
    
    def get_tile_servers(self, map_type: str) -> List[str]:
        """Get tile servers for a map type."""
        return self.TILE_SERVERS.get(map_type, self.TILE_SERVERS["osm"])
    
    def get_map_type_config(self, map_type: str) -> Dict:
        """Get configuration for a map type."""
        return self.MAP_TYPE_CONFIGS.get(map_type, self.MAP_TYPE_CONFIGS["osm"])
    
    def is_supported_map_type(self, map_type: str) -> bool:
        """Check if map type is supported."""
        return map_type.lower() in self.SUPPORTED_MAP_TYPES
    
    def get_next_server(self, map_type: str) -> str:
        """Get next server URL for load balancing."""
        servers = self.get_tile_servers(map_type)
        server = servers[self.current_server_index % len(servers)]
        self.current_server_index = (self.current_server_index + 1) % len(servers)
        return server
    
    def generate_tile_url(self, map_type: str, x: int, y: int, zoom: int) -> Optional[str]:
        """
        Generate tile URL for given parameters.
        
        Args:
            map_type: Type of map
            x: Tile X coordinate
            y: Tile Y coordinate
            zoom: Zoom level
            
        Returns:
            Tile URL or None if map type not supported
        """
        if not self.is_supported_map_type(map_type):
            return None
        
        config = self.get_map_type_config(map_type)
        server = self.get_next_server(map_type)
        
        if config["service_type"] == "xyz":
            # Simple XYZ tile service (like OSM)
            return config["tile_pattern"].format(
                server=server, x=x, y=y, zoom=zoom
            )
        elif config["service_type"] == "wmts":
            # WMTS service
            return config["tile_pattern"].format(
                server=server, x=x, y=y, zoom=zoom,
                layer=config["layer"],
                tilematrixset=config["tilematrixset"]
            )
        elif config["service_type"] == "wms":
            # WMS service (requires bbox calculation)
            return server  # Base URL, bbox will be added by tile service
        
        return None
    
    def get_overlay_layers(self, map_type: str) -> List[str]:
        """
        Get list of layers for overlay map types.
        
        Args:
            map_type: Map type
            
        Returns:
            List of component layer types
        """
        overlay_mappings = {
            "bgt-bg-bag": ["bgt-achtergrond", "bag"],
            "bgt-bg-omtrek": ["bgt-achtergrond", "bgt-omtrek"],
            "brta-bag": ["brta", "bag"],
            "brta-omtrek": ["brta", "bgt-omtrek"]
        }
        
        return overlay_mappings.get(map_type, [map_type])
    
    def is_overlay_type(self, map_type: str) -> bool:
        """Check if map type is an overlay combination."""
        overlay_types = ["bgt-bg-bag", "bgt-bg-omtrek", "brta-bag", "brta-omtrek"]
        return map_type in overlay_types
    
    def get_background_color(self, map_type: str) -> tuple:
        """
        Get appropriate background color for map type.
        
        Args:
            map_type: Map type
            
        Returns:
            RGB tuple for background color
        """
        # BGT omtrek and BAG layers need white background for better visibility
        if map_type in ["bgt-omtrek", "bag"]:
            return (255, 255, 255)  # White
        else:
            return (0, 0, 0)  # Black
    
    def requires_transparency_handling(self, map_type: str) -> bool:
        """Check if map type requires special transparency handling."""
        return map_type in ["bgt-omtrek", "bag"]
    
    def get_request_headers(self) -> Dict[str, str]:
        """Get standard request headers."""
        return {'User-Agent': self.user_agent}
    
    @staticmethod
    def get_supported_types_info() -> Dict[str, str]:
        """Get information about all supported map types."""
        return {
            "osm": "OpenStreetMap standard tiles",
            "bgt-achtergrond": "Dutch BGT background visualization (filled areas)",
            "bgt-omtrek": "Dutch BGT outline visualization (object boundaries)",
            "bgt-standaard": "Dutch BGT standard visualization (default BGT styling)",
            "luchtfoto": "Dutch aerial photography (current)",
            "luchtfoto-2022": "Dutch aerial photography (winter 2022)",
            "brta": "BRT-A standard topographic map",
            "brta-omtrek": "BRT-A outline topographic map",
            "top10": "BRT-TOP10 NL topographic map",
            "bag": "BAG buildings layer",
            "bgt-bg-bag": "BGT background with BAG buildings overlay",
            "bgt-bg-omtrek": "BGT background with outline overlay",
            "brta-bag": "BRT-A with BAG buildings overlay",
            "brta-omtrek": "BRT-A with BGT outline overlay"
        }