"""
Tile Service - Handles downloading and stitching map tiles from various sources
"""
import cv2
import numpy as np
import requests
import time
import io
import logging
from typing import Optional, List, Dict, Any
from PIL import Image

# Import our utilities
from utils.tile_server_utils import TileServerUtils
from utils.coordinate_utils import CoordinateUtils

logger = logging.getLogger(__name__)

# Optional progress tracking
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


class TileService:
    """Service for downloading and stitching map tiles from various sources."""
    
    def __init__(self, user_agent: str = "MapAI-FeatureMatching/1.0"):
        """
        Initialize tile service.
        
        Args:
            user_agent: User agent string for requests
        """
        self.tile_size = 256  # Standard tile size
        self.tile_server_utils = TileServerUtils(user_agent)
        self.coordinate_utils = CoordinateUtils()
    
    def download_and_stitch_tiles(
        self,
        min_tile_x: int,
        min_tile_y: int,
        max_tile_x: int,
        max_tile_y: int,
        zoom: int,
        map_type: str,
        timeout: float = 10.0
    ) -> Dict[str, Any]:
        """
        Download and stitch tiles into a single image.
        
        Args:
            min_tile_x: Minimum tile X coordinate
            min_tile_y: Minimum tile Y coordinate
            max_tile_x: Maximum tile X coordinate
            max_tile_y: Maximum tile Y coordinate
            zoom: Zoom level
            map_type: Type of map
            timeout: Request timeout in seconds
            
        Returns:
            Dict with stitched image and metadata
        """
        try:
            # Calculate dimensions
            width_tiles = max_tile_x - min_tile_x + 1
            height_tiles = max_tile_y - min_tile_y + 1
            image_width = width_tiles * self.tile_size
            image_height = height_tiles * self.tile_size
            
            logger.info(f"Downloading {width_tiles}×{height_tiles} tiles ({image_width}×{image_height} pixels)")
            
            # Create output image with appropriate background
            background_color = self.tile_server_utils.get_background_color(map_type)
            output_image = Image.new('RGB', (image_width, image_height), background_color)
            
            # Download and stitch tiles
            total_tiles = width_tiles * height_tiles
            downloaded_tiles = 0
            
            if TQDM_AVAILABLE:
                progress_bar = tqdm(total=total_tiles, desc="Downloading tiles")
            
            for y in range(min_tile_y, max_tile_y + 1):
                for x in range(min_tile_x, max_tile_x + 1):
                    tile = self._download_tile(x, y, zoom, map_type, timeout)
                    
                    if tile:
                        # Calculate position in output image
                        paste_x = (x - min_tile_x) * self.tile_size
                        paste_y = (y - min_tile_y) * self.tile_size
                        
                        # Handle transparency if needed
                        if self.tile_server_utils.requires_transparency_handling(map_type):
                            self._paste_tile_with_transparency(output_image, tile, paste_x, paste_y)
                        else:
                            output_image.paste(tile, (paste_x, paste_y))
                        
                        downloaded_tiles += 1
                    else:
                        logger.warning(f"Failed to download tile {x}/{y}/{zoom}")
                    
                    if TQDM_AVAILABLE:
                        progress_bar.update(1)
                    
                    # Be nice to tile servers
                    time.sleep(0.1)
            
            if TQDM_AVAILABLE:
                progress_bar.close()
            
            logger.info(f"Successfully downloaded {downloaded_tiles}/{total_tiles} tiles")
            
            if downloaded_tiles == 0:
                return {
                    "success": False,
                    "error": "Failed to download any tiles"
                }
            
            # Convert PIL image to OpenCV format
            cv_image = cv2.cvtColor(np.array(output_image), cv2.COLOR_RGB2BGR)
            
            return {
                "success": True,
                "image": cv_image,
                "image_width": image_width,
                "image_height": image_height,
                "downloaded_tiles": downloaded_tiles,
                "total_tiles": total_tiles
            }
            
        except Exception as e:
            logger.error(f"Error stitching tiles: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _download_tile(self, x: int, y: int, zoom: int, map_type: str, timeout: float = 10.0) -> Optional[Image.Image]:
        """
        Download a single tile from the appropriate service.
        
        Args:
            x: Tile X coordinate
            y: Tile Y coordinate
            zoom: Zoom level
            map_type: Type of map service
            timeout: Request timeout in seconds
            
        Returns:
            PIL Image or None if failed
        """
        # Handle overlay combinations
        if self.tile_server_utils.is_overlay_type(map_type):
            return self._download_overlay_tile(x, y, zoom, map_type, timeout)
        
        # Handle single layer types
        return self._download_single_tile(x, y, zoom, map_type, timeout)
    
    def _download_single_tile(self, x: int, y: int, zoom: int, map_type: str, timeout: float) -> Optional[Image.Image]:
        """Download a single tile from a specific service type."""
        try:
            config = self.tile_server_utils.get_map_type_config(map_type)
            
            if config["service_type"] == "xyz":
                return self._download_xyz_tile(x, y, zoom, map_type, timeout)
            elif config["service_type"] == "wmts":
                return self._download_wmts_tile(x, y, zoom, map_type, timeout)
            elif config["service_type"] == "wms":
                return self._download_wms_tile(x, y, zoom, map_type, timeout)
            else:
                logger.error(f"Unknown service type: {config['service_type']}")
                return None
                
        except Exception as e:
            logger.error(f"Error downloading single tile {x}/{y}/{zoom} for {map_type}: {e}")
            return None
    
    def _download_xyz_tile(self, x: int, y: int, zoom: int, map_type: str, timeout: float) -> Optional[Image.Image]:
        """Download a tile from XYZ service (like OSM)."""
        try:
            url = self.tile_server_utils.generate_tile_url(map_type, x, y, zoom)
            if not url:
                return None
            
            headers = self.tile_server_utils.get_request_headers()
            response = requests.get(url, headers=headers, timeout=timeout)
            
            if response.status_code == 200:
                return Image.open(io.BytesIO(response.content))
            elif response.status_code == 429:  # Rate limited
                logger.warning(f"Rate limited on {map_type} service, waiting...")
                time.sleep(1)
                return None
            else:
                logger.warning(f"HTTP {response.status_code} from {map_type} service")
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to download {map_type} tile {x}/{y}/{zoom}: {e}")
            return None
    
    def _download_wmts_tile(self, x: int, y: int, zoom: int, map_type: str, timeout: float) -> Optional[Image.Image]:
        """Download a tile from WMTS service."""
        try:
            url = self.tile_server_utils.generate_tile_url(map_type, x, y, zoom)
            if not url:
                return None
            
            headers = self.tile_server_utils.get_request_headers()
            response = requests.get(url, headers=headers, timeout=timeout)
            
            if response.status_code == 200:
                return Image.open(io.BytesIO(response.content))
            elif response.status_code == 429:  # Rate limited
                logger.warning(f"Rate limited on PDOK {map_type} service, waiting...")
                time.sleep(2)  # Be more patient with PDOK
                return None
            else:
                logger.warning(f"HTTP {response.status_code} from PDOK {map_type} service")
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to download {map_type} tile {x}/{y}/{zoom}: {e}")
            return None
    
    def _download_wms_tile(self, x: int, y: int, zoom: int, map_type: str, timeout: float) -> Optional[Image.Image]:
        """Download a tile from WMS service (used for BAG)."""
        try:
            # Convert tile coordinates to bounding box
            bbox = CoordinateUtils.tile_to_bbox(x, y, zoom)
            
            # Get base URL
            servers = self.tile_server_utils.get_tile_servers(map_type)
            base_url = servers[0]  # Use first server for WMS
            
            # WMS GetMap parameters
            params = {
                'SERVICE': 'WMS',
                'VERSION': '1.3.0',
                'REQUEST': 'GetMap',
                'LAYERS': 'pand',  # BAG buildings layer
                'STYLES': '',
                'CRS': 'EPSG:3857',
                'BBOX': f"{bbox['xmin']},{bbox['ymin']},{bbox['xmax']},{bbox['ymax']}",
                'WIDTH': str(self.tile_size),
                'HEIGHT': str(self.tile_size),
                'FORMAT': 'image/png',
                'TRANSPARENT': 'TRUE'
            }
            
            headers = self.tile_server_utils.get_request_headers()
            response = requests.get(base_url, params=params, headers=headers, timeout=timeout)
            
            if response.status_code == 200:
                return Image.open(io.BytesIO(response.content))
            elif response.status_code == 429:
                logger.warning(f"Rate limited on PDOK {map_type} WMS service, waiting...")
                time.sleep(2)
                return None
            else:
                logger.warning(f"HTTP {response.status_code} from PDOK {map_type} WMS service")
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to download {map_type} WMS tile {x}/{y}/{zoom}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error processing {map_type} WMS tile {x}/{y}/{zoom}: {e}")
            return None
    
    def _download_overlay_tile(self, x: int, y: int, zoom: int, map_type: str, timeout: float) -> Optional[Image.Image]:
        """Download and composite multiple layers into a single overlay tile."""
        try:
            layers = self.tile_server_utils.get_overlay_layers(map_type)
            tiles = []
            
            # Download all layers
            for layer in layers:
                tile = self._download_single_tile(x, y, zoom, layer, timeout)
                if tile:
                    tiles.append(tile)
                else:
                    logger.warning(f"Failed to download layer {layer} for overlay")
            
            if not tiles:
                return None
            
            # Composite tiles
            result = tiles[0].copy()
            
            # Overlay additional tiles
            for tile in tiles[1:]:
                # Convert to RGBA for alpha blending
                if result.mode != 'RGBA':
                    result = result.convert('RGBA')
                if tile.mode != 'RGBA':
                    tile = tile.convert('RGBA')
                
                # Composite the tiles
                result = Image.alpha_composite(result, tile)
            
            # Convert back to RGB for consistency
            if result.mode == 'RGBA':
                # Create white background for final image
                background = Image.new('RGB', result.size, (255, 255, 255))
                background.paste(result, mask=result.split()[3] if len(result.split()) > 3 else None)
                result = background
            
            return result
            
        except Exception as e:
            logger.error(f"Error creating overlay tile: {e}")
            return None
    
    def _paste_tile_with_transparency(self, output_image: Image.Image, tile: Image.Image, paste_x: int, paste_y: int):
        """Paste tile with proper transparency handling."""
        try:
            if tile.mode in ('RGBA', 'LA'):
                # Composite the tile with transparency onto white background
                white_bg = Image.new('RGB', tile.size, (255, 255, 255))
                if tile.mode == 'RGBA':
                    white_bg.paste(tile, mask=tile.split()[3])  # Use alpha channel as mask
                else:
                    white_bg.paste(tile, mask=tile.split()[1])  # Use alpha channel as mask
                output_image.paste(white_bg, (paste_x, paste_y))
            else:
                output_image.paste(tile, (paste_x, paste_y))
        except Exception as e:
            logger.error(f"Error pasting tile with transparency: {e}")
            # Fallback to simple paste
            output_image.paste(tile, (paste_x, paste_y))
    
    def get_service_info(self) -> Dict[str, Any]:
        """Get information about the tile service capabilities."""
        return {
            "tile_size": self.tile_size,
            "supported_map_types": self.tile_server_utils.SUPPORTED_MAP_TYPES,
            "map_type_info": self.tile_server_utils.get_supported_types_info(),
            "overlay_types": [
                "bgt-bg-bag", "bgt-bg-omtrek", "brta-bag", "brta-omtrek"
            ],
            "service_types": ["xyz", "wmts", "wms"]
        }