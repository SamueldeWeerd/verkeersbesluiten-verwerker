"""
Validation utilities for input validation
"""
import os
import json
import logging
from typing import Union, Any
from fastapi import HTTPException, UploadFile

logger = logging.getLogger(__name__)


class ValidationUtils:
    """Utility class for input validation"""
    
    # Allowed file extensions for images
    ALLOWED_IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp'}
    
    @staticmethod
    def validate_transparency(transparency: float) -> None:
        """Validate overlay transparency value"""
        if transparency < 0.0 or transparency > 1.0:
            raise HTTPException(
                status_code=400, 
                detail="overlay_transparency must be between 0.0 and 1.0"
            )
    
    @staticmethod
    def validate_output_format(output_format: str) -> None:
        """Validate output format"""
        if output_format not in ["json", "files"]:
            raise HTTPException(
                status_code=400, 
                detail="output_format must be 'json' or 'files'"
            )
    
    @staticmethod
    def validate_buffer(buffer: float, min_buffer: float = 0, max_buffer: float = 10000) -> None:
        """Validate buffer distance"""
        if buffer < min_buffer or buffer > max_buffer:
            raise HTTPException(
                status_code=400, 
                detail=f"Buffer must be between {min_buffer} and {max_buffer} meters"
            )
    
    @staticmethod
    def validate_image_file(file: UploadFile, file_type: str = "image") -> None:
        """Validate uploaded image file"""
        if not file or not file.filename:
            raise HTTPException(
                status_code=400, 
                detail=f"{file_type} file is required"
            )
        
        extension = ValidationUtils._get_file_extension(file.filename)
        if extension not in ValidationUtils.ALLOWED_IMAGE_EXTENSIONS:
            raise HTTPException(
                status_code=400, 
                detail=f"{file_type} must be PNG, JPG, TIFF, or BMP"
            )
    
    @staticmethod
    def validate_pgw_file(file: UploadFile) -> None:
        """Validate PGW file"""
        if file is not None and not file.filename.lower().endswith('.pgw'):
            raise HTTPException(
                status_code=400, 
                detail="PGW file must have .pgw extension"
            )
    
    @staticmethod
    def validate_image_url(url: str) -> None:
        """Validate image URL"""
        if not url or not url.startswith(('http://', 'https://')):
            raise HTTPException(
                status_code=400, 
                detail="Valid image URL is required"
            )
    
    @staticmethod
    def parse_geometry_input(geometry: str) -> Union[str, dict, list]:
        """
        Parse geometry input from string
        
        Args:
            geometry: Geometry as JSON string or WKT string
            
        Returns:
            Parsed geometry (dict for GeoJSON, list for coordinates, str for WKT)
            
        Raises:
            HTTPException: If geometry cannot be parsed
        """
        try:
            # Try to parse as JSON first (for GeoJSON or coordinate lists)
            geometry_input = json.loads(geometry)
            return geometry_input
        except json.JSONDecodeError:
            # If JSON parsing fails, treat as WKT string
            if not geometry.strip():
                raise HTTPException(
                    status_code=400, 
                    detail="Geometry input cannot be empty"
                )
            return geometry
    
    @staticmethod
    def validate_map_type(map_type: str) -> None:
        """
        Validate map type (basic validation)
        
        Args:
            map_type: Map type string
            
        Note: This is a basic validation. The actual map type validation
        is handled by the cutting service which knows about supported types.
        """
        if not map_type or not map_type.strip():
            raise HTTPException(
                status_code=400,
                detail="Map type is required"
            )
    
    @staticmethod
    def _get_file_extension(filename: str) -> str:
        """Get file extension in lowercase"""
        return os.path.splitext(filename.lower())[1]


class SessionValidator:
    """Validator for session-related operations"""
    
    @staticmethod
    def validate_session_id(session_id: str) -> None:
        """Validate session ID format for security"""
        # Allow both generated session IDs (session_*) and traffic decree IDs
        if ".." in session_id or "/" in session_id or len(session_id.strip()) == 0:
            raise HTTPException(status_code=400, detail="Invalid session ID")
    
    @staticmethod
    def validate_file_path(file_path: str) -> None:
        """Validate file path for security"""
        if ".." in file_path:
            raise HTTPException(status_code=400, detail="Invalid file path")


def validate_common_inputs(
    overlay_transparency: float,
    output_format: str,
    source_image: UploadFile = None,
    image_url: str = None
) -> None:
    """
    Validate common inputs used across multiple endpoints
    
    Args:
        overlay_transparency: Transparency value
        output_format: Output format
        source_image: Optional source image file
        image_url: Optional image URL
    """
    ValidationUtils.validate_transparency(overlay_transparency)
    ValidationUtils.validate_output_format(output_format)
    
    # Validate source (either file or URL)
    if source_image is not None:
        ValidationUtils.validate_image_file(source_image, "Source image")
    elif image_url is not None:
        ValidationUtils.validate_image_url(image_url)
    # If both are None, that's fine - caller should handle this case