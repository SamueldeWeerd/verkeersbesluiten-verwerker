"""
Request models for API input validation using Pydantic
"""
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Literal
from enum import Enum


class OutputFormat(str, Enum):
    """Valid output formats"""
    JSON = "json"
    FILES = "files"


class MapType(str, Enum):
    """Supported map types"""
    OSM = "osm"
    BGT_ACHTERGROND = "bgt-achtergrond"
    BGT_OMTREK = "bgt-omtrek"
    BGT_STANDAARD = "bgt-standaard"
    LUCHTFOTO = "luchtfoto"
    LUCHTFOTO_2022 = "luchtfoto-2022"
    BRTA = "brta"
    BRTA_OMTREK = "brta-omtrek"
    TOP10 = "top10"
    BAG = "bag"
    BGT_BG_BAG = "bgt-bg-bag"
    BGT_BG_OMTREK = "bgt-bg-omtrek"
    BRTA_BAG = "brta-bag"
    BRTA_OMTREK_COMBO = "brta-omtrek"


class FeatureMatchingRequest(BaseModel):
    """Request model for feature matching endpoints"""
    overlay_transparency: float = Field(
        default=0.6, 
        ge=0.0, 
        le=1.0, 
        description="Overlay transparency (0.0-1.0)"
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.JSON,
        description="Output format: 'json' or 'files'"
    )
    traffic_decree_id: Optional[str] = Field(
        default=None,
        description="Optional traffic decree ID for session naming"
    )
    
    @validator('traffic_decree_id')
    def validate_traffic_decree_id(cls, v):
        if v is not None and len(v.strip()) == 0:
            return None
        return v


class MapCuttingRequest(BaseModel):
    """Request model for map cutting endpoints"""
    geometry: str = Field(
        ...,
        description="Geometry as GeoJSON, WKT, or coordinate list JSON string"
    )
    map_type: MapType = Field(
        default=MapType.OSM,
        description="Map type"
    )
    buffer: float = Field(
        default=800,
        ge=0,
        le=10000,
        description="Buffer distance in meters around the geometry"
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.JSON,
        description="Output format: 'json' or 'files'"
    )
    traffic_decree_id: Optional[str] = Field(
        default=None,
        description="Optional traffic decree ID for session naming"
    )
    
    @validator('geometry')
    def validate_geometry(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Geometry cannot be empty')
        return v.strip()


class CutoutAndMatchRequest(BaseModel):
    """Request model for cutout-and-match endpoints"""
    geometry: str = Field(
        ...,
        description="Geometry as GeoJSON, WKT, or coordinate list JSON string"
    )
    map_type: MapType = Field(
        ...,
        description="Map type"
    )
    overlay_transparency: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Overlay transparency (0.0-1.0)"
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.JSON,
        description="Output format: 'json' or 'files'"
    )
    traffic_decree_id: Optional[str] = Field(
        default=None,
        description="Optional traffic decree ID for session naming"
    )
    
    @validator('geometry')
    def validate_geometry(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Geometry cannot be empty')
        return v.strip()


class CutoutAndMatchWithUrlRequest(BaseModel):
    """Request model for cutout-and-match-with-url endpoint"""
    image_url: str = Field(
        ...,
        description="URL of the source image to be warped and matched"
    )
    geometry: str = Field(
        ...,
        description="Geometry as GeoJSON, WKT, or coordinate list JSON string"
    )
    map_type: MapType = Field(
        ...,
        description="Map type"
    )
    overlay_transparency: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Overlay transparency (0.0-1.0)"
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.JSON,
        description="Output format: 'json' or 'files'"
    )
    traffic_decree_id: Optional[str] = Field(
        default=None,
        description="Optional traffic decree ID for session naming"
    )
    
    @validator('image_url')
    def validate_image_url(cls, v):
        if not v or not v.startswith(('http://', 'https://')):
            raise ValueError('Valid image URL is required')
        return v
    
    @validator('geometry')
    def validate_geometry(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Geometry cannot be empty')
        return v.strip()


class SessionRequest(BaseModel):
    """Request model for session operations"""
    session_id: str = Field(
        ...,
        description="Session ID"
    )
    
    @validator('session_id')
    def validate_session_id(cls, v):
        # Allow both generated session IDs (session_*) and traffic decree IDs
        if ".." in v or "/" in v or len(v.strip()) == 0:
            raise ValueError('Invalid session ID format')
        return v


class FileDownloadRequest(BaseModel):
    """Request model for file downloads"""
    session_id: str = Field(
        ...,
        description="Session ID"
    )
    file_path: str = Field(
        ...,
        description="File path within session directory"
    )
    
    @validator('session_id')
    def validate_session_id(cls, v):
        # Allow both generated session IDs (session_*) and traffic decree IDs
        if ".." in v or "/" in v or len(v.strip()) == 0:
            raise ValueError('Invalid session ID format')
        return v
    
    @validator('file_path')
    def validate_file_path(cls, v):
        if ".." in v:
            raise ValueError('Invalid file path - path traversal not allowed')
        return v