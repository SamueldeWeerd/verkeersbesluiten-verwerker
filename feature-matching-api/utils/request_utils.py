"""
Request parsing utilities for converting form data to Pydantic models
"""
from typing import Optional
from models.request_models import (
    FeatureMatchingRequest,
    MapCuttingRequest, 
    CutoutAndMatchRequest,
    CutoutAndMatchWithUrlRequest,
    MapType,
    OutputFormat
)


def parse_feature_matching_form(
    overlay_transparency: float,
    output_format: str,
    traffic_decree_id: Optional[str]
) -> FeatureMatchingRequest:
    """Parse form data to FeatureMatchingRequest model"""
    return FeatureMatchingRequest(
        overlay_transparency=overlay_transparency,
        output_format=OutputFormat(output_format),
        traffic_decree_id=traffic_decree_id
    )


def parse_map_cutting_form(
    geometry: str,
    map_type: str,
    buffer: float,
    output_format: str,
    traffic_decree_id: Optional[str]
) -> MapCuttingRequest:
    """Parse form data to MapCuttingRequest model"""
    return MapCuttingRequest(
        geometry=geometry,
        map_type=MapType(map_type),
        buffer=buffer,
        output_format=OutputFormat(output_format),
        traffic_decree_id=traffic_decree_id
    )


def parse_cutout_and_match_form(
    geometry: str,
    map_type: str,
    overlay_transparency: float,
    output_format: str,
    traffic_decree_id: Optional[str]
) -> CutoutAndMatchRequest:
    """Parse form data to CutoutAndMatchRequest model"""
    return CutoutAndMatchRequest(
        geometry=geometry,
        map_type=MapType(map_type),
        overlay_transparency=overlay_transparency,
        output_format=OutputFormat(output_format),
        traffic_decree_id=traffic_decree_id
    )


def parse_cutout_and_match_url_form(
    image_url: str,
    geometry: str,
    map_type: str,
    overlay_transparency: float,
    output_format: str,
    traffic_decree_id: Optional[str]
) -> CutoutAndMatchWithUrlRequest:
    """Parse form data to CutoutAndMatchWithUrlRequest model"""
    return CutoutAndMatchWithUrlRequest(
        image_url=image_url,
        geometry=geometry,
        map_type=MapType(map_type),
        overlay_transparency=overlay_transparency,
        output_format=OutputFormat(output_format),
        traffic_decree_id=traffic_decree_id
    )