"""
Response building utilities for creating standardized API responses using response models
"""
from typing import Dict, Any, List, Optional
from datetime import datetime
from models.response_models import (
    FeatureMatchingResponse,
    MapCuttingResponse, 
    CutoutAndMatchResponse,
    ErrorResponse,
    HealthCheckResponse,
    DetailedHealthResponse,
    SessionListResponse,
    SessionCleanupResponse,
    GeoreferencedFile,
    GeoreferencedFileType,
    SessionInfo
)


def build_health_response(
    service: str = "Map AI Processing Service",
    status: str = "healthy", 
    version: str = "1.0.0",
    available_endpoints: Optional[Dict[str, str]] = None
) -> HealthCheckResponse:
    """Build health check response"""
    if available_endpoints is None:
        available_endpoints = {
            "feature_matching": "/match-maps",
            "map_cutting_geometry": "/cut-out-georeferenced-map", 
            "cutout_and_match": "/cutout-and-match",
            "cutout_and_match_with_url": "/cutout-and-match-with-url",
            "health": "/health",
            "sessions": "/sessions",
            "download": "/download/{session_id}/{file_path:path}"
        }
    
    return HealthCheckResponse(
        service=service,
        status=status,
        version=version,
        timestamp=datetime.now().isoformat(),
        available_endpoints=available_endpoints
    )


def build_detailed_health_response() -> DetailedHealthResponse:
    """Build detailed health response with service status"""
    return DetailedHealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        services={
            "feature_matching": "operational",
            "map_cutting": "operational", 
            "file_management": "operational",
            "session_management": "operational"
        }
    )


def build_feature_matching_response(
    session_id: str,
    match_result: Dict[str, Any],
    overlay_transparency: float,
    pgw_provided: bool = False,
    georeferenced_files: Optional[List[str]] = None
) -> FeatureMatchingResponse:
    """Build feature matching response from service results"""
    
    # Convert georeferenced file names to GeoreferencedFile objects
    geo_files = []
    if georeferenced_files:
        for filename in georeferenced_files:
            if isinstance(filename, tuple):
                file_type, filename = filename
            else:
                if filename.lower().endswith('.tif'):
                    file_type = "GeoTIFF"
                elif filename.lower().endswith('.pgw'):
                    file_type = "World File" 
                else:
                    file_type = "Projection File"
            
            geo_files.append(GeoreferencedFile(
                type=file_type,
                url=f"/download/{session_id}/{filename}"
            ))
    
    return FeatureMatchingResponse.create_success_response(
        session_id=session_id,
        matches_count=match_result.get("matches_count", 0),
        inlier_ratio=match_result.get("inlier_ratio", 0.0),
        overlay_transparency=overlay_transparency,
        georeferenced=bool(match_result.get("georeferenced_files")),
        pgw_provided=pgw_provided,
        georeferenced_files=geo_files
    )


def build_map_cutting_response(
    session_id: str,
    cut_result: Dict[str, Any],
    map_type: str,
    buffer_meters: float,
    target_width: int = 2048
) -> MapCuttingResponse:
    """Build map cutting response from service results"""
    
    return MapCuttingResponse.create_success_response(
        session_id=session_id,
        map_type=map_type,
        bounds_rd=cut_result.get("bounds_rd", {}),
        output_name=cut_result.get("output_name", "map_cutout"),
        buffer_meters=buffer_meters,
        target_width=target_width,
        actual_size=cut_result.get("actual_size", {"width": target_width, "height": target_width}),
        geometry_info=cut_result.get("geometry_info")
    )


def build_cutout_and_match_response(
    session_id: str,
    match_result: Dict[str, Any],
    cut_result: Dict[str, Any],
    overlay_transparency: float,
    map_type: str,
    best_buffer: float,
    test_buffer_sizes: List[int],
    buffer_results: List[Dict[str, Any]],
    source_url: Optional[str] = None
) -> CutoutAndMatchResponse:
    """Build cutout and match response from service results"""
    
    # Build georeferenced files list 
    georeferenced_files = []
    
    # Always include the GeoTIFF file for cutout-and-match operations
    georeferenced_files.append(GeoreferencedFile(
        type=GeoreferencedFileType.GEOTIFF,
        url=f"/download/{session_id}/warped_source.tif"
    ))
    
    # Add other georeferenced files if available
    if match_result.get("georeferenced_files"):
        for filename in match_result["georeferenced_files"]:
            if isinstance(filename, tuple):
                file_type, filename = filename
            else:
                if filename.lower().endswith('.tif'):
                    continue  # Skip .tif files since we already added it above
                elif filename.lower().endswith('.pgw'):
                    file_type = "World File" 
                else:
                    file_type = "Projection File"
            
            # Only add if it's not a duplicate GeoTIFF
            if not filename.lower().endswith('.tif'):
                georeferenced_files.append(GeoreferencedFile(
                    type=file_type,
                    url=f"/download/{session_id}/{filename}"
                ))
    
    # Create buffer selection info
    from models.response_models import BufferSelection, BufferTestResult
    buffer_selection = BufferSelection(
        tested_buffers=test_buffer_sizes,
        results=[
            BufferTestResult(
                buffer_meters=result.get("buffer_size", 0),
                map_type=map_type,
                matches_count=result.get("matches_count", 0),
                inlier_count=int(result.get("matches_count", 0) * result.get("inlier_ratio", 0.0)),
                inlier_ratio=result.get("inlier_ratio", 0.0)
            )
            for result in buffer_results
        ],
        selection_criteria=f"Best buffer: {best_buffer}m based on match quality",
        inspection_folders=[f"buffer_{size}m_cutout" for size in test_buffer_sizes]
    )
    
    return CutoutAndMatchResponse.create_success_response(
        session_id=session_id,
        matches_count=match_result.get("matches_count", 0),
        inlier_ratio=match_result.get("inlier_ratio", 0.0),
        overlay_transparency=overlay_transparency,
        georeferenced=True,  # Always true for cutout-and-match since we generate PGW
        georeferenced_files=georeferenced_files,
        map_type=map_type,
        selected_buffer_meters=int(best_buffer),
        bounds_rd=cut_result.get("bounds_rd", {"min_x": 0, "min_y": 0, "max_x": 1000, "max_y": 1000}),
        destination_name=cut_result.get("destination_name", f"cutout_{session_id}"),
        buffer_results=[
            BufferTestResult(
                buffer_meters=result.get("buffer_size", 0),
                map_type=map_type,
                matches_count=result.get("matches_count", 0),
                inlier_count=int(result.get("matches_count", 0) * result.get("inlier_ratio", 0.0)),
                inlier_ratio=result.get("inlier_ratio", 0.0)
            )
            for result in buffer_results
        ],
        tested_buffers=test_buffer_sizes,
        target_width=cut_result.get("actual_size", {}).get("width", 2048),
        actual_size=cut_result.get("actual_size", {"width": 2048, "height": 2048}),
        source_url=source_url
    )


def build_error_response(
    error_message: str,
    session_id: Optional[str] = None
) -> ErrorResponse:
    """Build error response"""
    return ErrorResponse.create_error_response(
        error_message=error_message,
        session_id=session_id
    )


def build_session_list_response(sessions: List[Dict[str, Any]]) -> SessionListResponse:
    """Build session list response"""
    session_infos = [
        SessionInfo(
            session_id=session["session_id"],
            created_at=session.get("created_at", "unknown"),
            file_count=session.get("file_count", 0),
            size_mb=session.get("size_mb", 0.0)
        )
        for session in sessions
    ]
    
    return SessionListResponse(
        status="success",
        timestamp=datetime.now().isoformat(),
        sessions=session_infos
    )


def build_session_cleanup_response(session_id: str, files_deleted: int = 0) -> SessionCleanupResponse:
    """Build session cleanup response"""
    return SessionCleanupResponse(
        session_id=session_id,
        message=f"Session {session_id} cleaned up successfully",
        files_deleted=files_deleted,
        timestamp=datetime.now().isoformat()
    )