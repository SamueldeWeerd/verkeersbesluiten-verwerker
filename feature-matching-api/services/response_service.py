"""
Response Service - Creates standardized API responses using data models
"""
import logging
from typing import Dict, List, Any, Optional
from fastapi.responses import JSONResponse

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
    BufferTestResult,
    to_dict
)

logger = logging.getLogger(__name__)


class ResponseService:
    """Service for creating standardized API responses"""
    
    @staticmethod
    def create_feature_matching_response(
        session_id: str,
        matches_count: int,
        inlier_ratio: float,
        overlay_transparency: float,
        georeferenced: bool,
        pgw_provided: bool,
        georeferenced_files_data: List[tuple]
    ) -> JSONResponse:
        """
        Create standardized feature matching response.
        
        Args:
            session_id: Session identifier
            matches_count: Number of feature matches
            inlier_ratio: Ratio of inlier matches
            overlay_transparency: Transparency setting used
            georeferenced: Whether georeferencing was successful
            pgw_provided: Whether PGW file was provided
            georeferenced_files_data: List of (type, url) tuples
            
        Returns:
            JSONResponse with standardized structure
        """
        try:
            # Convert georeferenced files data
            georeferenced_files = [
                GeoreferencedFile(
                    type=GeoreferencedFileType(file_type), 
                    url=url
                )
                for file_type, url in georeferenced_files_data
            ]
            
            response = FeatureMatchingResponse.create_success_response(
                session_id=session_id,
                matches_count=matches_count,
                inlier_ratio=inlier_ratio,
                overlay_transparency=overlay_transparency,
                georeferenced=georeferenced,
                pgw_provided=pgw_provided,
                georeferenced_files=georeferenced_files
            )
            
            return JSONResponse(content=to_dict(response))
            
        except Exception as e:
            logger.error(f"Error creating feature matching response: {e}")
            return ResponseService.create_error_response(f"Failed to create response: {str(e)}")
    
    @staticmethod
    def create_map_cutting_response(
        session_id: str,
        map_type: str,
        bounds_rd: Dict[str, float],
        output_name: str,
        buffer_meters: float,
        target_width: int,
        actual_size: Dict[str, int],
        geometry_info: Optional[Dict[str, Any]] = None
    ) -> JSONResponse:
        """
        Create standardized map cutting response.
        
        Args:
            session_id: Session identifier
            map_type: Type of map that was cut
            bounds_rd: RD coordinate bounds
            output_name: Name of output files
            buffer_meters: Buffer distance used
            target_width: Target image width
            actual_size: Actual image dimensions
            geometry_info: Optional geometry information
            
        Returns:
            JSONResponse with standardized structure
        """
        try:
            response = MapCuttingResponse.create_success_response(
                session_id=session_id,
                map_type=map_type,
                bounds_rd=bounds_rd,
                output_name=output_name,
                buffer_meters=buffer_meters,
                target_width=target_width,
                actual_size=actual_size,
                geometry_info=geometry_info
            )
            
            return JSONResponse(content=to_dict(response))
            
        except Exception as e:
            logger.error(f"Error creating map cutting response: {e}")
            return ResponseService.create_error_response(f"Failed to create response: {str(e)}")
    
    @staticmethod
    def create_cutout_and_match_response(
        session_id: str,
        matches_count: int,
        inlier_ratio: float,
        overlay_transparency: float,
        georeferenced: bool,
        georeferenced_files_data: List[tuple],
        map_type: str,
        selected_buffer_meters: int,
        bounds_rd: Dict[str, float],
        destination_name: str,
        buffer_results_data: List[Dict[str, Any]],
        tested_buffers: List[int],
        source_url: Optional[str] = None
    ) -> JSONResponse:
        """
        Create standardized cutout-and-match response.
        
        Args:
            session_id: Session identifier
            matches_count: Number of feature matches
            inlier_ratio: Ratio of inlier matches
            overlay_transparency: Transparency setting used
            georeferenced: Whether georeferencing was successful
            georeferenced_files_data: List of (type, url) tuples
            map_type: Type of map used
            selected_buffer_meters: Selected buffer size
            bounds_rd: RD coordinate bounds
            destination_name: Name of destination file
            buffer_results_data: Results from buffer testing
            tested_buffers: List of tested buffer sizes
            source_url: Optional source URL for URL-based requests
            
        Returns:
            JSONResponse with standardized structure
        """
        try:
            # Convert georeferenced files data
            georeferenced_files = [
                GeoreferencedFile(
                    type=GeoreferencedFileType(file_type), 
                    url=url
                )
                for file_type, url in georeferenced_files_data
            ]
            
            # Convert buffer results data
            buffer_results = [
                BufferTestResult(
                    buffer_meters=result["buffer_meters"],
                    map_type=result["map_type"],
                    matches_count=result["matches_count"],
                    inlier_count=result["inlier_count"],
                    inlier_ratio=result["inlier_ratio"]
                )
                for result in buffer_results_data
            ]
            
            response = CutoutAndMatchResponse.create_success_response(
                session_id=session_id,
                matches_count=matches_count,
                inlier_ratio=inlier_ratio,
                overlay_transparency=overlay_transparency,
                georeferenced=georeferenced,
                georeferenced_files=georeferenced_files,
                map_type=map_type,
                selected_buffer_meters=selected_buffer_meters,
                bounds_rd=bounds_rd,
                destination_name=destination_name,
                buffer_results=buffer_results,
                tested_buffers=tested_buffers,
                source_url=source_url
            )
            
            return JSONResponse(content=to_dict(response))
            
        except Exception as e:
            logger.error(f"Error creating cutout-and-match response: {e}")
            return ResponseService.create_error_response(f"Failed to create response: {str(e)}")
    
    @staticmethod
    def create_error_response(
        error_message: str,
        session_id: Optional[str] = None,
        status_code: int = 500
    ) -> JSONResponse:
        """
        Create standardized error response.
        
        Args:
            error_message: Error description
            session_id: Optional session identifier
            status_code: HTTP status code
            
        Returns:
            JSONResponse with error structure
        """
        try:
            response = ErrorResponse.create_error_response(
                error_message=error_message,
                session_id=session_id
            )
            
            return JSONResponse(
                content=to_dict(response),
                status_code=status_code
            )
            
        except Exception as e:
            logger.error(f"Error creating error response: {e}")
            # Fallback basic error response
            return JSONResponse(
                content={
                    "success": False,
                    "error_message": "Internal server error",
                    "details": str(e)
                },
                status_code=500
            )
    
    @staticmethod
    def create_health_response() -> JSONResponse:
        """Create health check response."""
        try:
            from datetime import datetime
            
            response = HealthCheckResponse(
                service="Map AI Processing Service",
                status="healthy",
                version="1.0.0",
                timestamp=datetime.now().isoformat(),
                available_endpoints={
                    "feature_matching": "/match-maps",
                    "feature_matching_with_size_reduction": "/match-maps-with-size-reduction",
                    "map_cutting_geometry": "/cut-out-georeferenced-map",
                    "cutout_and_match": "/cutout-and-match",
                    "cutout_and_match_with_url": "/cutout-and-match-with-url",
                    "health": "/health"
                }
            )
            
            return JSONResponse(content=to_dict(response))
            
        except Exception as e:
            logger.error(f"Error creating health response: {e}")
            return ResponseService.create_error_response("Health check failed")
    
    @staticmethod
    def create_detailed_health_response() -> JSONResponse:
        """Create detailed health check response."""
        try:
            from datetime import datetime
            
            response = DetailedHealthResponse(
                status="healthy",
                timestamp=datetime.now().isoformat(),
                services={
                    "matcher": "available",
                    "cutter": "available",
                }
            )
            
            return JSONResponse(content=to_dict(response))
            
        except Exception as e:
            logger.error(f"Error creating detailed health response: {e}")
            return ResponseService.create_error_response("Detailed health check failed")
    
    @staticmethod
    def create_session_list_response(sessions_data: List[Dict[str, Any]]) -> JSONResponse:
        """Create session list response."""
        try:
            from models.response_models import SessionInfo
            
            sessions = [
                SessionInfo(
                    session_id=session["session_id"],
                    files=session["files"],
                    created=session["created"]
                )
                for session in sessions_data
            ]
            
            response = SessionListResponse(sessions=sessions)
            
            return JSONResponse(content=to_dict(response))
            
        except Exception as e:
            logger.error(f"Error creating session list response: {e}")
            return ResponseService.create_error_response("Failed to list sessions")
    
    @staticmethod
    def create_session_cleanup_response(session_id: str) -> JSONResponse:
        """Create session cleanup response."""
        try:
            response = SessionCleanupResponse(
                message=f"Session {session_id} cleaned up successfully"
            )
            
            return JSONResponse(content=to_dict(response))
            
        except Exception as e:
            logger.error(f"Error creating session cleanup response: {e}")
            return ResponseService.create_error_response("Failed to cleanup session")
    
    @staticmethod
    def create_validation_error_response(error_details: str) -> JSONResponse:
        """Create validation error response."""
        return ResponseService.create_error_response(
            error_message=f"Validation error: {error_details}",
            status_code=422
        )
    
    @staticmethod
    def create_not_found_response(resource: str) -> JSONResponse:
        """Create not found error response."""
        return ResponseService.create_error_response(
            error_message=f"{resource} not found",
            status_code=404
        )
    
    @staticmethod
    def create_bad_request_response(reason: str) -> JSONResponse:
        """Create bad request error response."""
        return ResponseService.create_error_response(
            error_message=f"Bad request: {reason}",
            status_code=400
        )