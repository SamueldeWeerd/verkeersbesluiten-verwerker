#!/usr/bin/env python3
"""
astAPI Orchestrator for Map AI Processing

Application for processing maps with map-cutting and feature matching.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Depends
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import logging
from datetime import datetime
from typing import Optional

# Import our services and utilities
from services.feature_matching_service import FeatureMatchingService
from services.map_cutting_service import MapCuttingService
from utils.file_utils import FileManager
from utils.validation_utils import ValidationUtils, SessionValidator, validate_common_inputs

from utils.session_utils import SessionManager
from utils.request_utils import (
    parse_feature_matching_form,
    parse_map_cutting_form,
    parse_cutout_and_match_form,
    parse_cutout_and_match_url_form
)
from utils.response_builders import (
    build_health_response,
    build_detailed_health_response,
    build_feature_matching_response,
    build_map_cutting_response,
    build_cutout_and_match_response,
    build_error_response
)

# Import request and response models
from models.request_models import (
    FeatureMatchingRequest, MapCuttingRequest, CutoutAndMatchRequest, 
    CutoutAndMatchWithUrlRequest, SessionRequest, FileDownloadRequest,
    OutputFormat, MapType
)
from models.response_models import (
    FeatureMatchingResponse, MapCuttingResponse, CutoutAndMatchResponse,
    ErrorResponse, HealthCheckResponse, DetailedHealthResponse,
    SessionListResponse, SessionCleanupResponse, GeoreferencedFile,
    to_dict
)

# Set environment variables to limit thread usage
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Map AI Processing Service",
    description="A service for processing images of maps and performing automaticfeature matching and georeferencing",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directory configuration
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"

# Initialize services
file_manager = FileManager(UPLOAD_DIR, OUTPUT_DIR)
session_manager = SessionManager(UPLOAD_DIR, OUTPUT_DIR)
feature_matching_service = FeatureMatchingService()
map_cutting_service = MapCuttingService()



@app.get("/", response_model=dict)
async def root():
    """Health check endpoint."""
    response = build_health_response()
    return to_dict(response)


@app.get("/health", response_model=dict)
async def health_check():
    """Detailed health check endpoint."""
    response = build_detailed_health_response()
    return to_dict(response)


@app.post("/match-maps", response_model=dict)
async def match_maps(
    source_image: UploadFile = File(..., description="Source image to be warped"),
    destination_image: UploadFile = File(..., description="Destination/reference image"),
    destination_pgw: UploadFile = File(..., description="Optional PGW file for destination image georeferencing"),
    overlay_transparency: float = Form(0.6, description="Overlay transparency (0.0-1.0)"),
    output_format: str = Form("json", description="Output format: 'json' or 'files'"),
    traffic_decree_id: str = Form(None, description="Optional traffic decree ID") 
):
    """Perform feature matching between two schematic maps with optional georeferencing."""
    
    # Parse and validate form data using request model
    try:
        request_data = parse_feature_matching_form(
            overlay_transparency=overlay_transparency,
            output_format=output_format,
            traffic_decree_id=traffic_decree_id
        )
    except Exception as e:
        error_response = build_error_response(f"Invalid request data: {str(e)}")
        return JSONResponse(content=to_dict(error_response), status_code=422)
    
    # Validate files
    ValidationUtils.validate_image_file(source_image, "Source image")
    ValidationUtils.validate_image_file(destination_image, "Destination image")
    ValidationUtils.validate_pgw_file(destination_pgw)
    
    # Create session
    session_id = session_manager.create_session_id(request_data.traffic_decree_id)
    session_upload_dir, session_output_dir = session_manager.setup_session_directories(session_id)
    
    try:
        # Save uploaded files
        source_path = file_manager.save_uploaded_file(source_image, session_id, "source")
        dest_path = file_manager.save_uploaded_file(destination_image, session_id, "destination")
        
        # Save PGW file if provided
        pgw_path = file_manager.save_pgw_file(destination_pgw, dest_path)
        
        logger.info(f"Processing feature matching for session {session_id}")
        logger.info(f"Source: {source_image.filename}, Destination: {destination_image.filename}")
        if destination_pgw:
            logger.info(f"PGW file provided: {destination_pgw.filename}")
        
        # Perform feature matching
        match_result = feature_matching_service.perform_feature_matching(
            source_image_path=source_path,
            destination_image_path=dest_path,
            output_dir=session_output_dir,
            overlay_transparency=request_data.overlay_transparency
        )
        
        if not match_result["success"]:
            error_response = ErrorResponse.create_error_response(
                error_message=f"Feature matching failed: {match_result['error_message']}",
                session_id=session_id
            )
            return JSONResponse(content=to_dict(error_response), status_code=422)
        
        # Handle response format
        if request_data.output_format == OutputFormat.JSON:
            # Create response using the response builder
            response = build_feature_matching_response(
                session_id=session_id,
                match_result=match_result,
                overlay_transparency=request_data.overlay_transparency,
                pgw_provided=destination_pgw is not None,
                georeferenced_files=match_result.get("georeferenced_files", [])
            )
            return JSONResponse(content=to_dict(response))
        
        else:  # output_format == "files"
            overlay_file = os.path.join(session_output_dir, "warped_overlay_result.png")
            if os.path.exists(overlay_file):
                return FileResponse(
                    overlay_file,
                    media_type="image/png",
                    filename=f"matched_overlay_{session_id}.png"
                )
            else:
                error_response = ErrorResponse.create_error_response(
                    error_message="Output file not generated",
                    session_id=session_id
                )
                return JSONResponse(content=to_dict(error_response), status_code=500)
    
    except HTTPException:
        raise
    
    except Exception as e:
        logger.error(f"Error in feature matching: {str(e)}")
        error_response = ErrorResponse.create_error_response(
            error_message=f"Internal server error: {str(e)}",
            session_id=session_id
        )
        return JSONResponse(content=to_dict(error_response), status_code=500)
    
    finally:
        # Clean up uploaded files (keep outputs for download)
        file_manager.cleanup_session_uploads(session_id)


@app.post("/cut-out-georeferenced-map", response_model=dict)
async def cut_osm_map_endpoint(
    geometry: str = Form(..., description="Geometry as GeoJSON, WKT, or coordinate list JSON string"),
    map_type: str = Form("osm", description="Map type (see endpoint documentation for full list of supported types)"),
    buffer: float = Form(800, description="Buffer distance in meters around the geometry"),
    output_format: str = Form("json", description="Output format: 'json' or 'files'"), 
    traffic_decree_id: str = Form(None, description="Optional traffic decree ID") 
):
    """Cut out a georeferenced map section from various map sources based on geometry input."""
    
    # Parse and validate form data using request model
    try:
        request_data = parse_map_cutting_form(
            geometry=geometry,
            map_type=map_type,
            buffer=buffer,
            output_format=output_format,
            traffic_decree_id=traffic_decree_id
        )
    except Exception as e:
        error_response = build_error_response(f"Invalid request data: {str(e)}")
        return JSONResponse(content=to_dict(error_response), status_code=422)
    
    # Parse geometry
    geometry_input = ValidationUtils.parse_geometry_input(request_data.geometry)
    
    # Create session
    session_id = session_manager.create_session_id(request_data.traffic_decree_id)
    _, session_output_dir = session_manager.setup_session_directories(session_id)
    
    try:
        logger.info(f"Processing map cutting ({request_data.map_type.value}) with geometry input for session {session_id}")
        logger.info(f"Buffer: {request_data.buffer}m")
        
        # Perform map cutting
        cut_result = map_cutting_service.cut_georeferenced_map(
            geometry_input=geometry_input,
            map_type=request_data.map_type.value,
            buffer_meters=request_data.buffer,
            output_dir=session_output_dir,
            target_width=2048,
            output_name="temp"  # Will be overridden by descriptive name
        )
        
        if not cut_result["success"]:
            error_response = ErrorResponse.create_error_response(
                error_message=f"Map cutting failed: {cut_result['error_message']}",
                session_id=session_id
            )
            return JSONResponse(content=to_dict(error_response), status_code=422)
        
        # Handle response format
        if request_data.output_format == OutputFormat.JSON:
            # Create response using the response builder
            response = build_map_cutting_response(
                session_id=session_id,
                cut_result=cut_result,
                map_type=request_data.map_type.value,
                buffer_meters=request_data.buffer,
                target_width=2048
            )
            return JSONResponse(content=to_dict(response))
        
        else:  # output_format == "files"
            map_file = os.path.join(session_output_dir, cut_result["files"]["map_image"])
            if os.path.exists(map_file):
                return FileResponse(
                    map_file,
                    media_type="image/png",
                    filename=f"{cut_result['output_name']}_{session_id}.png"
                )
            else:
                error_response = ErrorResponse.create_error_response(
                    error_message="Output file not generated",
                    session_id=session_id
                )
                return JSONResponse(content=to_dict(error_response), status_code=500)
    
    except HTTPException:
        raise
    
    except Exception as e:
        logger.error(f"Error in map cutting: {str(e)}")
        error_response = ErrorResponse.create_error_response(
            error_message=f"Internal server error: {str(e)}",
            session_id=session_id
        )
        return JSONResponse(content=to_dict(error_response), status_code=500)


@app.post("/cutout-and-match", response_model=dict)
async def cutout_and_match(
    source_image: UploadFile = File(..., description="Source image to be warped and matched"),
    geometry: str = Form(..., description="Geometry as GeoJSON, WKT, or coordinate list JSON string for map cutting"),
    map_type: str = Form(..., description="Map type (see endpoint documentation for full list of supported types)"),
    overlay_transparency: float = Form(0.6, description="Overlay transparency (0.0-1.0)"),
    output_format: str = Form("json", description="Output format: 'json' or 'files'"),
    traffic_decree_id: str = Form(None, description="Optional traffic decree ID") 
):
    """Cut out a map section and perform feature matching with automatic buffer optimization."""
    
    # Parse and validate form data using request model
    try:
        request_data = parse_cutout_and_match_form(
            geometry=geometry,
            map_type=map_type,
            overlay_transparency=overlay_transparency,
            output_format=output_format,
            traffic_decree_id=traffic_decree_id
        )
    except Exception as e:
        error_response = build_error_response(f"Invalid request data: {str(e)}")
        return JSONResponse(content=to_dict(error_response), status_code=422)
    
    # Validate file
    ValidationUtils.validate_image_file(source_image, "Source image")
    
    # Parse geometry
    geometry_input = ValidationUtils.parse_geometry_input(request_data.geometry)
    
    # Get buffer sizes to test based on map type
    test_buffer_sizes = map_cutting_service.get_buffer_sizes_for_map_type(request_data.map_type.value)
    
    # Create session
    session_id = session_manager.create_session_id(request_data.traffic_decree_id)
    session_upload_dir, session_output_dir = session_manager.setup_session_directories(session_id)
    
    try:
        # Save source image
        source_path = file_manager.save_uploaded_file(source_image, session_id, "source")
        
        logger.info(f"Processing cutout-and-match for session {session_id}")
        logger.info(f"Source: {source_image.filename}, Map type: {request_data.map_type.value}")
        
        # Test multiple buffer sizes and find the best one
        buffer_test_result = feature_matching_service.test_multiple_buffers(
            source_image_path=source_path,
            geometry_input=geometry_input,
            map_type=request_data.map_type.value,
            test_buffer_sizes=test_buffer_sizes,
            overlay_transparency=request_data.overlay_transparency,
            session_output_dir=session_output_dir,
            map_cutting_service=map_cutting_service
        )
        
        if not buffer_test_result["success"]:
            error_response = ErrorResponse.create_error_response(
                error_message=buffer_test_result["error_message"],
                session_id=session_id
            )
            return JSONResponse(content=to_dict(error_response), status_code=422)
        
        # Copy best results to final output directory
        feature_matching_service.copy_best_results(buffer_test_result["best_result"], session_output_dir)
        
        # Handle response format
        if output_format == "json":
            response = build_cutout_and_match_response(
                session_id=session_id,
                match_result=buffer_test_result["best_result"]["match_result"],
                cut_result=buffer_test_result["best_result"]["cut_result"],
                overlay_transparency=overlay_transparency,
                map_type=map_type,
                best_buffer=buffer_test_result["best_buffer"],
                test_buffer_sizes=test_buffer_sizes,
                buffer_results=buffer_test_result["buffer_results"]
            )
            return JSONResponse(content=to_dict(response))
        
        else:  # output_format == "files"
            tif_file = os.path.join(session_output_dir, "warped_source.tif")
            if os.path.exists(tif_file):
                return FileResponse(
                    tif_file,
                    media_type="image/tiff",
                    filename=f"warped_source_{session_id}.tif"
                )
            else:
                raise HTTPException(status_code=500, detail="GeoTIFF file not generated")
    
    except HTTPException:
        raise
    
    except Exception as e:
        logger.error(f"Error in cutout-and-match: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    
    finally:
        file_manager.cleanup_session_uploads(session_id)


@app.post("/cutout-and-match-with-url")
async def cutout_and_match_with_url(
    image_url: str = Form(..., description="URL of the source image to be warped and matched"),
    geometry: str = Form(..., description="Geometry as GeoJSON, WKT, or coordinate list JSON string for map cutting"),
    map_type: str = Form(..., description="Map type (see endpoint documentation for full list of supported types)"),
    overlay_transparency: float = Form(0.6, description="Overlay transparency (0.0-1.0)"),
    output_format: str = Form("json", description="Output format: 'json' or 'files'"),
    traffic_decree_id: str = Form(None, description="Optional traffic decree ID") 
):
    """Cut out a map section and perform feature matching with a source image from URL."""
    
    # Validate inputs
    ValidationUtils.validate_output_format(output_format)
    ValidationUtils.validate_transparency(overlay_transparency)
    ValidationUtils.validate_map_type(map_type)
    ValidationUtils.validate_image_url(image_url)
    
    # Parse geometry
    geometry_input = ValidationUtils.parse_geometry_input(geometry)
    
    # Get buffer sizes to test based on map type
    test_buffer_sizes = map_cutting_service.get_buffer_sizes_for_map_type(map_type)
    
    # Create session
    session_id = session_manager.create_session_id(traffic_decree_id)
    session_upload_dir, session_output_dir = session_manager.setup_session_directories(session_id)
    
    try:
        # Download source image from URL
        source_path = file_manager.download_image_from_url(image_url, session_id, "source")
        
        logger.info(f"Processing cutout-and-match-with-url for session {session_id}")
        logger.info(f"Source URL: {image_url}, Map type: {map_type}")
        
        # Test multiple buffer sizes and find the best one
        buffer_test_result = feature_matching_service.test_multiple_buffers(
            source_image_path=source_path,
            geometry_input=geometry_input,
            map_type=map_type,
            test_buffer_sizes=test_buffer_sizes,
            overlay_transparency=overlay_transparency,
            session_output_dir=session_output_dir,
            map_cutting_service=map_cutting_service
        )
        
        if not buffer_test_result["success"]:
            raise HTTPException(
                status_code=422, 
                detail=buffer_test_result["error_message"]
            )
        
        # Copy best results to final output directory
        feature_matching_service.copy_best_results(buffer_test_result["best_result"], session_output_dir)
        
        # Handle response format
        if output_format == "json":
            response = build_cutout_and_match_response(
                session_id=session_id,
                match_result=buffer_test_result["best_result"]["match_result"],
                cut_result=buffer_test_result["best_result"]["cut_result"],
                overlay_transparency=overlay_transparency,
                map_type=map_type,
                best_buffer=buffer_test_result["best_buffer"],
                test_buffer_sizes=test_buffer_sizes,
                buffer_results=buffer_test_result["buffer_results"]
            )
            return JSONResponse(content=to_dict(response))
        
        else:  # output_format == "files"
            tif_file = os.path.join(session_output_dir, "warped_source.tif")
            if os.path.exists(tif_file):
                return FileResponse(
                    tif_file,
                    media_type="image/tiff",
                    filename=f"warped_source_{session_id}.tif"
                )
            else:
                raise HTTPException(status_code=500, detail="GeoTIFF file not generated")
    
    except HTTPException:
        raise
    
    except Exception as e:
        logger.error(f"Error in cutout-and-match-with-url: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    
    finally:
        file_manager.cleanup_session_uploads(session_id)


@app.get("/download/{session_id}/{file_path:path}")
async def download_file(session_id: str, file_path: str):
    """Download processed files by session ID and file path (supports subdirectories)."""
    try:
        # Validate and get full file path
        full_file_path = file_manager.validate_file_path_security(session_id, file_path)
        
        # Get filename and media type
        filename = os.path.basename(file_path)
        media_type = file_manager.get_media_type(filename)
        
        return FileResponse(full_file_path, media_type=media_type, filename=filename)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in file download: {str(e)}")
        raise HTTPException(status_code=500, detail="File download failed")


@app.delete("/sessions/{session_id}", response_model=dict)
async def cleanup_session(session_id: str):
    """Clean up a specific session's files."""
    try:
        SessionValidator.validate_session_id(session_id)
        success = file_manager.cleanup_session_outputs(session_id)
        
        if success:
            response = SessionCleanupResponse(message=f"Session {session_id} cleaned up successfully")
            return JSONResponse(content=to_dict(response))
        else:
            error_response = ErrorResponse.create_error_response(
                error_message="Session not found",
                session_id=session_id
            )
            return JSONResponse(content=to_dict(error_response), status_code=404)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cleaning up session {session_id}: {str(e)}")
        error_response = ErrorResponse.create_error_response(
            error_message=f"Failed to cleanup session: {str(e)}",
            session_id=session_id
        )
        return JSONResponse(content=to_dict(error_response), status_code=500)


@app.get("/sessions", response_model=dict)
async def list_sessions():
    """List all active sessions."""
    try:
        sessions = file_manager.list_all_sessions()
        # Convert to SessionInfo objects
        from models.response_models import SessionInfo
        session_infos = []
        for session in sessions:
            session_infos.append(SessionInfo(
                session_id=session["session_id"],
                files=session["files"],
                created=session["created"]
            ))
        
        response = SessionListResponse(sessions=session_infos)
        return JSONResponse(content=to_dict(response))
    
    except Exception as e:
        logger.error(f"Error listing sessions: {str(e)}")
        error_response = ErrorResponse.create_error_response(f"Failed to list sessions: {str(e)}")
        return JSONResponse(content=to_dict(error_response), status_code=500)

if __name__ == "__main__":
    # Run the server with dynamic port for cloud deployment
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )