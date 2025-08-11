"""
Response models for standardized API outputs
"""
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
from datetime import datetime
from enum import Enum


class QualityStatus(str, Enum):
    """Quality assessment levels"""
    EXCELLENT = "excellent"
    GOOD = "good" 
    FAIR = "fair"
    POOR = "poor"


class GeoreferencedFileType(str, Enum):
    """Types of georeferenced files"""
    GEOTIFF = "GeoTIFF"
    WORLD_FILE = "World File"
    PROJECTION_FILE = "Projection File"


@dataclass
class FeatureMatchingProcessingInfo:
    """Processing information specific to feature matching operations"""
    matches_count: int
    inlier_ratio: float
    overlay_transparency: float
    georeferenced: bool
    pgw_provided: bool


@dataclass
class MapCuttingProcessingInfo:
    """Processing information specific to map cutting operations"""
    input_type: str
    buffer_meters: float
    map_type: str
    target_width: int
    actual_size: Dict[str, int]  # {"width": 2048, "height": 1536}
    geometry_type: Optional[str] = None
    geometry_format: Optional[str] = None
    point_count: Optional[int] = None


@dataclass
class QualityAssessment:
    """Quality assessment for feature matching results"""
    status: QualityStatus
    inlier_ratio: float
    matches_count: int
    
    @classmethod
    def from_inlier_ratio(cls, inlier_ratio: float, matches_count: int) -> 'QualityAssessment':
        """Create quality assessment from inlier ratio"""
        if inlier_ratio >= 0.3:
            status = QualityStatus.EXCELLENT
        elif inlier_ratio >= 0.2:
            status = QualityStatus.GOOD
        elif inlier_ratio >= 0.1:
            status = QualityStatus.FAIR
        else:
            status = QualityStatus.POOR
            
        return cls(
            status=status,
            inlier_ratio=inlier_ratio,
            matches_count=matches_count
        )


@dataclass
class FeatureMatchingFiles:
    """File paths for feature matching outputs"""
    all_matches: str
    inlier_matches: str
    outlier_matches: str
    analysis: str
    warped_overlay: str
    warped_source: Optional[str] = None


@dataclass
class MapCuttingFiles:
    """File paths for map cutting outputs"""
    map_image: str
    world_file: str


@dataclass
class GeoreferencedFile:
    """Information about a georeferenced file"""
    type: GeoreferencedFileType
    url: str


# MapCuttingInfo removed - now using MapCuttingProcessingInfo directly


@dataclass
class BufferTestResult:
    """Result from testing a specific buffer size"""
    buffer_meters: int
    map_type: str
    matches_count: int
    inlier_count: int
    inlier_ratio: float


@dataclass
class BufferSelection:
    """Information about buffer selection process"""
    tested_buffers: List[int]
    results: List[BufferTestResult]
    selection_criteria: str
    inspection_folders: List[str]


# MapFiles removed - now using MapCuttingFiles


@dataclass
class BaseResponse:
    """Base response structure for all API endpoints"""
    success: bool
    session_id: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    error_message: Optional[str] = None


@dataclass
class FeatureMatchingResponse:
    """Standardized response for feature matching operations"""
    success: bool
    session_id: str
    timestamp: str
    processing_info: FeatureMatchingProcessingInfo
    output_files: Dict[str, FeatureMatchingFiles] 
    georeferenced_files: List[GeoreferencedFile]
    quality_assessment: QualityAssessment
    error_message: Optional[str] = None
    
    @classmethod
    def create_success_response(
        cls,
        session_id: str,
        matches_count: int,
        inlier_ratio: float,
        overlay_transparency: float,
        georeferenced: bool,
        pgw_provided: bool,
        georeferenced_files: List[GeoreferencedFile]
    ) -> 'FeatureMatchingResponse':
        """Create successful feature matching response"""
        
        processing_info = FeatureMatchingProcessingInfo(
            matches_count=matches_count,
            inlier_ratio=inlier_ratio,
            overlay_transparency=overlay_transparency,
            georeferenced=georeferenced,
            pgw_provided=pgw_provided
        )
        
        feature_matching_files = FeatureMatchingFiles(
            all_matches=f"/download/{session_id}/all_feature_matches.png",
            inlier_matches=f"/download/{session_id}/inlier_matches.png",
            outlier_matches=f"/download/{session_id}/outlier_matches.png",
            analysis=f"/download/{session_id}/feature_matching_analysis.png",
            warped_overlay=f"/download/{session_id}/warped_overlay_result.png",
            warped_source=f"/download/{session_id}/warped_source.png"
        )
        
        output_files = {"feature_matching": feature_matching_files}
        
        quality_assessment = QualityAssessment.from_inlier_ratio(inlier_ratio, matches_count)
        
        return cls(
            success=True,
            session_id=session_id,
            timestamp=datetime.now().isoformat(),
            processing_info=processing_info,
            output_files=output_files,
            georeferenced_files=georeferenced_files,
            quality_assessment=quality_assessment
        )


@dataclass
class MapCuttingResponse:
    """Standardized response for map cutting operations - COMPLETELY DIFFERENT from feature matching"""
    success: bool
    session_id: str
    timestamp: str
    processing_info: MapCuttingProcessingInfo
    bounds_rd: Dict[str, float]
    files: MapCuttingFiles
    error_message: Optional[str] = None
    
    @classmethod
    def create_success_response(
        cls,
        session_id: str,
        map_type: str,
        bounds_rd: Dict[str, float],
        output_name: str,
        buffer_meters: float,
        target_width: int,
        actual_size: Dict[str, int],
        geometry_info: Optional[Dict[str, Any]] = None
    ) -> 'MapCuttingResponse':
        """Create successful map cutting response"""
        
        processing_info = MapCuttingProcessingInfo(
            input_type="geometry",
            buffer_meters=buffer_meters,
            map_type=map_type,
            target_width=target_width,
            actual_size=actual_size
        )
        
        # Add geometry-specific information
        if geometry_info:
            processing_info.geometry_type = geometry_info.get("geometry_type")
            processing_info.geometry_format = geometry_info.get("geometry_format")
            processing_info.point_count = geometry_info.get("point_count")
        
        files = MapCuttingFiles(
            map_image=f"/download/{session_id}/{output_name}.png",
            world_file=f"/download/{session_id}/{output_name}.pgw"
        )
        
        return cls(
            success=True,
            session_id=session_id,
            timestamp=datetime.now().isoformat(),
            processing_info=processing_info,
            bounds_rd=bounds_rd,
            files=files
        )


@dataclass
class CutoutAndMatchResponse:
    """Standardized response for cutout-and-match operations (combines both feature matching AND map cutting)"""
    success: bool
    session_id: str
    timestamp: str
    processing_info: FeatureMatchingProcessingInfo  # Feature matching info
    output_files: Dict[str, FeatureMatchingFiles]   # {"feature_matching": FeatureMatchingFiles}
    georeferenced_files: List[GeoreferencedFile]
    map_cutting: MapCuttingProcessingInfo           # Map cutting info 
    error_message: Optional[str] = None
    buffer_selection: Optional[BufferSelection] = None  # This one can actually be optional
    
    @classmethod
    def create_success_response(
        cls,
        session_id: str,
        matches_count: int,
        inlier_ratio: float,
        overlay_transparency: float,
        georeferenced: bool,
        georeferenced_files: List[GeoreferencedFile],
        map_type: str,
        selected_buffer_meters: int,
        bounds_rd: Dict[str, float],
        destination_name: str,
        buffer_results: List[BufferTestResult],
        tested_buffers: List[int],
        target_width: int = 2048,
        actual_size: Optional[Dict[str, int]] = None,
        source_url: Optional[str] = None
    ) -> 'CutoutAndMatchResponse':
        """Create successful cutout-and-match response"""
        
        # Feature matching processing info
        processing_info = FeatureMatchingProcessingInfo(
            matches_count=matches_count,
            inlier_ratio=inlier_ratio,
            overlay_transparency=overlay_transparency,
            georeferenced=georeferenced,
            pgw_provided=True  # Always true for cutout operations
        )
        
        # Feature matching output files
        feature_matching_files = FeatureMatchingFiles(
            all_matches=f"/download/{session_id}/all_feature_matches.png",
            inlier_matches=f"/download/{session_id}/inlier_matches.png",
            outlier_matches=f"/download/{session_id}/outlier_matches.png",
            analysis=f"/download/{session_id}/feature_matching_analysis.png",
            warped_overlay=f"/download/{session_id}/warped_overlay_result.png"
        )
        
        output_files = {"feature_matching": feature_matching_files}
        
        # Map cutting processing info
        map_cutting_info = MapCuttingProcessingInfo(
            input_type="geometry",
            buffer_meters=float(selected_buffer_meters),
            map_type=map_type,
            target_width=target_width,
            actual_size=actual_size or {"width": target_width, "height": target_width}
        )
        
        buffer_selection = BufferSelection(
            tested_buffers=tested_buffers,
            results=buffer_results,
            selection_criteria="maximum_inlier_count",
            inspection_folders=[f"/download/{session_id}/buffer_{size}m_cutout/" for size in tested_buffers]
        )
        
        return cls(
            success=True,
            session_id=session_id,
            timestamp=datetime.now().isoformat(),
            processing_info=processing_info,
            output_files=output_files,
            georeferenced_files=georeferenced_files,
            map_cutting=map_cutting_info,
            buffer_selection=buffer_selection
        )


@dataclass
class ErrorResponse:
    """Standardized error response"""
    success: bool
    session_id: str
    timestamp: str
    error_message: Optional[str] = None
    
    @classmethod
    def create_error_response(
        cls,
        error_message: str,
        session_id: Optional[str] = None
    ) -> 'ErrorResponse':
        """Create error response"""
        return cls(
            success=False,
            session_id=session_id or "unknown",
            timestamp=datetime.now().isoformat(),
            error_message=error_message
        )


@dataclass
class HealthCheckResponse:
    """Health check response"""
    service: str
    status: str
    version: str
    timestamp: str
    available_endpoints: Dict[str, str]


@dataclass
class DetailedHealthResponse:
    """Detailed health check response"""
    status: str
    timestamp: str
    services: Dict[str, str]


@dataclass
class SessionInfo:
    """Information about a session"""
    session_id: str
    files: List[str]
    created: str


@dataclass
class SessionListResponse:
    """Response for listing sessions"""
    sessions: List[SessionInfo]


@dataclass
class SessionCleanupResponse:
    """Response for session cleanup"""
    message: str


# Utility functions for converting dataclasses to dicts for JSON responses
def to_dict(obj) -> Dict[str, Any]:
    """Convert dataclass to dictionary for JSON serialization"""
    if hasattr(obj, '__dataclass_fields__'):
        result = {}
        for field_name, field_value in obj.__dict__.items():
            if isinstance(field_value, list):
                result[field_name] = [to_dict(item) for item in field_value]
            elif isinstance(field_value, dict):
                result[field_name] = {k: to_dict(v) for k, v in field_value.items()}
            elif hasattr(field_value, '__dataclass_fields__'):
                result[field_name] = to_dict(field_value)
            elif isinstance(field_value, Enum):
                result[field_name] = field_value.value
            else:
                result[field_name] = field_value
        return result
    elif isinstance(obj, list):
        return [to_dict(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: to_dict(v) for k, v in obj.items()}
    elif hasattr(obj, '__dataclass_fields__'):
        return to_dict(obj)  # Handle dataclasses at any level
    elif isinstance(obj, Enum):
        return obj.value
    else:
        return obj