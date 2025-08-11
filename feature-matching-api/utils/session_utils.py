"""
Session utilities for managing sessions and directories
"""
import os
import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


class SessionManager:
    """Manages session creation and directory setup"""
    
    def __init__(self, upload_dir: str, output_dir: str):
        self.upload_dir = upload_dir
        self.output_dir = output_dir
    
    def create_session_id(self, traffic_decree_id: Optional[str] = None) -> str:
        """
        Create a unique session ID
        
        Args:
            traffic_decree_id: Optional traffic decree ID to use as session ID
            
        Returns:
            Session ID string
        """
        if traffic_decree_id:
            return traffic_decree_id
        else:
            return f"session_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    
    def setup_session_directories(self, session_id: str) -> tuple[str, str]:
        """
        Set up session directories for uploads and outputs
        
        Args:
            session_id: Session identifier
            
        Returns:
            Tuple of (upload_dir, output_dir) paths
        """
        session_upload_dir = os.path.join(self.upload_dir, session_id)
        session_output_dir = os.path.join(self.output_dir, session_id)
        
        # Create directories
        os.makedirs(session_upload_dir, exist_ok=True)
        os.makedirs(session_output_dir, exist_ok=True)
        
        logger.info(f"Created session directories for {session_id}")
        logger.info(f"Upload dir: {session_upload_dir}")
        logger.info(f"Output dir: {session_output_dir}")
        
        return session_upload_dir, session_output_dir
    
    def get_session_upload_dir(self, session_id: str) -> str:
        """Get session upload directory path"""
        return os.path.join(self.upload_dir, session_id)
    
    def get_session_output_dir(self, session_id: str) -> str:
        """Get session output directory path"""
        return os.path.join(self.output_dir, session_id)