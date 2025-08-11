"""
File utilities for handling uploads, downloads, and file operations
"""
import os
import shutil
import tempfile
import logging
import requests
from typing import Dict, Any, Optional
from fastapi import UploadFile, HTTPException
from PIL import Image

logger = logging.getLogger(__name__)


class FileManager:
    """Handles file operations for the API"""
    
    def __init__(self, upload_dir: str, output_dir: str):
        self.upload_dir = upload_dir
        self.output_dir = output_dir
        
        # Create directories if they don't exist
        os.makedirs(upload_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
    
    def save_uploaded_file(self, file: UploadFile, session_id: str, prefix: str = "") -> str:
        """
        Save an uploaded file to the session directory
        
        Args:
            file: Uploaded file
            session_id: Session identifier
            prefix: Optional prefix for filename
            
        Returns:
            Path to saved file
        """
        session_upload_dir = os.path.join(self.upload_dir, session_id)
        os.makedirs(session_upload_dir, exist_ok=True)
        
        filename = f"{prefix}_{file.filename}" if prefix else file.filename
        file_path = os.path.join(session_upload_dir, filename)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"Saved uploaded file: {file_path}")
        return file_path
    
    def download_image_from_url(self, url: str, session_id: str, prefix: str = "source") -> str:
        """
        Download an image from URL and save it to session directory
        
        Args:
            url: Image URL
            session_id: Session identifier  
            prefix: Prefix for filename
            
        Returns:
            Path to downloaded file
            
        Raises:
            HTTPException: If download fails or file is not a valid image
        """
        logger.info(f"Downloading image from URL: {url}")
        
        session_upload_dir = os.path.join(self.upload_dir, session_id)
        os.makedirs(session_upload_dir, exist_ok=True)
        
        try:
            response = requests.get(url, timeout=30, stream=True)
            response.raise_for_status()
            
            # Get filename from URL or use default
            url_filename = os.path.basename(url.split('?')[0])  # Remove query parameters
            if not url_filename or '.' not in url_filename:
                url_filename = "source_image.jpg"
            
            file_path = os.path.join(session_upload_dir, f"{prefix}_{url_filename}")
            
            with open(file_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info(f"Downloaded image: {len(open(file_path, 'rb').read())} bytes")
            
            # Validate that the downloaded file is an image
            try:
                with Image.open(file_path) as img:
                    img.verify()
                logger.info("Image validation successful")
            except Exception as e:
                logger.error(f"Downloaded file is not a valid image: {str(e)}")
                os.remove(file_path)  # Clean up invalid file
                raise HTTPException(status_code=400, detail="Downloaded file is not a valid image")
            
            return file_path
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to download image from URL: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Failed to download image from URL: {str(e)}")
    
    def save_pgw_file(self, pgw_file: UploadFile, destination_path: str) -> Optional[str]:
        """
        Save PGW file with matching name to destination image
        
        Args:
            pgw_file: Uploaded PGW file
            destination_path: Path to destination image
            
        Returns:
            Path to saved PGW file or None if no file provided
        """
        if pgw_file is None:
            return None
        
        # Create PGW filename based on destination image name
        dest_base_name = os.path.splitext(destination_path)[0]
        pgw_path = f"{dest_base_name}.pgw"
        
        with open(pgw_path, "wb") as buffer:
            shutil.copyfileobj(pgw_file.file, buffer)
        
        logger.info(f"Saved PGW file: {pgw_path}")
        return pgw_path
    
    def cleanup_session_uploads(self, session_id: str) -> None:
        """Clean up uploaded files for a session (keep outputs)"""
        session_upload_dir = os.path.join(self.upload_dir, session_id)
        
        try:
            if os.path.exists(session_upload_dir):
                shutil.rmtree(session_upload_dir)
                logger.info(f"Cleaned up upload directory: {session_upload_dir}")
        except Exception as e:
            logger.warning(f"Failed to clean up upload directory: {str(e)}")
    
    def cleanup_session_outputs(self, session_id: str) -> bool:
        """Clean up output files for a session"""
        session_output_dir = os.path.join(self.output_dir, session_id)
        
        try:
            if os.path.exists(session_output_dir):
                shutil.rmtree(session_output_dir)
                logger.info(f"Cleaned up output directory: {session_output_dir}")
                return True
            else:
                return False
        except Exception as e:
            logger.error(f"Failed to clean up output directory: {str(e)}")
            raise
    
    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a session"""
        session_path = os.path.join(self.output_dir, session_id)
        
        if not os.path.exists(session_path) or not os.path.isdir(session_path):
            return None
        
        try:
            files = os.listdir(session_path)
            return {
                "session_id": session_id,
                "files": files,
                "created": os.path.getctime(session_path)
            }
        except Exception as e:
            logger.error(f"Error getting session info: {str(e)}")
            return None
    
    def list_all_sessions(self) -> list:
        """List all active sessions"""
        sessions = []
        
        try:
            if os.path.exists(self.output_dir):
                for session_dir in os.listdir(self.output_dir):
                    # Skip hidden files and system files
                    if session_dir.startswith('.') or not os.path.isdir(os.path.join(self.output_dir, session_dir)):
                        continue
                    # Include all directories (both session_* and traffic decree IDs)
                    session_info = self.get_session_info(session_dir)
                    if session_info:
                        sessions.append(session_info)
            
            return sessions
        except Exception as e:
            logger.error(f"Error listing sessions: {str(e)}")
            raise
    
    def validate_file_path_security(self, session_id: str, file_path: str) -> str:
        """
        Validate file path for security and return full path
        
        Args:
            session_id: Session identifier
            file_path: Requested file path
            
        Returns:
            Full validated file path
            
        Raises:
            HTTPException: If path is invalid or insecure
        """
        # Validate session_id format for security
        # Allow both generated session IDs (session_*) and traffic decree IDs
        if ".." in session_id or "/" in session_id or len(session_id.strip()) == 0:
            raise HTTPException(status_code=400, detail="Invalid session ID")
        
        # Validate file_path for security (allow forward slashes for subdirectories)
        if ".." in file_path:
            raise HTTPException(status_code=400, detail="Invalid file path")
        
        # Build the full file path
        full_file_path = os.path.join(self.output_dir, session_id, file_path)
        
        # Security check: ensure the resolved path is still within the session directory
        session_dir = os.path.join(self.output_dir, session_id)
        if not full_file_path.startswith(session_dir):
            raise HTTPException(status_code=400, detail="Invalid file path")
        
        if not os.path.exists(full_file_path):
            raise HTTPException(status_code=404, detail="File not found")
        
        return full_file_path
    
    def get_media_type(self, filename: str) -> str:
        """Get appropriate media type for file"""
        filename_lower = filename.lower()
        
        if filename_lower.endswith(('.png', '.jpg', '.jpeg')):
            extension = filename_lower.split('.')[-1].replace('jpg', 'jpeg')
            return f"image/{extension}"
        elif filename_lower.endswith(('.tiff', '.tif')):
            return "image/tiff"
        elif filename_lower.endswith(('.pgw', '.prj')):
            return "text/plain"
        else:
            return "application/octet-stream"