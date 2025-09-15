"""
Cleanup utilities for managing disk space and memory
"""
import os
import shutil
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any

logger = logging.getLogger(__name__)


class CleanupManager:
    """Manages automatic cleanup of files and directories to prevent memory/disk issues"""
    
    def __init__(self, max_age_hours: int = 24, max_files: int = 1000, max_size_mb: int = 500):
        """
        Initialize cleanup manager
        
        Args:
            max_age_hours: Maximum age of files before cleanup (default 24 hours)
            max_files: Maximum number of files to keep (default 1000)
            max_size_mb: Maximum total size in MB before cleanup (default 500MB)
        """
        self.max_age_hours = max_age_hours
        self.max_files = max_files
        self.max_size_mb = max_size_mb
    
    def cleanup_old_files(self, directory: str, file_pattern: str = "*") -> Dict[str, Any]:
        """
        Clean up old files based on age, count, and size limits
        
        Args:
            directory: Directory to clean
            file_pattern: File pattern to match (default all files)
            
        Returns:
            Dictionary with cleanup statistics
        """
        try:
            dir_path = Path(directory)
            if not dir_path.exists():
                return {"success": True, "message": "Directory does not exist", "files_removed": 0}
            
            # Get all files matching pattern
            files = list(dir_path.glob(file_pattern))
            if not files:
                return {"success": True, "message": "No files to clean", "files_removed": 0}
            
            # Sort by modification time (oldest first)
            files.sort(key=lambda f: f.stat().st_mtime)
            
            files_to_remove = []
            total_size = sum(f.stat().st_size for f in files if f.is_file())
            total_size_mb = total_size / (1024 * 1024)
            
            # Check if cleanup is needed
            cutoff_time = datetime.now() - timedelta(hours=self.max_age_hours)
            cutoff_timestamp = cutoff_time.timestamp()
            
            # Remove files based on age
            for file in files:
                if file.is_file() and file.stat().st_mtime < cutoff_timestamp:
                    files_to_remove.append(file)
            
            # Remove oldest files if count exceeds limit
            if len(files) > self.max_files:
                remaining_files = [f for f in files if f not in files_to_remove]
                excess_count = len(remaining_files) - self.max_files
                if excess_count > 0:
                    files_to_remove.extend(remaining_files[:excess_count])
            
            # Remove oldest files if size exceeds limit
            if total_size_mb > self.max_size_mb:
                remaining_files = [f for f in files if f not in files_to_remove]
                current_size = sum(f.stat().st_size for f in remaining_files) / (1024 * 1024)
                
                for file in remaining_files:
                    if current_size <= self.max_size_mb:
                        break
                    if file not in files_to_remove:
                        files_to_remove.append(file)
                        current_size -= file.stat().st_size / (1024 * 1024)
            
            # Remove the files
            removed_count = 0
            removed_size = 0
            for file in files_to_remove:
                try:
                    file_size = file.stat().st_size
                    file.unlink()
                    removed_count += 1
                    removed_size += file_size
                    logger.info(f"🗑️ Removed old file: {file.name}")
                except Exception as e:
                    logger.warning(f"Failed to remove {file}: {e}")
            
            removed_size_mb = removed_size / (1024 * 1024)
            
            return {
                "success": True,
                "files_removed": removed_count,
                "size_freed_mb": round(removed_size_mb, 2),
                "total_files_before": len(files),
                "total_size_before_mb": round(total_size_mb, 2)
            }
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            return {"success": False, "error": str(e)}
    
    def get_directory_stats(self, directory: str) -> Dict[str, Any]:
        """Get statistics about a directory"""
        try:
            dir_path = Path(directory)
            if not dir_path.exists():
                return {"exists": False}
            
            files = [f for f in dir_path.iterdir() if f.is_file()]
            total_size = sum(f.stat().st_size for f in files)
            total_size_mb = total_size / (1024 * 1024)
            
            return {
                "exists": True,
                "file_count": len(files),
                "total_size_mb": round(total_size_mb, 2),
                "oldest_file": min(files, key=lambda f: f.stat().st_mtime).name if files else None,
                "newest_file": max(files, key=lambda f: f.stat().st_mtime).name if files else None
            }
            
        except Exception as e:
            return {"exists": True, "error": str(e)}

