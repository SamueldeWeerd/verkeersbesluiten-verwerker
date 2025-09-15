"""
Cleanup utilities for managing session files and memory
"""
import os
import shutil
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any

logger = logging.getLogger(__name__)


class SessionCleanupManager:
    """Manages automatic cleanup of session files to prevent disk space issues"""
    
    def __init__(self, max_age_hours: int = 24, max_sessions: int = 100, max_size_gb: int = 5):
        """
        Initialize session cleanup manager
        
        Args:
            max_age_hours: Maximum age of sessions before cleanup (default 24 hours)
            max_sessions: Maximum number of sessions to keep (default 100)
            max_size_gb: Maximum total size in GB before cleanup (default 5GB)
        """
        self.max_age_hours = max_age_hours
        self.max_sessions = max_sessions
        self.max_size_gb = max_size_gb
    
    def cleanup_old_sessions(self, outputs_dir: str = "outputs") -> Dict[str, Any]:
        """
        Clean up old session directories based on age, count, and size limits
        
        Args:
            outputs_dir: Directory containing session folders
            
        Returns:
            Dictionary with cleanup statistics
        """
        try:
            outputs_path = Path(outputs_dir)
            if not outputs_path.exists():
                return {"success": True, "message": "Outputs directory does not exist", "sessions_removed": 0}
            
            # Get all session directories
            session_dirs = [d for d in outputs_path.iterdir() if d.is_dir() and d.name.startswith("session_")]
            if not session_dirs:
                return {"success": True, "message": "No sessions to clean", "sessions_removed": 0}
            
            # Sort by modification time (oldest first)
            session_dirs.sort(key=lambda d: d.stat().st_mtime)
            
            sessions_to_remove = []
            total_size = self._calculate_directory_size(outputs_path)
            total_size_gb = total_size / (1024 * 1024 * 1024)
            
            # Check if cleanup is needed
            cutoff_time = datetime.now() - timedelta(hours=self.max_age_hours)
            cutoff_timestamp = cutoff_time.timestamp()
            
            # Remove sessions based on age
            for session_dir in session_dirs:
                if session_dir.stat().st_mtime < cutoff_timestamp:
                    sessions_to_remove.append(session_dir)
            
            # Remove oldest sessions if count exceeds limit
            if len(session_dirs) > self.max_sessions:
                remaining_sessions = [d for d in session_dirs if d not in sessions_to_remove]
                excess_count = len(remaining_sessions) - self.max_sessions
                if excess_count > 0:
                    sessions_to_remove.extend(remaining_sessions[:excess_count])
            
            # Remove oldest sessions if size exceeds limit
            if total_size_gb > self.max_size_gb:
                remaining_sessions = [d for d in session_dirs if d not in sessions_to_remove]
                current_size_gb = total_size_gb
                
                for session_dir in remaining_sessions:
                    if current_size_gb <= self.max_size_gb:
                        break
                    if session_dir not in sessions_to_remove:
                        session_size = self._calculate_directory_size(session_dir) / (1024 * 1024 * 1024)
                        sessions_to_remove.append(session_dir)
                        current_size_gb -= session_size
            
            # Remove the sessions
            removed_count = 0
            removed_size = 0
            for session_dir in sessions_to_remove:
                try:
                    session_size = self._calculate_directory_size(session_dir)
                    shutil.rmtree(session_dir)
                    removed_count += 1
                    removed_size += session_size
                    logger.info(f"🗑️ Removed old session: {session_dir.name}")
                except Exception as e:
                    logger.warning(f"Failed to remove session {session_dir}: {e}")
            
            removed_size_gb = removed_size / (1024 * 1024 * 1024)
            
            return {
                "success": True,
                "sessions_removed": removed_count,
                "size_freed_gb": round(removed_size_gb, 2),
                "total_sessions_before": len(session_dirs),
                "total_size_before_gb": round(total_size_gb, 2)
            }
            
        except Exception as e:
            logger.error(f"Error during session cleanup: {e}")
            return {"success": False, "error": str(e)}
    
    def _calculate_directory_size(self, directory: Path) -> int:
        """Calculate total size of a directory in bytes"""
        total_size = 0
        try:
            for dirpath, dirnames, filenames in os.walk(directory):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    try:
                        total_size += os.path.getsize(filepath)
                    except OSError:
                        continue
        except Exception:
            pass
        return total_size
    
    def get_outputs_stats(self, outputs_dir: str = "outputs") -> Dict[str, Any]:
        """Get statistics about the outputs directory"""
        try:
            outputs_path = Path(outputs_dir)
            if not outputs_path.exists():
                return {"exists": False}
            
            session_dirs = [d for d in outputs_path.iterdir() if d.is_dir() and d.name.startswith("session_")]
            total_size = self._calculate_directory_size(outputs_path)
            total_size_gb = total_size / (1024 * 1024 * 1024)
            
            return {
                "exists": True,
                "session_count": len(session_dirs),
                "total_size_gb": round(total_size_gb, 2),
                "oldest_session": min(session_dirs, key=lambda d: d.stat().st_mtime).name if session_dirs else None,
                "newest_session": max(session_dirs, key=lambda d: d.stat().st_mtime).name if session_dirs else None
            }
            
        except Exception as e:
            return {"exists": True, "error": str(e)}

