"""
Memory utilities for monitoring and managing memory usage
"""
import os
import gc
import psutil
import logging

logger = logging.getLogger(__name__)


class MemoryManager:
    """Utilities for memory monitoring and management"""
    
    @staticmethod
    def get_memory_usage() -> float:
        """
        Get current memory usage of the process in MB.
        
        Returns:
            Memory usage in megabytes
        """
        try:
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024  # Convert to MB
        except Exception as e:
            logger.warning(f"Could not get memory usage: {str(e)}")
            return 0.0
    
    @staticmethod
    def log_memory_usage(step: str) -> None:
        """
        Log current memory usage for debugging.
        
        Args:
            step: Description of the current processing step
        """
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            logger.info(f"[{step}] Memory usage: {memory_mb:.1f} MB")
        except Exception as e:
            logger.warning(f"Could not log memory usage for {step}: {str(e)}")
    
    @staticmethod
    def force_garbage_collection() -> float:
        """
        Force garbage collection and return memory freed.
        
        Returns:
            Amount of memory freed in MB
        """
        memory_before = MemoryManager.get_memory_usage()
        gc.collect()
        memory_after = MemoryManager.get_memory_usage()
        memory_freed = memory_before - memory_after
        
        if memory_freed > 0:
            logger.info(f"Garbage collection freed {memory_freed:.1f} MB")
        
        return memory_freed
    
    @staticmethod
    def check_memory_threshold(threshold_mb: float = 1000.0) -> bool:
        """
        Check if memory usage exceeds a threshold.
        
        Args:
            threshold_mb: Memory threshold in MB
            
        Returns:
            True if memory usage exceeds threshold
        """
        current_memory = MemoryManager.get_memory_usage()
        if current_memory > threshold_mb:
            logger.warning(f"Memory usage ({current_memory:.1f} MB) exceeds threshold ({threshold_mb:.1f} MB)")
            return True
        return False
    
    @staticmethod
    def get_system_memory_info() -> dict:
        """
        Get system memory information.
        
        Returns:
            Dictionary with system memory stats
        """
        try:
            memory = psutil.virtual_memory()
            return {
                "total": memory.total / 1024 / 1024 / 1024,  # GB
                "available": memory.available / 1024 / 1024 / 1024,  # GB
                "percent_used": memory.percent,
                "free": memory.free / 1024 / 1024 / 1024  # GB
            }
        except Exception as e:
            logger.error(f"Could not get system memory info: {str(e)}")
            return {}
    
    @staticmethod
    def memory_cleanup() -> dict:
        """
        Perform comprehensive memory cleanup.
        
        Returns:
            Dictionary with cleanup results
        """
        memory_before = MemoryManager.get_memory_usage()
        
        # Force garbage collection multiple times
        for i in range(3):
            gc.collect()
        
        memory_after = MemoryManager.get_memory_usage()
        memory_freed = memory_before - memory_after
        
        result = {
            "memory_before": memory_before,
            "memory_after": memory_after,
            "memory_freed": memory_freed,
            "cleanup_effective": memory_freed > 0
        }
        
        logger.info(f"Memory cleanup: {memory_before:.1f} MB → {memory_after:.1f} MB (freed: {memory_freed:.1f} MB)")
        
        return result


# Convenience functions for backward compatibility
def get_memory_usage() -> float:
    """Get current memory usage of the process."""
    return MemoryManager.get_memory_usage()


def log_memory_usage(step: str) -> None:
    """Log current memory usage for debugging."""
    MemoryManager.log_memory_usage(step)