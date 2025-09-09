"""
Configuration settings for the Feature Matching API
"""
import os
import psutil
from typing import Dict, Any


class FeatureMatchingConfig:
    """Configuration class for feature matching settings."""
    
    def __init__(self):
        """Initialize configuration from environment variables with sensible defaults."""
        
        # CPU Configuration - Auto-detect or manual override
        self.cpu_cores = self._get_int_env("CPU_CORES", psutil.cpu_count())
        
        # GPU Detection
        self.has_gpu = self._detect_gpu()
        
        # Configure threading based on GPU availability
        if self.has_gpu:
            # With GPU, use fewer CPU threads to avoid competition
            default_threads = min(4, self.cpu_cores // 2)
        else:
            # Without GPU, use all CPU cores for maximum performance
            default_threads = min(self.cpu_cores, 14)  # Cap at 14 for stability
        
        self.torch_threads = self._get_int_env("TORCH_THREADS", default_threads)
        
        # ROMA Model Configuration - Auto-detect GPU vs CPU mode
        self.roma_force_cpu = self._get_bool_env("ROMA_FORCE_CPU", not self.has_gpu)
        self.roma_memory_efficient = self._get_bool_env("ROMA_MEMORY_EFFICIENT", not self.has_gpu)
        
        # Scale image sizes and keypoints based on GPU/CPU capabilities
        if self.has_gpu:
            # GPU can handle much larger images and more keypoints for better accuracy
            base_size = 840  # Significantly larger for better matching
            base_keypoints = 2000  # Much more keypoints for better accuracy
        else:
            # Increase CPU values significantly for better matching accuracy
            # Scale based on CPU power - more cores = can handle larger images
            base_size = 560 if self.cpu_cores < 8 else (700 if self.cpu_cores < 12 else 840)
            base_keypoints = 1000 if self.cpu_cores < 8 else (1500 if self.cpu_cores < 12 else 2000)

        self.roma_max_image_size = self._get_int_env("ROMA_MAX_IMAGE_SIZE", base_size)
        self.roma_max_keypoints = self._get_int_env("ROMA_MAX_KEYPOINTS", base_keypoints)
        
        # Coarse resolution (must be multiple of 14)
        coarse_res = self._get_int_env("ROMA_COARSE_RES", self.roma_max_image_size)
        self.roma_coarse_res = (coarse_res, coarse_res)
        
        # Upsample resolution (must be multiple of 14) - increased for better accuracy
        upsample_res = self._get_int_env("ROMA_UPSAMPLE_RES", min(self.roma_max_image_size + 56, 896))  # Increased max from 420 to 896
        self.roma_upsample_res = (upsample_res, upsample_res)
        
        # Multi-processing settings for CPU optimization
        self.enable_multiprocessing = self._get_bool_env("ENABLE_MULTIPROCESSING", True)
        self.batch_size = self._get_int_env("BATCH_SIZE", min(self.cpu_cores // 2, 8))
        
        # Feature Matching Configuration
        self.buffer_sizes = self._get_list_env("BUFFER_SIZES", [20, 800, 5000])
        self.min_match_count = self._get_int_env("MIN_MATCH_COUNT", 10)
        self.min_inlier_ratio = self._get_float_env("MIN_INLIER_RATIO", 0.1)
        
        # Memory and Performance - scale with GPU/CPU capabilities
        if self.has_gpu:
            # GPU systems typically have more RAM and can handle larger images
            base_memory = 12000  # 12GB for GPU systems
            base_pixels = 8000000  # 8M pixels
        else:
            # Scale with CPU cores for CPU-only systems
            base_memory = 4000 if self.cpu_cores < 8 else (6000 if self.cpu_cores < 12 else 8000)
            base_pixels = 4200000  # 4.2M pixels
            
        self.max_memory_mb = self._get_int_env("MAX_MEMORY_MB", base_memory)
        self.image_resize_max_pixels = self._get_int_env("IMAGE_RESIZE_MAX_PIXELS", base_pixels)
        
    def _get_bool_env(self, key: str, default: bool) -> bool:
        """Get boolean environment variable."""
        value = os.getenv(key, str(default)).lower()
        return value in ('true', '1', 'yes', 'on')
    
    def _get_int_env(self, key: str, default: int) -> int:
        """Get integer environment variable."""
        try:
            return int(os.getenv(key, str(default)))
        except ValueError:
            return default
    
    def _get_float_env(self, key: str, default: float) -> float:
        """Get float environment variable."""
        try:
            return float(os.getenv(key, str(default)))
        except ValueError:
            return default
    
    def _get_list_env(self, key: str, default: list) -> list:
        """Get list environment variable (comma-separated)."""
        value = os.getenv(key)
        if value:
            try:
                return [int(x.strip()) for x in value.split(',')]
            except ValueError:
                return default
        return default
    
    def _detect_gpu(self) -> bool:
        """Detect if CUDA GPU is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def get_roma_config(self) -> Dict[str, Any]:
        """Get ROMA-specific configuration dictionary."""
        return {
            'force_cpu': self.roma_force_cpu,
            'memory_efficient': self.roma_memory_efficient,
            'max_image_size': self.roma_max_image_size,
            'max_keypoints': self.roma_max_keypoints,
            'coarse_res': self.roma_coarse_res,
            'upsample_res': self.roma_upsample_res,
            'torch_threads': self.torch_threads,
            'enable_multiprocessing': self.enable_multiprocessing,
            'batch_size': self.batch_size
        }
    
    def setup_torch_threading(self):
        """Setup PyTorch for optimal CPU threading."""
        import torch
        
        # Set number of threads for PyTorch operations
        torch.set_num_threads(self.torch_threads)
        
        # Enable optimized CPU operations
        torch.set_grad_enabled(False)  # Disable gradients for inference
        
        # Set threading for different backends
        if hasattr(torch, 'set_num_interop_threads'):
            torch.set_num_interop_threads(max(1, self.torch_threads // 2))
        
        return self.torch_threads
    
    def log_config(self, logger):
        """Log current configuration settings."""
        logger.info("🔧 Feature Matching Configuration:")
        logger.info(f"  CPU Cores Available: {self.cpu_cores}")
        logger.info(f"  GPU Available: {'✅ Yes' if self.has_gpu else '❌ No'}")
        logger.info(f"  PyTorch Threads: {self.torch_threads}")
        logger.info(f"  ROMA Force CPU: {self.roma_force_cpu}")
        logger.info(f"  ROMA Memory Efficient: {self.roma_memory_efficient}")
        logger.info(f"  ROMA Max Image Size: {self.roma_max_image_size}px")
        logger.info(f"  ROMA Max Keypoints: {self.roma_max_keypoints}")
        logger.info(f"  ROMA Coarse Resolution: {self.roma_coarse_res}")
        logger.info(f"  ROMA Upsample Resolution: {self.roma_upsample_res}")
        logger.info(f"  Enable Multiprocessing: {self.enable_multiprocessing}")
        logger.info(f"  Batch Size: {self.batch_size}")
        logger.info(f"  Buffer Sizes: {self.buffer_sizes}m")
        logger.info(f"  Max Memory: {self.max_memory_mb}MB")
        
        # Log mode summary
        if self.has_gpu:
            logger.info("🚀 Running in GPU-accelerated mode")
        else:
            logger.info("🖥️  Running in CPU-optimized mode")


# Global configuration instance
config = FeatureMatchingConfig()