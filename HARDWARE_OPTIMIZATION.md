# 🤖 Automatic Hardware Detection & Configuration

The Feature Matching API automatically detects your hardware and optimizes settings accordingly. No manual configuration needed!

## 🔍 **What Gets Auto-Detected**

### 1. **CPU Cores**
```python
CPU_CORES = psutil.cpu_count()  # Auto-detected on any computer
```

### 2. **GPU Availability**
```python
# Automatically checks for CUDA GPU
has_gpu = torch.cuda.is_available()
```

### 3. **Optimal Settings**
Based on detection, the system automatically configures:
- Image processing sizes
- Threading settings
- Memory limits
- ROMA model parameters

## 📊 **Auto-Configuration Examples**

### **4-Core Laptop (No GPU)**
```
🔧 Auto-Detection Results:
✅ CPU Cores: 4
🔍 GPU Available: False
⚙️  PyTorch Threads: 4
🖥️  Force CPU: True
📷 Max Image Size: 112px
🎯 Max Keypoints: 100
💾 Max Memory: 4000MB
🖥️  CPU mode: Multi-core optimization enabled
```

### **8-Core Desktop (No GPU)**
```
🔧 Auto-Detection Results:
✅ CPU Cores: 8
🔍 GPU Available: False
⚙️  PyTorch Threads: 8
🖥️  Force CPU: True
📷 Max Image Size: 196px
🎯 Max Keypoints: 200
💾 Max Memory: 6000MB
🖥️  CPU mode: Multi-core optimization enabled
```

### **14-Core Mac (No GPU) - Your System**
```
🔧 Auto-Detection Results:
✅ CPU Cores: 14
🔍 GPU Available: False
⚙️  PyTorch Threads: 14
🖥️  Force CPU: True
📷 Max Image Size: 280px
🎯 Max Keypoints: 300
💾 Max Memory: 8000MB
🖥️  CPU mode: Multi-core optimization enabled
```

### **8-Core Desktop (With RTX 4080)**
```
🔧 Auto-Detection Results:
✅ CPU Cores: 8
🔍 GPU Available: True
⚙️  PyTorch Threads: 4  (fewer threads with GPU)
🖥️  Force CPU: False
📷 Max Image Size: 560px
🎯 Max Keypoints: 1000
💾 Max Memory: 12000MB
🚀 GPU mode: High-performance settings enabled
```

### **32-Core Server (With Multiple GPUs)**
```
🔧 Auto-Detection Results:
✅ CPU Cores: 32
🔍 GPU Available: True
⚙️  PyTorch Threads: 4  (GPU handles main workload)
🖥️  Force CPU: False
📷 Max Image Size: 560px
🎯 Max Keypoints: 1000
💾 Max Memory: 12000MB
🚀 GPU mode: High-performance settings enabled
```

## 🔧 **Setup Process**

### **1. Docker Build Time**
During `docker-compose up --build`, the setup script automatically:

#### **No GPU Detected:**
```
🔍 Detecting GPU availability...
  ⚠️ No CUDA GPU detected
🔧 Applying CPU compatibility patch (no GPU detected)...
  ✅ Applied CPU compatibility patch to kde.py
```

#### **GPU Detected:**
```
🔍 Detecting GPU availability...
  ✅ GPU detected: 1x NVIDIA RTX 4080
🚀 GPU mode enabled - skipping CPU patch (GPU: 1x NVIDIA RTX 4080)
  ℹ️ ROMA will use GPU acceleration with half precision
```

### **2. Runtime Configuration**
At startup, the system automatically:
- Detects CPU cores
- Checks for GPU availability
- Sets optimal threading
- Configures memory limits
- Scales image processing parameters

## 🎛️ **Manual Override (Optional)**

Users can override auto-detection if needed:

```env
# Override auto-detection
CPU_CORES=16                    # Force specific core count
ROMA_FORCE_CPU=false           # Force GPU mode
ROMA_MEMORY_EFFICIENT=false    # Disable memory efficiency
ROMA_MAX_IMAGE_SIZE=420        # Custom image size
TORCH_THREADS=8                # Custom thread count
```

## 🚀 **Performance Scaling**

The system automatically scales performance based on hardware:

| Hardware Type | Image Size | Keypoints | Threads | Memory | Performance |
|---------------|------------|-----------|---------|---------|-------------|
| **4-core CPU** | 112px | 100 | 4 | 4GB | Basic |
| **8-core CPU** | 196px | 200 | 8 | 6GB | Good |
| **14-core CPU** | 280px | 300 | 14 | 8GB | Excellent |
| **8-core + GPU** | 560px | 1000 | 4 | 12GB | Ultra |
| **32-core + GPU** | 560px | 1000 | 4 | 12GB | Maximum |

## 🔄 **Cross-Platform Compatibility**

The auto-detection works on:
- ✅ **macOS** (Intel & Apple Silicon)
- ✅ **Linux** (Ubuntu, CentOS, etc.)
- ✅ **Windows** (with Docker Desktop)
- ✅ **Cloud instances** (AWS, GCP, Azure)
- ✅ **Kubernetes clusters**

## 📝 **Verification**

To check what was detected on your system:

```bash
# Check detection results
docker exec feature-matching-api python3 -c "
from memory_config import config
config.log_config(__import__('logging').getLogger())
"
```

## 🛠️ **Troubleshooting**

### **GPU Not Detected (But You Have One)**
1. Check NVIDIA drivers are installed
2. Verify Docker has GPU access:
   ```bash
   docker run --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi
   ```
3. Add GPU support to docker-compose.yml:
   ```yaml
   runtime: nvidia
   environment:
     - NVIDIA_VISIBLE_DEVICES=all
   ```

### **Too Many/Few CPU Threads**
```env
# Reduce threads if system becomes unresponsive
TORCH_THREADS=8

# Increase if you have more cores available
CPU_CORES=32
TORCH_THREADS=16
```

### **Memory Issues**
```env
# Reduce memory limits
MAX_MEMORY_MB=4000
ROMA_MAX_IMAGE_SIZE=112

# Or increase for powerful systems
MAX_MEMORY_MB=16000
```

## 🎯 **Benefits**

1. **Zero Configuration**: Works out-of-the-box on any system
2. **Optimal Performance**: Automatically uses best settings for your hardware
3. **Resource Efficiency**: Never over-allocates resources
4. **Future Proof**: Adapts to hardware upgrades automatically
5. **Portable**: Same code works on laptop and server

The system is designed to **"just work"** regardless of the underlying hardware! 🚀