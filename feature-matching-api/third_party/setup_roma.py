#!/usr/bin/env python3
"""
Setup script for the standalone GIM(Roma) matcher.

This script:
1. Creates the necessary directory structure
2. Clones the RoMa repository 
3. Installs required dependencies
"""

import os
import subprocess
import sys
from pathlib import Path

def run_command(cmd, description=""):
    """Run a shell command and handle errors."""
    print(f"Running: {cmd}")
    if description:
        print(f"  {description}")
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return False
    else:
        if result.stdout.strip():
            print(f"  Output: {result.stdout.strip()}")
        return True

def setup_roma():
    """Set up the Roma environment."""
    
    print("Setting up standalone GIM(Roma) matcher...")
    print("=" * 50)
    
    # We're already in third_party directory
    script_dir = Path(__file__).parent
    third_party_dir = script_dir  # We're in third_party already
    
    roma_dir = third_party_dir / "RoMa"
    
    # Clone RoMa repository if it doesn't exist
    if not roma_dir.exists():
        print("Cloning RoMa repository...")
        success = run_command(
            f"git clone https://github.com/Vincentqyw/RoMa.git {roma_dir}",
            "Downloading RoMa source code"
        )
        if not success:
            print("Failed to clone RoMa repository")
            return False
    else:
        print(f"RoMa repository already exists at {roma_dir}")
    
    # Auto-detect GPU and apply patches conditionally
    print("🔍 Detecting GPU availability...")
    has_gpu = False
    gpu_info = ""
    
    try:
        import torch
        if torch.cuda.is_available():
            has_gpu = True
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
            gpu_info = f"{gpu_count}x {gpu_name}"
            print(f"  ✅ GPU detected: {gpu_info}")
        else:
            print("  ⚠️ No CUDA GPU detected")
    except ImportError:
        print("  ⚠️ PyTorch not available for GPU detection")
    
    # Only apply CPU compatibility patch if no GPU is available
    if not has_gpu:
        print("🔧 Applying CPU compatibility patch (no GPU detected)...")
        kde_file = roma_dir / "romatch" / "utils" / "kde.py"
        if kde_file.exists():
            try:
                content = kde_file.read_text()
                # Check if patch is already applied
                if "half = False" in content and "CPU compatibility" in content:
                    print("  ✅ CPU compatibility patch already applied")
                else:
                    # Patch the half precision default to False for CPU compatibility
                    content = content.replace(
                        "def kde(x, std = 0.1, half = True, down = None):",
                        "def kde(x, std = 0.1, half = False, down = None):  # Changed default to False for CPU compatibility"
                    )
                    kde_file.write_text(content)
                    print("  ✅ Applied CPU compatibility patch to kde.py")
            except Exception as e:
                print(f"  ⚠️ Warning: Could not apply patch to kde.py: {e}")
        else:
            print("  ⚠️ Warning: kde.py not found, patch not applied")
    else:
        print(f"🚀 GPU mode enabled - skipping CPU patch (GPU: {gpu_info})")
        print("  ℹ️ ROMA will use GPU acceleration with half precision")
    
    # Check if PyTorch is properly installed
    print("\nChecking PyTorch installation...")
    try:
        import torch
        print(f"  PyTorch version: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA device: {torch.cuda.get_device_name()}")
    except ImportError:
        print("  Warning: PyTorch not properly installed")
        return False
    
    print("\n" + "=" * 50)
    print("Setup completed successfully!")
    print("\nYou can now use the Roma matcher:")
    print("  python example_usage.py")
    print("\nOr import it in your own code:")
    print("  from roma_matcher import create_roma_matcher")
    
    return True

def cleanup():
    """Remove downloaded files (optional cleanup function)."""
    import shutil
    
    response = input("Do you want to remove the third_party directory? (y/N): ")
    if response.lower() == 'y':
        third_party_dir = Path("third_party")
        if third_party_dir.exists():
            shutil.rmtree(third_party_dir)
            print("Removed third_party directory")
        else:
            print("third_party directory does not exist")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "cleanup":
        cleanup()
    else:
        success = setup_roma()
        if not success:
            sys.exit(1) 