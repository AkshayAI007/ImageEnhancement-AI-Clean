#!/usr/bin/env python3
"""
Setup script for Real-ESRGAN Image Enhancement API
Compatible with Linux/Render deployment
"""

import subprocess
import sys
import importlib.util

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔧 {description}")
    print("=" * 50)
    
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed")
        print(f"Error: {e.stderr}")
        return False

def check_import(module_name, package_name=None):
    """Check if a module can be imported"""
    try:
        importlib.import_module(module_name)
        print(f"✅ {package_name or module_name} imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to import {package_name or module_name}: {e}")
        return False

def main():
    print("🔧 Setting up Real-ESRGAN Image Enhancement API")
    print("=" * 60)
    
    # Test basic imports
    print("🧪 Testing basic imports...")
    
    # Check if torch is available
    if not check_import("torch"):
        print("❌ PyTorch not found. Please check your requirements.txt")
        return False
    
    # Check torch version and CUDA
    try:
        import torch
        print(f"✅ PyTorch version: {torch.__version__}")
        print(f"✅ CUDA available: {torch.cuda.is_available()}")
    except Exception as e:
        print(f"❌ PyTorch check failed: {e}")
    
    # Check other critical imports
    imports_to_check = [
        ("torchvision", "torchvision"),
        ("cv2", "opencv-python"),
        ("PIL", "Pillow"),
        ("numpy", "numpy"),
        ("gradio", "gradio"),
    ]
    
    for module, package in imports_to_check:
        check_import(module, package)
    
    # Check Real-ESRGAN specific imports
    print("\n🎯 Testing Real-ESRGAN specific imports...")
    
    try:
        from basicsr.archs.rrdbnet_arch import RRDBNet
        print("✅ basicsr.archs.rrdbnet_arch imported successfully")
    except ImportError as e:
        print(f"❌ basicsr import failed: {e}")
        print("💡 This might be due to the basicsr version issue")
    
    try:
        from realesrgan import RealESRGANer
        print("✅ RealESRGANer imported successfully")
    except ImportError as e:
        print(f"❌ RealESRGANer import failed: {e}")
    
    print("\n🎉 Setup completed!")
    print("💡 If you see import errors above, check your requirements.txt")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
