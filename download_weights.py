#!/usr/bin/env python3
import os
import subprocess
import sys
from pathlib import Path
import requests

def download_weights():
    weights_dir = Path("weights")
    weights_dir.mkdir(exist_ok=True)
    
    url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
    output_path = weights_dir / "RealESRGAN_x4plus.pth"
    
    if output_path.exists():
        print("✅ Weights already exist!")
        return True
    
    print("📥 Downloading weights...")
    try:
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\rProgress: {percent:.1f}%", end="")
        
        print(f"\n✅ Download complete!")
        return True
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return False

if __name__ == "__main__":
    download_weights()
