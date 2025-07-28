#!/usr/bin/env python3
"""
Image Enhancement AI - Gradio Interface for Hugging Face Spaces
"""

import gradio as gr
import numpy as np
from PIL import Image
import torch
import cv2
import os
from pathlib import Path
import requests
import subprocess
import sys

# Download weights function
def download_weights():
    """Download RealESRGAN weights if not present"""
    weights_dir = Path("weights")
    weights_dir.mkdir(exist_ok=True)
    
    weights_path = weights_dir / "RealESRGAN_x4plus.pth"
    
    if weights_path.exists():
        print("✅ Weights already exist!")
        return True
    
    print("📥 Downloading RealESRGAN weights...")
    try:
        url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
        
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(weights_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\rDownloading: {percent:.1f}%", end="", flush=True)
        
        print(f"\n✅ Downloaded successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return False

# Initialize model globally
model = None

def load_model():
    """Load the RealESRGAN model"""
    global model
    
    if model is not None:
        return model
    
    try:
        # Download weights if needed
        if not download_weights():
            raise Exception("Failed to download weights")
        
        # Import RealESRGAN
        from realesrgan import RealESRGAN
        
        # Load model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = RealESRGAN(device, scale=4)
        model.load_weights('weights/RealESRGAN_x4plus.pth', download=False)
        
        print("✅ Model loaded successfully!")
        return model
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

def enhance_image(input_image, scale_factor=4):
    """
    Enhance image using RealESRGAN
    
    Args:
        input_image: PIL Image or numpy array
        scale_factor: Upscaling factor (2 or 4)
    
    Returns:
        Enhanced PIL Image
    """
    
    if input_image is None:
        return None, "Please upload an image first!"
    
    try:
        # Load model
        current_model = load_model()
        if current_model is None:
            return None, "❌ Failed to load AI model. Please try again."
        
        # Convert PIL to numpy if needed
        if isinstance(input_image, Image.Image):
            image_np = np.array(input_image)
        else:
            image_np = input_image
        
        # Convert RGB to BGR for OpenCV
        if len(image_np.shape) == 3 and image_np.shape[2] == 3:
            image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        else:
            image_bgr = image_np
        
        print("🔄 Enhancing image...")
        
        # Enhance image
        enhanced_bgr = current_model.predict(image_bgr)
        
        # Convert back to RGB
        enhanced_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        enhanced_pil = Image.fromarray(enhanced_rgb)
        
        # Get dimensions for info
        original_size = f"{image_np.shape[1]}x{image_np.shape[0]}"
        enhanced_size = f"{enhanced_pil.width}x{enhanced_pil.height}"
        
        success_message = f"✅ Enhancement complete!\nOriginal: {original_size} → Enhanced: {enhanced_size}"
        
        return enhanced_pil, success_message
        
    except Exception as e:
        error_message = f"❌ Enhancement failed: {str(e)}"
        print(error_message)
        return None, error_message

def create_gradio_interface():
    """Create the Gradio interface"""
    
    # Custom CSS for better styling
    css = """
    .gradio-container {
        font-family: 'Arial', sans-serif;
    }
    .title {
        text-align: center;
        margin-bottom: 1rem;
    }
    .description {
        text-align: center;
        margin-bottom: 2rem;
        color: #666;
    }
    """
    
    # Create interface
    with gr.Blocks(css=css, title="Image Enhancement AI") as interface:
        
        # Header
        gr.Markdown(
            """
            # 🎨 Image Enhancement AI
            ### Enhance your images using Real-ESRGAN super-resolution
            Upload any image and watch it get enhanced with AI-powered upscaling!
            """,
            elem_classes=["title"]
        )
        
        with gr.Row():
            with gr.Column():
                # Input components
                input_image = gr.Image(
                    label="📤 Upload Image",
                    type="pil",
                    format="png"
                )
                
                enhance_btn = gr.Button(
                    "✨ Enhance Image",
                    variant="primary",
                    size="lg"
                )
                
                # Information
                gr.Markdown(
                    """
                    **Supported formats:** PNG, JPG, JPEG, WEBP  
                    **Max file size:** 10MB  
                    **Enhancement:** 4x super-resolution
                    """
                )
            
            with gr.Column():
                # Output components
                output_image = gr.Image(
                    label="✨ Enhanced Image",
                    type="pil"
                )
                
                status_text = gr.Textbox(
                    label="Status",
                    interactive=False,
                    max_lines=3
                )
        
        # Examples section
        gr.Markdown("### 📸 Try these examples:")
        
        gr.Examples(
            examples=[
                ["https://raw.githubusercontent.com/xinntao/Real-ESRGAN/master/inputs/00.jpg"],
                ["https://raw.githubusercontent.com/xinntao/Real-ESRGAN/master/inputs/01.jpg"],
            ],
            inputs=input_image,
            label="Sample Images"
        )
        
        # Event handlers
        enhance_btn.click(
            fn=enhance_image,
            inputs=[input_image],
            outputs=[output_image, status_text],
            show_progress=True
        )
        
        # Footer
        gr.Markdown(
            """
            ---
            Made with ❤️ using [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) and [Gradio](https://gradio.app/)
            """
        )
    
    return interface

# Launch the app
if __name__ == "__main__":
    print("🚀 Starting Image Enhancement AI...")
    
    # Create and launch interface
    demo = create_gradio_interface()
    
    # Launch with public sharing for Hugging Face Spaces
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,  # Standard port for HF Spaces
        share=False,  # Don't need share=True on HF Spaces
        show_error=True
    )