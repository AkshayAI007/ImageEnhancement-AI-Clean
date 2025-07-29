import sys
sys.stdout.flush()
import faulthandler; faulthandler.enable()
print("DEBUG: Script started")

try:
    from flask import Flask, request, send_file, jsonify, render_template
    print("DEBUG: Imported Flask")
    from flask_cors import CORS
    print("DEBUG: Imported flask_cors")
    from PIL import Image, ImageEnhance, ImageFilter, ImageOps
    print("DEBUG: Imported PIL")
    import numpy as np
    print("DEBUG: Imported numpy")
    import io
    print("DEBUG: Imported io")
    import os
    print("DEBUG: Imported os")
    import gc
    print("DEBUG: Imported gc")
    import time
    print("DEBUG: Imported time")
    from concurrent.futures import ThreadPoolExecutor, as_completed
    print("DEBUG: Imported concurrent.futures")
    import threading
    print("DEBUG: Imported threading")
    import cv2
    print("DEBUG: Imported cv2")
    import subprocess
    print("DEBUG: Imported subprocess")
    import tempfile
    print("DEBUG: Imported tempfile")
    import shutil
    print("DEBUG: Imported shutil")
    import glob
    print("DEBUG: Imported glob")
    
    # Try to import Real-ESRGAN components
    REALESRGAN_AVAILABLE = False
    realesrgan_error = None
    
    try:
        import torch
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer
        from gfpgan import GFPGANer
        print("DEBUG: Imported Real-ESRGAN components")
        REALESRGAN_AVAILABLE = True
    except Exception as e:
        realesrgan_error = str(e)
        print(f"DEBUG: Real-ESRGAN components not available: {e}")
        print("DEBUG: Make sure you've installed Real-ESRGAN properly")
        
except Exception as e:
    print(f"Import error: {e}")
    exit(1)

print("DEBUG: All imports successful")

# Configuration
MAX_IMAGE_SIZE_AI = 1024
MAX_FILE_SIZE = 10 * 1024 * 1024
WEIGHTS_FOLDER = 'weights'
TEMP_FOLDER = 'temp_processing'
REAL_ESRGAN_PATH = 'Real-ESRGAN'  # Path to Real-ESRGAN repository

# Create necessary directories
os.makedirs(WEIGHTS_FOLDER, exist_ok=True)
os.makedirs(TEMP_FOLDER, exist_ok=True)

# Parallel processing settings
MAX_WORKERS = min(4, os.cpu_count() or 1)
THREAD_LOCAL = threading.local()

app = Flask(__name__)
CORS(app, origins="*")

# Initialize RealESRGAN model
realesrgan_upsampler = None
face_enhancer = None
device = None

# Model configurations
MODEL_CONFIGS = {
    'RealESRGAN_x4plus': {
        'model_path': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
        'model_name': 'RealESRGAN_x4plus',
        'scale': 4,
        'arch': 'RRDB',
        'num_block': 23,
        'num_feat': 64
    },
    'RealESRGAN_x2plus': {
        'model_path': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x2plus.pth',
        'model_name': 'RealESRGAN_x2plus', 
        'scale': 2,
        'arch': 'RRDB',
        'num_block': 23,
        'num_feat': 64
    }
}

def download_model_weights():
    """Download model weights if they don't exist"""
    model_config = MODEL_CONFIGS['RealESRGAN_x4plus']
    model_path = os.path.join(WEIGHTS_FOLDER, f"{model_config['model_name']}.pth")
    
    if not os.path.exists(model_path):
        print(f"DEBUG: Downloading {model_config['model_name']} weights...")
        try:
            import urllib.request
            urllib.request.urlretrieve(model_config['model_path'], model_path)
            print(f"DEBUG: Downloaded weights to {model_path}")
        except Exception as e:
            print(f"DEBUG: Failed to download weights: {e}")
            return False
    else:
        print(f"DEBUG: Using existing weights at {model_path}")
    
    return os.path.exists(model_path)

if REALESRGAN_AVAILABLE:
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"DEBUG: Using device: {device}")
        
        # Download weights if needed
        if download_model_weights():
            model_config = MODEL_CONFIGS['RealESRGAN_x4plus']
            model_path = os.path.join(WEIGHTS_FOLDER, f"{model_config['model_name']}.pth")
            
            # Initialize RealESRGAN
            model = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=model_config['num_feat'],
                num_block=model_config['num_block'],
                num_grow_ch=32,
                scale=model_config['scale']
            )
            
            realesrgan_upsampler = RealESRGANer(
                scale=model_config['scale'],
                model_path=model_path,
                model=model,
                tile=256,  # Tile size for processing large images
                tile_pad=10,
                pre_pad=0,
                half=device.type == 'cuda',  # Use half precision on GPU
                device=device
            )
            
            print("DEBUG: RealESRGAN upsampler initialized successfully!")
            
            # Try to initialize face enhancer (optional)
            try:
                face_enhancer = GFPGANer(
                    model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
                    upscale=model_config['scale'],
                    arch='clean',
                    channel_multiplier=2,
                    bg_upsampler=realesrgan_upsampler,
                    device=device
                )
                print("DEBUG: GFPGAN face enhancer initialized successfully!")
            except Exception as e:
                print(f"DEBUG: Could not initialize face enhancer: {e}")
                face_enhancer = None
        else:
            REALESRGAN_AVAILABLE = False
            realesrgan_error = "Could not download model weights"
            
    except Exception as e:
        print(f"ERROR initializing RealESRGAN: {e}")
        REALESRGAN_AVAILABLE = False
        realesrgan_error = str(e)

def advanced_preprocessing(img):
    """Advanced image preprocessing before AI enhancement"""
    print("DEBUG: Starting advanced preprocessing...")
    
    # Convert to numpy array for processing
    img_array = np.array(img)
    img = Image.fromarray(img_array)
    
    # Step 1: Noise reduction using median filter
    img = img.filter(ImageFilter.MedianFilter(size=3))
    
    # Step 2: Enhance contrast adaptively
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.1)
    
    # Step 3: Enhance sharpness slightly
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(1.15)
    
    # Step 4: Color enhancement
    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(1.05)
    
    # Step 5: Histogram equalization for better dynamic range
    img = ImageOps.equalize(img)
    
    # Step 6: Unsharp mask for edge enhancement
    img = img.filter(ImageFilter.UnsharpMask(radius=1, percent=120, threshold=3))
    
    print("DEBUG: Advanced preprocessing completed")
    return img

def resize_if_too_large(img, max_size):
    """Resize image if it's too large while maintaining aspect ratio"""
    width, height = img.size
    if max(width, height) > max_size:
        if width > height:
            new_width = max_size
            new_height = int((height * max_size) / width)
        else:
            new_height = max_size
            new_width = int((width * max_size) / height)
        
        print(f"DEBUG: Resizing from {img.size} to ({new_width}, {new_height})")
        return img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return img

def enhance_with_realesrgan(img, enhance_face=False):
    """Enhance image using Real-ESRGAN"""
    if not REALESRGAN_AVAILABLE or realesrgan_upsampler is None:
        raise ValueError("RealESRGAN not available")
    
    print("DEBUG: Starting Real-ESRGAN enhancement...")
    
    # Convert PIL to numpy array (BGR format for OpenCV)
    img_array = np.array(img)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    try:
        with torch.no_grad():
            if enhance_face and face_enhancer is not None:
                print("DEBUG: Using GFPGAN for face enhancement...")
                # Use GFPGAN for face enhancement
                cropped_faces, restored_faces, restored_img = face_enhancer.enhance(
                    img_bgr,
                    has_aligned=False,
                    only_center_face=False,
                    paste_back=True,
                    weight=0.5
                )
                enhanced_img = restored_img
            else:
                print("DEBUG: Using RealESRGAN upsampler...")
                # Use RealESRGAN for general enhancement
                enhanced_img, _ = realesrgan_upsampler.enhance(img_bgr, outscale=4)
        
        # Convert back to RGB
        enhanced_img_rgb = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB)
        return Image.fromarray(enhanced_img_rgb)
        
    except Exception as e:
        print(f"DEBUG: Real-ESRGAN enhancement failed: {e}")
        raise

def super_enhance_image(img, enhance_face=False):
    """Super enhancement pipeline: Preprocessing + RealESRGAN + Post-processing"""
    if not REALESRGAN_AVAILABLE:
        raise ValueError("RealESRGAN model not available")
    
    print("DEBUG: Starting super enhancement pipeline...")
    start_time = time.time()
    
    # Step 1: Advanced preprocessing
    preprocessing_start = time.time()
    preprocessed_img = advanced_preprocessing(img)
    preprocessing_time = time.time() - preprocessing_start
    print(f"DEBUG: Preprocessing completed in {preprocessing_time:.2f}s")
    
    # Step 2: Apply RealESRGAN
    ai_start = time.time()
    enhanced_img = enhance_with_realesrgan(preprocessed_img, enhance_face=enhance_face)
    ai_time = time.time() - ai_start
    print(f"DEBUG: RealESRGAN processing completed in {ai_time:.2f}s")
    
    # Step 3: Post-processing (optional fine-tuning)
    postprocessing_start = time.time()
    
    # Light post-processing to refine the AI output
    enhancer = ImageEnhance.Sharpness(enhanced_img)
    enhanced_img = enhancer.enhance(1.02)  # Very slight sharpening
    
    enhancer = ImageEnhance.Contrast(enhanced_img)
    enhanced_img = enhancer.enhance(1.01)  # Very slight contrast boost
    
    postprocessing_time = time.time() - postprocessing_start
    print(f"DEBUG: Post-processing completed in {postprocessing_time:.2f}s")
    
    total_time = time.time() - start_time
    print(f"DEBUG: Super enhancement pipeline completed in {total_time:.2f}s")
    
    # Clean up GPU memory
    if device and device.type == 'cuda':
        torch.cuda.empty_cache()
    
    return enhanced_img, {
        'total_time': total_time,
        'preprocessing_time': preprocessing_time,
        'ai_time': ai_time,
        'postprocessing_time': postprocessing_time
    }

def fallback_pil_enhancement(img):
    """Fallback PIL enhancement when RealESRGAN is not available"""
    print("DEBUG: Using PIL fallback enhancement...")
    
    # Advanced preprocessing
    enhanced_img = advanced_preprocessing(img)
    
    # 4x upscaling with high-quality resampling
    width, height = enhanced_img.size
    enhanced_img = enhanced_img.resize((width * 4, height * 4), Image.Resampling.LANCZOS)
    
    # Additional enhancement
    enhancer = ImageEnhance.Sharpness(enhanced_img)
    enhanced_img = enhancer.enhance(1.3)
    
    enhancer = ImageEnhance.Contrast(enhanced_img)
    enhanced_img = enhancer.enhance(1.2)
    
    # Apply unsharp mask
    enhanced_img = enhanced_img.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
    
    return enhanced_img

@app.route('/api/enhance', methods=['POST'])
def enhance_image():
    print("DEBUG: /api/enhance called")
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided.'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected.'}), 400
    
    # Get enhancement options
    enhance_face = request.form.get('face_enhance', 'false').lower() == 'true'
    
    # Check file size
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    
    if file_size > MAX_FILE_SIZE:
        return jsonify({'error': f'File too large. Maximum size is {MAX_FILE_SIZE//1024//1024}MB'}), 400
    
    try:
        img = Image.open(file.stream).convert('RGB')
        print(f"DEBUG: Image opened - Original size: {img.size}")
        
        # Resize if too large for AI processing
        img = resize_if_too_large(img, MAX_IMAGE_SIZE_AI)
        print(f"DEBUG: Processing size: {img.size}")
        
    except Exception as e:
        print(f"DEBUG: Invalid image: {e}")
        return jsonify({'error': 'Invalid image format.'}), 400
    
    try:
        if REALESRGAN_AVAILABLE:
            print(f"DEBUG: Starting Real-ESRGAN enhancement (face_enhance={enhance_face})...")
            enhanced_img, timing_info = super_enhance_image(img, enhance_face=enhance_face)
            method_used = f'Real-ESRGAN 4x {"+ GFPGAN Face Enhancement" if enhance_face else ""}'
        else:
            print("DEBUG: RealESRGAN not available, using PIL fallback...")
            enhanced_img = fallback_pil_enhancement(img)
            timing_info = {'total_time': 0}
            method_used = f'Advanced PIL Enhancement 4x (RealESRGAN unavailable: {realesrgan_error})'
        
        print(f"DEBUG: Enhancement completed using {method_used}")
        
        # Clean up memory
        del img
        gc.collect()
            
    except Exception as e:
        print(f"DEBUG: Enhancement failed: {e}")
        import traceback
        print(f"DEBUG: Full traceback: {traceback.format_exc()}")
        return jsonify({'error': f'Enhancement failed: {str(e)}'}), 500
    
    try:
        byte_io = io.BytesIO()
        
        # Save as JPEG for very large images to reduce file size
        if enhanced_img.size[0] * enhanced_img.size[1] > 8000000:  # ~8MP
            enhanced_img.save(byte_io, 'JPEG', quality=95, optimize=True)
            mimetype = 'image/jpeg'
            print("DEBUG: Saved as JPEG for very large image")
        else:
            enhanced_img.save(byte_io, 'PNG', optimize=True)
            mimetype = 'image/png'
            print("DEBUG: Saved as PNG")
        
        byte_io.seek(0)
        
        # Add processing info to response headers
        response = send_file(byte_io, mimetype=mimetype)
        response.headers['X-Processing-Method'] = method_used
        response.headers['X-Processing-Time'] = f"{timing_info.get('total_time', 0):.2f}s"
        if 'preprocessing_time' in timing_info:
            response.headers['X-Preprocessing-Time'] = f"{timing_info['preprocessing_time']:.2f}s"
            response.headers['X-AI-Time'] = f"{timing_info['ai_time']:.2f}s"
            response.headers['X-Postprocessing-Time'] = f"{timing_info['postprocessing_time']:.2f}s"
        
        # Clean up
        del enhanced_img
        gc.collect()
        
        return response
        
    except Exception as e:
        print(f"DEBUG: Error saving image: {e}")
        return jsonify({'error': f'Error saving enhanced image: {str(e)}'}), 500

@app.route('/api/status')
def get_status():
    """Get current system status"""
    status_info = {
        'status': 'healthy',
        'realesrgan_available': REALESRGAN_AVAILABLE,
        'face_enhancer_available': face_enhancer is not None,
        'realesrgan_error': realesrgan_error if not REALESRGAN_AVAILABLE else None,
        'max_image_size': MAX_IMAGE_SIZE_AI,
        'max_file_size_mb': MAX_FILE_SIZE // (1024 * 1024),
        'model_source': 'Official Real-ESRGAN + GFPGAN' if REALESRGAN_AVAILABLE else 'N/A',
        'enhancement_pipeline': [
            'Advanced Preprocessing (Noise Reduction, Contrast, Sharpness, Color, Histogram Equalization, Unsharp Mask)',
            'Real-ESRGAN 4x Super Resolution' if REALESRGAN_AVAILABLE else 'Advanced PIL 4x Upscaling (RealESRGAN fallback)',
            'GFPGAN Face Enhancement (Optional)' if face_enhancer else 'No Face Enhancement',
            'Post-processing (Fine-tuning)'
        ]
    }
    
    if REALESRGAN_AVAILABLE and device:
        status_info['device'] = str(device)
        status_info['cuda_available'] = torch.cuda.is_available()
        if torch.cuda.is_available():
            try:
                status_info['cuda_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            except:
                status_info['cuda_memory_gb'] = 'Unknown'
    
    return jsonify(status_info)

@app.route('/api/models')
def get_models():
    """Get available models"""
    models = []
    for model_name, config in MODEL_CONFIGS.items():
        models.append({
            'name': model_name,
            'scale': config['scale'],
            'architecture': config['arch'],
            'available': REALESRGAN_AVAILABLE
        })
    
    return jsonify({'models': models})

@app.route('/test')
def test():
    return jsonify({'status': 'Flask is working!', 'message': 'Official Real-ESRGAN API'})

@app.route('/health')
def health_check():
    return jsonify({'status': 'healthy', 'message': 'Official Real-ESRGAN API is running'})

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    print("DEBUG: Starting Official Real-ESRGAN Flask app on port 5000")
    print(f"DEBUG: RealESRGAN available: {REALESRGAN_AVAILABLE}")
    print(f"DEBUG: Face enhancer available: {face_enhancer is not None}")
    if not REALESRGAN_AVAILABLE:
        print(f"DEBUG: RealESRGAN error: {realesrgan_error}")
        print("DEBUG: Using advanced PIL processing as fallback")
    else:
        print("DEBUG: Using official Real-ESRGAN implementation")
    if REALESRGAN_AVAILABLE and device:
        print(f"DEBUG: Using device: {device}")
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
