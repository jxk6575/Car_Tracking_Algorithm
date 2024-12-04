import os
import gdown
import torch
from pathlib import Path

def download_deepsort_model():
    """Download DeepSORT ReID model."""
    # Create models directory if it doesn't exist
    model_dir = Path(__file__).resolve().parent.parent / 'data' / 'models'
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Model paths
    model_path = model_dir / 'ckpt.t7'
    
    if not model_path.exists():
        print("Downloading DeepSORT model...")
        # Google Drive link to the model
        url = 'https://drive.google.com/uc?id=1_qwTWdzT9dWNudpusgKavj_4elGgbkUN'
        gdown.download(url, str(model_path), quiet=False)
        print(f"Model downloaded to: {model_path}")
    else:
        print("DeepSORT model already exists.")
    
    # Verify the model
    try:
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        print("Model loaded successfully!")
        print(f"Model info: {checkpoint.keys()}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("=== Downloading DeepSORT Models ===")
    success = download_deepsort_model()
    if success:
        print("\nAll models downloaded successfully!")
    else:
        print("\nError downloading models. Please check the error messages above.") 