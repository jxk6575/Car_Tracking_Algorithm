from typing import List
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
from ...utils.detection import Detection

class FeatureExtractor:
    def __init__(self, model_path: str = None):
        """Initialize feature extractor with optional pretrained weights."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Use a more suitable model for vehicle features
        self.model = resnet18(pretrained=True)
        self.model.fc = torch.nn.Sequential(
            torch.nn.Linear(512, 128),  # Reduce feature dimension
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64)    # Final feature dimension
        )
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Normalize using vehicle dataset statistics
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 128)),  # Larger size for better features
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
        
    def extract(self, frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """Extract features from detected regions."""
        if not detections:
            return np.array([])
            
        patches = []
        valid_detections = []
        for det in detections:
            x1, y1, x2, y2 = map(int, det.bbox)
            # Add padding around detection
            pad_x = int((x2 - x1) * 0.1)
            pad_y = int((y2 - y1) * 0.1)
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(frame.shape[1], x2 + pad_x)
            y2 = min(frame.shape[0], y2 + pad_y)
            
            patch = frame[y1:y2, x1:x2]
            if patch.size == 0:
                continue
            patch = self.transform(patch).unsqueeze(0)
            patches.append(patch)
            valid_detections.append(det)
            
        if not patches:
            return np.array([])
            
        # Stack all patches and extract features
        patches = torch.cat(patches).to(self.device)
        with torch.no_grad():
            features = self.model(patches)
            
        # L2 normalize features
        features = features.cpu().numpy()
        features = features / np.linalg.norm(features, axis=1, keepdims=True)
        return features
