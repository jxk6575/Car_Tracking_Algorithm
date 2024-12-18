from ultralytics import YOLO
import numpy as np
from .detection import Detection
from typing import List

class HybridYOLODetector:
    """YOLOv8 detector wrapper for HybridTracker."""
    
    def __init__(self, model_name: str = 'yolov8n.pt', 
                 conf_threshold: float = 0.5,
                 nms_threshold: float = 0.45):
        """Initialize YOLOv8 detector."""
        self.model = YOLO(model_name)
        self.target_class = 0  # For custom model with single class (car)
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        print(f"HybridYOLODetector initialized with model: {model_name}")
    
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Detect cars in frame using YOLOv8."""
        try:
            results = self.model(frame, verbose=False)[0]
            detections = []
            
            for box in results.boxes:
                # For single class model, we don't need to check class
                try:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0].cpu().numpy())
                    
                    if confidence >= self.conf_threshold:
                        detections.append(Detection(
                            bbox=np.array([x1, y1, x2, y2]),
                            confidence=confidence,
                            class_id=0  # Single class (car)
                        ))
                except Exception as e:
                    print(f"Error processing detection: {e}")
                    continue
                    
            return detections
            
        except Exception as e:
            print(f"Error in detection: {e}")
            return [] 