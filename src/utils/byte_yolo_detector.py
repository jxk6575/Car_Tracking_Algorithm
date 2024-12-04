from typing import List
import numpy as np
from ultralytics import YOLO

from .detection import Detection

class ByteYOLODetector:
    """YOLOv8 detector wrapper for ByteTrack."""
    
    def __init__(self, model_name: str = 'yolov8n.pt'):
        """Initialize YOLOv8 detector."""
        self.model = YOLO(model_name)
        # Set to only detect cars (class 2 in COCO dataset)
        self.target_class = 2  # car class
        self.conf_threshold = 0.1  # Add lower confidence threshold for ByteTrack
        
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Detect cars in frame using YOLOv8."""
        try:
            results = self.model(frame, verbose=False, conf=self.conf_threshold)[0]
            detections = []
            
            for box in results.boxes:
                if int(box.cls) == self.target_class:
                    try:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = float(box.conf[0].cpu().numpy())
                        
                        # Remove size check to allow ByteTrack to handle small detections
                        detections.append(Detection(
                            bbox=np.array([x1, y1, x2, y2]),
                            confidence=confidence,
                            class_id=self.target_class
                        ))
                    except Exception as e:
                        print(f"Error processing detection: {e}")
                        continue
                        
            return detections
        except Exception as e:
            print(f"Error in detection: {e}")
            return [] 