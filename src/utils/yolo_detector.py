from ultralytics import YOLO
import numpy as np
from .detection import Detection
from typing import List

class YOLODetector:
    """YOLOv8 detector wrapper for DeepSORT."""
    
    def __init__(self, model_name: str = 'yolov8n.pt', 
                 conf_threshold: float = 0.5,
                 nms_threshold: float = 0.45):
        """Initialize YOLOv8 detector."""
        self.model = YOLO(model_name)
        self.target_class = 2  # cars
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        print(f"YOLODetector initialized with model: {model_name}")
    
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Detect cars in frame using YOLOv8."""
        # Run inference with confidence threshold
        results = self.model(frame, 
                           conf=self.conf_threshold,
                           iou=self.nms_threshold,
                           verbose=False)[0]
        
        detections = []
        for box in results.boxes:
            # Only process car detections
            if int(box.cls) == self.target_class:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0].cpu().numpy())
                
                # Additional size-based filtering
                width = x2 - x1
                height = y2 - y1
                min_size = 20  # Minimum detection size
                
                if width >= min_size and height >= min_size:
                    detections.append(Detection(
                        bbox=np.array([x1, y1, x2, y2]),
                        confidence=confidence,
                        class_id=self.target_class
                    ))
        
        return detections 