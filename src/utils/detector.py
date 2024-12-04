import cv2
import numpy as np
from .detection import Detection

class SimpleDetector:
    """Simple car detector using background subtraction."""
    
    def __init__(self):
        self.fgbg = cv2.createBackgroundSubtractorMOG2()
        self.min_area = 500  # Minimum contour area to be considered a car
        
    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Detect cars in frame using background subtraction."""
        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply background subtraction
        fgmask = self.fgbg.apply(gray)
        
        # Apply threshold to clean up the mask
        _, fgmask = cv2.threshold(fgmask, 128, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        for contour in contours:
            if cv2.contourArea(contour) < self.min_area:
                continue
                
            x, y, w, h = cv2.boundingRect(contour)
            bbox = np.array([x, y, x+w, y+h])
            detections.append(Detection(bbox=bbox, confidence=0.9, class_id=1))
            
        return detections 