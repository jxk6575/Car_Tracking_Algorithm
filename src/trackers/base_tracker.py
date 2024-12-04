from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np

from ..utils.detection import Detection


class BaseTracker(ABC):
    """Abstract base class for all tracking algorithms."""
    
    def __init__(self, config: dict):
        """Initialize the tracker with configuration parameters.
        
        Args:
            config (dict): Configuration parameters for the tracker
        """
        self.config = config
        self.frame_count = 0
        self.tracks = []
    
    @abstractmethod
    def update(self, detections: List[Detection]) -> np.ndarray:
        """Update the tracker with new detections.
        
        Args:
            detections (List[Detection]): List of new detections in the current frame
            
        Returns:
            np.ndarray: Array of track results in format [track_id, x1, y1, x2, y2]
        """
        pass
    
    @abstractmethod
    def predict(self):
        """Predict new locations of tracks."""
        pass