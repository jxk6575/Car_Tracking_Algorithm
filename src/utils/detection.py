from dataclasses import dataclass
import numpy as np

@dataclass
class Detection:
    """Detection class for storing detection results."""
    bbox: np.ndarray  # [x1, y1, x2, y2]
    confidence: float
    class_id: int

    @property
    def center(self) -> np.ndarray:
        """Calculate center point of bounding box."""
        return (self.bbox[:2] + self.bbox[2:]) / 2


def iou(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
    """Calculate intersection over union between two boxes.
    
    Args:
        bbox1: First bounding box [x1, y1, x2, y2]
        bbox2: Second bounding box [x1, y1, x2, y2]
        
    Returns:
        float: IOU score between 0 and 1
    """
    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = bbox1
    b2_x1, b2_y1, b2_x2, b2_y2 = bbox2

    # Get the coordinates of intersection rectangle
    inter_rect_x1 = max(b1_x1, b2_x1)
    inter_rect_y1 = max(b1_y1, b2_y1)
    inter_rect_x2 = min(b1_x2, b2_x2)
    inter_rect_y2 = min(b1_y2, b2_y2)

    # Intersection area
    inter_area = max(0, inter_rect_x2 - inter_rect_x1) * \
                 max(0, inter_rect_y2 - inter_rect_y1)

    # Union Area
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    
    union = b1_area + b2_area - inter_area

    return inter_area / union if union > 0 else 0


def convert_bbox_to_z(bbox: np.ndarray) -> np.ndarray:
    """Convert bounding box format from [x1,y1,x2,y2] to [x,y,s,r].
    
    Where: x,y is the center of the box,
           s is the scale/area, and r is the aspect ratio
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w/2.
    y = bbox[1] + h/2.
    s = w * h  # scale is area
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_z_to_bbox(z: np.ndarray) -> np.ndarray:
    """Convert Kalman state [x,y,s,r] to bounding box [x1,y1,x2,y2]."""
    w = np.sqrt(z[2] * z[3])
    h = z[2] / w
    x1 = z[0] - w/2.
    y1 = z[1] - h/2.
    x2 = z[0] + w/2.
    y2 = z[1] + h/2.
    return np.array([x1, y1, x2, y2])