from typing import List, Dict
import numpy as np
from ..utils.bbox import iou
from scipy.optimize import linear_sum_assignment

# Add default configuration
DEFAULT_METRICS_CONFIG = {
    'iou_threshold': 0.5,  # IOU threshold for considering a match
    'confidence_threshold': 0.5,  # Confidence threshold for predictions
    'max_age': 5,  # Maximum frames to keep unmatched tracks for MOTA
}

def initialize_metrics(config: dict = None) -> dict:
    """Initialize metrics configuration.
    
    Args:
        config: Optional configuration dictionary to override defaults
        
    Returns:
        dict: Final configuration with defaults and overrides
    """
    metrics_config = DEFAULT_METRICS_CONFIG.copy()
    if config:
        metrics_config.update(config)
    return metrics_config

def calculate_metrics(gt_tracks: np.ndarray, pred_tracks: np.ndarray, 
                     config: dict = None) -> Dict[str, float]:
    """Calculate MOT metrics.
    
    Args:
        gt_tracks: Ground truth tracks array
        pred_tracks: Predicted tracks array
        config: Optional configuration dictionary
        
    Returns:
        dict: Dictionary of calculated metrics
    """
    # Initialize configuration
    cfg = initialize_metrics(config)
    
    # Apply confidence threshold if specified
    if cfg['confidence_threshold'] is not None:
        pred_tracks = pred_tracks[pred_tracks[:, 5] >= cfg['confidence_threshold']]
    
    # Initialize counters
    matches = 0
    misses = 0
    false_positives = 0
    id_switches = 0
    total_iou = 0.0
    
    # Track assignments from previous frame
    prev_assignments = {}
    
    # Process each frame
    unique_frames = np.unique(gt_tracks[:, 0])
    for frame_id in unique_frames:
        frame_gt = gt_tracks[gt_tracks[:, 0] == frame_id]
        frame_pred = pred_tracks[pred_tracks[:, 0] == frame_id]
        
        # Calculate IoU matrix
        iou_matrix = np.zeros((len(frame_gt), len(frame_pred)))
        for i, gt in enumerate(frame_gt):
            for j, pred in enumerate(frame_pred):
                iou_matrix[i, j] = iou(gt[2:6], pred[2:6])
        
        # Find matches using Hungarian algorithm
        if len(frame_gt) > 0 and len(frame_pred) > 0:
            matched_indices = linear_sum_assignment(-iou_matrix)
            matched_indices = np.array(list(zip(*matched_indices)))
            
            # Filter matches by IoU threshold
            valid_matches = matched_indices[iou_matrix[matched_indices[:, 0], 
                                                    matched_indices[:, 1]] > 0.5]
            
            # Update metrics
            matches += len(valid_matches)
            misses += len(frame_gt) - len(valid_matches)
            false_positives += len(frame_pred) - len(valid_matches)
            
            # Calculate ID switches
            for gt_idx, pred_idx in valid_matches:
                gt_id = frame_gt[gt_idx, 1]
                pred_id = frame_pred[pred_idx, 1]
                
                if gt_id in prev_assignments and prev_assignments[gt_id] != pred_id:
                    id_switches += 1
                prev_assignments[gt_id] = pred_id
                
                # Add IoU for MOTP calculation
                total_iou += iou_matrix[gt_idx, pred_idx]
        else:
            misses += len(frame_gt)
            false_positives += len(frame_pred)
    
    # Calculate final metrics
    total_gt = len(gt_tracks)
    mota = 1 - (misses + false_positives + id_switches) / total_gt if total_gt > 0 else 0
    motp = total_iou / matches if matches > 0 else 0
    
    # Calculate IDF1 score
    num_detections = matches
    num_gt = len(gt_tracks)
    num_pred = len(pred_tracks)
    idf1 = 2 * num_detections / (num_gt + num_pred) if (num_gt + num_pred) > 0 else 0
    
    return {
        "MOTA": mota,
        "MOTP": motp,
        "IDF1": idf1,
        "Num_Switches": id_switches
    }
