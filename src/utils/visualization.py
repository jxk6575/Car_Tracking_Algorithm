import cv2
import numpy as np
from typing import List, Dict

def draw_tracks(frame: np.ndarray, tracks: np.ndarray, color: tuple) -> np.ndarray:
    """Draw tracking boxes on frame with specified color."""
    frame_copy = frame.copy()
    
    if len(tracks) == 0:
        return frame_copy
        
    for track in tracks:
        track_id = int(track[0])
        bbox = track[1:5].astype(int)
        
        # Draw bounding box
        cv2.rectangle(frame_copy, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        
        # Draw track ID
        cv2.putText(frame_copy, f"ID: {track_id}", (bbox[0], bbox[1]-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
    return frame_copy

def create_comparison_view(frame, tracker_results, tracker_metrics):
    """Create a 2x2 grid comparing different trackers."""
    height, width = frame.shape[:2]
    grid = np.zeros((height * 2, width * 2, 3), dtype=np.uint8)
    
    # Define positions for each tracker
    positions = {
        'Original': (0, 0),
        'SORTTracker': (0, width),
        'DeepSORTTracker': (height, 0),
        'ByteTracker': (height, width)
    }
    
    # Colors for each tracker
    colors = {
        'SORTTracker': (0, 255, 0),      # Green
        'DeepSORTTracker': (255, 0, 0),  # Red
        'ByteTracker': (0, 0, 255)       # Blue
    }
    
    # Place original frame
    grid[0:height, 0:width] = frame.copy()
    cv2.putText(grid, "Original", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Draw each tracker's results
    for tracker_name, tracks in tracker_results.items():
        if tracker_name not in positions:
            continue
            
        y, x = positions[tracker_name]
        color = colors.get(tracker_name, (0, 255, 0))
        
        # Create copy of frame for this tracker
        tracker_frame = frame.copy()
        
        # Draw tracks if any exist
        if tracks is not None and len(tracks) > 0:
            tracker_frame = draw_tracks(tracker_frame, tracks, color)
        
        # Add to grid
        grid[y:y+height, x:x+width] = tracker_frame
        
        # Add tracker name and metrics
        metrics = tracker_metrics.get(tracker_name, {})
        fps = metrics.get('fps', 0)
        num_tracks = metrics.get('num_tracks', 0)
        
        cv2.putText(grid, f"{tracker_name}", (x+10, y+30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(grid, f"FPS: {fps:.1f}", (x+10, y+60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.putText(grid, f"Tracks: {num_tracks}", (x+10, y+80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return grid 