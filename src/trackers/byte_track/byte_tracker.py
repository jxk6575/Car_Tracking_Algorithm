from typing import List
import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

from ..base_tracker import BaseTracker
from ...utils.detection import Detection
from ...utils.bbox import convert_bbox_to_z, convert_z_to_bbox, iou

class ByteTrackTrack:
    """Track class for ByteTrack."""
    
    count = 0
    
    def __init__(self, detection: Detection):
        """Initialize track with detection."""
        ByteTrackTrack.count += 1
        self.id = ByteTrackTrack.count
        
        # Initialize Kalman filter
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        
        # State transition matrix (constant velocity model)
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],  # x
            [0, 1, 0, 0, 0, 1, 0],  # y
            [0, 0, 1, 0, 0, 0, 1],  # s (scale)
            [0, 0, 0, 1, 0, 0, 0],  # r (aspect ratio)
            [0, 0, 0, 0, 1, 0, 0],  # dx
            [0, 0, 0, 0, 0, 1, 0],  # dy
            [0, 0, 0, 0, 0, 0, 1]   # ds
        ])
        
        # Measurement matrix
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ])
        
        # Measurement noise
        self.kf.R *= 10.0
        self.kf.R[2:, 2:] *= 10.0
        
        # Process noise
        self.kf.Q[4:, 4:] *= 0.01
        self.kf.Q[2:4, 2:4] *= 0.01
        self.kf.Q[:-1, -1] *= 0.01
        
        # Error covariance
        self.kf.P[4:, 4:] *= 100.0
        self.kf.P *= 10.0
        
        # Initialize state with first detection
        bbox = detection.bbox
        self.kf.x[:4] = convert_bbox_to_z(bbox).reshape(4, 1)
        self.kf.x[4:] = 0  # Initialize velocities to 0
        
        self.time_since_update = 0
        self.hits = 1
        self.hit_streak = 1
        self.age = 0
        self.last_bbox = bbox
        self.confidence = detection.confidence
        self.class_id = detection.class_id
        
        # Store original aspect ratio
        self.original_ar = (bbox[3] - bbox[1]) / (bbox[2] - bbox[0])
    
    def predict(self):
        """Predict next state."""
        if self.time_since_update > 0:
            self.hit_streak = 0
        
        # Predict
        self.kf.predict()
        
        # Get predicted bbox
        bbox = convert_z_to_bbox(self.kf.x[:4].reshape(-1))
        
        # Maintain aspect ratio
        current_ar = (bbox[3] - bbox[1]) / (bbox[2] - bbox[0])
        if abs(current_ar - self.original_ar) > 0.5:
            h = bbox[3] - bbox[1]
            w = h / self.original_ar
            cx = (bbox[2] + bbox[0]) / 2
            bbox[0] = cx - w/2
            bbox[2] = cx + w/2
        
        # Ensure bbox is within image bounds
        bbox[2:] = np.maximum(bbox[2:], bbox[:2])  # width/height must be positive
        bbox[:2] = np.maximum(bbox[:2], 0)  # limit to image bounds
        
        self.last_bbox = bbox  # Update last_bbox with prediction
        return bbox

    def update(self, detection: Detection):
        """Update track with detection."""
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        
        bbox = detection.bbox
        self.confidence = detection.confidence
        self.class_id = detection.class_id
        
        # Update Kalman filter
        measurement = convert_bbox_to_z(bbox).reshape((4, 1))
        self.kf.update(measurement)
        
        # Update last_bbox with the measurement
        self.last_bbox = bbox
    
    def get_state(self):
        """Get current state."""
        return self.last_bbox

class ByteTracker(BaseTracker):
    """ByteTrack implementation."""
    
    def __init__(self, config: dict):
        """Initialize ByteTracker with config."""
        super().__init__(config)  # Pass config to parent class
        self.max_age = config.get('max_age', 30)
        self.min_hits = config.get('min_hits', 3)
        self.iou_threshold = config.get('iou_threshold', 0.3)
        self.low_thresh = config.get('low_thresh', 0.1)
        self.high_thresh = config.get('high_thresh', 0.5)
        self.tracks = []
        self.frame_count = 0
    
    def reset(self):
        """Reset tracker state."""
        self.tracks = []
        self.frame_count = 0
        ByteTrackTrack.count = 0

    def predict(self):
        """Predict new locations of tracks."""
        for track in self.tracks:
            track.predict()
    
    def _match_detections(self, tracks: List[ByteTrackTrack], detections: List[Detection], threshold: float):
        """Match detections to tracks using IoU."""
        if len(tracks) == 0 or len(detections) == 0:
            return np.empty((0, 2), dtype=int), np.arange(len(tracks)), np.arange(len(detections))
        
        # Calculate IoU matrix
        iou_matrix = np.zeros((len(tracks), len(detections)))
        for t, track in enumerate(tracks):
            track_bbox = track.predict()
            for d, det in enumerate(detections):
                iou_val = iou(track_bbox, det.bbox)
                iou_matrix[t, d] = iou_val
        
        # Hungarian algorithm
        track_indices, det_indices = linear_sum_assignment(-iou_matrix)
        matches = np.column_stack((track_indices, det_indices))
        
        # Filter matches by threshold
        valid_matches = matches[iou_matrix[matches[:, 0], matches[:, 1]] >= threshold]
        
        unmatched_tracks = np.array([i for i in range(len(tracks)) 
                                   if i not in valid_matches[:, 0]]) if len(valid_matches) > 0 else np.arange(len(tracks))
        unmatched_dets = np.array([i for i in range(len(detections)) 
                                 if i not in valid_matches[:, 1]]) if len(valid_matches) > 0 else np.arange(len(detections))
        
        return valid_matches, unmatched_tracks, unmatched_dets
    
    def update(self, detections: List[Detection]) -> np.ndarray:
        """Update tracks with new detections."""
        self.predict()
        self.frame_count += 1
        
        # Split detections by confidence
        high_dets = [d for d in detections if d.confidence >= self.high_thresh]
        low_dets = [d for d in detections if self.low_thresh <= d.confidence < self.high_thresh]
        
        # First associate with high confidence detections
        matches, unmatched_tracks, unmatched_dets = self._match_detections(
            self.tracks, high_dets, self.iou_threshold)
        
        # Update matched tracks
        for track_idx, det_idx in matches:
            self.tracks[track_idx].update(high_dets[det_idx])
        
        # Try to match remaining tracks with low confidence detections
        remaining_unmatched_tracks = list(unmatched_tracks)  # Convert to list for easier indexing
        if len(remaining_unmatched_tracks) > 0 and len(low_dets) > 0:
            remaining_tracks = [self.tracks[i] for i in remaining_unmatched_tracks]
            matches_low, unmatched_tracks_low, _ = self._match_detections(
                remaining_tracks,
                low_dets,
                self.iou_threshold
            )
            
            # Update tracks matched with low confidence detections
            matched_track_indices = set()  # Keep track of which tracks were matched
            for local_track_idx, det_idx in matches_low:
                original_track_idx = remaining_unmatched_tracks[local_track_idx]
                self.tracks[original_track_idx].update(low_dets[det_idx])
                matched_track_indices.add(local_track_idx)
            
            # Update remaining unmatched tracks
            remaining_unmatched_tracks = [
                track_idx for i, track_idx in enumerate(remaining_unmatched_tracks)
                if i not in matched_track_indices
            ]
        
        # Update time since update for unmatched tracks
        for track_idx in remaining_unmatched_tracks:
            self.tracks[track_idx].time_since_update += 1
        
        # Create new tracks from unmatched high confidence detections
        for det_idx in unmatched_dets:
            new_track = ByteTrackTrack(high_dets[det_idx])
            self.tracks.append(new_track)
        
        # Remove old tracks
        self.tracks = [t for t in self.tracks if t.time_since_update < self.max_age]
        
        # Get results
        results = []
        for track in self.tracks:
            if track.time_since_update < 1 and (track.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                bbox = track.get_state()
                results.append([track.id, *bbox])
        
        return np.array(results) if results else np.empty((0, 5))