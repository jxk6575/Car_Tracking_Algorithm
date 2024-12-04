from typing import List, Optional
import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

from ..base_tracker import BaseTracker
from ...utils.detection import Detection
from ...utils.bbox import convert_bbox_to_z, convert_z_to_bbox, iou
from .feature_extractor import FeatureExtractor

class DeepSORTTrack:
    """Track class for DeepSORT."""
    
    count = 0
    
    def __init__(self, detection: Detection, feature: np.ndarray):
        """Initialize track with detection and appearance feature."""
        # Initialize Kalman filter (same as SORT)
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ])
        
        # Initialize measurement matrix
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ])
        
        # Initialize state with detection
        z = convert_bbox_to_z(detection.bbox)
        self.kf.x[:4] = z
        
        # Track metadata
        self.id = DeepSORTTrack.count
        DeepSORTTrack.count += 1
        self.hits = 1
        self.time_since_update = 0
        self.age = 1
        
        # Feature history (for appearance matching)
        self.features = [feature]
        self.max_features = 100
        
    def update(self, detection: Detection, feature: np.ndarray):
        """Update track with new detection and feature."""
        self.hits += 1
        self.time_since_update = 0
        
        # Update Kalman filter
        z = convert_bbox_to_z(detection.bbox)
        self.kf.update(z)
        
        # Update feature history
        self.features.append(feature)
        if len(self.features) > self.max_features:
            self.features.pop(0)
            
    def get_feature(self) -> np.ndarray:
        """Get mean feature vector."""
        return np.mean(self.features, axis=0)
            
    def predict(self) -> np.ndarray:
        """Predict new location."""
        if self.kf.x[6] + self.kf.x[2] <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        return convert_z_to_bbox(self.kf.x)

class DeepSORTTracker(BaseTracker):
    """DeepSORT implementation."""
    
    def __init__(self, config: dict):
        """Initialize DeepSORT tracker."""
        super().__init__(config)
        self.max_age = config.get('max_age', 30)
        self.min_hits = config.get('min_hits', 3)
        self.iou_threshold = config.get('iou_threshold', 0.3)
        self.feature_threshold = config.get('feature_threshold', 0.5)
        self.feature_extractor = FeatureExtractor(config.get('model_path', None))
        self.tracks = []
        self.frame_count = 0
        
    def predict(self):
        """Predict next state of all tracks."""
        for track in self.tracks:
            track.predict()
            
    def update(self, detections: List[Detection], frame: np.ndarray = None) -> np.ndarray:
        """Update tracks with new detections."""
        self.frame_count += 1
        
        # Predict new locations of tracks
        self.predict()
        
        # Extract features for all detections
        if detections and frame is not None:
            features = self.feature_extractor.extract(frame, detections)
        else:
            features = np.array([])
            
        # Get predicted locations from existing tracks
        track_boxes = np.array([track.predict() for track in self.tracks])
        
        # Match detections to tracks
        if len(self.tracks) > 0 and len(detections) > 0:
            # Calculate cost matrix using both motion and appearance
            motion_cost = self._motion_distance(track_boxes, detections)
            appearance_cost = self._appearance_distance(self.tracks, features)
            
            # Combine costs
            cost_matrix = motion_cost + appearance_cost
            
            # Use Hungarian algorithm to find matches
            track_indices, det_indices = linear_sum_assignment(cost_matrix)
            
            # Filter matches using thresholds
            valid_matches = []
            for track_idx, det_idx in zip(track_indices, det_indices):
                if (motion_cost[track_idx, det_idx] <= self.iou_threshold and 
                    appearance_cost[track_idx, det_idx] <= self.feature_threshold):
                    valid_matches.append([track_idx, det_idx])
                    
            matches = np.array(valid_matches)
            
            if len(matches) == 0:
                matches = np.empty((0, 2), dtype=int)
                unmatched_tracks = list(range(len(self.tracks)))
                unmatched_detections = list(range(len(detections)))
            else:
                matched_track_indices = matches[:, 0]
                matched_det_indices = matches[:, 1]
                unmatched_tracks = [i for i in range(len(self.tracks)) 
                                  if i not in matched_track_indices]
                unmatched_detections = [i for i in range(len(detections)) 
                                      if i not in matched_det_indices]
        else:
            matches = np.empty((0, 2), dtype=int)
            unmatched_tracks = list(range(len(self.tracks)))
            unmatched_detections = list(range(len(detections)))
            
        # Update matched tracks
        for track_idx, det_idx in matches:
            self.tracks[track_idx].update(detections[det_idx], features[det_idx])
            
        # Create new tracks
        for det_idx in unmatched_detections:
            if len(features) > 0:  # Make sure we have features for the detection
                self.tracks.append(DeepSORTTrack(detections[det_idx], features[det_idx]))
            
        # Delete old tracks
        tracks_to_keep = []
        for track in self.tracks:
            if track.time_since_update < self.max_age:
                tracks_to_keep.append(track)
        self.tracks = tracks_to_keep
        
        # Get results
        results = []
        for track in self.tracks:
            if track.hits >= self.min_hits:
                bbox = track.predict()  # Use predict instead of get_state
                results.append(np.concatenate(([track.id], bbox)))
                
        return np.array(results) if results else np.empty((0, 5))
    
    def _motion_distance(self, tracks, detections):
        """Calculate IOU distance matrix."""
        cost_matrix = np.zeros((len(tracks), len(detections)))
        for i, track in enumerate(tracks):
            for j, det in enumerate(detections):
                cost_matrix[i, j] = 1 - iou(track, det.bbox)
        return cost_matrix
    
    def _appearance_distance(self, tracks, features):
        """Calculate feature distance matrix."""
        cost_matrix = np.zeros((len(tracks), len(features)))
        for i, track in enumerate(tracks):
            track_feat = track.get_feature()
            cost_matrix[i, :] = 1 - np.dot(track_feat, features.T)
        return cost_matrix
