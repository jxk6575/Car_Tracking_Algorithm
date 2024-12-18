from typing import List
import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

from ..base_tracker import BaseTracker
from ...utils.detection import Detection
from ...utils.bbox import convert_bbox_to_z, convert_z_to_bbox, iou

class HybridTrack:
    """Track class for Hybrid Tracker."""
    
    def __init__(self, detection: Detection):
        """Initialize track with detection."""
        self.id = None  # ID will be assigned by tracker
        
        # Initialize Kalman filter with more appropriate parameters
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        
        # State transition matrix
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ])
        
        # Measurement matrix
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ])
        
        # Initialize state
        self.kf.x[:4] = convert_bbox_to_z(detection.bbox)
        
        # Initialize uncertainties
        self.kf.P *= 10
        self.kf.P[4:, 4:] *= 1000
        self.kf.Q[4:, 4:] *= 0.1  # Increased from 0.01
        
        # Track metrics
        self.hits = 1
        self.hit_streak = 1
        self.age = 1
        self.time_since_update = 0
        self.confidence = detection.confidence
        self.score = detection.confidence
        
        # Reduced minimum hits requirement
        self.min_hits = 2  # Changed from 3
        
    def predict(self):
        """Predict next state using Kalman filter."""
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        return convert_z_to_bbox(self.kf.x)
        
    def update(self, detection: Detection):
        """Update track with detection."""
        self.kf.update(convert_bbox_to_z(detection.bbox))
        
        # Update track metrics
        self.hits += 1
        self.hit_streak += 1
        self.time_since_update = 0
        
        # Smooth confidence update
        self.confidence = 0.8 * self.confidence + 0.2 * detection.confidence
        
        # Update track score based on history
        self.score = (self.hits / self.age) * self.confidence
        
    def mark_missed(self):
        """Mark track as missed (no detection)."""
        self.hit_streak = 0
        self.time_since_update += 1
        
    def is_valid(self):
        """Check if track is valid based on history."""
        # More lenient validation criteria
        return (self.hits >= self.min_hits and 
                self.hit_streak >= 1 and 
                self.time_since_update < 3)
        
    def get_state(self) -> np.ndarray:
        """Get current state as bounding box."""
        return convert_z_to_bbox(self.kf.x)

class HybridTracker(BaseTracker):
    """Hybrid tracker implementation."""
    
    def __init__(self, config: dict):
        """Initialize tracker with configuration parameters."""
        super().__init__(config)
        self.high_thresh = config.get('high_thresh', 0.3)
        self.low_thresh = config.get('low_thresh', 0.1)
        self.max_time_lost = config.get('max_age', 60)
        
        # Matching parameters
        self.kalman_weight = config.get('kalman_weight', 0.3)
        self.iou_threshold = config.get('iou_threshold', 0.5)
        self.min_confidence = config.get('min_confidence', 0.2)
        
        self.tracks = []
        self.frame_count = 0
        self.next_id = 1  # Sequential ID counter
        
    def _get_next_id(self):
        """Get next sequential ID."""
        id_val = self.next_id
        self.next_id += 1
        return id_val
        
    def _match_high_confidence(self, detections):
        """Match high confidence detections to tracks."""
        if not self.tracks or not detections:
            return [], list(range(len(self.tracks))), list(range(len(detections)))
            
        # Calculate both IoU and Kalman distances
        iou_dists = self._iou_distance(self.tracks, detections)
        kalman_dists = self._kalman_distance(self.tracks, detections)
        
        # Combine distances with weights
        combined_dists = (1 - self.kalman_weight) * iou_dists + self.kalman_weight * kalman_dists
        
        # More lenient threshold for matching
        return self._hungarian_matching(combined_dists, 0.7)  # Increased threshold
        
    def update(self, detections: List[Detection]) -> np.ndarray:
        """Update tracks with new detections."""
        self.frame_count += 1
        
        # Predict track states
        for track in self.tracks:
            track.predict()
        
        # Split detections by confidence
        high_dets = [d for d in detections if d.confidence >= self.high_thresh]
        low_dets = [d for d in detections 
                   if self.low_thresh <= d.confidence < self.high_thresh]
        
        # First stage: match with high confidence detections
        matches_a, unmatched_tracks_a, unmatched_detections_a = \
            self._match_high_confidence(high_dets)
            
        # Second stage: match remaining tracks with low confidence detections
        matches_b, unmatched_tracks_b, unmatched_detections_b = \
            self._match_low_confidence(low_dets, unmatched_tracks_a)
        
        # Update matched tracks
        for track_idx, det_idx in matches_a:
            self.tracks[track_idx].update(high_dets[det_idx])
            
        for track_idx, det_idx in matches_b:
            self.tracks[track_idx].update(low_dets[det_idx])
            
        # Mark unmatched tracks as missed
        for track_idx in unmatched_tracks_b:
            self.tracks[track_idx].mark_missed()
        
        # Create new tracks from unmatched high confidence detections
        for det_idx in unmatched_detections_a:
            det = high_dets[det_idx]
            if det.confidence > self.min_confidence:
                new_track = HybridTrack(det)
                new_track.id = self._get_next_id()
                self.tracks.append(new_track)
        
        # Remove dead tracks
        self.tracks = [t for t in self.tracks 
                      if t.time_since_update < self.max_time_lost]
        
        # Return valid tracks
        results = []
        for track in self.tracks:
            if track.is_valid() and track.time_since_update < 1:
                bbox = track.get_state()
                results.append([track.id, *bbox])
        
        # Sort results by ID for consistent visualization
        if results:
            results = sorted(results, key=lambda x: x[0])
            
        return np.array(results) if results else np.empty((0, 5))

    def _match_low_confidence(self, detections, unmatched_tracks):
        """Match low confidence detections to remaining tracks."""
        if not detections or not unmatched_tracks:
            print("No detections or unmatched tracks for low confidence matching")
            return [], unmatched_tracks, []
        
        print(f"Low confidence matching: {len(detections)} detections, {len(unmatched_tracks)} unmatched tracks")
        
        # Get the subset of tracks that were unmatched
        track_subset = [self.tracks[i] for i in unmatched_tracks]
        
        # Use pure IoU matching for low confidence detections
        iou_dists = self._iou_distance(track_subset, detections)
        print(f"IoU distance matrix shape: {iou_dists.shape}")
        
        # Use lower threshold for second association
        matches, remaining_track_indices, unmatched_detections = \
            self._hungarian_matching(iou_dists, self.iou_threshold * 0.5)
        print(f"After Hungarian matching: {len(matches)} matches found")
        
        # Convert track indices back to original indexing
        converted_matches = []
        for track_idx, det_idx in matches:
            original_track_idx = unmatched_tracks[track_idx]
            converted_matches.append((original_track_idx, det_idx))
            print(f"Converted match: subset idx {track_idx} -> original idx {original_track_idx}")
        
        remaining_unmatched = [unmatched_tracks[i] for i in remaining_track_indices]
        
        return converted_matches, remaining_unmatched, unmatched_detections

    def _kalman_distance(self, tracks, detections):
        """Calculate distance between Kalman predictions and detections."""
        distance_matrix = np.zeros((len(tracks), len(detections)))
        
        for i, track in enumerate(tracks):
            track_pred = track.predict()  # Get Kalman prediction
            for j, det in enumerate(detections):
                # Calculate Euclidean distance between predicted location and detection
                pred_center = (track_pred[:2] + track_pred[2:]) / 2
                det_center = (det.bbox[:2] + det.bbox[2:]) / 2
                distance_matrix[i, j] = np.linalg.norm(pred_center - det_center)
        
        # Normalize distances to [0, 1] range
        if distance_matrix.size > 0:
            distance_matrix = distance_matrix / np.max(distance_matrix)
        
        return distance_matrix

    def _hungarian_matching(self, distance_matrix, threshold):
        """Perform matching using Hungarian algorithm."""
        if distance_matrix.size == 0:
            return [], list(range(distance_matrix.shape[0])), list(range(distance_matrix.shape[1]))
        
        # Use scipy's linear_sum_assignment for Hungarian algorithm
        track_indices, det_indices = linear_sum_assignment(distance_matrix)
        
        # Filter matches using threshold
        matches = []
        unmatched_tracks = list(range(distance_matrix.shape[0]))
        unmatched_detections = list(range(distance_matrix.shape[1]))
        
        for track_idx, det_idx in zip(track_indices, det_indices):
            if distance_matrix[track_idx, det_idx] <= threshold:
                matches.append((track_idx, det_idx))
                unmatched_tracks.remove(track_idx)
                unmatched_detections.remove(det_idx)
        
        return matches, unmatched_tracks, unmatched_detections

    def _iou_distance(self, tracks, detections):
        """Calculate IoU distance matrix between tracks and detections."""
        if not tracks or not detections:
            return np.empty((0, 0))
        
        distance_matrix = np.zeros((len(tracks), len(detections)))
        
        for i, track in enumerate(tracks):
            track_bbox = track.get_state()  # Get current bounding box
            for j, det in enumerate(detections):
                distance_matrix[i, j] = 1 - iou(track_bbox, det.bbox)  # Convert IoU to distance
        
        return distance_matrix

    def predict(self):
        """Predict the state of all tracks."""
        results = []
        for track in self.tracks:
            bbox = track.predict()  # This calls the track's predict method
            if track.is_valid():
                results.append([track.id, *bbox])
        
        # Sort results by ID for consistent visualization
        if results:
            results = sorted(results, key=lambda x: x[0])
            
        return np.array(results) if results else np.empty((0, 5))