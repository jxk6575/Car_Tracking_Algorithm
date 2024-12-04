from typing import List, Optional

import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

from ..base_tracker import BaseTracker
from ...utils.bbox import convert_bbox_to_z, convert_z_to_bbox, iou
from ...utils.detection import Detection


class Track:
    """Track class for SORT."""
    
    count = 0
    
    def __init__(self, detection: Detection):
        """Initialize a track from detection."""
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        
        # Initialize state transition matrix
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
        
        # Initialize covariance matrices
        self.kf.R[2:, 2:] *= 10.  # Measurement uncertainty
        self.kf.P[4:, 4:] *= 1000. # Initial velocity uncertainty
        self.kf.P *= 10.
        
        # Initialize state vector
        z = convert_bbox_to_z(detection.bbox)
        self.kf.x = np.zeros((7, 1))  # Initialize state vector
        self.kf.x[:4] = z  # Set position and size (z is already shape (4,1))
        
        self.time_since_update = 0
        self.id = Track.count
        Track.count += 1
        self.hits = 1
        self.hit_streak = 1
        self.age = 1
        
    def update(self, detection: Detection):
        """Update the track with new detection."""
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        
        z = convert_bbox_to_z(detection.bbox)
        self.kf.update(z)
        
    def predict(self) -> np.ndarray:
        """Predict new location and return bounding box."""
        if self.kf.x[6] + self.kf.x[2] <= 0:
            self.kf.x[6] *= 0.0
            
        self.kf.predict()
        self.age += 1
        
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        
        # Convert prediction to bbox and ensure correct shape (4,)
        bbox = convert_z_to_bbox(self.kf.x)
        return bbox.reshape(-1)  # Reshape to 1D array


class SORTTracker(BaseTracker):
    """SORT (Simple Online and Realtime Tracking) implementation."""
    
    def __init__(self, config: dict):
        """Initialize SORT tracker.
        
        Args:
            config: Configuration dictionary with following keys:
                - max_age: Maximum frames to keep alive a track without associated detections
                - min_hits: Minimum hits to start a track
                - iou_threshold: IOU threshold for detection association
        """
        super().__init__(config)
        self.max_age = config.get('max_age', 1)
        self.min_hits = config.get('min_hits', 3)
        self.iou_threshold = config.get('iou_threshold', 0.3)
        
    def predict(self):
        """Predict new locations of tracks."""
        for track in self.tracks:
            track.predict()
    
    def update(self, detections: List[Detection]) -> np.ndarray:
        """Update the tracker with new detections."""
        self.frame_count += 1
        
        # Predict new locations of tracks
        self.predict()
        
        # Get predicted locations from existing tracks
        if len(self.tracks) > 0:
            trks = np.zeros((len(self.tracks), 4))
            for i, track in enumerate(self.tracks):
                trks[i] = track.predict()
            
            # Print debug info about tracks
            print("\nDebug - Tracks:")
            for i, track in enumerate(self.tracks):
                print(f"Track {i}: ID={track.id}, State={trks[i]}")
        else:
            trks = np.empty((0, 4))
        
        # Get current detections
        if len(detections) > 0:
            dets = np.array([d.bbox for d in detections])
            
            # Print debug info about detections
            print("\nDebug - Detections:")
            for i, det in enumerate(dets):
                print(f"Detection {i}: {det}")
        else:
            dets = np.empty((0, 4))
        
        # Match detections to tracks
        matched, unmatched_dets, unmatched_trks = self._match_detections_to_tracks(dets, trks)
        
        # Print debug info about matching
        print("\nDebug - Matching:")
        print(f"Matched pairs: {matched}")
        print(f"Unmatched detections: {unmatched_dets}")
        print(f"Unmatched tracks: {unmatched_trks}")
        
        # Update matched tracks
        for m in matched:
            self.tracks[m[1]].update(detections[m[0]])  # Changed index order
        
        # Create new tracks for unmatched detections
        for i in unmatched_dets:
            self.tracks.append(Track(detections[i]))
        
        # Delete old tracks
        ret = []
        i = len(self.tracks)
        for track in reversed(self.tracks):
            i -= 1
            if track.time_since_update > self.max_age:
                self.tracks.pop(i)
                continue
            
            bbox = track.predict()
            # Print debug info about track output
            print(f"\nDebug - Track output:")
            print(f"Track ID: {track.id}")
            print(f"Bbox shape: {bbox.shape}")
            print(f"Bbox: {bbox}")
            
            if track.hits >= self.min_hits or self.frame_count <= self.min_hits:
                try:
                    ret.append(np.concatenate(([track.id], bbox)))
                    print(f"Successfully concatenated track {track.id}")
                except ValueError as e:
                    print(f"Error concatenating track {track.id}:")
                    print(f"track.id shape: {np.array([track.id]).shape}")
                    print(f"bbox shape: {bbox.shape}")
                    raise e
        
        if len(ret) > 0:
            return np.stack(ret)
        return np.empty((0, 5))
    
    def _match_detections_to_tracks(self, detections: np.ndarray, tracks: np.ndarray):
        """Match detections to tracks using IOU."""
        if len(tracks) == 0 or len(detections) == 0:
            return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty(0, dtype=int)
        
        # Calculate IOU between all detections and tracks
        iou_matrix = np.zeros((len(detections), len(tracks)))
        for d, det in enumerate(detections):
            for t, trk in enumerate(tracks):
                iou_matrix[d, t] = iou(det, trk)
                
        # Use Hungarian algorithm to find best matches
        if min(iou_matrix.shape) > 0:
            matched_indices = np.array(linear_sum_assignment(-iou_matrix))
            matched_indices = matched_indices.transpose()
        else:
            matched_indices = np.empty((0, 2), dtype=int)
            
        # Filter matches with low IOU
        matches = []
        unmatched_detections = []
        for d, det in enumerate(detections):
            if d not in matched_indices[:, 0]:
                unmatched_detections.append(d)
                
        unmatched_tracks = []
        for t, trk in enumerate(tracks):
            if t not in matched_indices[:, 1]:
                unmatched_tracks.append(t)
                
        for m in matched_indices:
            if iou_matrix[m[0], m[1]] < self.iou_threshold:
                unmatched_detections.append(m[0])
                unmatched_tracks.append(m[1])
            else:
                matches.append(m)
                
        return np.array(matches), np.array(unmatched_detections), np.array(unmatched_tracks)