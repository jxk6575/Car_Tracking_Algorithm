import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest

from src.trackers.sort import SORTTracker
from src.utils.detection import Detection
from src.trackers.byte_track import ByteTracker

def test_sort_tracker_initialization():
    config = {
        'max_age': 1,
        'min_hits': 3,
        'iou_threshold': 0.3
    }
    tracker = SORTTracker(config)
    assert tracker.max_age == 1
    assert tracker.min_hits == 3
    assert tracker.iou_threshold == 0.3

def test_sort_tracker_single_detection():
    config = {
        'max_age': 1,
        'min_hits': 1,
        'iou_threshold': 0.3
    }
    tracker = SORTTracker(config)
    
    # Create a single detection
    bbox = np.array([100, 100, 200, 200])
    detection = Detection(bbox=bbox, confidence=0.9, class_id=1)
    
    # Update tracker
    tracks = tracker.update([detection])
    
    assert len(tracks) == 1
    assert tracks[0][0] == 1  # First track should have ID 1
    np.testing.assert_array_almost_equal(tracks[0][1:], bbox, decimal=0)

def test_sort_tracker_track_continuation():
    config = {
        'max_age': 2,
        'min_hits': 1,
        'iou_threshold': 0.3
    }
    tracker = SORTTracker(config)
    
    # First frame
    bbox1 = np.array([100, 100, 200, 200])
    detection1 = Detection(bbox=bbox1, confidence=0.9, class_id=1)
    tracks1 = tracker.update([detection1])
    track_id = tracks1[0][0]
    
    # Second frame - slightly moved
    bbox2 = np.array([110, 110, 210, 210])
    detection2 = Detection(bbox=bbox2, confidence=0.9, class_id=1)
    tracks2 = tracker.update([detection2])
    
    assert tracks2[0][0] == track_id  # Should maintain same ID

def test_sort_tracker_track_termination():
    config = {
        'max_age': 1,
        'min_hits': 1,
        'iou_threshold': 0.3
    }
    tracker = SORTTracker(config)
    
    # First frame
    bbox = np.array([100, 100, 200, 200])
    detection = Detection(bbox=bbox, confidence=0.9, class_id=1)
    tracker.update([detection])
    
    # Second frame - no detections
    tracks = tracker.update([])
    assert len(tracks) == 0  # Track should be terminated

def test_sort_tracker_multiple_tracks():
    config = {
        'max_age': 1,
        'min_hits': 1,
        'iou_threshold': 0.3
    }
    tracker = SORTTracker(config)
    
    # Create two detections
    bbox1 = np.array([100, 100, 200, 200])
    bbox2 = np.array([300, 300, 400, 400])
    detection1 = Detection(bbox=bbox1, confidence=0.9, class_id=1)
    detection2 = Detection(bbox=bbox2, confidence=0.9, class_id=1)
    
    tracks = tracker.update([detection1, detection2])
    assert len(tracks) == 2
    assert tracks[0][0] != tracks[1][0]  # Should have different IDs

def test_byte_tracker_initialization():
    config = {
        'max_age': 30,
        'min_hits': 3,
        'iou_threshold': 0.3,
        'low_thresh': 0.1,
        'high_thresh': 0.5,
        'match_thresh': 0.8,
        'second_match_thresh': 0.5
    }
    tracker = ByteTracker(config)
    assert tracker.max_age == 30
    assert tracker.min_hits == 3
    assert tracker.iou_threshold == 0.3
    assert tracker.low_thresh == 0.1
    assert tracker.high_thresh == 0.5

def test_byte_tracker_high_confidence_detection():
    config = {
        'max_age': 30,
        'min_hits': 1,
        'iou_threshold': 0.3,
        'low_thresh': 0.1,
        'high_thresh': 0.5
    }
    tracker = ByteTracker(config)
    
    # Create high confidence detection
    bbox = np.array([100, 100, 200, 200])
    detection = Detection(bbox=bbox, confidence=0.9, class_id=1)
    
    tracks = tracker.update([detection])
    assert len(tracks) == 1
    assert tracks[0][0] == 1  # First track should have ID 1
    np.testing.assert_array_almost_equal(tracks[0][1:], bbox, decimal=0)

def test_byte_tracker_low_confidence_detection():
    config = {
        'max_age': 30,
        'min_hits': 1,
        'iou_threshold': 0.3,
        'low_thresh': 0.1,
        'high_thresh': 0.5
    }
    tracker = ByteTracker(config)
    
    # Create low confidence detection
    bbox = np.array([100, 100, 200, 200])
    detection = Detection(bbox=bbox, confidence=0.3, class_id=1)
    
    tracks = tracker.update([detection])
    assert len(tracks) == 0  # Should not create track from low confidence detection

def test_byte_tracker_confidence_threshold():
    config = {
        'max_age': 30,
        'min_hits': 1,
        'iou_threshold': 0.3,
        'low_thresh': 0.1,
        'high_thresh': 0.5
    }
    tracker = ByteTracker(config)
    
    # First frame - high confidence detection
    bbox1 = np.array([100, 100, 200, 200])
    det1 = Detection(bbox=bbox1, confidence=0.9, class_id=1)
    tracks1 = tracker.update([det1])
    track_id = tracks1[0][0]
    
    # Second frame - low confidence detection at similar location
    bbox2 = np.array([110, 110, 210, 210])
    det2 = Detection(bbox=bbox2, confidence=0.3, class_id=1)
    tracks2 = tracker.update([det2])
    
    assert len(tracks2) == 1
    assert tracks2[0][0] == track_id  # Should maintain track with low confidence match

def test_byte_tracker_track_recovery():
    config = {
        'max_age': 30,
        'min_hits': 1,
        'iou_threshold': 0.3,
        'low_thresh': 0.1,
        'high_thresh': 0.5
    }
    tracker = ByteTracker(config)
    
    # First frame - high confidence detection
    bbox1 = np.array([100, 100, 200, 200])
    det1 = Detection(bbox=bbox1, confidence=0.9, class_id=1)
    tracks1 = tracker.update([det1])
    track_id = tracks1[0][0]
    
    # Second frame - no detection
    tracks2 = tracker.update([])
    assert len(tracks2) == 1  # Should maintain track due to high max_age
    
    # Third frame - low confidence detection
    bbox3 = np.array([120, 120, 220, 220])
    det3 = Detection(bbox=bbox3, confidence=0.2, class_id=1)
    tracks3 = tracker.update([det3])
    assert len(tracks3) == 1
    assert tracks3[0][0] == track_id  # Should recover track with low confidence

def test_byte_tracker_multiple_confidence_levels():
    config = {
        'max_age': 30,
        'min_hits': 1,
        'iou_threshold': 0.3,
        'low_thresh': 0.1,
        'high_thresh': 0.5
    }
    tracker = ByteTracker(config)
    
    # Create mixed confidence detections
    bbox1 = np.array([100, 100, 200, 200])
    bbox2 = np.array([300, 300, 400, 400])
    det1 = Detection(bbox=bbox1, confidence=0.9, class_id=1)  # High confidence
    det2 = Detection(bbox=bbox2, confidence=0.3, class_id=1)  # Low confidence
    
    tracks = tracker.update([det1, det2])
    assert len(tracks) == 1  # Should only create track for high confidence detection
    np.testing.assert_array_almost_equal(tracks[0][1:], bbox1, decimal=0)
