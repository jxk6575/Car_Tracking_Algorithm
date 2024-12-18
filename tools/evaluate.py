import argparse
from pathlib import Path
import yaml
import numpy as np
import cv2
import time

from src.trackers import SORTTracker, ByteTracker
from src.evaluation.metrics import calculate_metrics

def evaluate_tracker(tracker, data_path: str, output_path: str = None):
    """Evaluate tracker on dataset."""
    # Load ground truth
    gt_data = np.load(data_path)
    
    # Initialize metrics
    frame_metrics = []
    all_tracks = []
    
    # Process each frame
    unique_frames = np.unique(gt_data[:, 0])
    for frame_id in unique_frames:
        # Get ground truth for this frame
        frame_gt = gt_data[gt_data[:, 0] == frame_id]
        
        # Convert to detections
        detections = []
        for gt in frame_gt:
            bbox = gt[2:6]  # [x1, y1, x2, y2]
            confidence = 1.0  # Ground truth has confidence 1
            class_id = int(gt[1])  # Assuming track_id is class_id
            detections.append(Detection(bbox=bbox, confidence=confidence, class_id=class_id))
        
        # Update tracker
        tracks = tracker.update(detections)
        all_tracks.append(np.column_stack((np.full((len(tracks), 1), frame_id), tracks)))
        
        # Calculate frame metrics
        frame_metrics.append(calculate_metrics(frame_gt, tracks))
        
    # Combine all tracks
    all_tracks = np.vstack(all_tracks) if all_tracks else np.empty((0, 6))
    
    # Calculate overall metrics
    overall_metrics = {
        "MOTA": np.mean([m["MOTA"] for m in frame_metrics]),
        "MOTP": np.mean([m["MOTP"] for m in frame_metrics]),
        "IDF1": np.mean([m["IDF1"] for m in frame_metrics]),
        "Num_Switches": sum(m["Num_Switches"] for m in frame_metrics)
    }
    
    # Save results if output path provided
    if output_path:
        np.save(output_path, all_tracks)
        
    return overall_metrics

def evaluate_tracker_performance(tracker, detector, video_path, num_frames=1500):
    """Evaluate tracker performance metrics."""
    cap = cv2.VideoCapture(str(video_path))
    
    total_time = 0
    total_det_time = 0
    total_track_time = 0
    total_tracks = 0
    frame_count = 0
    
    while frame_count < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
            
        start_time = time.time()
        
        # Detection time
        det_start = time.time()
        detections = detector.detect(frame)
        det_time = time.time() - det_start
        
        # Tracking time
        track_start = time.time()
        tracks = tracker.update(detections)
        track_time = time.time() - track_start
        
        frame_time = time.time() - start_time
        
        total_time += frame_time
        total_det_time += det_time
        total_track_time += track_time
        total_tracks += len(tracks)
        frame_count += 1
    
    cap.release()
    
    # Calculate metrics
    avg_fps = frame_count / total_time
    avg_tracks = total_tracks / frame_count
    avg_det_time = (total_det_time / frame_count) * 1000  # Convert to ms
    avg_track_time = (total_track_time / frame_count) * 1000  # Convert to ms
    
    return {
        'avg_fps': avg_fps,
        'avg_num_tracks': avg_tracks,
        'avg_det_time': avg_det_time,
        'avg_track_time': avg_track_time
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/sort_config.yaml')
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--output', type=str, help='Path to save tracking results')
    args = parser.parse_args()
    
    # Initialize tracker
    tracker = get_tracker(args.config)
    print(f"\nInitialized {tracker.__class__.__name__}")
    
    # Run evaluation
    print("\nRunning evaluation...")
    metrics = evaluate_tracker(tracker, args.data, args.output)
    
    # Print results
    print("\nEvaluation Results:")
    print("-----------------")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

def get_tracker(config_path):
    """Get appropriate tracker based on config file."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    if 'byte_track' in config_path.name.lower():
        return ByteTracker(config)
    elif 'deep_sort' in config_path.name.lower():
        return DeepSORTTracker(config)
    else:
        return SORTTracker(config)

if __name__ == "__main__":
    main()
