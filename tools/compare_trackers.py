import os
import sys
import time
import yaml
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.trackers.sort import SORTTracker
from src.trackers.deep_sort import DeepSORTTracker
from src.trackers.byte_track import ByteTracker
from src.utils.visualization import draw_tracks, create_comparison_view
from src.utils.detector import SimpleDetector
from src.utils.yolo_detector import YOLODetector
from src.utils.byte_yolo_detector import ByteYOLODetector

def get_tracker_and_detector(config_file):
    """Initialize appropriate tracker and detector based on config file."""
    # Load config
    with open(config_file) as f:
        config = yaml.safe_load(f)
    print(f"Loaded config: {config}")
    
    # Get config filename as string
    config_name = config_file.name.lower()
    
    if 'byte_track' in config_name:
        print(f"Initializing bytetrack tracker with config")
        tracker = ByteTracker(config)
        detector = ByteYOLODetector(config.get('yolo_model', 'yolov8n.pt'))
        
    elif 'deep_sort' in config_name:
        print(f"Initializing deepsort tracker with config")
        tracker = DeepSORTTracker(config)
        detector = YOLODetector()
        
    else:  # Default to SORT
        print(f"Initializing sort tracker with config")
        tracker = SORTTracker(config)
        detector = SimpleDetector()
    
    return tracker, detector

def evaluate_tracker(tracker, detector, video_path):
    """Evaluate a single tracker on a video."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Failed to open video: {video_path}")
    
    metrics = {
        'frame_times': [],
        'num_tracks': [],
        'detection_times': [],
        'tracking_times': [],
    }
    
    frame_count = 0
    
    with tqdm(desc=f"Processing {tracker.__class__.__name__}", unit='frames') as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            frame_start = time.time()
            
            # Detection
            det_start = time.time()
            detections = detector.detect(frame)
            det_time = time.time() - det_start
            
            # Tracking
            track_start = time.time()
            tracks = tracker.update(detections)
            track_time = time.time() - track_start
            
            frame_time = time.time() - frame_start
            
            # Store metrics
            metrics['frame_times'].append(frame_time)
            metrics['num_tracks'].append(len(tracks))
            metrics['detection_times'].append(det_time)
            metrics['tracking_times'].append(track_time)
            
            pbar.update(1)
    
    cap.release()
    
    # Calculate summary statistics
    summary = {
        'avg_fps': 1.0 / np.mean(metrics['frame_times']),
        'avg_num_tracks': np.mean(metrics['num_tracks']),
        'avg_det_time': np.mean(metrics['detection_times']) * 1000,  # Convert to ms
        'avg_track_time': np.mean(metrics['tracking_times']) * 1000,  # Convert to ms
        'total_frames': frame_count
    }
    
    return metrics, summary

def compare_trackers(video_path):
    """Compare all trackers on a single video."""
    config_dir = project_root / 'configs'
    results = []
    
    # Initialize all trackers and detectors at once
    trackers = {}
    detectors = {}
    for config_file in config_dir.glob('*.yaml'):
        try:
            tracker, detector = get_tracker_and_detector(config_file)
            trackers[tracker.__class__.__name__] = tracker
            detectors[tracker.__class__.__name__] = detector
            print(f"Initialized {tracker.__class__.__name__} with {config_file.name}")
        except Exception as e:
            print(f"Error initializing {config_file.name}: {str(e)}")
            continue
    
    # Initialize video capture
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Failed to open video: {video_path}")
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Initialize video writer for the 2x2 grid
    output_dir = project_root / 'data' / 'outputs'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f'comparison_{video_path.name}'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, 
                         (frame_width*2, frame_height*2))  # 2x2 grid size
    
    frame_count = 0
    metrics = {name: {'frame_times': [], 'num_tracks': [], 
                     'detection_times': [], 'tracking_times': []} 
              for name in trackers.keys()}
    
    with tqdm(desc="Processing frames", unit='frames') as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            tracker_results = {}
            tracker_metrics = {}
            
            # Process frame with each tracker
            for name, tracker in trackers.items():
                start_time = time.time()
                
                # Detection
                det_start = time.time()
                detections = detectors[name].detect(frame)
                det_time = time.time() - det_start
                
                # Tracking
                track_start = time.time()
                if isinstance(tracker, DeepSORTTracker):
                    tracks = tracker.update(detections, frame)
                else:
                    tracks = tracker.update(detections)
                track_time = time.time() - track_start
                
                # Calculate metrics
                frame_time = time.time() - start_time
                fps = 1.0 / frame_time if frame_time > 0 else 0
                
                # Store results and metrics
                tracker_results[name] = tracks
                tracker_metrics[name] = {
                    'fps': fps,
                    'num_tracks': len(tracks)
                }
                
                # Store detailed metrics
                metrics[name]['frame_times'].append(frame_time)
                metrics[name]['num_tracks'].append(len(tracks))
                metrics[name]['detection_times'].append(det_time)
                metrics[name]['tracking_times'].append(track_time)
            
            # Create comparison visualization
            vis_frame = create_comparison_view(frame, tracker_results, tracker_metrics)
            
            # Write frame to video
            out.write(vis_frame)
            
            # Display frame
            cv2.imshow('Tracker Comparison', vis_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            pbar.update(1)
    
    # Calculate summary statistics for each tracker
    for name in trackers.keys():
        summary = {
            'tracker': name,
            'config': config_file.name,
            'avg_fps': 1.0 / np.mean(metrics[name]['frame_times']),
            'avg_num_tracks': np.mean(metrics[name]['num_tracks']),
            'avg_det_time': np.mean(metrics[name]['detection_times']) * 1000,
            'avg_track_time': np.mean(metrics[name]['tracking_times']) * 1000,
            'total_frames': frame_count
        }
        results.append(summary)
    
    # Cleanup
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    # Save results
    if results:
        df = pd.DataFrame(results)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        csv_path = project_root / 'data' / 'results' / f'tracker_comparison_{timestamp}.csv'
        df.to_csv(csv_path, index=False)
        print(f"\nResults saved to: {csv_path}")
    
    print(f"\nComparison video saved to: {output_path}")

def main():
    print("=== Car Tracking Algorithms Comparison ===")
    
    # List available videos
    data_dir = project_root / 'data' / 'datasets'
    videos = list(data_dir.glob('*.avi'))
    videos.extend(list(data_dir.glob('*.mp4')))
    
    if not videos:
        raise FileNotFoundError("No video files found in data/datasets directory!")
    
    print("\nAvailable videos:")
    for i, video in enumerate(videos):
        print(f"{i+1}. {video.name}")
    
    # Get video selection
    while True:
        try:
            choice = int(input("\nSelect video number: ")) - 1
            if 0 <= choice < len(videos):
                video_path = videos[choice]
                break
        except ValueError:
            pass
        print("Invalid choice. Please try again.")
    
    # Run comparison
    compare_trackers(video_path)

if __name__ == "__main__":
    main() 