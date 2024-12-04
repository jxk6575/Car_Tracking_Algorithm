import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import cv2
import yaml
import numpy as np

from src.trackers.sort import SORTTracker
from src.trackers.deep_sort import DeepSORTTracker
from src.trackers.byte_track import ByteTracker
from src.utils.visualization import draw_tracks
from src.utils.detector import SimpleDetector
from src.utils.yolo_detector import YOLODetector
from src.utils.byte_yolo_detector import ByteYOLODetector

def select_config():
    """Let user select tracking configuration."""
    config_dir = project_root / 'configs'
    if not config_dir.exists():
        raise FileNotFoundError(f"configs directory not found at {config_dir}!")
    
    configs = list(config_dir.glob('*.yaml'))
    if not configs:
        raise FileNotFoundError("No configuration files found in configs directory!")
    
    print("\nAvailable configurations:")
    for i, config in enumerate(configs):
        print(f"{i+1}. {config.name}")
    
    while True:
        try:
            choice = int(input("\nSelect configuration number: ")) - 1
            if 0 <= choice < len(configs):
                return configs[choice]
        except ValueError:
            pass
        print("Invalid choice. Please try again.")

def select_video():
    """Let user select input video."""
    data_dir = project_root / 'data' / 'datasets'
    if not data_dir.exists():
        raise FileNotFoundError(f"data/datasets directory not found at {data_dir}!")
    
    videos = list(data_dir.glob('*.avi'))
    videos.extend(list(data_dir.glob('*.mp4')))
    if not videos:
        raise FileNotFoundError("No video files found in data/datasets directory!")
    
    print("\nAvailable videos:")
    for i, video in enumerate(videos):
        print(f"{i+1}. {video.name}")
    
    while True:
        try:
            choice = int(input("\nSelect video number: ")) - 1
            if 0 <= choice < len(videos):
                return videos[choice]
        except ValueError:
            pass
        print("Invalid choice. Please try again.")

def run_sort_tracker(tracker, detector, video_path, output_path):
    """Run SORT tracker with simple detector."""
    cap = cv2.VideoCapture(str(video_path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Get detections and update tracker
        detections = detector.detect(frame)
        tracks = tracker.update(detections)
        
        # Draw and display results
        frame = draw_tracks(frame, tracks)
        cv2.imshow('Tracking', frame)
        out.write(frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    out.release()
    cv2.destroyAllWindows()

def run_deep_sort_tracker(tracker, detector, video_path, output_path):
    """Run DeepSORT tracker with YOLOv8 detector."""
    cap = cv2.VideoCapture(str(video_path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Get detections and update tracker
        detections = detector.detect(frame)
        tracks = tracker.update(detections, frame)
        
        # Draw and display results
        frame = draw_tracks(frame, tracks)
        cv2.imshow('Tracking', frame)
        out.write(frame)
        
        # Print progress
        frame_count += 1
        if frame_count % 30 == 0:  # Show progress every 30 frames
            print(f"Processed {frame_count} frames")
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    out.release()
    cv2.destroyAllWindows()

def run_byte_track_tracker(tracker, detector, video_path, output_path):
    """Run ByteTrack tracker with YOLOv8 detector."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Get detections and update tracker
        detections = detector.detect(frame)
        print(f"\nFrame {frame_count}: {len(detections)} detections")
        
        if detections:
            tracks = tracker.update(detections)
            print(f"Tracks returned: {len(tracks)}")
            if len(tracks) > 0:
                print(f"First track format: {tracks[0]}")  # Debug track format
                frame = draw_tracks(frame, tracks)
        
        cv2.imshow('Tracking', frame)
        out.write(frame)
        
        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    out.release()
    cv2.destroyAllWindows()

def get_tracker(config_path):
    """Initialize appropriate tracker based on config file."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    if 'byte_track' in config_path.name.lower():
        return ByteTracker(config)
    elif 'deep_sort' in config_path.name.lower():
        return DeepSORTTracker(config)
    else:
        return SORTTracker(config)

def draw_tracks(frame: np.ndarray, tracks: np.ndarray, base_color: tuple = (0, 255, 0)) -> np.ndarray:
    """Draw tracking results on frame with consistent colors per track ID."""
    frame_copy = frame.copy()
    
    if len(tracks) == 0:  # Handle empty tracks
        return frame_copy
        
    for track in tracks:
        if len(track) < 5:  # Need at least [id, x1, y1, x2, y2]
            continue
            
        track_id = int(track[0])
        bbox = track[1:5].astype(int)
        
        # Generate consistent color based on track ID
        np.random.seed(track_id)  # Set seed based on track ID for consistent colors
        color = (
            int((track_id * 123) % 255),
            int((track_id * 50) % 255),
            int((track_id * 182) % 255)
        )
        
        # Draw bounding box
        cv2.rectangle(frame_copy, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        
        # Draw track ID
        label = f"{track_id}"
        cv2.putText(frame_copy, label, (bbox[0], bbox[1]-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
    return frame_copy

def main():
    print("=== Car Tracking Demo ===")
    
    # Let user select config and video
    config_path = select_config()
    video_path = select_video()
    
    # Initialize appropriate tracker and detector
    tracker = get_tracker(config_path)
    
    if isinstance(tracker, DeepSORTTracker):
        detector = YOLODetector()  # Use original YOLODetector for DeepSORT
        print("\nInitialized YOLOv8 detector for DeepSORT")
    elif isinstance(tracker, ByteTracker):
        detector = ByteYOLODetector()  # Use new detector for ByteTrack
        print("\nInitialized YOLOv8 detector for ByteTrack")
    else:
        detector = SimpleDetector()  # Use simple detector for SORT
        print("\nInitialized simple detector")
        
    print(f"Initialized {tracker.__class__.__name__} tracker")
    print(f"Loaded video: {video_path.name}")
    
    # Create output directory if it doesn't exist
    output_dir = project_root / 'data' / 'outputs'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create output filename with tracker type
    tracker_name = tracker.__class__.__name__.replace('Tracker', '').lower()
    output_path = output_dir / f'{tracker_name}_{video_path.name}'
    print(f"Output will be saved to: {output_path}")
    
    # Run appropriate tracker
    if isinstance(tracker, DeepSORTTracker):
        run_deep_sort_tracker(tracker, detector, video_path, output_path)
    elif isinstance(tracker, ByteTracker):
        run_byte_track_tracker(tracker, detector, video_path, output_path)
    else:
        run_sort_tracker(tracker, detector, video_path, output_path)
    
    print(f"\nDemo finished - Output saved to {output_path}")

if __name__ == "__main__":
    main()
