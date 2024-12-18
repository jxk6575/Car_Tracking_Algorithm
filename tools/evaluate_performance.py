import os
import sys
from pathlib import Path
import yaml
import time
import cv2

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from tools.evaluate import evaluate_tracker_performance
from src.trackers import SORTTracker, DeepSORTTracker, ByteTracker, HybridTracker
from src.utils import SimpleDetector, YOLODetector, ByteYOLODetector, HybridYOLODetector

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

def get_tracker_and_detector(config_path):
    """Initialize appropriate tracker and detector based on config."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    if 'byte_track' in str(config_path):
        tracker = ByteTracker(config)
        detector = ByteYOLODetector()
        print("\nInitialized ByteTrack with ByteYOLO detector")
    elif 'deep_sort' in str(config_path):
        tracker = DeepSORTTracker(config)
        detector = YOLODetector()
        print("\nInitialized DeepSORT with YOLO detector")
    elif 'hybrid' in str(config_path):
        tracker = HybridTracker(config)
        detector = HybridYOLODetector()
        print("\nInitialized HybridTrack with HybridYOLO detector")
    else:
        tracker = SORTTracker(config)
        detector = SimpleDetector()
        print("\nInitialized SORT with Simple detector")
    
    return tracker, detector

def main():
    print("=== Tracker Performance Evaluation ===")
    
    # Let user select config and video
    config_path = select_config()
    video_path = select_video()
    
    # Get number of frames to evaluate
    while True:
        try:
            frames = int(input("\nEnter number of frames to evaluate (default 1500): ") or "1500")
            if frames > 0:
                break
        except ValueError:
            pass
        print("Invalid number. Please try again.")
    
    # Initialize tracker and detector
    tracker, detector = get_tracker_and_detector(config_path)
    
    # Run evaluation
    print(f"\nEvaluating on {frames} frames...")
    metrics = evaluate_tracker_performance(tracker, detector, video_path, frames)
    
    # Print results
    print("\nPerformance Results:")
    print(f"Average FPS: {metrics['avg_fps']:.2f}")
    print(f"Average Tracks per Frame: {metrics['avg_num_tracks']:.2f}")
    print(f"Average Detection Time: {metrics['avg_det_time']:.2f}ms")
    print(f"Average Tracking Time: {metrics['avg_track_time']:.2f}ms")

if __name__ == "__main__":
    main() 