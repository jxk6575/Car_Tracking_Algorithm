from pathlib import Path

def create_project_structure():
    """Create the basic project structure."""
    # Define the root directory
    root = Path(__file__).resolve().parent.parent
    
    # Create main directories
    directories = [
        'configs',
        'data/datasets',
        'data/models',
        'demo',
        'src/evaluation',
        'src/trackers/byte_track',
        'src/trackers/deep_sort',
        'src/trackers/sort',
        'src/utils',
        'tests',
        'tools'
    ]
    
    # Add data subdirectories
    directories.extend([
        'data/outputs',
        'data/models',
        'data/datasets'
    ])
    
    # Create directories
    for dir_path in directories:
        (root / dir_path).mkdir(parents=True, exist_ok=True)
    
    # Create config files
    configs = {
        'sort_config.yaml': """# SORT tracker configuration
max_age: 1          # Maximum number of frames to keep alive a track without associated detections
min_hits: 3         # Minimum number of associated detections before track is initialized
iou_threshold: 0.3  # Minimum IOU for match""",

        'deep_sort_config.yaml': """# DeepSORT tracker configuration
max_age: 1          # Maximum number of frames to keep alive a track without associated detections
min_hits: 3         # Minimum number of associated detections before track is initialized
iou_threshold: 0.3  # Minimum IOU for match
feature_threshold: 0.5  # Minimum feature similarity for match""",

        'byte_track_config.yaml': """# ByteTrack configuration
max_age: 1          # Maximum number of frames to keep alive a track without associated detections
min_hits: 3         # Minimum number of associated detections before track is initialized
iou_threshold: 0.3  # Minimum IOU for match
low_thresh: 0.1     # Low detection confidence threshold
high_thresh: 0.5    # High detection confidence threshold"""
    }
    
    # Write config files
    for filename, content in configs.items():
        with open(root / 'configs' / filename, 'w') as f:
            f.write(content)
            
    print("Project structure created successfully!")

if __name__ == "__main__":
    create_project_structure() 