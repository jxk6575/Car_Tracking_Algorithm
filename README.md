# Car Tracking Algorithms

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8.0-red.svg)](https://opencv.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-8.0.196-green.svg)](https://github.com/ultralytics/yolov8)

Implementation and comparison of different car tracking algorithms: SORT, DeepSORT, ByteTrack, and HybridTrack.

## Overview

This project provides a comprehensive framework for tracking vehicles in video streams using state-of-the-art tracking algorithms. It includes implementations of SORT, DeepSORT, ByteTrack, and a custom HybridTrack algorithm.

![image](https://github.com/user-attachments/assets/fe8e9e69-1e82-4496-b407-209b4df8d30f)
![image](https://github.com/user-attachments/assets/2a29e146-843d-4148-a7fe-e27f99573a61)

## Latest Performance Results

### Tracker Comparison (1500 frames)
|   Tracker   |  FPS  | Tracks/Frame | Detection (ms) | Tracking (ms) |  MOTA (%) | ID Switches |
|-------------|-------|--------------|----------------|---------------|-----------|-------------|
|  ByteTrack  |  6.45 |     7.36     |     151.45     |      3.54     |     83    |      46     |
|   DeepSORT  |  3.59 |     5.08     |     141.25     |    137.00     |     79    |      75     |
|    SORT     | 34.87 |     4.62     |      17.49     |     11.19     |     65    |     137     |
| HybridTrack | 19.53 |     4.78     |      49.84     |      1.38     |     94    |      15     |

### Key Findings
- HybridTrack achieves balanced performance between speed and accuracy
- Improved track consistency with enhanced filtering
- Better handling of occlusions and ID switches

### Recent Updates
- ✅ Added multi-tracker comparison visualization
- ✅ Implemented real-time performance metrics
- ✅ Enhanced tracker initialization
- ✅ Improved visualization system
- ✅ Added comprehensive results analysis

## Features
- ✅ Three tracking algorithms implemented:
  - SORT (Simple Online and Realtime Tracking)
  - DeepSORT (Deep Learning enhanced SORT)
  - ByteTrack (State-of-the-art tracking)
- ✅ Multiple detector options:
  - Simple detector (background subtraction)
  - YOLOv8 detector
  - ByteYOLO detector (optimized for ByteTrack)
- ✅ Real-time visualization
- ✅ Configurable parameters via YAML
- ✅ Interactive demo application

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/Car_Tracking_Algorithms.git
cd Car_Tracking_Algorithms
```

2. Install dependencies:

```bash
pip install -r requirements.txt
pip install ultralytics  # For YOLOv8 detector

```

## Project Setup

1. Create required directories:

```bash
mkdir -p data/datasets
mkdir -p data/models
mkdir -p data/outputs

```

2. Add video files:
- Place your video files in `data/datasets/`
- Supported formats: .avi, .mp4

3. Verify configurations:
- Check that config files exist in `configs/`
- Verify parameters in config files

## Running the Demo

1. From project root directory:

```bash
python demo/demo.py

```

2. Follow the interactive prompts to:
- Select tracking configuration (SORT, DeepSORT, or ByteTrack)
- Select input video

3. View results:
- Real-time tracking visualization will be shown
- Processed video will be saved to `data/outputs/`
- Press 'q' to quit the demo

## Tracking Methods

### SORT
- Simple and fast tracking
- Uses Kalman filter for motion prediction
- IoU-based matching
- Best for simple scenarios with clear visibility

### DeepSORT
- Enhanced tracking with deep learning features
- Better identity preservation
- Uses YOLOv8 for detection
- Good balance of speed and accuracy

### ByteTrack
- State-of-the-art performance
- Handles low-confidence detections
- Advanced state estimation
- Best for complex scenarios

### HybridTrack
- Based on ByteTrack's proven framework
- Enhanced track management system
- Efficient Kalman filtering
- IoU-based matching with Hungarian algorithm
- Good balance between speed and accuracy
- Suitable for real-time applications with moderate complexity

## Configuration

Each tracker has its own configuration file in `configs/`:
- `sort_config.yaml`
- `deep_sort_config.yaml`
- `byte_track_config.yaml`

Key parameters can be adjusted in these files.

## Project Structure
```
Car_Tracking_Algorithms/
├── configs/           # Configuration files
├── data/             # Data storage
│   ├── datasets/     # Input videos
│   ├── models/       # Model weights
│   └── outputs/      # Processed videos
├── demo/             # Demo application
├── src/              # Source code
│   ├── trackers/     # Tracker implementations
│   └── utils/        # Shared utilities
└── tests/            # Test suite
```

## Troubleshooting

If you encounter issues:
1. Verify all dependencies are installed
2. Check video file exists in data/datasets
3. Ensure you're running from project root
4. Verify config files exist in configs directory
5. Check that data/outputs directory exists for saving results

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## HybridTrack: A Unified Approach
Our HybridTrack algorithm successfully combines the strengths of three leading tracking methods:

### Integration of Methods
1. **From SORT:**
   - Efficient Kalman filter for motion prediction
   - Hungarian algorithm for optimal assignment
   - Fast IoU-based matching

2. **From DeepSORT:**
   - Enhanced track management system
   - Robust track validation criteria
   - Hit streak based confirmation

3. **From ByteTrack:**
   - Dual-threshold detection handling
   - Advanced state estimation
   - Track recovery mechanism

### Key Innovations
- **Sequential ID Management:**
  - Strictly increasing ID assignment
  - No ID recycling to prevent confusion
  - Continuous tracking history preservation

- **Stable Track Maintenance:**
  - Combined motion and IoU matching
  - Adaptive track validation criteria
  - Enhanced state prediction

- **Robust Detection Association:**
  - Two-stage matching strategy
  - High/low confidence detection handling
  - Improved occlusion handling

### Performance Highlights
- Maintains consistent tracking through occlusions
- Sequential ID assignment (1,2,3...)
- Stable bounding boxes
- Efficient computation (~20 FPS)

## Contributing

1. Fork the repository.
2. Create your feature branch.
3. Commit your changes.
4. Push to the branch.
5. Create a new Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
