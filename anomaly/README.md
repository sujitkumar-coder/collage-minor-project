# Crowd Density Based Anomaly Detection

## Overview
Real-time video analysis for detecting anomalies in crowd density using YOLOv8 object detection and supervision tracking.

**Key Features:**
- People detection and tracking (person class only).
- Bidirectional line zone counting (IN/OUT).
- 5x5 grid-based crowd density heatmap (color-coded: low-green to high-red).
- Anomaly detection: Flags overcrowded zones (> 3 standard deviations from average density), console alerts, red borders.
- Visual overlays: Boxes, lines, texts for counts/density/anomalies.
- Project title in window and console.

## Setup
1. Install dependencies:
   
```
   pip install -r requirements.txt
   
```
   (First run downloads YOLOv8m model ~100MB automatically.)

2. Place your input video as `test.mp4` in the project directory (or edit `video_path` in app.py).

## Usage
```
python app.py
```
- Press **ESC** to exit.
- Console shows alerts for anomalies.
- Window titled "Crowd Density Based Anomaly Detection".

## Customization
Edit `app.py`:
- `video_path`: Input video file.
- `GRID_SIZE = 5`: Grid resolution.
- `DENSITY_THRESHOLD = 3.0`: Anomaly sensitivity (lower = more sensitive).

## Dependencies
- opencv-python
- ultralytics (YOLOv8)
- supervision
- numpy

## Files
- `app.py`: Main application.
- `requirements.txt`: Dependencies.
- `TODO.md`: Improvement tracker.
