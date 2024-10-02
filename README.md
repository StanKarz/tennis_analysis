# Tennis Analysis with Computer Vision

## Overview
This project utilizes computer vision techniques to analyze tennis matches, providing in-depth insights and visualizations such as player tracking, ball trajectory, and shot statistics. By leveraging state-of-the-art deep learning models, this tool aims to assist in game analysis, player performance evaluation, and strategy formulation.

## Features

- **Ball Detector and Tracker**:  Detects and tracks the tennis ball across frames using a fine-tuned YOLOv8 model.
- **Tennis Player Detector and Tracker**: Identifies and tracks players on the court, using proximity to court keypoints for consistent player identification. 
- **Court Keypoints Identfication**: Detects court lines and keypoints to accurately map player and ball positions.
- **Mini Court Diagram**: Visualizes player and ball movements on a mini court schematic.
- **Player Statistics**: 
	- Ball hit speed
	- Average ball hit speed
	- Player speed
	- Average player speed
- **Ball Trajectory**: Visualizes the trajectory of the ball across the court
- Player movement visualisation via heatmap
- **Player Movement Heatmap**: Displays a heatmap showing areas of high player activity on the court.

## Examples
**Player and Ball Detection**

![Frame-by-frame tracking of the tennis ball and players, with bounding boxes.](examples/ball+player_detections.gif)
**Court Mapping**

![Identification of court keypoints, such as baselines, service boxes, and sidelines.](examples/court_kps.gif)
**Mini Court Diagram**

![Real-time visualization of player and ball movements on a court diagram.](examples/minicourt_keypoints.gif)

**Player Statistics**
![Logging player and balll speed statistics at each ball hit.](examples/player_stats.gif)

**Ball Trajectory**

![Ball path visualized with colored trails showing speed and direction changes.](examples/ball_trajectory.gif)

## Data and Models 

### Ball Detection 
-   **Model**: YOLOv8
-   **Dataset**: Custom dataset from [Roboflow Tennis Ball Detection](https://universe.roboflow.com/viren-dhanwani/tennis-ball-detection)
-   **Augmentations**:
    -   Flip: Horizontal
    -   Crop: 0% Minimum Zoom, 12% Maximum Zoom
    -   Brightness: Between -12% and +12%
    -   Exposure: Between -8% and +8%
   
   ### Player Detection
   -   **Model**: YOLOv8x Tracking
   - **Filtering Method**: Players are filtered using proximity to court keypoints for stable tracking.

### Court Keypoints Detection
-   **Model**: ResNet101
-   **Dataset**: Pretrained on court keypoints defined for hard, clay, and grass courts using a dataset from this [Google Drive link](https://drive.google.com/file/d/1lhAaeQCmk2y440PmagA0KmIVBIysVMwu/view).
-   **Source Repository**: [TennisCourtDetector Repo](https://github.com/yastrebksv/TennisCourtDetector?tab=readme-ov-file)
