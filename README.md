# 🎯 User-Selected Object Tracking System Using OpenCV

## 📌 Project Overview

The **User-Selected Object Tracking System** is an interactive Computer Vision application that enables users to manually select any object within a video and track it accurately across subsequent frames.

Unlike traditional tracking systems that may incorrectly switch to another object after losing the target, this project introduces a **Safe Tracking Mode** that detects tracking failures, stops tracking immediately, and waits for the user to manually reselect the desired object.

The project also provides preprocessing visualizations, multiple tracking algorithms, real-time performance monitoring, and automatic report generation.

---

# 🚀 Features

### 🎯 Interactive Object Selection

- Manual ROI (Region of Interest) selection.
- ROI validation to prevent accidental small selections.
- Ability to reset and select a new target at any time.

---

### 📹 Multiple Tracking Algorithms

The system supports three OpenCV tracking algorithms:

- **CSRT** – Highest tracking accuracy.
- **KCF** – Balanced speed and accuracy.
- **MOSSE** – Lightweight tracker optimized for speed.

Users can switch between trackers during execution without restarting the application.

---

### 🖼 Image Processing Pipeline

The project includes an image preprocessing stage for visualization purposes:

- Grayscale Conversion
- Edge Detection (Canny)
- Contour Detection

These preprocessing steps help visualize object boundaries and scene structure.

---

### 📦 Bounding Box Stabilization

To improve tracking stability, the project implements:

- Bounding box size stabilization.
- Growth limitation between consecutive frames.
- Minimum bounding box size constraints.
- Smooth bounding box resizing.

These techniques significantly reduce unstable tracking caused by sudden size fluctuations.

---

### ⚠️ Safe Target Lost Detection

Instead of incorrectly tracking another object, the system:

- Detects when the object leaves the frame.
- Monitors target visibility.
- Displays **TARGET LOST**.
- Stops tracking safely.
- Waits for manual target reselection.

---

### 📊 Automatic Report Generation

The application automatically generates:

- Annotated output video.
- CSV tracking report.
- Bounding box coordinates.
- Tracking status.
- FPS measurements.

---

### 🎮 Interactive Controls

| Key | Action |
|-----|--------|
| **Q / 1** | Quit |
| **S / 2** | Save current frame |
| **R / 3** | Select new target |
| **P / 4** | Pause / Resume |
| **T** | Change tracking algorithm |

---

# 🛠 Technologies Used

## Programming Language

- Python

## Computer Vision

- OpenCV

## Data Processing

- NumPy
- CSV

---

# 🔄 Project Workflow

1. Load the input video.
2. Resize video frames.
3. Select the target object.
4. Validate ROI.
5. Choose the tracking algorithm.
6. Initialize the tracker.
7. Track the selected object frame by frame.
8. Stabilize the bounding box.
9. Detect tracking failures.
10. Generate annotated video.
11. Save tracking results to CSV.

---

# 📈 Tracking Algorithms

| Tracker | Accuracy | Speed | Best Use Case |
|----------|----------|-------|---------------|
| **CSRT** | ⭐⭐⭐⭐⭐ | ⭐⭐ | High accuracy tracking |
| **KCF** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Balanced performance |
| **MOSSE** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Real-time applications |

---

# 📂 Project Outputs

The system generates:

- ✅ Real-time object tracking
- ✅ Bounding box visualization
- ✅ Tracking status monitoring
- ✅ FPS measurement
- ✅ Annotated output video
- ✅ CSV tracking report
- ✅ Saved image frames

---

# ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/yourusername/User-Selected-Object-Tracking-System.git
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the project

```bash
python main.py
```

---

# 📂 Project Structure

```
├── data/
│   └── test1.mp4
│
├── output/
│   ├── tracking_output.mp4
│   ├── tracking_report.csv
│   └── saved_frames/
│
├── main.py
├── requirements.txt
└── README.md
```

---

# 🎓 Learning Outcomes

This project provided practical experience in:

- Computer Vision
- OpenCV Tracking Algorithms
- Object Tracking
- Image Preprocessing
- ROI Selection
- Bounding Box Stabilization
- Video Processing
- Real-Time Performance Monitoring
- CSV Report Generation
- Interactive Application Development

---

# 👨‍💻 My Contribution

During this project, I contributed to:

- Implementing the object tracking pipeline.
- Integrating multiple OpenCV trackers (CSRT, KCF, and MOSSE).
- Developing ROI validation and interactive target selection.
- Implementing bounding box stabilization techniques.
- Designing the safe target-loss detection mechanism.
- Developing automatic CSV report generation.
- Testing and comparing tracker performance under different scenarios.

---

# 🚀 Future Improvements

- Integrate YOLO for automatic object detection.
- Add DeepSORT for multi-object tracking.
- Support GPU acceleration.
- Deploy as a desktop application.
- Add trajectory visualization and object analytics.

---

# ⭐ Support

If you found this project helpful, consider giving it a ⭐ on GitHub.
