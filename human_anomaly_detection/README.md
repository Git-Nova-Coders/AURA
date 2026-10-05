# Human Anomaly Detection Using Computer Vision

A modular, high-performance Human Anomaly Detection subsystem built with PyTorch and OpenCV, specifically architected for integration into the **AURA** project.

---

## 1. Project Architecture

The system is designed with a strict decoupled boundary between upstream Person Detection and downstream Anomaly Classification:

```text
               LIVE CAMERA / VIDEO / IMAGE STREAM
                               │
                               ▼
               UPSTREAM PERSON DETECTOR
         (Standalone Torchvision or future AURA ANN)
                               │
                               ▼
                        PERSON CROPS
                               │
                               ▼
               ANOMALY DETECTOR INFERENCE ENGINE
                  (src.anomaly_detector.AnomalyDetector)
                               │
                  ┌────────────┴────────────┐
                  ▼                         ▼
            NORMAL (Label: 0)         ANOMALY (Label: 1)
```

This ensures the anomaly module never depends on the specific camera source or object detection model; it simply expects a cropped person NumPy array.

---

## 2. Directory Structure

```text
human_anomaly_detection/
├── dataset/
│   ├── train/        # (1500 normal, 1500 anomaly)
│   ├── validation/   # (1500 normal, 1500 anomaly)
│   └── test/         # (1500 normal, 1500 anomaly)
├── models/
│   └── anomaly_model.pth    # Trained CNN weights
├── results/
│   ├── classification_report.txt
│   ├── confusion_matrix.png
│   └── training_history.png
├── src/
│   ├── __init__.py
│   ├── config.py             # Centralized settings
│   ├── preprocessing.py      # BGR->RGB, resize, normalize, augment
│   ├── data_loader.py        # Framework-agnostic generator
│   ├── model.py              # AnomalyCNN architecture
│   ├── train.py              # Training loop with Early Stopping
│   ├── evaluate.py           # Objective test evaluation
│   ├── anomaly_detector.py   # Reusable AnomalyDetector class
│   ├── person_detector.py    # Standalone person crop extractor
│   ├── predict.py            # CLI for single image inference
│   └── predict_video.py      # Real-time video/webcam inference
├── tests/
│   ├── test_preprocessing.py
│   ├── test_model.py
│   ├── test_prediction.py
│   ├── test_person_detector.py
│   └── test_suite.py         # Comprehensive unit/integration suite
├── app.py                    # Complete application interface
└── requirements.txt
```

---

## 3. Quick Start & Setup

### Environment Setup
```powershell
# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 4. Usage Guide

### A. Run Application Layer (`app.py`)
The application supports image, video file, or live camera inputs:
```powershell
# 1. Single Image with person detection & anomaly labeling
python app.py --image dataset/test/anomaly/anom_test_00000.jpg --save

# 2. Live Webcam feed (using default camera 0)
python app.py --camera

# 3. Pre-recorded Video file
python app.py --video path/to/video.mp4
```

### B. Single Image CLI (`src/predict.py`)
```powershell
python -m src.predict --image dataset/test/anomaly/anom_test_00000.jpg --save
```

### C. Video Inference CLI (`src/predict_video.py`)
```powershell
python -m src.predict_video --video 0
```

---

## 5. Model Training & Evaluation

### Train Model:
```powershell
python -m src.train --epochs 30 --patience 5
```
Automatically maps to NVIDIA GPU if CUDA is available, triggers early stopping when validation loss plateaus, and saves the best model state to `models/anomaly_model.pth`.

### Evaluate Model:
```powershell
python -m src.evaluate
```
Evaluates on 3,000 unseen test samples and logs metrics (Accuracy, Precision, Recall, Specificity, F1-Score, ROC-AUC) to `results/classification_report.txt` and exports `results/confusion_matrix.png`.

---

## 6. Testing

Run the entire test suite covering unit tests, image robustness (lighting/blur/resolutions), and end-to-end person detection pipelines:
```powershell
python -m unittest tests/test_suite.py
```

---

## 7. Integration into AURA Project

When integrating with AURA's upstream ANN Object Detector:
```python
from src.anomaly_detector import AnomalyDetector

# 1. Load model once at startup
detector = AnomalyDetector("models/anomaly_model.pth")

# 2. During your detection loop (e.g. YOLO/SSD/ANN):
# person_crop is a standard cv2 / numpy BGR image slice
prediction = detector.predict(person_crop)

# Returns clean dictionary:
# {
#     "label": "Normal" | "Anomaly",
#     "confidence": 0.875
# }
```
