# AURA Human Activity Recognition (HAR) Pipeline

Comprehensive Human Activity Recognition module for the AURA (Adaptive Understanding and Reasoning Architecture) system.

## Pipeline Architecture
```text
UCF-101 Video Streams
        ↓
OpenCV Frame Extraction
        ↓
Google MediaPipe Pose Landmarker
        ↓
33 3D Body Landmarks (x, y, z, visibility = 132 features/frame)
        ↓
Landmark Normalization (Mid-hip translation + Torso Euclidean scale invariance)
        ↓
Uniform Temporal Resampling (30 frames fixed temporal sequence)
        ↓
  ┌───────────────────────┬────────────────────────┬──────────────────────┐
  ↓                       ↓                        ↓                      ↓
Random Forest       Spatial Pose CNN        Temporal Pose LSTM     Hybrid CNN + LSTM
  │                       │                        │                      │
  └───────────────────────┴────────────────────────┴──────────────────────┘
                                  ↓
                        Unified Scientific Evaluation
                                  ↓
                     Real-Time Webcam Inference (HUD)
```

## Supported 8 UCF101 Classes & Class IDs
1. `BodyWeightSquats` (ID: 15)
2. `JumpRope` (ID: 48)
3. `JumpingJack` (ID: 47)
4. `Lunges` (ID: 52)
5. `Punch` (ID: 71)
6. `PushUps` (ID: 72)
7. `TaiChi` (ID: 91)
8. `WallPushups` (ID: 99)

## Directory Structure
```text
AURA/
├── data/
│   └── har/
│       ├── raw/UCF-101/          # 998 videos (724 train, 274 test) [gitignored]
│       ├── splits/               # Official UCF101 train/test split files
│       └── processed/            # Preprocessed sequences & raw landmarks [gitignored]
├── ml/
│   └── har/
│       ├── dataset.py            # Dataset loader and split validator
│       ├── extract_landmarks.py  # MediaPipe landmark extraction CLI
│       ├── preprocess.py         # Normalization & temporal sequence builder
│       ├── features.py           # Handcrafted kinematic & angular feature extraction
│       ├── models.py             # PyTorch CNN, LSTM, and Hybrid CNN-LSTM architectures
│       ├── train_utils.py        # Checkpointing, early stopping, and training loop
│       ├── train_baseline.py     # Random Forest baseline trainer
│       ├── train_cnn.py          # Spatial CNN trainer
│       ├── train_lstm.py         # Temporal LSTM trainer
│       ├── train_cnn_lstm.py     # Hybrid CNN+LSTM trainer
│       ├── evaluate.py           # Unified metrics & comparative evaluation
│       └── inference.py          # Real-time webcam inference engine
├── vision/
│   └── pose/
│       ├── pose_detector.py      # Reusable MediaPipe PoseDetector wrapper
│       └── landmark_extractor.py # Video-to-landmark decoder
├── models/
│   └── har/                      # Trained model checkpoints
└── results/
    └── har/                      # Metrics, confusion matrices, and training curves
```

## How to Run Each Phase

### 1. Dataset Verification
```powershell
.\.venv\Scripts\python.exe ml/har/dataset.py
```

### 2. Single-Video Pose Extraction Test
```powershell
.\.venv\Scripts\python.exe ml/har/extract_landmarks.py --test-one
```

### 3. Full Batch Pose Extraction
```powershell
.\.venv\Scripts\python.exe ml/har/extract_landmarks.py --batch
```

### 4. Pose Normalization & Sequence Generation
```powershell
.\.venv\Scripts\python.exe -c "from ml.har.preprocess import create_preprocessed_dataset; create_preprocessed_dataset()"
```

### 5. Random Forest Baseline Training
```powershell
.\.venv\Scripts\python.exe ml/har/train_baseline.py
```

### 6. Spatial CNN Training
```powershell
.\.venv\Scripts\python.exe ml/har/train_cnn.py
```

### 7. Temporal LSTM Training
```powershell
.\.venv\Scripts\python.exe ml/har/train_lstm.py
```

### 8. Hybrid CNN + LSTM Training
```powershell
.\.venv\Scripts\python.exe ml/har/train_cnn_lstm.py
```

### 9. Unified Model Evaluation & Comparison
```powershell
.\.venv\Scripts\python.exe ml/har/evaluate.py
```

### 10. Real-Time Webcam Activity Inference
```powershell
.\.venv\Scripts\python.exe ml/har/inference.py
```
