# UCF101 Human Activity Recognition Dataset Specification

This document details the UCF101 dataset configuration, split methodology, and landmark representations used in the AURA Human Activity Recognition (HAR) subsystem.

## 1. Selected Activity Classes
We utilize 8 selected action classes from the UCF101 action recognition benchmark dataset:

| Class Index | Activity Class | Category | Motion Characteristics |
|---|---|---|---|
| **15** | `BodyWeightSquats` | Calisthenics | Vertical hip/knee flexion, sagittal symmetry |
| **48** | `JumpRope` | Aerobic | Rapid cyclical ankle/wrist elevation & jumping |
| **47** | `JumpingJack` | Aerobic | Lateral coronal limb abduction & adduction |
| **52** | `Lunges` | Calisthenics | Asymmetric anteroposterior lower-limb extension |
| **71** | `Punch` | Combat | Rapid unilateral upper-limb ballistic extension |
| **72** | `PushUps` | Calisthenics | Prone sagittal upper-limb flexion & extension |
| **91** | `TaiChi` | Martial Arts | Slow smooth kinematic trajectories & stance shifts |
| **99** | `WallPushups` | Calisthenics | Upright angled planar upper-limb pushing |

## 2. Official UCF101 Splits & Data Leakage Prevention
- **Official Split 1** (`trainlist01.txt`, `testlist01.txt`):
  - **724 Training Videos**
  - **274 Testing Videos**
  - **998 Total Videos**
  - **0 Missing Videos**
- **Data Leakage Rules**:
  - Training and testing partitions are segmented strictly at the **video clip / recording session group level**.
  - No frame-level random splitting is permitted.
  - Normalization parameters are calculated independently per pose frame (relative to mid-hip and torso scale) and never pooled across test sets.
  - Validation splits for deep learning training are generated strictly from the official training partition using stratified split.

## 3. Pose Representation & Normalization
- **Landmark Extractor**: Google MediaPipe Pose Landmarker (33 body landmarks, 3D coordinates + visibility confidence score = 132 features/frame).
- **Body-Relative Normalization**:
  1. **Translation Invariance**: Joint coordinates are centered by subtracting the mid-hip point $\frac{\text{LeftHip} + \text{RightHip}}{2}$.
  2. **Scale Invariance**: Coordinates are divided by the torso length $\|\text{MidShoulder} - \text{MidHip}\|_2$.
- **Temporal Window**: Fixed 30-frame sequence ($T=30$) via uniform temporal resampling.

## 4. Pipeline Execution
See [`ml/har/README.md`](../../ml/har/README.md) for full instructions on running dataset verification, extraction, preprocessing, model training, evaluation, and live webcam inference.
