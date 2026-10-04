"""
AURA HAR Handcrafted Feature Extraction
Generates statistical, kinematic, and angular features from 3D pose sequences
for classical machine learning baselines (e.g., Random Forest).
"""

import math
from typing import List, Tuple

import numpy as np

# MediaPipe Landmark Indices
NOSE = 0
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28


def calculate_joint_angle_3d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Calculates angle (in degrees) at joint b formed by vectors ba and bc."""
    v1 = a - b
    v2 = c - b
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 < 1e-6 or norm2 < 1e-6:
        return 180.0
    cosine = np.dot(v1, v2) / (norm1 * norm2)
    cosine = np.clip(cosine, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def extract_features_from_sequence(seq: np.ndarray) -> np.ndarray:
    """
    Extracts an aggregated 1D feature vector from a sequence of shape (T, 33, 4).

    Features extracted:
    1. Statistical Landmark Moments (mean, std, min, max) of (x, y, z) coordinates:
       33 landmarks * 3 coords * 4 stats = 396 features
    2. Kinematic Velocity:
       Frame-to-frame coordinate differences delta = seq[t+1] - seq[t]
       Velocity mean and std for key joints: 12 key joints * 3 coords * 2 = 72 features
    3. Angular Dynamics:
       - Left & Right Elbow angle (Shoulder - Elbow - Wrist)
       - Left & Right Knee angle (Hip - Knee - Ankle)
       - Left & Right Shoulder angle (Elbow - Shoulder - Hip)
       - Left & Right Hip angle (Shoulder - Hip - Knee)
       For each of these 8 angles: mean, std, min, max across time = 32 features
    4. Distance Features:
       - Wrist-to-Wrist distance (mean, std, min, max) = 4 features
       - Ankle-to-Ankle distance (mean, std, min, max) = 4 features

    Total Feature Dimension: ~508 features per sequence.
    """
    T = seq.shape[0]
    coords = seq[:, :, :3]  # (T, 33, 3)

    # 1. Global Landmark Statistics
    mean_coords = np.mean(coords, axis=0).flatten()
    std_coords = np.std(coords, axis=0).flatten()
    min_coords = np.min(coords, axis=0).flatten()
    max_coords = np.max(coords, axis=0).flatten()

    feature_list = [mean_coords, std_coords, min_coords, max_coords]

    # 2. Velocity Statistics
    if T > 1:
        velocities = np.diff(coords, axis=0)  # (T-1, 33, 3)
        key_joints = [
            NOSE, LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW,
            LEFT_WRIST, RIGHT_WRIST, LEFT_HIP, RIGHT_HIP,
            LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE
        ]
        key_vel = velocities[:, key_joints, :]
        vel_mean = np.mean(key_vel, axis=0).flatten()
        vel_std = np.std(key_vel, axis=0).flatten()
        feature_list.extend([vel_mean, vel_std])
    else:
        # Fallback for single-frame sequence
        zeros = np.zeros(13 * 3, dtype=np.float32)
        feature_list.extend([zeros, zeros])

    # 3. Dynamic Joint Angles
    angle_series = {
        "l_elbow": [], "r_elbow": [],
        "l_knee": [], "r_knee": [],
        "l_shoulder": [], "r_shoulder": [],
        "l_hip": [], "r_hip": [],
    }

    for t in range(T):
        frame = coords[t]
        # Elbows
        angle_series["l_elbow"].append(
            calculate_joint_angle_3d(frame[LEFT_SHOULDER], frame[LEFT_ELBOW], frame[LEFT_WRIST])
        )
        angle_series["r_elbow"].append(
            calculate_joint_angle_3d(frame[RIGHT_SHOULDER], frame[RIGHT_ELBOW], frame[RIGHT_WRIST])
        )
        # Knees
        angle_series["l_knee"].append(
            calculate_joint_angle_3d(frame[LEFT_HIP], frame[LEFT_KNEE], frame[LEFT_ANKLE])
        )
        angle_series["r_knee"].append(
            calculate_joint_angle_3d(frame[RIGHT_HIP], frame[RIGHT_KNEE], frame[RIGHT_ANKLE])
        )
        # Shoulders
        angle_series["l_shoulder"].append(
            calculate_joint_angle_3d(frame[LEFT_ELBOW], frame[LEFT_SHOULDER], frame[LEFT_HIP])
        )
        angle_series["r_shoulder"].append(
            calculate_joint_angle_3d(frame[RIGHT_ELBOW], frame[RIGHT_SHOULDER], frame[RIGHT_HIP])
        )
        # Hips
        angle_series["l_hip"].append(
            calculate_joint_angle_3d(frame[LEFT_SHOULDER], frame[LEFT_HIP], frame[LEFT_KNEE])
        )
        angle_series["r_hip"].append(
            calculate_joint_angle_3d(frame[RIGHT_SHOULDER], frame[RIGHT_HIP], frame[RIGHT_KNEE])
        )

    angle_features = []
    for k in sorted(angle_series.keys()):
        vals = np.array(angle_series[k])
        angle_features.extend([
            float(np.mean(vals)),
            float(np.std(vals)),
            float(np.min(vals)),
            float(np.max(vals))
        ])
    feature_list.append(np.array(angle_features, dtype=np.float32))

    # 4. Key Distance Features
    wrist_dist = np.linalg.norm(coords[:, LEFT_WRIST, :] - coords[:, RIGHT_WRIST, :], axis=-1)
    ankle_dist = np.linalg.norm(coords[:, LEFT_ANKLE, :] - coords[:, RIGHT_ANKLE, :], axis=-1)

    dist_features = [
        float(np.mean(wrist_dist)), float(np.std(wrist_dist)),
        float(np.min(wrist_dist)), float(np.max(wrist_dist)),
        float(np.mean(ankle_dist)), float(np.std(ankle_dist)),
        float(np.min(ankle_dist)), float(np.max(ankle_dist)),
    ]
    feature_list.append(np.array(dist_features, dtype=np.float32))

    return np.concatenate(feature_list, axis=0).astype(np.float32)


def extract_features_dataset(X: np.ndarray) -> np.ndarray:
    """
    Extracts 1D feature vectors for all sequences in a dataset.
    Input shape: (N, T, 33, 4)
    Output shape: (N, D) where D ~ 508
    """
    N = X.shape[0]
    feats = [extract_features_from_sequence(X[i]) for i in range(N)]
    return np.stack(feats, axis=0)
