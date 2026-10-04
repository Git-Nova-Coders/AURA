"""
AURA HAR Landmark Preprocessing and Sequence Generation
Normalizes pose landmarks to a body-relative coordinate frame and
generates fixed-length temporal sequences (e.g. 30 frames) for downstream ML/DL models.

Normalization Strategy:
1. Mid-hip translation: Translation invariance by centering the origin at the midpoint
   of the left and right hip joints (landmarks 23 and 24).
2. Torso-scale normalization: Scale invariance by dividing coordinates by the distance
   between the mid-hip and mid-shoulder (landmarks 11 and 12). If torso distance is
   near zero, fallback to bounding box or unit scaling.
3. Temporal Resampling / Padding:
   - For a target sequence length of L frames:
     * If video length T == L: exact match.
     * If video length T > L: uniform temporal subsampling or multi-window extraction.
     * If video length T < L: zero-padding or edge-reflection padding with valid mask.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("AURA.HAR.Preprocess")

# Key Landmark Indices in MediaPipe Pose
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

NUM_LANDMARKS = 33
LANDMARK_DIM = 4  # x, y, z, visibility
SEQUENCE_LENGTH = 30  # Standard fixed temporal sequence length


def normalize_pose_frame(frame_landmarks: np.ndarray) -> np.ndarray:
    """
    Normalizes a single frame of 33 pose landmarks (shape: (33, 4)).

    Steps:
    1. Compute mid-hip position: center = (left_hip + right_hip) / 2
    2. Subtract center from all landmark (x, y, z) coordinates (Translation Invariance)
    3. Compute mid-shoulder position: (left_shoulder + right_shoulder) / 2
    4. Compute torso length: Euclidean distance between mid-hip and mid-shoulder
    5. Scale (x, y, z) by torso length (Scale / Distance Invariance)
    6. Preserve visibility confidence as feature channel 4.
    """
    normed = frame_landmarks.copy().astype(np.float32)

    # Check if hips are valid/detected
    left_hip = frame_landmarks[LEFT_HIP, :3]
    right_hip = frame_landmarks[RIGHT_HIP, :3]
    mid_hip = (left_hip + right_hip) / 2.0

    # Subtract origin (mid-hip)
    normed[:, :3] -= mid_hip

    # Torso scale
    left_shoulder = frame_landmarks[LEFT_SHOULDER, :3]
    right_shoulder = frame_landmarks[RIGHT_SHOULDER, :3]
    mid_shoulder = (left_shoulder + right_shoulder) / 2.0
    torso_size = np.linalg.norm(mid_shoulder - mid_hip)

    if torso_size > 1e-4:
        normed[:, :3] /= torso_size
    else:
        # Fallback to standard deviation of non-zero points or 1.0
        std = np.std(normed[:, :3])
        if std > 1e-4:
            normed[:, :3] /= std

    return normed


def normalize_sequence(landmarks_seq: np.ndarray, detected_mask: np.ndarray) -> np.ndarray:
    """
    Normalizes a sequence of pose landmarks of shape (T, 33, 4).
    Frames with detected_mask == False remain zeroed out.
    """
    T = landmarks_seq.shape[0]
    out = np.zeros_like(landmarks_seq, dtype=np.float32)

    for t in range(T):
        if detected_mask[t]:
            out[t] = normalize_pose_frame(landmarks_seq[t])
        else:
            # Undetected frame remains zeroed
            out[t] = 0.0

    return out


def resample_or_pad_sequence(
    sequence: np.ndarray,
    target_length: int = SEQUENCE_LENGTH
) -> np.ndarray:
    """
    Converts a sequence of shape (T, 33, 4) into a fixed-length sequence of shape (target_length, 33, 4).

    Strategies:
    - If T == target_length: return exact sequence.
    - If T > target_length: uniformly sample target_length indices.
    - If T < target_length: pad trailing frames with zeros (or last valid frame).
    """
    T = sequence.shape[0]
    dim_shape = sequence.shape[1:]  # (33, 4)

    if T == 0:
        return np.zeros((target_length, *dim_shape), dtype=np.float32)

    if T == target_length:
        return sequence.astype(np.float32)

    if T > target_length:
        # Uniform temporal sampling (preserves motion dynamics over the entire clip)
        indices = np.linspace(0, T - 1, target_length, dtype=int)
        return sequence[indices].astype(np.float32)

    # T < target_length: Repeat/pad frames
    pad_needed = target_length - T
    padding = np.zeros((pad_needed, *dim_shape), dtype=np.float32)
    # Option: repeat last frame or zero pad. Zero-padding with valid mask is standard.
    padded = np.concatenate([sequence, padding], axis=0)
    return padded.astype(np.float32)


def process_dataset_split(
    raw_landmarks_dir: Path,
    split_name: str,
    target_length: int = SEQUENCE_LENGTH,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """
    Loads all .npz files for a given split ('train' or 'test'),
    normalizes landmarks, and generates fixed-length sequences.

    Returns:
        X: np.ndarray of shape (N, target_length, 33, 4)
        y: np.ndarray of shape (N,) integer class labels (0 to 7)
        metadata: List of metadata dicts for each sample
    """
    split_dir = raw_landmarks_dir / split_name
    npz_files = sorted(list(split_dir.glob("*.npz")))

    if not npz_files:
        raise FileNotFoundError(f"No .npz landmark files found in {split_dir}")

    # Build consistent label mapping from sorted class names
    classes = [
        "BodyWeightSquats", "JumpRope", "JumpingJack", "Lunges",
        "Punch", "PushUps", "TaiChi", "WallPushups"
    ]
    class_to_label = {cls_name: idx for idx, cls_name in enumerate(sorted(classes))}

    X_list = []
    y_list = []
    meta_list = []

    for fpath in npz_files:
        data = np.load(fpath, allow_pickle=True)
        raw_lm = data["landmarks"]  # (T, 33, 4)
        mask = data["detected_mask"]  # (T,)
        activity = str(data["activity"])
        video_name = str(data["video_name"])

        if activity not in class_to_label:
            continue

        label = class_to_label[activity]

        # 1. Normalize sequence
        norm_seq = normalize_sequence(raw_lm, mask)

        # 2. Resample / pad to target length
        fixed_seq = resample_or_pad_sequence(norm_seq, target_length=target_length)

        X_list.append(fixed_seq)
        y_list.append(label)
        meta_list.append({
            "video_name": video_name,
            "activity": activity,
            "label": label,
            "original_frames": len(raw_lm),
            "detected_frames": int(np.sum(mask)),
            "sequence_length": target_length,
            "split": split_name,
        })

    X = np.stack(X_list, axis=0)  # (N, target_length, 33, 4)
    y = np.array(y_list, dtype=np.int64)

    return X, y, meta_list


def create_preprocessed_dataset(
    processed_dir: Optional[Path] = None,
    target_length: int = SEQUENCE_LENGTH
):
    """
    Creates and saves preprocessed dataset arrays:
    - X_train.npy, y_train.npy
    - X_test.npy, y_test.npy
    - dataset_meta.json
    """
    if processed_dir is None:
        processed_dir = Path(__file__).resolve().parents[2] / "data" / "har" / "processed"

    raw_dir = processed_dir / "raw_landmarks"

    logger.info("Processing training set...")
    X_train, y_train, meta_train = process_dataset_split(raw_dir, "train", target_length)

    logger.info("Processing test set...")
    X_test, y_test, meta_test = process_dataset_split(raw_dir, "test", target_length)

    logger.info("=" * 60)
    logger.info(f"PREPROCESSED DATASET GENERATED:")
    logger.info(f"  X_train: {X_train.shape} | y_train: {y_train.shape}")
    logger.info(f"  X_test:  {X_test.shape}  | y_test:  {y_test.shape}")
    logger.info("=" * 60)

    # Save to disk
    np.save(processed_dir / "X_train.npy", X_train)
    np.save(processed_dir / "y_train.npy", y_train)
    np.save(processed_dir / "X_test.npy", X_test)
    np.save(processed_dir / "y_test.npy", y_test)

    meta_payload = {
        "sequence_length": target_length,
        "features_per_frame": NUM_LANDMARKS * LANDMARK_DIM,
        "classes": sorted([
            "BodyWeightSquats", "JumpRope", "JumpingJack", "Lunges",
            "Punch", "PushUps", "TaiChi", "WallPushups"
        ]),
        "num_train": len(X_train),
        "num_test": len(X_test),
        "normalization": "mid-hip origin centering + torso Euclidean distance scaling",
        "train_samples": meta_train,
        "test_samples": meta_test,
    }

    with open(processed_dir / "dataset_meta.json", "w") as f:
        json.dump(meta_payload, f, indent=2)

    logger.info("Saved preprocessed dataset to %s", processed_dir)
