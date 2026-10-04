"""
AURA Landmark Extractor Utility
Handles video decoding, frame-by-frame pose landmark extraction,
and metadata recording for videos.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import cv2
import numpy as np

from vision.pose.pose_detector import PoseDetector, NUM_LANDMARKS, LANDMARK_DIM


def extract_landmarks_from_video(
    video_path: Path,
    detector: Optional[PoseDetector] = None,
    max_frames: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Extracts 33 pose landmarks for every frame of a video file.

    Args:
        video_path: Path to the input video (.avi/.mp4).
        detector: Optional pre-initialized PoseDetector instance (reuses model).
        max_frames: Optional cap on frames to process (useful for quick debugging).

    Returns:
        landmarks: np.ndarray of shape (T, 33, 4) with [x, y, z, visibility]
        detected_mask: np.ndarray of shape (T,) boolean, indicating if pose was found
        stats: dictionary with video and extraction statistics
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    close_detector_after = False
    if detector is None:
        detector = PoseDetector(running_mode="video")
        close_detector_after = True
    else:
        # Reset detector internal time sequence for new video
        detector.reset()

    landmarks_list = []
    detected_mask_list = []

    frame_idx = 0
    try:
        while cap.isOpened():
            if max_frames is not None and frame_idx >= max_frames:
                break

            ret, frame = cap.read()
            if not ret:
                break

            timestamp_ms = int(frame_idx * (1000.0 / fps))
            pose_data = detector.process_frame(frame, timestamp_ms=timestamp_ms)

            landmarks_list.append(pose_data.landmarks)
            detected_mask_list.append(pose_data.detected)

            frame_idx += 1
    finally:
        cap.release()
        if close_detector_after:
            detector.close()

    if len(landmarks_list) == 0:
        landmarks = np.zeros((0, NUM_LANDMARKS, LANDMARK_DIM), dtype=np.float32)
        detected_mask = np.zeros((0,), dtype=bool)
    else:
        landmarks = np.stack(landmarks_list, axis=0)  # (T, 33, 4)
        detected_mask = np.array(detected_mask_list, dtype=bool)

    num_frames = len(landmarks_list)
    num_detected = int(np.sum(detected_mask))
    num_missing = num_frames - num_detected
    detection_rate = (num_detected / num_frames) if num_frames > 0 else 0.0

    stats = {
        "video_name": video_path.name,
        "video_path": str(video_path),
        "total_video_frames": total_video_frames,
        "processed_frames": num_frames,
        "detected_frames": num_detected,
        "missing_pose_frames": num_missing,
        "detection_rate": round(detection_rate, 4),
        "fps": fps,
        "resolution": (width, height),
        "feature_shape": list(landmarks.shape),
    }

    return landmarks, detected_mask, stats
