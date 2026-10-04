"""
AURA MediaPipe Pose Detector Module
Extracts 33 body landmarks per frame (x, y, z, visibility -> 132 features).
Supports both single image and sequential video processing modes.
"""

import os
import logging
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger("AURA.Vision.Pose")

# Official MediaPipe Pose Landmark Names (33 landmarks)
POSE_LANDMARK_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index"
]

NUM_LANDMARKS = 33
LANDMARK_DIM = 4  # x, y, z, visibility
TOTAL_FEATURE_DIM = NUM_LANDMARKS * LANDMARK_DIM  # 132


@dataclass
class PoseLandmarkData:
    """Container for 33 pose landmarks extracted from a single frame."""
    detected: bool
    landmarks: np.ndarray  # Shape: (33, 4) - [x, y, z, visibility]
    world_landmarks: Optional[np.ndarray] = None  # Shape: (33, 4) if available

    @property
    def flat(self) -> np.ndarray:
        """Returns 1D feature array of length 132."""
        return self.landmarks.flatten()


class PoseDetector:
    """
    Production-ready MediaPipe Pose Landmarker wrapper for AURA.
    Extracts 33 body landmarks (x, y, z, visibility) per frame.
    """

    MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
    DEFAULT_MODEL_REL_PATH = Path("models") / "pose_landmarker_full.task"

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        running_mode: str = "video",  # 'video' or 'image'
        min_detection_confidence: float = 0.5,
        min_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        if model_path is None:
            # Resolve relative to project root
            project_root = Path(__file__).resolve().parents[2]
            self.model_path = project_root / self.DEFAULT_MODEL_REL_PATH
        else:
            self.model_path = Path(model_path)

        self.running_mode_str = running_mode.lower()
        self.min_detection_confidence = min_detection_confidence
        self.min_presence_confidence = min_presence_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self._landmarker = None
        self._initialized = False
        self._init_detector()

    def _ensure_model(self) -> bool:
        """Ensures the .task model file is downloaded and valid."""
        if self.model_path.exists() and self.model_path.stat().st_size > 5_000_000:
            return True

        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            logger.info("Downloading MediaPipe Pose Landmarker model to %s...", self.model_path)
            urllib.request.urlretrieve(self.MODEL_URL, str(self.model_path))
            logger.info("Pose Landmarker downloaded successfully (%d bytes).", self.model_path.stat().st_size)
            return True
        except Exception as e:
            logger.error("Failed to download MediaPipe Pose Landmarker model: %s", e)
            return False

    def _init_detector(self) -> None:
        """Initializes the MediaPipe PoseLandmarker task."""
        if not self._ensure_model():
            logger.warning("MediaPipe pose model could not be initialized.")
            return

        try:
            import mediapipe as mp
            from mediapipe.tasks.python import BaseOptions
            from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode

            mode = RunningMode.VIDEO if self.running_mode_str == "video" else RunningMode.IMAGE

            options = PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(self.model_path)),
                running_mode=mode,
                num_poses=1,
                min_pose_detection_confidence=self.min_detection_confidence,
                min_pose_presence_confidence=self.min_presence_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
                output_segmentation_masks=False,
            )
            self._landmarker = PoseLandmarker.create_from_options(options)
            self._initialized = True
            logger.debug("MediaPipe Pose Landmarker initialized in %s mode.", self.running_mode_str.upper())
        except Exception as e:
            logger.error("MediaPipe PoseLandmarker initialization failed: %s", e)
            self._initialized = False

    def process_frame(
        self,
        frame: np.ndarray,
        timestamp_ms: Optional[int] = None
    ) -> PoseLandmarkData:
        """
        Processes a single BGR video frame and extracts 33 landmarks.
        Returns:
            PoseLandmarkData with detected flag and (33, 4) array.
        """
        if not self._initialized or self._landmarker is None:
            return PoseLandmarkData(
                detected=False,
                landmarks=np.zeros((NUM_LANDMARKS, LANDMARK_DIM), dtype=np.float32),
            )

        import mediapipe as mp

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        if self.running_mode_str == "video":
            if timestamp_ms is None:
                timestamp_ms = 0
            result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
        else:
            result = self._landmarker.detect(mp_image)

        if result.pose_landmarks and len(result.pose_landmarks) > 0:
            raw_landmarks = result.pose_landmarks[0]
            arr = np.zeros((NUM_LANDMARKS, LANDMARK_DIM), dtype=np.float32)
            for i, lm in enumerate(raw_landmarks):
                if i < NUM_LANDMARKS:
                    arr[i, 0] = lm.x
                    arr[i, 1] = lm.y
                    arr[i, 2] = lm.z
                    arr[i, 3] = getattr(lm, "visibility", 1.0) or 0.0

            world_arr = None
            if result.pose_world_landmarks and len(result.pose_world_landmarks) > 0:
                world_raw = result.pose_world_landmarks[0]
                world_arr = np.zeros((NUM_LANDMARKS, LANDMARK_DIM), dtype=np.float32)
                for i, lm in enumerate(world_raw):
                    if i < NUM_LANDMARKS:
                        world_arr[i, 0] = lm.x
                        world_arr[i, 1] = lm.y
                        world_arr[i, 2] = lm.z
                        world_arr[i, 3] = getattr(lm, "visibility", 1.0) or 0.0

            return PoseLandmarkData(
                detected=True,
                landmarks=arr,
                world_landmarks=world_arr,
            )

        # No pose detected: return zeros
        return PoseLandmarkData(
            detected=False,
            landmarks=np.zeros((NUM_LANDMARKS, LANDMARK_DIM), dtype=np.float32),
        )

    def close(self) -> None:
        """Closes the underlying landmarker instance."""
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None
            self._initialized = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
