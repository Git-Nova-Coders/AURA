"""
AURA HAR Phase 8: Real-Time Webcam Activity Inference
Integrates:
- Live OpenCV camera stream
- MediaPipe Pose landmark extraction (33 joints)
- Body-relative pose normalization
- Rolling sequence buffer (30 frames)
- Trained Hybrid CNN + LSTM PyTorch model
- Real-time Cybernetic HUD with activity label and confidence display
"""

import collections
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vision.pose.pose_detector import PoseDetector
from ml.har.preprocess import normalize_pose_frame, SEQUENCE_LENGTH
from ml.har.models import HybridPoseCNNLSTM

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("AURA.HAR.Inference")

MODEL_PATH = PROJECT_ROOT / "models" / "har" / "cnn_lstm_har_model.pt"
META_PATH = PROJECT_ROOT / "data" / "har" / "processed" / "dataset_meta.json"


class RealTimeHARInference:
    """
    Real-Time HAR engine with rolling sequence buffer.
    """

    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        meta_path: Path = META_PATH,
        sequence_length: int = SEQUENCE_LENGTH,
        confidence_threshold: float = 0.55
    ):
        self.sequence_length = sequence_length
        self.confidence_threshold = confidence_threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load classes
        if meta_path.exists():
            with open(meta_path, "r") as f:
                meta = json.load(f)
            self.classes = meta["classes"]
        else:
            self.classes = [
                "BodyWeightSquats", "JumpRope", "JumpingJack", "Lunges",
                "Punch", "PushUps", "TaiChi", "WallPushups"
            ]

        # Load Model
        logger.info(f"Loading trained CNN+LSTM model from {model_path} onto {self.device}...")
        self.model = HybridPoseCNNLSTM(
            num_landmarks=33,
            dim_per_lm=4,
            seq_len=sequence_length,
            cnn_out_dim=128,
            lstm_hidden=128,
            num_classes=len(self.classes),
            dropout=0.0  # Eval mode
        )
        if model_path.exists():
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        else:
            logger.warning(f"Model checkpoint not found at {model_path}. Running with uninitialized weights for testing.")

        self.model.to(self.device)
        self.model.eval()

        # Rolling Buffer: keeps last 30 normalized frames (each is 33, 4)
        self.buffer = collections.deque(maxlen=sequence_length)
        self.detector = PoseDetector(running_mode="image")

        self.last_activity = "Initializing..."
        self.last_confidence = 0.0

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, str, float]:
        """
        Processes single camera frame:
        - Detects pose
        - Normalizes landmarks
        - Appends to rolling buffer
        - Runs model inference if buffer is full
        - Overlays HUD
        """
        h, w = frame.shape[:2]
        pose_data = self.detector.process_frame(frame)

        if pose_data.detected:
            normed = normalize_pose_frame(pose_data.landmarks)
            self.buffer.append(normed)
        else:
            # Undetected: maintain zeros or last state
            self.buffer.append(np.zeros((33, 4), dtype=np.float32))

        # Predict when buffer has full sequence
        if len(self.buffer) == self.sequence_length:
            seq_arr = np.array(self.buffer, dtype=np.float32)  # (30, 33, 4)
            tensor_seq = torch.from_numpy(seq_arr).unsqueeze(0).to(self.device)  # (1, 30, 33, 4)

            with torch.no_grad():
                logits = self.model(tensor_seq)
                probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
                best_idx = int(np.argmax(probs))
                best_conf = float(probs[best_idx])

                if best_conf >= self.confidence_threshold:
                    self.last_activity = self.classes[best_idx]
                    self.last_confidence = best_conf
                else:
                    self.last_activity = "Uncertain"
                    self.last_confidence = best_conf

        # Render HUD Overlay
        annotated = frame.copy()
        # Header banner
        cv2.rectangle(annotated, (20, 20), (450, 95), (15, 15, 20), -1)
        cv2.rectangle(annotated, (20, 20), (450, 95), (0, 230, 255), 2)

        # Activity text
        text_act = f"ACTIVITY: {self.last_activity}"
        text_conf = f"CONFIDENCE: {self.last_confidence * 100:.1f}%"
        cv2.putText(annotated, text_act, (35, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        cv2.putText(annotated, text_conf, (35, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 230, 255), 2)

        # Buffer status meter
        meter_w = int((len(self.buffer) / self.sequence_length) * 410)
        cv2.rectangle(annotated, (20, 97), (20 + meter_w, 102), (0, 255, 128), -1)

        return annotated, self.last_activity, self.last_confidence

    def run_webcam(self, camera_id: int = 0):
        """Runs live webcam loop until 'q' is pressed."""
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            logger.error("Could not open camera device %d", camera_id)
            return

        logger.info("Starting AURA-HAR Real-Time Inference. Press 'q' to exit.")
        prev_time = time.time()

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                curr_time = time.time()
                fps = 1.0 / max(curr_time - prev_time, 1e-4)
                prev_time = curr_time

                annotated, act, conf = self.process_frame(frame)
                cv2.putText(annotated, f"FPS: {fps:.1f}", (annotated.shape[1] - 120, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                cv2.imshow("AURA Human Activity Recognition", annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.detector.close()


if __name__ == "__main__":
    engine = RealTimeHARInference()
    engine.run_webcam()
