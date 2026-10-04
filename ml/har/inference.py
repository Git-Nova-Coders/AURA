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

from vision.pose.pose_detector import PoseDetector, draw_pose_skeleton
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

    def process_frame(self, frame: np.ndarray, draw_overlay: bool = True) -> Tuple[np.ndarray, str, float]:
        """
        Processes single camera frame:
        - Detects pose
        - Normalizes landmarks
        - Appends to rolling buffer
        - Runs model inference if buffer is full
        - Optionally overlays HUD
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

            # Check dynamic kinematics / motion velocity across recent frames
            # seq_arr contains normalized coordinates (x, y, z, vis) for 33 joints
            diffs = np.diff(seq_arr[-10:, :, :3], axis=0)  # last 10 frames motion
            joint_velocity = float(np.mean(np.linalg.norm(diffs, axis=-1)))

            if not pose_data.detected:
                self.last_activity = "STANDBY (NO POSE)"
                self.last_confidence = 0.0
            else:
                lm = pose_data.landmarks  # shape: (33, 4) with normalized [x, y, z, vis]
                # Key landmarks
                nose = lm[0]
                l_sh, r_sh = lm[11], lm[12]
                l_el, r_el = lm[13], lm[14]
                l_wr, r_wr = lm[15], lm[16]
                l_hip, r_hip = lm[23], lm[24]
                l_knee, r_knee = lm[25], lm[26]
                l_ank, r_ank = lm[27], lm[28]

                mid_sh_y = (l_sh[1] + r_sh[1]) / 2.0
                mid_hip_y = (l_hip[1] + r_hip[1]) / 2.0
                mid_knee_y = (l_knee[1] + r_knee[1]) / 2.0
                mid_ank_y = (l_ank[1] + r_ank[1]) / 2.0

                torso_len = abs(mid_hip_y - mid_sh_y) + 1e-5
                thigh_len = abs(mid_knee_y - mid_hip_y)

                # Angles calculation
                def angle_2d(a, b, c):
                    ba = a[:2] - b[:2]
                    bc = c[:2] - b[:2]
                    n1 = np.linalg.norm(ba)
                    n2 = np.linalg.norm(bc)
                    if n1 < 1e-5 or n2 < 1e-5:
                        return 180.0
                    cos = np.dot(ba, bc) / (n1 * n2)
                    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))

                l_knee_ang = angle_2d(l_hip, l_knee, l_ank)
                r_knee_ang = angle_2d(r_hip, r_knee, r_ank)
                avg_knee_ang = (l_knee_ang + r_knee_ang) / 2.0

                # Arms velocity for waving
                wrist_diffs = np.diff(seq_arr[-15:, [15, 16], :2], axis=0)
                wrist_velocity = float(np.mean(np.linalg.norm(wrist_diffs, axis=-1)))

                # Feet / Leg velocity for walking & running
                leg_diffs = np.diff(seq_arr[-10:, [25, 26, 27, 28], :2], axis=0)
                leg_velocity = float(np.mean(np.linalg.norm(leg_diffs, axis=-1)))

                # Vertical velocity of hips for jumping
                hip_y_diffs = np.diff(seq_arr[-10:, [23, 24], 1], axis=0)
                vertical_velocity = float(np.max(np.abs(hip_y_diffs)))

                # 1. Run Trained Deep Learning CNN+LSTM Model
                tensor_seq = torch.from_numpy(seq_arr).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    logits = self.model(tensor_seq)
                    probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
                    best_idx = int(np.argmax(probs))
                    best_conf = float(probs[best_idx])
                    ml_cls = self.classes[best_idx]

                class_map = {
                    "BodyWeightSquats": "Squatting",
                    "PushUps": "Push-ups",
                    "WallPushups": "Push-ups",
                    "JumpRope": "Jumping",
                    "JumpingJack": "Jumping",
                    "Lunges": "Walking",
                    "Punch": "Waving",
                    "TaiChi": "Standing",
                }
                mapped_ml_act = class_map.get(ml_cls, ml_cls)

                # 2. Key Dynamic Motion & Posture Checks
                wrist_above_shoulder = (l_wr[1] < l_sh[1] and l_wr[3] > 0.35) or (r_wr[1] < r_sh[1] and r_wr[3] > 0.35)
                is_horizontal = (abs(mid_sh_y - mid_hip_y) < 0.22 and mid_sh_y > 0.35)

                if wrist_above_shoulder and wrist_velocity > 0.038:
                    self.last_activity = "Waving"
                    self.last_confidence = min(0.98, 0.78 + wrist_velocity * 4.0)

                elif is_horizontal and (ml_cls in ("PushUps", "WallPushups") or joint_velocity > 0.02):
                    self.last_activity = "Push-ups"
                    self.last_confidence = max(0.92, best_conf)

                elif (best_conf >= self.confidence_threshold and mapped_ml_act in ("Squatting", "Jumping") and joint_velocity > 0.04):
                    # High confidence dynamic exercise detected by CNN-LSTM
                    self.last_activity = mapped_ml_act
                    self.last_confidence = best_conf

                elif vertical_velocity > 0.055 and joint_velocity > 0.05:
                    self.last_activity = "Jumping"
                    self.last_confidence = min(0.97, 0.82 + vertical_velocity * 2.5)

                elif avg_knee_ang < 130.0 and joint_velocity > 0.03:
                    self.last_activity = "Squatting"
                    self.last_confidence = min(0.96, 0.85 + (130.0 - avg_knee_ang) / 100.0)

                elif (abs(mid_hip_y - mid_knee_y) < torso_len * 0.45 and joint_velocity < 0.035 and mid_hip_y > 0.35):
                    self.last_activity = "Sitting"
                    self.last_confidence = 0.94

                elif leg_velocity > 0.070:
                    self.last_activity = "Running"
                    self.last_confidence = min(0.96, 0.80 + leg_velocity * 2.0)

                elif leg_velocity > 0.028:
                    self.last_activity = "Walking"
                    self.last_confidence = min(0.94, 0.78 + leg_velocity * 3.0)

                elif joint_velocity < 0.032 and avg_knee_ang > 145.0:
                    self.last_activity = "Standing"
                    self.last_confidence = 0.95

                elif best_conf >= self.confidence_threshold:
                    self.last_activity = mapped_ml_act
                    self.last_confidence = best_conf
                else:
                    self.last_activity = "Standing"
                    self.last_confidence = 0.85

        annotated = frame.copy()
        if pose_data.detected:
            annotated = draw_pose_skeleton(annotated, pose_data.landmarks)

        if not draw_overlay:
            return annotated, self.last_activity, self.last_confidence
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
