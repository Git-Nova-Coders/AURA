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

        # Temporal Smoothing: 7-frame sliding window majority vote & EMA confidence
        self.vote_history = collections.deque(maxlen=7)
        self.last_activity = "Initializing..."
        self.last_confidence = 0.0

    def process_frame(self, frame: np.ndarray, draw_overlay: bool = True) -> Tuple[np.ndarray, str, float]:
        """
        Processes single camera frame:
        - Detects pose
        - Normalizes landmarks
        - Appends to rolling buffer
        - Runs model inference + kinematics + desk-aware heuristics
        - Applies 7-frame majority voting temporal filter for rock-solid stability
        - Optionally overlays HUD
        """
        h, w = frame.shape[:2]
        pose_data = self.detector.process_frame(frame)

        if pose_data.detected:
            normed = normalize_pose_frame(pose_data.landmarks)
            self.buffer.append(normed)
        else:
            self.buffer.append(np.zeros((33, 4), dtype=np.float32))

        # Instant frame classification if buffer is still warming up or full
        if not pose_data.detected:
            self.vote_history.append(("STANDBY (NO POSE)", 0.0))
        else:
            lm = pose_data.landmarks  # shape: (33, 4) with normalized [x, y, z, vis]
            nose = lm[0]
            l_sh, r_sh = lm[11], lm[12]
            l_el, r_el = lm[13], lm[14]
            l_wr, r_wr = lm[15], lm[16]
            l_hip, r_hip = lm[23], lm[24]
            l_knee, r_knee = lm[25], lm[26]
            l_ank, r_ank = lm[27], lm[28]

            mid_sh_y = float((l_sh[1] + r_sh[1]) / 2.0)
            mid_hip_y = float((l_hip[1] + r_hip[1]) / 2.0)
            mid_knee_y = float((l_knee[1] + r_knee[1]) / 2.0)

            # Visibilities
            sh_vis = float((l_sh[3] + r_sh[3]) / 2.0)
            hip_vis = float((l_hip[3] + r_hip[3]) / 2.0)
            knee_vis = float((l_knee[3] + r_knee[3]) / 2.0)
            wrist_vis = float((l_wr[3] + r_wr[3]) / 2.0)

            # Kinematic velocities from buffer if enough frames exist
            buf_len = len(self.buffer)
            if buf_len >= 5:
                buf_arr = np.array(self.buffer, dtype=np.float32)
                sub_len = min(buf_len, 10)
                diffs = np.diff(buf_arr[-sub_len:, :, :3], axis=0)
                joint_velocity = float(np.mean(np.linalg.norm(diffs, axis=-1)))

                w_sub = min(buf_len, 12)
                wrist_diffs = np.diff(buf_arr[-w_sub:, [15, 16], :2], axis=0)
                wrist_velocity = float(np.mean(np.linalg.norm(wrist_diffs, axis=-1)))

                l_sub = min(buf_len, 10)
                leg_diffs = np.diff(buf_arr[-l_sub:, [25, 26, 27, 28], :2], axis=0)
                leg_velocity = float(np.mean(np.linalg.norm(leg_diffs, axis=-1)))

                hip_y_diffs = np.diff(buf_arr[-l_sub:, [23, 24], 1], axis=0)
                vertical_velocity = float(np.max(np.abs(hip_y_diffs)))
            else:
                joint_velocity = 0.01
                wrist_velocity = 0.01
                leg_velocity = 0.01
                vertical_velocity = 0.01

            # Knee angle calculation
            def angle_2d(a, b, c):
                ba = a[:2] - b[:2]
                bc = c[:2] - b[:2]
                n1 = np.linalg.norm(ba)
                n2 = np.linalg.norm(bc)
                if n1 < 1e-5 or n2 < 1e-5:
                    return 180.0
                cos = np.dot(ba, bc) / (n1 * n2)
                return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))

            l_knee_ang = angle_2d(l_hip, l_knee, l_ank) if knee_vis > 0.3 else 180.0
            r_knee_ang = angle_2d(r_hip, r_knee, r_ank) if knee_vis > 0.3 else 180.0
            avg_knee_ang = (l_knee_ang + r_knee_ang) / 2.0

            # 1. Run Trained Deep Learning CNN+LSTM Model (when buffer is full)
            ml_cls = None
            best_conf = 0.0
            mapped_ml_act = None
            if buf_len == self.sequence_length:
                seq_arr = np.array(self.buffer, dtype=np.float32)
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
                    "Punch": "Punching",
                    "TaiChi": "Standing",
                }
                mapped_ml_act = class_map.get(ml_cls, ml_cls)

            # 2. Key Dynamic Motion & Posture Checks
            wrist_above_shoulder = (
                (l_wr[1] < l_sh[1] and l_wr[3] > 0.35) or
                (r_wr[1] < r_sh[1] and r_wr[3] > 0.35)
            )
            is_horizontal = (abs(mid_sh_y - mid_hip_y) < 0.22 and mid_sh_y > 0.35)

            current_act = "Sitting"
            current_conf = 0.90

            # ── A. FULL-BODY SCENARIO (Knees and hips visible in frame) ────────
            if knee_vis >= 0.35 and hip_vis >= 0.35:
                # High confidence DL exercise
                if (
                    best_conf >= 0.70 and mapped_ml_act in ("Squatting", "Jumping", "Push-ups", "Punching")
                    and joint_velocity > 0.03
                ):
                    current_act = mapped_ml_act
                    current_conf = best_conf

                elif wrist_above_shoulder and wrist_velocity > 0.038:
                    current_act = "Waving"
                    current_conf = min(0.98, 0.80 + wrist_velocity * 3.5)

                elif is_horizontal and (ml_cls in ("PushUps", "WallPushups") or joint_velocity > 0.02):
                    current_act = "Push-ups"
                    current_conf = max(0.92, best_conf)

                elif vertical_velocity > 0.050 and joint_velocity > 0.045:
                    current_act = "Jumping"
                    current_conf = min(0.97, 0.82 + vertical_velocity * 2.5)

                elif avg_knee_ang < 135.0 and joint_velocity > 0.025:
                    current_act = "Squatting"
                    current_conf = min(0.96, 0.85 + (135.0 - avg_knee_ang) / 100.0)

                elif leg_velocity > 0.070:
                    current_act = "Running"
                    current_conf = min(0.96, 0.82 + leg_velocity * 2.0)

                elif leg_velocity > 0.028:
                    current_act = "Walking"
                    current_conf = min(0.94, 0.80 + leg_velocity * 2.5)

                elif avg_knee_ang > 145.0 and joint_velocity < 0.035:
                    current_act = "Standing"
                    current_conf = 0.95

                elif abs(mid_hip_y - mid_knee_y) < 0.20 and joint_velocity < 0.035:
                    current_act = "Sitting"
                    current_conf = 0.94

                elif best_conf >= self.confidence_threshold and mapped_ml_act:
                    current_act = mapped_ml_act
                    current_conf = best_conf
                else:
                    current_act = "Standing"
                    current_conf = 0.85

            # ── B. UPPER-BODY / DESK SCENARIO (Knees occluded / not in frame) ──
            else:
                # User is sitting at a laptop or desk in front of camera
                if wrist_above_shoulder and wrist_velocity > 0.035:
                    current_act = "Waving"
                    current_conf = min(0.98, 0.82 + wrist_velocity * 3.5)

                elif wrist_velocity > 0.065 and joint_velocity > 0.045:
                    current_act = "Punching"
                    current_conf = min(0.95, 0.80 + wrist_velocity * 2.0)

                elif joint_velocity < 0.040:
                    # Low motion upper body at desk
                    if mid_sh_y > 0.28:
                        current_act = "Sitting"
                        current_conf = 0.95
                    else:
                        # Shoulders are high up near top of frame -> person is standing
                        current_act = "Standing"
                        current_conf = 0.90
                else:
                    # Moderate upper body motion (typing, gesturing, adjusting position)
                    if mid_sh_y > 0.28:
                        current_act = "Sitting"
                        current_conf = 0.88
                    else:
                        current_act = "Standing"
                        current_conf = 0.85

            self.vote_history.append((current_act, current_conf))

        # 3. Temporal Majority Voting Filter (Smooths out all frame jitter)
        if self.vote_history:
            votes = [act for act, _ in self.vote_history]
            confidences = [c for _, c in self.vote_history]

            # Find most common activity in sliding window
            counts = collections.Counter(votes)
            top_act, top_count = counts.most_common(1)[0]

            # Require >= 3 votes out of up to 7 frames to switch
            if top_count >= 3 or self.last_activity in ("Initializing...", "STANDBY (NO POSE)"):
                self.last_activity = top_act

            # Exponential Moving Average for smooth confidence display
            avg_conf = float(np.mean([c for a, c in self.vote_history if a == self.last_activity] or [0.85]))
            if self.last_confidence <= 0.01:
                self.last_confidence = avg_conf
            else:
                self.last_confidence = 0.40 * avg_conf + 0.60 * self.last_confidence

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
