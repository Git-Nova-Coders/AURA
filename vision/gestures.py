"""
AURA 21-Landmark Hand Tracking & 3D Kinematic Gesture Control Subsystem
Powered by Google MediaPipe Hand Landmarker, 3D Vector Kinematics, Raycasting & Cybernetic HUD.
"""

import os
import time
import math
import logging
import urllib.request
from enum import Enum
from typing import List, Tuple, Optional, Dict, Any, Union
from dataclasses import dataclass, field

import cv2
import numpy as np

from vision.detector import Detection

logger = logging.getLogger("AURA.Gestures")

# 21 Hand Landmark Indices
WRIST = 0
THUMB_CMC = 1
THUMB_MCP = 2
THUMB_IP = 3
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_PIP = 6
INDEX_DIP = 7
INDEX_TIP = 8
MIDDLE_MCP = 9
MIDDLE_PIP = 10
MIDDLE_DIP = 11
MIDDLE_TIP = 12
RING_MCP = 13
RING_PIP = 14
RING_DIP = 15
RING_TIP = 16
PINKY_MCP = 17
PINKY_PIP = 18
PINKY_DIP = 19
PINKY_TIP = 20

# 21 Landmark Connections (Skeleton Bones)
HAND_CONNECTIONS = [
    # Palm base
    (0, 1), (0, 5), (5, 9), (9, 13), (13, 17), (0, 17),
    # Thumb
    (1, 2), (2, 3), (3, 4),
    # Index finger
    (5, 6), (6, 7), (7, 8),
    # Middle finger
    (9, 10), (10, 11), (11, 12),
    # Ring finger
    (13, 14), (14, 15), (15, 16),
    # Pinky finger
    (17, 18), (18, 19), (19, 20),
]


class GestureType(str, Enum):
    """Supported 3D Hand Gestures."""
    NONE = "none"
    OPEN_PALM = "open_palm"          # 🖐️ All 5 fingers extended -> Hide all boxes (Clean View)
    POINTING = "pointing"            # 👉 Index extended -> Cast laser ray & lock target
    PINCH = "pinch"                  # 👌 Thumb + Index touching -> Select / Inspect target
    PEACE_SIGN = "peace_sign"        # ✌️ Victory sign (Index + Middle) -> Capture Screenshot / Snapshot
    THUMBS_UP = "thumbs_up"          # 👍 Thumb up -> Restore all boxes / Confirm view
    THUMBS_DOWN = "thumbs_down"      # 👎 Thumb down -> Deselect / Cancel target lock
    FIST = "fist"                    # ✊ All fingers curled -> Freeze overlay
    ROCK_ON = "rock_on"              # 🤘 Index + Pinky extended -> Toggle SAHI High-Res
    CALL_ME = "call_me"              # 🤙 Thumb + Pinky extended -> Trigger Voice Assistant


class GestureMode(str, Enum):
    """Operational visualization & interaction modes."""
    ALL_OBJECTS = "ALL_OBJECTS"      # Default: Display all bounding boxes
    HIDE_BOXES = "HIDE_BOXES"        # Clean view: Hide all overlay boxes
    FOCUS_OBJECT = "FOCUS_OBJECT"    # Target locked: Highlight only pointed item
    FROZEN = "FROZEN"                # Frame / overlay frozen in place
    INSPECT_OBJECT = "INSPECT"       # Multi-modal RAG inspection active


@dataclass
class HandLandmark3D:
    """Normalized 3D hand landmark."""
    x: float
    y: float
    z: float = 0.0


@dataclass
class GestureResult:
    """Full kinematic analysis result for a tracked hand."""
    gesture: GestureType = GestureType.NONE
    confidence: float = 0.0
    finger_count: int = 0
    fingers_extended: Dict[str, bool] = field(default_factory=lambda: {
        "thumb": False, "index": False, "middle": False, "ring": False, "pinky": False
    })
    pinch_distance: float = 1.0
    is_pinching: bool = False
    pointing_tip: Optional[Tuple[float, float]] = None      # (px_x, px_y) in image space
    pointing_vector: Optional[Tuple[float, float]] = None   # Unit 2D/3D ray direction
    landmarks: List[HandLandmark3D] = field(default_factory=list)
    hand_bbox: Optional[List[float]] = None                 # [x1, y1, x2, y2]
    handedness: str = "Right"                               # "Left" or "Right"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gesture": self.gesture.value,
            "confidence": round(self.confidence, 2),
            "finger_count": self.finger_count,
            "fingers_extended": self.fingers_extended,
            "is_pinching": self.is_pinching,
            "pinch_distance": round(self.pinch_distance, 3),
            "pointing_tip": [round(c, 1) for c in self.pointing_tip] if self.pointing_tip else None,
            "hand_bbox": [round(c, 1) for c in self.hand_bbox] if self.hand_bbox else None,
            "handedness": self.handedness,
        }


def compute_3d_angle(a: HandLandmark3D, b: HandLandmark3D, c: HandLandmark3D) -> float:
    """Computes the 3D angle (in degrees) at joint b formed by vectors ba and bc."""
    v1 = np.array([a.x - b.x, a.y - b.y, a.z - b.z])
    v2 = np.array([c.x - b.x, c.y - b.y, c.z - b.z])
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 < 1e-6 or norm2 < 1e-6:
        return 180.0
    cosine = np.dot(v1, v2) / (norm1 * norm2)
    cosine = np.clip(cosine, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def compute_euclidean_dist(p1: HandLandmark3D, p2: HandLandmark3D) -> float:
    """Calculates 3D Euclidean distance between two landmarks."""
    return float(math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2))


class MediaPipeHandTracker:
    """
    Production-grade 21-landmark 3D Hand Tracker using Google MediaPipe HandLandmarker.
    Automatically ensures model asset availability and performs sub-millimeter joint extraction.
    """

    MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
    DEFAULT_MODEL_PATH = "models/hand_landmarker.task"

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH):
        self.model_path = model_path
        self._landmarker = None
        self._initialized = False
        self._init_engine()

    def _ensure_model_file(self) -> bool:
        """Downloads hand_landmarker.task if missing."""
        if os.path.exists(self.model_path) and os.path.getsize(self.model_path) > 1000000:
            return True
        os.makedirs(os.path.dirname(os.path.abspath(self.model_path)), exist_ok=True)
        try:
            logger.info(f"Downloading MediaPipe Hand Landmarker model to '{self.model_path}'...")
            urllib.request.urlretrieve(self.MODEL_URL, self.model_path)
            logger.info(f"MediaPipe Hand Landmarker downloaded successfully ({os.path.getsize(self.model_path)} bytes).")
            return True
        except Exception as e:
            logger.error(f"Failed to download MediaPipe Hand Landmarker model: {e}")
            return False

    def _init_engine(self) -> None:
        """Initializes the MediaPipe Tasks HandLandmarker."""
        if not self._ensure_model_file():
            logger.warning("MediaPipe model file not available. Hand tracking fallback will be active.")
            return

        try:
            import mediapipe as mp
            from mediapipe.tasks.python import BaseOptions
            from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode

            options = HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=self.model_path),
                running_mode=RunningMode.IMAGE,
                num_hands=2,
                min_hand_detection_confidence=0.45,
                min_hand_presence_confidence=0.45,
                min_tracking_confidence=0.45,
            )
            self._landmarker = HandLandmarker.create_from_options(options)
            self._initialized = True
            logger.info("MediaPipe 21-Landmark Hand Tracker initialized successfully.")
        except Exception as e:
            logger.error(f"MediaPipe HandLandmarker initialization failed: {e}")
            self._initialized = False

    @property
    def is_available(self) -> bool:
        return self._initialized and self._landmarker is not None

    def process_frame(self, frame_bgr: np.ndarray) -> List[List[HandLandmark3D]]:
        """
        Executes 21-landmark detection on a BGR image frame.
        Returns a list of 21 3D landmarks for each detected hand.
        """
        if not self.is_available or frame_bgr is None or frame_bgr.size == 0:
            return []

        try:
            import mediapipe as mp
            # Convert BGR to RGB
            rgb_img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)
            detection_result = self._landmarker.detect(mp_image)

            all_hands_landmarks: List[List[HandLandmark3D]] = []
            if detection_result.hand_landmarks:
                for hand_lms in detection_result.hand_landmarks:
                    landmarks_3d = [
                        HandLandmark3D(x=float(lm.x), y=float(lm.y), z=float(lm.z))
                        for lm in hand_lms
                    ]
                    all_hands_landmarks.append(landmarks_3d)

            return all_hands_landmarks
        except Exception as e:
            logger.debug(f"MediaPipe inference exception: {e}")
            return []


class HandGestureRecognizer:
    """
    Kinematic 3D joint angle and spatial vector classifier.
    Analyzes 21 landmarks and classifies gestures into rich interaction vocabulary.
    """

    def __init__(self, hand_tracker: Optional[MediaPipeHandTracker] = None):
        self.tracker = hand_tracker or MediaPipeHandTracker()

    def classify_landmarks(
        self,
        landmarks: List[HandLandmark3D],
        img_shape: Tuple[int, int],
        handedness: str = "Right",
    ) -> GestureResult:
        """
        Calculates joint extension angles, pinch distance, and classifies the 3D gesture.
        """
        if not landmarks or len(landmarks) < 21:
            return GestureResult()

        h, w = img_shape[:2]

        # 1. Calculate joint angles across each finger
        # Straight finger: angle > 155° at PIP and DIP
        thumb_angle = compute_3d_angle(landmarks[THUMB_CMC], landmarks[THUMB_MCP], landmarks[THUMB_IP])
        index_angle = compute_3d_angle(landmarks[INDEX_MCP], landmarks[INDEX_PIP], landmarks[INDEX_DIP])
        middle_angle = compute_3d_angle(landmarks[MIDDLE_MCP], landmarks[MIDDLE_PIP], landmarks[MIDDLE_DIP])
        ring_angle = compute_3d_angle(landmarks[RING_MCP], landmarks[RING_PIP], landmarks[RING_DIP])
        pinky_angle = compute_3d_angle(landmarks[PINKY_MCP], landmarks[PINKY_PIP], landmarks[PINKY_DIP])

        # Distance from fingertip to wrist vs knuckle (MCP) to wrist
        wrist = landmarks[WRIST]
        index_tip_dist = compute_euclidean_dist(landmarks[INDEX_TIP], wrist)
        index_pip_dist = compute_euclidean_dist(landmarks[INDEX_PIP], wrist)
        middle_tip_dist = compute_euclidean_dist(landmarks[MIDDLE_TIP], wrist)
        middle_pip_dist = compute_euclidean_dist(landmarks[MIDDLE_PIP], wrist)
        ring_tip_dist = compute_euclidean_dist(landmarks[RING_TIP], wrist)
        ring_pip_dist = compute_euclidean_dist(landmarks[RING_PIP], wrist)
        pinky_tip_dist = compute_euclidean_dist(landmarks[PINKY_TIP], wrist)
        pinky_pip_dist = compute_euclidean_dist(landmarks[PINKY_PIP], wrist)

        index_ext = (index_tip_dist > index_pip_dist * 1.06 or landmarks[INDEX_TIP].y < landmarks[INDEX_PIP].y) and index_angle > 130.0
        middle_ext = (middle_tip_dist > middle_pip_dist * 1.06 or landmarks[MIDDLE_TIP].y < landmarks[MIDDLE_PIP].y) and middle_angle > 130.0
        ring_ext = (ring_tip_dist > ring_pip_dist * 1.06 or landmarks[RING_TIP].y < landmarks[RING_PIP].y) and ring_angle > 130.0
        pinky_ext = (pinky_tip_dist > pinky_pip_dist * 1.06 or landmarks[PINKY_TIP].y < landmarks[PINKY_PIP].y) and pinky_angle > 130.0

        # Thumb extension (angle at IP joint + distance to index MCP)
        thumb_angle = compute_3d_angle(landmarks[THUMB_MCP], landmarks[THUMB_IP], landmarks[THUMB_TIP])
        thumb_tip_dist = compute_euclidean_dist(landmarks[THUMB_TIP], landmarks[INDEX_MCP])
        thumb_ext = thumb_tip_dist > 0.13 and thumb_angle > 140.0

        # Pinch detection (Thumb tip to Index tip distance when index is active)
        pinch_dist = compute_euclidean_dist(landmarks[THUMB_TIP], landmarks[INDEX_TIP])
        is_pinching = pinch_dist < 0.055 and (index_ext or index_angle > 125.0)

        fingers_ext = {
            "thumb": thumb_ext,
            "index": index_ext,
            "middle": middle_ext,
            "ring": ring_ext,
            "pinky": pinky_ext,
        }
        extended_count = sum(1 for ext in [thumb_ext, index_ext, middle_ext, ring_ext, pinky_ext] if ext)

        # 2. Bounding Box in Pixel Space
        xs = [lm.x * w for lm in landmarks]
        ys = [lm.y * h for lm in landmarks]
        hand_bbox = [
            max(0.0, min(xs) - 15.0),
            max(0.0, min(ys) - 15.0),
            min(float(w), max(xs) + 15.0),
            min(float(h), max(ys) + 15.0),
        ]

        # 3. Pointing Ray calculation (Index Tip & Vector from Index PIP)
        idx_tip_px = (float(landmarks[INDEX_TIP].x * w), float(landmarks[INDEX_TIP].y * h))
        idx_pip_px = (float(landmarks[INDEX_PIP].x * w), float(landmarks[INDEX_PIP].y * h))
        ray_dx = idx_tip_px[0] - idx_pip_px[0]
        ray_dy = idx_tip_px[1] - idx_pip_px[1]
        norm = math.sqrt(ray_dx**2 + ray_dy**2)
        pointing_vec = (ray_dx / max(norm, 1e-5), ray_dy / max(norm, 1e-5))

        # 4. Gesture Classification Matrix
        gesture = GestureType.NONE
        conf = 0.90

        is_fist = (extended_count == 0 or (not index_ext and not middle_ext and not ring_ext and not pinky_ext and not thumb_ext))

        if is_fist:
            gesture = GestureType.FIST
            conf = 0.90
        elif is_pinching:
            gesture = GestureType.PINCH
            conf = 0.95
        elif extended_count == 5 or (index_ext and middle_ext and ring_ext and pinky_ext and thumb_ext):
            gesture = GestureType.OPEN_PALM
            conf = 0.95
        elif index_ext and middle_ext and not ring_ext and not pinky_ext:
            gesture = GestureType.PEACE_SIGN
            conf = 0.95
        elif index_ext and pinky_ext and not middle_ext and not ring_ext:
            gesture = GestureType.ROCK_ON
            conf = 0.92
        elif thumb_ext and pinky_ext and not index_ext and not middle_ext and not ring_ext:
            gesture = GestureType.CALL_ME
            conf = 0.92
        elif index_ext and not middle_ext and not ring_ext and not pinky_ext:
            # Strictly POINTING whenever index is extended and other 3 fingers are curled
            gesture = GestureType.POINTING
            conf = 0.94
        elif thumb_ext and not index_ext and not middle_ext and not ring_ext and not pinky_ext:
            # Strictly Thumb only
            if landmarks[THUMB_TIP].y < landmarks[WRIST].y:
                gesture = GestureType.THUMBS_UP
            else:
                gesture = GestureType.THUMBS_DOWN
            conf = 0.91

        return GestureResult(
            gesture=gesture,
            confidence=conf,
            finger_count=extended_count,
            fingers_extended=fingers_ext,
            pinch_distance=pinch_dist,
            is_pinching=is_pinching,
            pointing_tip=idx_tip_px,
            pointing_vector=pointing_vec,
            landmarks=landmarks,
            hand_bbox=hand_bbox,
            handedness=handedness,
        )

    def analyze_frame(self, frame_bgr: np.ndarray) -> List[GestureResult]:
        """Runs MediaPipe landmark detection and classifies all hands in the frame."""
        if frame_bgr is None or frame_bgr.size == 0:
            return []

        all_lms = self.tracker.process_frame(frame_bgr)
        results: List[GestureResult] = []
        for lms in all_lms:
            res = self.classify_landmarks(lms, frame_bgr.shape)
            results.append(res)
        return results


def find_pointed_object(
    pointing_tip: Tuple[float, float],
    pointing_vector: Tuple[float, float],
    scene_detections: List[Detection],
    frame_shape: Tuple[int, int],
    max_distance_px: float = 650.0,
) -> Optional[Detection]:
    """
    Performs raycasting intersection from index fingertip along the pointing vector
    to identify the targeted object in the scene.
    """
    if not scene_detections or not pointing_tip or not pointing_vector:
        return None

    px, py = pointing_tip
    vx, vy = pointing_vector
    h, w = frame_shape[:2]

    # Exclude hands/persons from candidate target list to lock onto real objects
    candidates = [
        d for d in scene_detections
        if d.class_name.lower() not in ("hand", "human hand", "open palm", "pointing hand", "fist", "person")
    ]
    if not candidates:
        candidates = [d for d in scene_detections if "hand" not in d.class_name.lower()]

    if not candidates:
        return None

    best_cand: Optional[Detection] = None
    best_score = float("inf")

    for cand in candidates:
        cx1, cy1, cx2, cy2 = cand.bbox
        ccx, ccy = (cx1 + cx2) / 2.0, (cy1 + cy2) / 2.0

        # Vector from fingertip to object center
        to_obj_x = ccx - px
        to_obj_y = ccy - py
        dist = math.sqrt(to_obj_x**2 + to_obj_y**2)

        if dist < 1e-3 or dist > max_distance_px:
            continue

        # Normalized direction to object
        dir_x, dir_y = to_obj_x / dist, to_obj_y / dist

        # Alignment cosine similarity
        alignment = (vx * dir_x) + (vy * dir_y)

        # Combined cost: penalize angle deviation and distance
        if alignment > 0.40:
            angle_penalty = (1.0 - alignment) * 400.0
            total_cost = dist + angle_penalty
            if total_cost < best_score:
                best_score = total_cost
                best_cand = cand

    # Fallback to closest bounding box if no straight line hit
    if best_cand is None and candidates:
        best_cand = min(
            candidates,
            key=lambda c: math.sqrt(((c.bbox[0] + c.bbox[2]) / 2.0 - px)**2 + ((c.bbox[1] + c.bbox[3]) / 2.0 - py)**2)
        )

    return best_cand


def draw_hand_skeleton(
    image: np.ndarray,
    gesture_result: GestureResult,
    draw_bones: bool = True,
    draw_hud_badge: bool = True,
) -> np.ndarray:
    """
    Renders a cybernetic futuristic HUD hand skeleton:
    Neon cyan bone linkages, glowing joint nodes, fingertip pulse rings, and gesture pill tags.
    """
    if not gesture_result or not gesture_result.landmarks or image is None:
        return image

    h, w = image.shape[:2]
    lms = gesture_result.landmarks

    # 1. Convert normalized coordinates to pixel coordinates
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in lms]

    # 2. Draw Bone Connectors (Neon Cyan with Anti-Aliasing)
    if draw_bones:
        for p1_idx, p2_idx in HAND_CONNECTIONS:
            if p1_idx < len(pts) and p2_idx < len(pts):
                cv2.line(image, pts[p1_idx], pts[p2_idx], (255, 220, 0), 2, cv2.LINE_AA)  # Cyan glow
                cv2.line(image, pts[p1_idx], pts[p2_idx], (200, 150, 0), 1, cv2.LINE_AA)

    # 3. Draw Joint Nodes (Glowing Green / Cyan Circles)
    for idx, pt in enumerate(pts):
        is_tip = idx in (THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP)
        if is_tip:
            # Highlighted glowing fingertip
            cv2.circle(image, pt, 6, (0, 255, 255), -1, cv2.LINE_AA)  # Yellow tip
            cv2.circle(image, pt, 9, (0, 200, 255), 1, cv2.LINE_AA)
        else:
            cv2.circle(image, pt, 3, (0, 255, 120), -1, cv2.LINE_AA)  # Neon Green node

    # 4. Render Floating Gesture Pill Tag
    if draw_hud_badge and gesture_result.gesture != GestureType.NONE:
        wrist_pt = pts[WRIST] if len(pts) > WRIST else (50, 50)
        badge_x = max(10, min(wrist_pt[0] - 60, w - 180))
        badge_y = max(30, min(wrist_pt[1] + 35, h - 20))

        label_str = f"🖐️ {gesture_result.gesture.value.upper()}"
        if gesture_result.gesture == GestureType.POINTING:
            label_str = "👉 POINTING RAY"
        elif gesture_result.gesture == GestureType.PINCH:
            label_str = "👌 PINCH SELECT"
        elif gesture_result.gesture == GestureType.OPEN_PALM:
            label_str = "🖐️ OPEN PALM (CLEAR)"
        elif gesture_result.gesture == GestureType.PEACE_SIGN:
            label_str = "✌️ VICTORY (SNAPSHOT)"
        elif gesture_result.gesture == GestureType.THUMBS_UP:
            label_str = "👍 THUMBS UP (RESTORE ALL)"
        elif gesture_result.gesture == GestureType.THUMBS_DOWN:
            label_str = "👎 THUMBS DOWN (DESELECT)"
        elif gesture_result.gesture == GestureType.FIST:
            label_str = "✊ FIST (FREEZE)"
        elif gesture_result.gesture == GestureType.ROCK_ON:
            label_str = "🤘 ROCK ON (SAHI)"
        elif gesture_result.gesture == GestureType.CALL_ME:
            label_str = "🤙 CALL ME (VOICE)"

        # Semi-transparent pill background
        pill_w = len(label_str) * 10 + 20
        cv2.rectangle(image, (badge_x - 6, badge_y - 18), (badge_x + pill_w, badge_y + 8), (20, 20, 20), -1)
        cv2.rectangle(image, (badge_x - 6, badge_y - 18), (badge_x + pill_w, badge_y + 8), (0, 240, 255), 1)
        cv2.putText(image, label_str, (badge_x, badge_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 240, 255), 1, cv2.LINE_AA)

    return image


def draw_action_toast(
    image: np.ndarray,
    toast_text: str,
) -> np.ndarray:
    """
    Renders a prominent, glowing glassmorphism action toast banner at the top-center of the HUD.
    """
    if not toast_text or image is None:
        return image

    h, w = image.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.65
    thickness = 2
    (text_w, text_h), baseline = cv2.getTextSize(toast_text, font, scale, thickness)

    pad_x, pad_y = 24, 12
    box_w = text_w + pad_x * 2
    box_h = text_h + pad_y * 2
    box_x = (w - box_w) // 2
    box_y = 48

    overlay = image.copy()
    # Dark translucent box
    cv2.rectangle(
        overlay,
        (box_x, box_y),
        (box_x + box_w, box_y + box_h),
        (15, 20, 25),
        -1,
    )
    # Blend overlay
    cv2.addWeighted(overlay, 0.85, image, 0.15, 0, image)

    # Neon border and glow
    cv2.rectangle(
        image,
        (box_x, box_y),
        (box_x + box_w, box_y + box_h),
        (0, 240, 255),
        2,
        cv2.LINE_AA,
    )
    # Text
    cv2.putText(
        image,
        toast_text,
        (box_x + pad_x, box_y + box_h - pad_y),
        font,
        scale,
        (0, 255, 240),
        thickness,
        cv2.LINE_AA,
    )
    return image


class GestureActionController:
    """
    State machine and interactive action dispatcher.
    Coordinates gesture transitions with temporal debouncing, action execution,
    snapshot capture, and on-screen toast notifications.
    """

    def __init__(
        self,
        debounce_frames: int = 2,
        on_inspect_callback: Optional[Any] = None,
        on_toggle_sahi_callback: Optional[Any] = None,
        on_voice_trigger_callback: Optional[Any] = None,
    ):
        self.debounce_frames = debounce_frames
        self.recognizer = HandGestureRecognizer()
        
        # Callbacks
        self.on_inspect = on_inspect_callback
        self.on_toggle_sahi = on_toggle_sahi_callback
        self.on_voice_trigger = on_voice_trigger_callback

        # State
        self.current_mode: GestureMode = GestureMode.ALL_OBJECTS
        self.last_raw_gesture: GestureType = GestureType.NONE
        self.consecutive_gesture_count: int = 0
        self.confirmed_gesture: GestureType = GestureType.NONE
        self.last_executed_gesture: GestureType = GestureType.NONE
        self.last_action_time: float = 0.0
        
        # Raycasting & Target
        self.targeted_object: Optional[Detection] = None
        self.smoothed_pointing_tip: Optional[Tuple[float, float]] = None
        self.frozen_detections: List[Detection] = []

        # On-screen HUD Toast
        self.active_toast: Optional[str] = None
        self.toast_expiry_time: float = 0.0

    def trigger_toast(self, text: str, duration: float = 2.0) -> None:
        """Triggers a visible HUD action toast."""
        self.active_toast = text
        self.toast_expiry_time = time.time() + duration
        logger.info(f"Gesture Action: {text}")

    def update(
        self,
        frame: np.ndarray,
        all_detections: List[Detection],
    ) -> Tuple[List[Detection], GestureMode, GestureResult, Optional[Detection]]:
        """
        Executes MediaPipe 21-landmark tracking, classifies gestures,
        dispatches concrete actions (snapshots, inspections, mode toggles),
        and filters visible detections.
        """
        if frame is None or frame.size == 0:
            return all_detections, self.current_mode, GestureResult(), None

        now = time.time()

        # 1. MediaPipe 21-Landmark Analysis
        gesture_results = self.recognizer.analyze_frame(frame)
        active_res = gesture_results[0] if gesture_results else GestureResult()

        # 2. Temporal Debouncing
        if active_res.gesture != GestureType.NONE:
            if active_res.gesture == self.last_raw_gesture:
                self.consecutive_gesture_count += 1
            else:
                self.last_raw_gesture = active_res.gesture
                self.consecutive_gesture_count = 1

            if self.consecutive_gesture_count >= self.debounce_frames:
                self.confirmed_gesture = active_res.gesture
        else:
            self.consecutive_gesture_count = 0
            self.confirmed_gesture = GestureType.NONE

        # 3. Action Execution & Edge Triggering
        is_new_gesture = (self.confirmed_gesture != self.last_executed_gesture)

        if self.confirmed_gesture != GestureType.NONE and is_new_gesture:
            self.last_executed_gesture = self.confirmed_gesture

            if self.confirmed_gesture == GestureType.OPEN_PALM:
                self.current_mode = GestureMode.HIDE_BOXES
                self.targeted_object = None
                self.trigger_toast("🖐️ CLEAN VIEW (ALL BOXES HIDDEN)")

            elif self.confirmed_gesture == GestureType.PEACE_SIGN:
                # Victory Sign captures high-res snapshot with 1.2s cooldown
                if (now - self.last_action_time) > 1.2:
                    self.last_action_time = now
                    os.makedirs("captures", exist_ok=True)
                    filename = f"captures/aura_snapshot_{int(now)}.jpg"
                    try:
                        cv2.imwrite(filename, frame)
                        self.trigger_toast(f"📸 SNAPSHOT SAVED: {filename}", duration=2.5)
                    except Exception as e:
                        self.trigger_toast(f"📸 SNAPSHOT ERROR: {e}")

            elif self.confirmed_gesture == GestureType.THUMBS_UP:
                # Thumbs Up restores all bounding boxes and resets target selection
                self.current_mode = GestureMode.ALL_OBJECTS
                self.targeted_object = None
                self.trigger_toast("👍 ALL BOUNDING BOXES RESTORED")

            elif self.confirmed_gesture == GestureType.THUMBS_DOWN:
                self.targeted_object = None
                self.current_mode = GestureMode.ALL_OBJECTS
                self.trigger_toast("❌ TARGET DESELECTED")

            elif self.confirmed_gesture == GestureType.PINCH:
                target_to_inspect = self.targeted_object
                # If no object locked yet, find closest detection to pinch point or first detected entity
                if target_to_inspect is None and all_detections:
                    if active_res.landmarks:
                        px = active_res.landmarks[INDEX_TIP].x * frame.shape[1]
                        py = active_res.landmarks[INDEX_TIP].y * frame.shape[0]
                        target_to_inspect = min(
                            all_detections,
                            key=lambda c: math.sqrt(((c.bbox[0] + c.bbox[2]) / 2.0 - px)**2 + ((c.bbox[1] + c.bbox[3]) / 2.0 - py)**2)
                        )
                    else:
                        target_to_inspect = all_detections[0]

                if target_to_inspect is not None:
                    self.targeted_object = target_to_inspect
                    self.current_mode = GestureMode.INSPECT_OBJECT
                    self.trigger_toast(f"👌 INSPECTING: {target_to_inspect.class_name.upper()}", duration=2.5)
                    if self.on_inspect and (now - self.last_action_time) > 1.2:
                        self.last_action_time = now
                        try:
                            self.on_inspect(target_to_inspect)
                        except Exception as e:
                            logger.error(f"Inspect callback error: {e}")
                else:
                    self.trigger_toast("👌 PINCH (NO OBJECTS IN SCENE TO INSPECT)")

            elif self.confirmed_gesture == GestureType.ROCK_ON:
                if (now - self.last_action_time) > 1.2:
                    self.last_action_time = now
                    self.trigger_toast("🤘 SAHI HIGH-RES MODE TOGGLED")
                    if self.on_toggle_sahi:
                        try:
                            self.on_toggle_sahi()
                        except Exception as e:
                            logger.error(f"SAHI toggle callback error: {e}")

            elif self.confirmed_gesture == GestureType.CALL_ME:
                if (now - self.last_action_time) > 1.2:
                    self.last_action_time = now
                    self.trigger_toast("🤙 VOICE ASSISTANT LISTENING...")
                    if self.on_voice_trigger:
                        try:
                            self.on_voice_trigger()
                        except Exception as e:
                            logger.error(f"Voice trigger callback error: {e}")

            elif self.confirmed_gesture == GestureType.FIST:
                if self.current_mode == GestureMode.FROZEN:
                    self.current_mode = GestureMode.ALL_OBJECTS
                    self.trigger_toast("✊ OVERLAY UNFREEZED")
                else:
                    self.current_mode = GestureMode.FROZEN
                    self.frozen_detections = list(all_detections)
                    self.trigger_toast("✊ OVERLAY FROZEN")

        # Continuous Pointing Ray & Object Lock Update
        if self.confirmed_gesture == GestureType.POINTING:
            self.current_mode = GestureMode.FOCUS_OBJECT
            if active_res.pointing_tip and active_res.pointing_vector:
                if self.smoothed_pointing_tip is None:
                    self.smoothed_pointing_tip = active_res.pointing_tip
                else:
                    alpha = 0.55
                    sx = alpha * active_res.pointing_tip[0] + (1 - alpha) * self.smoothed_pointing_tip[0]
                    sy = alpha * active_res.pointing_tip[1] + (1 - alpha) * self.smoothed_pointing_tip[1]
                    self.smoothed_pointing_tip = (sx, sy)

                new_target = find_pointed_object(
                    self.smoothed_pointing_tip,
                    active_res.pointing_vector,
                    all_detections,
                    frame.shape,
                )
                if new_target != self.targeted_object and new_target is not None:
                    self.targeted_object = new_target
                    self.trigger_toast(f"👉 LOCKED: {new_target.class_name.upper()}", duration=1.2)

        # Reset last executed gesture when hand is lowered
        if self.confirmed_gesture == GestureType.NONE:
            self.last_executed_gesture = GestureType.NONE

        # 4. Filter Visible Detections
        if self.current_mode == GestureMode.HIDE_BOXES:
            visible_detections = []
        elif self.current_mode in (GestureMode.FOCUS_OBJECT, GestureMode.INSPECT_OBJECT) and self.targeted_object:
            visible_detections = [self.targeted_object]
        elif self.current_mode == GestureMode.FROZEN:
            visible_detections = self.frozen_detections
        else:
            visible_detections = all_detections

        return visible_detections, self.current_mode, active_res, self.targeted_object
