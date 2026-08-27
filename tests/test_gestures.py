"""
Unit Tests for 21-Landmark Hand Tracking & 3D Kinematic Gesture Control Subsystem
"""

import unittest
import numpy as np
import cv2

from vision.detector import Detection
from vision.gestures import (
    GestureType,
    GestureMode,
    GestureResult,
    HandLandmark3D,
    HandGestureRecognizer,
    GestureActionController,
    compute_3d_angle,
    compute_euclidean_dist,
    find_pointed_object,
    draw_hand_skeleton,
    WRIST,
    THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP,
    INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP,
    MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP,
    RING_MCP, RING_PIP, RING_DIP, RING_TIP,
    PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP,
)


def create_mock_landmarks(
    index_straight: bool = True,
    middle_straight: bool = True,
    ring_straight: bool = True,
    pinky_straight: bool = True,
    thumb_straight: bool = True,
    pinch: bool = False,
) -> list[HandLandmark3D]:
    """Helper to synthesize 21 3D landmarks for testing."""
    lms = [HandLandmark3D(0.5, 0.8, 0.0) for _ in range(21)]
    lms[WRIST] = HandLandmark3D(0.5, 0.8, 0.0)

    # Thumb
    lms[THUMB_CMC] = HandLandmark3D(0.42, 0.72, 0.0)
    lms[THUMB_MCP] = HandLandmark3D(0.38, 0.65, 0.0)
    lms[THUMB_IP] = HandLandmark3D(0.34, 0.58, 0.0) if thumb_straight else HandLandmark3D(0.42, 0.62, 0.0)
    lms[THUMB_TIP] = HandLandmark3D(0.26, 0.48, 0.0) if thumb_straight else HandLandmark3D(0.45, 0.60, 0.0)

    # Index
    lms[INDEX_MCP] = HandLandmark3D(0.46, 0.60, 0.0)
    lms[INDEX_PIP] = HandLandmark3D(0.46, 0.48, 0.0)
    lms[INDEX_DIP] = HandLandmark3D(0.46, 0.38, 0.0)
    lms[INDEX_TIP] = HandLandmark3D(0.46, 0.28, 0.0) if index_straight else HandLandmark3D(0.46, 0.65, 0.0)

    # Middle
    lms[MIDDLE_MCP] = HandLandmark3D(0.50, 0.59, 0.0)
    lms[MIDDLE_PIP] = HandLandmark3D(0.50, 0.46, 0.0)
    lms[MIDDLE_DIP] = HandLandmark3D(0.50, 0.35, 0.0)
    lms[MIDDLE_TIP] = HandLandmark3D(0.50, 0.25, 0.0) if middle_straight else HandLandmark3D(0.50, 0.66, 0.0)

    # Ring
    lms[RING_MCP] = HandLandmark3D(0.54, 0.60, 0.0)
    lms[RING_PIP] = HandLandmark3D(0.54, 0.48, 0.0)
    lms[RING_DIP] = HandLandmark3D(0.54, 0.38, 0.0)
    lms[RING_TIP] = HandLandmark3D(0.54, 0.28, 0.0) if ring_straight else HandLandmark3D(0.54, 0.67, 0.0)

    # Pinky
    lms[PINKY_MCP] = HandLandmark3D(0.58, 0.62, 0.0)
    lms[PINKY_PIP] = HandLandmark3D(0.58, 0.52, 0.0)
    lms[PINKY_DIP] = HandLandmark3D(0.58, 0.42, 0.0)
    lms[PINKY_TIP] = HandLandmark3D(0.58, 0.32, 0.0) if pinky_straight else HandLandmark3D(0.58, 0.68, 0.0)

    if pinch:
        # Move thumb tip close to index tip
        lms[THUMB_TIP] = HandLandmark3D(lms[INDEX_TIP].x + 0.02, lms[INDEX_TIP].y + 0.01, 0.0)

    return lms


class Test21LandmarkGestures(unittest.TestCase):
    """Test suite for 21-landmark 3D kinematic gesture recognition."""

    def setUp(self):
        self.recognizer = HandGestureRecognizer()
        self.controller = GestureActionController(debounce_frames=1)

    def test_compute_3d_angle(self):
        """Angle between collinear points should be ~180 degrees."""
        a = HandLandmark3D(0.0, 0.0, 0.0)
        b = HandLandmark3D(0.0, 1.0, 0.0)
        c = HandLandmark3D(0.0, 2.0, 0.0)
        angle = compute_3d_angle(a, b, c)
        self.assertAlmostEqual(angle, 180.0, places=1)

        # Right angle
        d = HandLandmark3D(1.0, 1.0, 0.0)
        angle_90 = compute_3d_angle(a, b, d)
        self.assertAlmostEqual(angle_90, 90.0, places=1)

    def test_compute_euclidean_dist(self):
        """3D Euclidean distance calculation."""
        p1 = HandLandmark3D(0.0, 0.0, 0.0)
        p2 = HandLandmark3D(3.0, 4.0, 0.0)
        self.assertAlmostEqual(compute_euclidean_dist(p1, p2), 5.0, places=2)

    def test_classify_open_palm(self):
        """All 5 fingers straight should be classified as OPEN_PALM."""
        lms = create_mock_landmarks(True, True, True, True, True)
        res = self.recognizer.classify_landmarks(lms, (480, 640))
        self.assertEqual(res.gesture, GestureType.OPEN_PALM)
        self.assertEqual(res.finger_count, 5)

    def test_classify_pointing(self):
        """Only index finger straight should be classified as POINTING."""
        lms = create_mock_landmarks(index_straight=True, middle_straight=False, ring_straight=False, pinky_straight=False, thumb_straight=False)
        res = self.recognizer.classify_landmarks(lms, (480, 640))
        self.assertEqual(res.gesture, GestureType.POINTING)
        self.assertEqual(res.finger_count, 1)
        self.assertTrue(res.fingers_extended["index"])
        self.assertFalse(res.fingers_extended["middle"])
        self.assertIsNotNone(res.pointing_tip)
        self.assertIsNotNone(res.pointing_vector)

    def test_classify_pinch(self):
        """Touching thumb and index tips triggers PINCH."""
        lms = create_mock_landmarks(index_straight=True, middle_straight=False, ring_straight=False, pinky_straight=False, pinch=True)
        res = self.recognizer.classify_landmarks(lms, (480, 640))
        self.assertEqual(res.gesture, GestureType.PINCH)
        self.assertTrue(res.is_pinching)
        self.assertLess(res.pinch_distance, 0.055)

    def test_classify_peace_sign(self):
        """Index and Middle straight triggers PEACE_SIGN."""
        lms = create_mock_landmarks(index_straight=True, middle_straight=True, ring_straight=False, pinky_straight=False, thumb_straight=False)
        res = self.recognizer.classify_landmarks(lms, (480, 640))
        self.assertEqual(res.gesture, GestureType.PEACE_SIGN)
        self.assertEqual(res.finger_count, 2)

    def test_classify_fist(self):
        """All fingers curled triggers FIST."""
        lms = create_mock_landmarks(index_straight=False, middle_straight=False, ring_straight=False, pinky_straight=False, thumb_straight=False)
        res = self.recognizer.classify_landmarks(lms, (480, 640))
        self.assertEqual(res.gesture, GestureType.FIST)
        self.assertEqual(res.finger_count, 0)

    def test_find_pointed_object_raycasting(self):
        """Ray along pointing direction associates with candidate target object."""
        laptop = Detection(class_id=1, class_name="laptop", confidence=0.9, bbox=[100.0, 50.0, 250.0, 180.0])
        bottle = Detection(class_id=2, class_name="water bottle", confidence=0.85, bbox=[450.0, 300.0, 520.0, 420.0])
        
        # Pointing upwards directly at laptop
        pointing_tip = (175.0, 260.0)
        pointing_vec = (0.0, -1.0)
        
        target = find_pointed_object(pointing_tip, pointing_vec, [laptop, bottle], (480, 640))
        self.assertIsNotNone(target)
        self.assertEqual(target.class_name, "laptop")

    def test_gesture_action_controller_transitions(self):
        """Controller state transitions from Open Palm to Peace Sign to Pointing."""
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        laptop = Detection(class_id=1, class_name="laptop", confidence=0.9, bbox=[100.0, 50.0, 250.0, 180.0])
        
        # 1. Open Palm -> HIDE_BOXES
        palm_res = GestureResult(gesture=GestureType.OPEN_PALM, confidence=0.95)
        self.controller.recognizer.analyze_frame = lambda f: [palm_res]
        vis, mode, _, _ = self.controller.update(img, [laptop])
        self.assertEqual(mode, GestureMode.HIDE_BOXES)
        self.assertEqual(len(vis), 0)

        # 2. Peace Sign -> ALL_OBJECTS
        peace_res = GestureResult(gesture=GestureType.PEACE_SIGN, confidence=0.95)
        self.controller.recognizer.analyze_frame = lambda f: [peace_res]
        vis, mode, _, _ = self.controller.update(img, [laptop])
        self.assertEqual(mode, GestureMode.ALL_OBJECTS)
        self.assertEqual(len(vis), 1)

    def test_draw_hand_skeleton(self):
        """Skeleton overlay draws smoothly on image."""
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        lms = create_mock_landmarks(True, True, True, True, True)
        res = self.recognizer.classify_landmarks(lms, (480, 640))
        
        drawn = draw_hand_skeleton(img, res, draw_bones=True, draw_hud_badge=True)
        self.assertIsNotNone(drawn)
        self.assertEqual(drawn.shape, (480, 640, 3))


if __name__ == "__main__":
    unittest.main()
