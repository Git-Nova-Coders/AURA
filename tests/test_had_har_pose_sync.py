"""
Unit and Integration Tests for HAR and HAD Unified Pose Synchronization.
Verifies that dual bone-links and duplicate skeleton overlays are eliminated
when both HAR and HAD engines are active concurrently.
"""

import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import cv2

from vision.pose.pose_detector import PoseLandmarkData, draw_pose_skeleton, NUM_LANDMARKS
from ml.har.inference import RealTimeHARInference
from ml.had.inference import RealTimeHADInference


def create_synthetic_landmarks(detected: bool = True) -> PoseLandmarkData:
    """Creates a mock 33-point pose landmark structure for deterministic testing."""
    if not detected:
        return PoseLandmarkData(
            detected=False,
            landmarks=np.zeros((NUM_LANDMARKS, 4), dtype=np.float32),
        )

    # 33 landmarks with normalized coords (x in 0.4..0.6, y in 0.2..0.8, vis = 0.95)
    lms = np.zeros((NUM_LANDMARKS, 4), dtype=np.float32)
    for i in range(NUM_LANDMARKS):
        lms[i, 0] = 0.5 + 0.1 * np.sin(i * 0.5)
        lms[i, 1] = 0.2 + 0.6 * (i / float(NUM_LANDMARKS))
        lms[i, 2] = 0.0
        lms[i, 3] = 0.95  # High visibility so all bones render

    return PoseLandmarkData(detected=True, landmarks=lms)


class TestPoseSkeletonDeduplication(unittest.TestCase):
    """Verifies that HAR and HAD skeleton rendering coordinates without dual bone-links."""

    def setUp(self):
        self.frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.mock_pose = create_synthetic_landmarks(detected=True)

    @patch("ml.har.inference.PoseDetector")
    def test_har_draw_skeleton_toggle(self, mock_detector_cls):
        """Verify RealTimeHARInference honors draw_skeleton flag and stores last_pose_data."""
        mock_detector = MagicMock()
        mock_detector.process_frame.return_value = self.mock_pose
        mock_detector_cls.return_value = mock_detector

        engine = RealTimeHARInference()

        # 1. With draw_skeleton = True
        frame_in = self.frame.copy()
        annotated_with, act, conf = engine.process_frame(
            frame_in,
            draw_overlay=False,
            draw_skeleton=True,
        )
        self.assertIsNotNone(engine.last_pose_data)
        self.assertTrue(engine.last_pose_data.detected)
        self.assertGreater(np.count_nonzero(annotated_with), 0, "Bones must be drawn when draw_skeleton=True")

        # 2. With draw_skeleton = False
        frame_in = self.frame.copy()
        annotated_without, act, conf = engine.process_frame(
            frame_in,
            draw_overlay=False,
            draw_skeleton=False,
        )
        self.assertEqual(np.count_nonzero(annotated_without), 0, "No bones should be drawn when draw_skeleton=False")

    @patch("ml.had.inference.PoseDetector")
    def test_had_draw_skeleton_toggle_and_external_pose(self, mock_detector_cls):
        """Verify RealTimeHADInference honors draw_skeleton, stores last_pose_data, and reuses external_pose_data."""
        mock_detector = MagicMock()
        mock_detector.process_frame.return_value = self.mock_pose
        mock_detector_cls.return_value = mock_detector

        engine = RealTimeHADInference()

        # 1. With draw_skeleton = True
        frame_in = self.frame.copy()
        annotated_with, stat, conf = engine.process_frame(
            frame_in,
            draw_overlay=False,
            draw_skeleton=True,
        )
        self.assertIsNotNone(engine.last_pose_data)
        self.assertTrue(engine.last_pose_data.detected)
        self.assertGreater(np.count_nonzero(annotated_with), 0, "Bones must be drawn when draw_skeleton=True")

        # 2. With draw_skeleton = False
        frame_in = self.frame.copy()
        annotated_without, stat, conf = engine.process_frame(
            frame_in,
            draw_overlay=False,
            draw_skeleton=False,
        )
        self.assertEqual(np.count_nonzero(annotated_without), 0, "No bones should be drawn when draw_skeleton=False")

        # 3. With external_pose_data: internal detector must NOT be called
        mock_detector.process_frame.reset_mock()
        frame_in = self.frame.copy()
        engine.process_frame(
            frame_in,
            draw_overlay=False,
            draw_skeleton=False,
            external_pose_data=self.mock_pose,
        )
        mock_detector.process_frame.assert_not_called()
        self.assertEqual(engine.last_pose_data, self.mock_pose)

    @patch("ml.had.inference.PoseDetector")
    @patch("ml.har.inference.PoseDetector")
    def test_co_active_har_and_had_prevents_dual_bone_links(
        self,
        mock_har_detector_cls,
        mock_had_detector_cls,
    ):
        """
        Simulate the exact pipeline loop in main.py and interface/bridge.py:
        Both HAR and HAD active concurrently must produce exactly ONE skeleton,
        identical to single-skeleton rendering without dual bone-links or ghosting.
        """
        mock_har_det = MagicMock()
        mock_har_det.process_frame.return_value = self.mock_pose
        mock_har_detector_cls.return_value = mock_har_det

        mock_had_det = MagicMock()
        mock_had_det.process_frame.return_value = self.mock_pose
        mock_had_detector_cls.return_value = mock_had_det

        har_engine = RealTimeHARInference()
        had_engine = RealTimeHADInference()

        raw_img = self.frame.copy()
        annotated = raw_img.copy()

        # Step 1: HAR processes frame and renders skeleton
        annotated, har_act, har_conf = har_engine.process_frame(
            annotated,
            draw_overlay=False,
            draw_skeleton=True,
            clean_frame=raw_img,
        )
        har_pose = getattr(har_engine, "last_pose_data", None)
        self.assertIsNotNone(har_pose)
        self.assertTrue(har_pose.detected)

        # Baseline single skeleton snapshot
        single_skeleton_pixels = annotated.copy()

        # Step 2: HAD processes frame with deduplication logic
        already_rendered_skeleton = bool(har_pose and getattr(har_pose, "detected", False))
        annotated, had_stat, had_conf = had_engine.process_frame(
            annotated,
            draw_overlay=False,
            draw_skeleton=not already_rendered_skeleton,
            external_pose_data=har_pose,
            clean_frame=raw_img,
        )

        # Assertions:
        # 1. Output pixels after both HAR and HAD match the single skeleton exactly (ZERO dual bones!)
        np.testing.assert_array_equal(
            annotated,
            single_skeleton_pixels,
            "Dual bone-links detected! Pixels modified when HAD should have suppressed skeleton drawing.",
        )

        # 2. Redundant pose detector in HAD was bypassed because external_pose_data was shared
        mock_had_det.process_frame.assert_not_called()

    @patch("ml.had.inference.PoseDetector")
    def test_had_only_active_renders_skeleton(self, mock_had_detector_cls):
        """When HAR is inactive (or har_pose is None), HAD must render the skeleton."""
        mock_had_det = MagicMock()
        mock_had_det.process_frame.return_value = self.mock_pose
        mock_had_detector_cls.return_value = mock_had_det

        had_engine = RealTimeHADInference()
        raw_img = self.frame.copy()
        annotated = raw_img.copy()

        har_pose = None
        already_rendered_skeleton = bool(har_pose and getattr(har_pose, "detected", False))
        self.assertFalse(already_rendered_skeleton)

        annotated, stat, conf = had_engine.process_frame(
            annotated,
            draw_overlay=False,
            draw_skeleton=not already_rendered_skeleton,
            external_pose_data=har_pose,
            clean_frame=raw_img,
        )

        self.assertGreater(np.count_nonzero(annotated), 0, "HAD must render skeleton when HAR is not active.")


if __name__ == "__main__":
    unittest.main()
