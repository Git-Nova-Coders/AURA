"""
Unit tests for AURA FeatureBuilder and DetectionFeatures data structures.
"""

import unittest
import numpy as np
import cv2

from vision.camera import Frame
from vision.detector import Detection
from vision.features import DetectionFeatures, FeatureBuilder


class TestFeatureBuilder(unittest.TestCase):
    def setUp(self):
        self.builder = FeatureBuilder(
            enable_blur=True,
            enable_brightness=True,
            enable_contrast=True,
        )

    def test_feature_normalization_bounds(self):
        """Verify geometric features are strictly bounded in [0.0, 1.0]."""
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        det = Detection(
            class_id=0,
            class_name="person",
            confidence=0.85,
            bbox=[64.0, 48.0, 320.0, 240.0],
        )

        feat = self.builder.extract(img, det)

        self.assertAlmostEqual(feat.norm_x1, 0.1, places=3)
        self.assertAlmostEqual(feat.norm_y1, 0.1, places=3)
        self.assertAlmostEqual(feat.norm_x2, 0.5, places=3)
        self.assertAlmostEqual(feat.norm_y2, 0.5, places=3)
        self.assertAlmostEqual(feat.norm_width, 0.4, places=3)
        self.assertAlmostEqual(feat.norm_height, 0.4, places=3)
        self.assertAlmostEqual(feat.norm_area, 0.16, places=3)
        self.assertAlmostEqual(feat.norm_center_x, 0.3, places=3)
        self.assertAlmostEqual(feat.norm_center_y, 0.3, places=3)
        self.assertAlmostEqual(feat.aspect_ratio, 1.0, places=3)

    def test_to_vector_shapes_and_types(self):
        """Verify to_vector produces correct 1D float32 numpy arrays."""
        det = Detection(
            class_id=5,
            class_name="bus",
            confidence=0.92,
            bbox=[10.0, 20.0, 100.0, 200.0],
        )
        img = np.full((300, 300, 3), 128, dtype=np.uint8)
        feat = self.builder.extract(img, det)

        # Base feature vector (14 features)
        vec_base = feat.to_vector(include_class_onehot=False)
        self.assertEqual(vec_base.shape, (14,))
        self.assertEqual(vec_base.dtype, np.float32)
        self.assertFalse(np.isnan(vec_base).any())

        # Vector with 80-class one-hot (14 + 80 = 94 features)
        vec_onehot = feat.to_vector(include_class_onehot=True, num_classes=80)
        self.assertEqual(vec_onehot.shape, (94,))
        self.assertEqual(vec_onehot[14 + 5], 1.0)  # Class 5 is hot

    def test_visual_quality_metrics(self):
        """Verify brightness, contrast, and blur score calculations."""
        # Create an image with known high-contrast checkerboard pattern
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        img[0:100, 0:100] = 255
        img[100:200, 100:200] = 255

        det = Detection(class_id=0, class_name="person", confidence=0.9, bbox=[0, 0, 200, 200])
        feat = self.builder.extract(img, det)

        self.assertGreater(feat.blur_score, 0.0)  # High sharpness/variance
        self.assertAlmostEqual(feat.brightness, 127.5, delta=1.0)  # 50% white, 50% black
        self.assertGreater(feat.contrast, 100.0)  # High contrast

    def test_edge_case_boundary_clamping(self):
        """Verify out-of-bounds bounding boxes are clamped gracefully without errors."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        det = Detection(
            class_id=0,
            class_name="person",
            confidence=0.5,
            bbox=[-50.0, -20.0, 200.0, 150.0],  # Out of bounds
        )

        feat = self.builder.extract(img, det)
        self.assertTrue(0.0 <= feat.norm_x1 <= 1.0)
        self.assertTrue(0.0 <= feat.norm_y1 <= 1.0)
        self.assertTrue(0.0 <= feat.norm_x2 <= 1.0)
        self.assertTrue(0.0 <= feat.norm_y2 <= 1.0)
        self.assertFalse(np.isnan(feat.to_vector()).any())

    def test_edge_case_zero_area_crop(self):
        """Verify 0-width or 0-height bbox does not crash or produce NaNs."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        det = Detection(
            class_id=0,
            class_name="person",
            confidence=0.5,
            bbox=[50.0, 50.0, 50.0, 50.0],  # 0 area
        )

        feat = self.builder.extract(img, det)
        self.assertEqual(feat.norm_width, 0.0)
        self.assertEqual(feat.norm_height, 0.0)
        self.assertEqual(feat.aspect_ratio, 0.0)
        self.assertFalse(np.isnan(feat.to_vector()).any())

    def test_to_dict_serialization(self):
        """Verify to_dict produces expected structured JSON dictionary."""
        det = Detection(
            class_id=63,
            class_name="laptop",
            confidence=0.9123,
            bbox=[10.0, 10.0, 100.0, 100.0],
            track_id=3,
        )
        img = np.full((200, 200, 3), 100, dtype=np.uint8)
        feat = self.builder.extract(img, det)

        d = feat.to_dict()
        self.assertEqual(d["class_id"], 63)
        self.assertEqual(d["class_name"], "laptop")
        self.assertEqual(d["confidence"], 0.9123)
        self.assertIn("norm_geometry", d)
        self.assertIn("visual_quality", d)
        self.assertEqual(d["track_id"], 3)


if __name__ == "__main__":
    unittest.main()
