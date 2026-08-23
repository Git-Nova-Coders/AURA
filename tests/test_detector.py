"""
Unit tests for AURA ObjectDetector and Detection structured schemas.
"""

import unittest
import numpy as np
from vision.detector import (
    Detection,
    ObjectDetector,
    draw_detections,
    get_class_color,
    ModelLoadError,
)
from vision.camera import Frame


class TestObjectDetector(unittest.TestCase):
    def test_detection_dataclass_properties(self):
        """Verify Detection geometry computations and normalized bounding boxes."""
        det = Detection(
            class_id=0,
            class_name="person",
            confidence=0.895,
            bbox=[100.0, 50.0, 300.0, 450.0],
            track_id=1,
        )

        self.assertEqual(det.x1, 100.0)
        self.assertEqual(det.y1, 50.0)
        self.assertEqual(det.x2, 300.0)
        self.assertEqual(det.y2, 450.0)
        self.assertEqual(det.width, 200.0)
        self.assertEqual(det.height, 400.0)
        self.assertEqual(det.area, 80000.0)
        self.assertEqual(det.center, (200.0, 250.0))

        # Test normalized bbox on 1000x1000 image
        norm_bbox = det.normalized_bbox(img_width=1000, img_height=1000)
        self.assertEqual(norm_bbox, [0.1, 0.05, 0.3, 0.45])

    def test_detection_serialization_matching_ssd(self):
        """Verify to_dict() matches the format required by the AURA SSD."""
        det = Detection(
            class_id=63,
            class_name="laptop",
            confidence=0.9312,
            bbox=[10.5, 20.25, 200.0, 180.75],
            track_id=None,
        )

        d = det.to_dict()
        self.assertIsNone(d["track_id"])
        self.assertEqual(d["class_id"], 63)
        self.assertEqual(d["class_name"], "laptop")
        self.assertEqual(d["confidence"], 0.9312)
        self.assertEqual(d["bbox"], [10.5, 20.25, 200.0, 180.75])

        # Test round-trip reconstruction
        det_rebuilt = Detection.from_dict(d)
        self.assertEqual(det_rebuilt.class_id, det.class_id)
        self.assertEqual(det_rebuilt.class_name, det.class_name)
        self.assertEqual(det_rebuilt.confidence, det.confidence)
        self.assertEqual(det_rebuilt.bbox, det.bbox)
        self.assertEqual(det_rebuilt.track_id, det.track_id)

    def test_get_class_color_deterministic(self):
        """Verify class color generation is deterministic and within valid BGR range."""
        col1 = get_class_color(0)
        col2 = get_class_color(0)
        self.assertEqual(col1, col2)
        self.assertEqual(len(col1), 3)
        for channel in col1:
            self.assertTrue(0 <= channel <= 255)

    def test_draw_detections(self):
        """Verify draw_detections modifies image and retains dimensions."""
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = [
            Detection(
                class_id=0,
                class_name="person",
                confidence=0.95,
                bbox=[50.0, 50.0, 200.0, 300.0],
            ),
            Detection(
                class_id=67,
                class_name="cell phone",
                confidence=0.78,
                bbox=[250.0, 150.0, 320.0, 280.0],
                track_id=5,
            ),
        ]

        annotated = draw_detections(img, detections)
        self.assertEqual(annotated.shape, img.shape)
        self.assertEqual(annotated.dtype, np.uint8)
        # Image should no longer be completely black
        self.assertTrue(np.any(annotated > 0))

    def test_object_detector_initialization(self):
        """Verify ObjectDetector loads pretrained model and exposes class names."""
        detector = ObjectDetector(model_name="yolo11n.pt", confidence_threshold=0.3)
        self.assertEqual(detector.confidence_threshold, 0.3)
        self.assertTrue(len(detector.class_names) > 0)
        self.assertIn(0, detector.class_names)  # Person class in COCO

    def test_object_detector_invalid_model(self):
        """Verify ObjectDetector raises ModelLoadError for invalid model path."""
        with self.assertRaises(ModelLoadError):
            ObjectDetector(model_name="invalid_non_existent_model_xyz.pt")

    def test_object_detector_inference_on_blank_frame(self):
        """Verify inference on blank frame returns list of valid detections without crash."""
        detector = ObjectDetector(model_name="yolo11n.pt", confidence_threshold=0.25)
        blank_img = np.zeros((480, 640, 3), dtype=np.uint8)
        frame = Frame(image=blank_img, timestamp=0.0, source_id="test_blank")

        detections = detector.detect(frame)
        self.assertIsInstance(detections, list)
        for det in detections:
            self.assertIsInstance(det, Detection)
            self.assertTrue(0.0 <= det.confidence <= 1.0)


if __name__ == "__main__":
    unittest.main()
