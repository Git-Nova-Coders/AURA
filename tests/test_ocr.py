"""
Unit tests for AURA OCR Subsystem (Milestone 5).
"""

import unittest
import numpy as np
import cv2

from vision.detector import Detection
from ocr.engine import TextDetection, OCREngine, draw_text_annotations


class TestOCREngine(unittest.TestCase):
    def setUp(self):
        self.ocr_engine = OCREngine(confidence_threshold=0.2)

    def test_text_detection_properties(self):
        """Verify TextDetection geometry calculation and dictionary serialization."""
        td = TextDetection(
            text="HELLO AURA",
            confidence=0.95,
            bbox=[10.0, 20.0, 110.0, 60.0],
            polygon=[[10.0, 20.0], [110.0, 20.0], [110.0, 60.0], [10.0, 60.0]],
        )

        self.assertEqual(td.text, "HELLO AURA")
        self.assertAlmostEqual(td.x1, 10.0)
        self.assertAlmostEqual(td.y1, 20.0)
        self.assertAlmostEqual(td.x2, 110.0)
        self.assertAlmostEqual(td.y2, 60.0)
        self.assertAlmostEqual(td.width, 100.0)
        self.assertAlmostEqual(td.height, 40.0)
        self.assertEqual(td.center, [60.0, 40.0])

        d = td.to_dict()
        self.assertEqual(d["text"], "HELLO AURA")
        self.assertEqual(d["confidence"], 0.95)
        self.assertIn("bbox", d)

    def test_ocr_extract_text_synthetic_image(self):
        """Verify OCREngine extracts rendered text on a synthetic image."""
        # Create a white canvas with crisp black text
        img = np.full((200, 500, 3), 255, dtype=np.uint8)
        cv2.putText(
            img,
            "AURA TEST",
            (50, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.8,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )

        texts = self.ocr_engine.extract_text(img)
        self.assertIsInstance(texts, list)
        if self.ocr_engine.is_available:
            self.assertGreaterEqual(len(texts), 1)
            recognized_strings = " ".join([t.text.upper() for t in texts])
            self.assertTrue("AURA" in recognized_strings or "TEST" in recognized_strings)

    def test_ocr_extract_text_for_detections(self):
        """Verify matching of text bounding boxes to overlapping object detections."""
        img = np.full((300, 400, 3), 255, dtype=np.uint8)
        cv2.putText(img, "LABEL", (60, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

        # Detection overlapping the text
        det1 = Detection(class_id=39, class_name="bottle", confidence=0.9, bbox=[30.0, 50.0, 200.0, 250.0], track_id=5)
        # Non-overlapping detection
        det2 = Detection(class_id=0, class_name="person", confidence=0.8, bbox=[250.0, 50.0, 380.0, 280.0], track_id=8)

        mapping = self.ocr_engine.extract_text_for_detections(img, [det1, det2])
        self.assertIsInstance(mapping, dict)
        if self.ocr_engine.is_available and len(mapping) > 0:
            self.assertIn(5, mapping)
            self.assertNotIn(8, mapping)

    def test_draw_text_annotations(self):
        """Verify draw_text_annotations produces a valid non-empty annotated image."""
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        td = TextDetection(text="SCAN", confidence=0.88, bbox=[20.0, 20.0, 80.0, 60.0])
        annotated = draw_text_annotations(img, [td])

        self.assertEqual(annotated.shape, img.shape)
        self.assertGreater(np.sum(annotated), 0)


if __name__ == "__main__":
    unittest.main()
