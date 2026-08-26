"""
End-to-end integration tests for AURA Milestone 5 (Tracking & OCR Pipeline).
"""

import unittest
import numpy as np
import cv2

from config.config import AuraConfig
from vision.camera import Frame
from vision.pipeline import VisionPipeline, PipelineResult


class TestMilestone5Integration(unittest.TestCase):
    def setUp(self):
        self.config = AuraConfig()
        self.config.tracker.enabled = True
        self.config.ocr.enabled = True
        self.pipeline = VisionPipeline(config=self.config)

    def test_end_to_end_m5_pipeline(self):
        """
        Validates complete Milestone 5 data flow:
        Frame -> YOLO Object Detection -> ObjectTracker -> FeatureBuilder -> Reliability ANN -> OCR -> PipelineResult
        """
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        # Render a synthetic object and text
        cv2.rectangle(img, (100, 100), (300, 300), (120, 120, 120), -1)
        cv2.putText(img, "AURA VISION", (120, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        frame = Frame(image=img, timestamp=100.0, source_id="m5_integration_test")

        # Frame 1
        result1 = self.pipeline.process_frame(frame, extract_features=True, annotate=True, run_ocr=True)
        self.assertIsInstance(result1, PipelineResult)
        self.assertIsInstance(result1.detections, list)
        self.assertIsInstance(result1.features, list)
        self.assertIsInstance(result1.text_detections, list)
        self.assertIsInstance(result1.object_texts, dict)
        self.assertIsNotNone(result1.annotated_frame)
        self.assertGreaterEqual(result1.latency_ms, 0.0)

        # Frame 2: verify tracking persistence
        result2 = self.pipeline.process_frame(frame, extract_features=True, annotate=True)
        self.assertIsInstance(result2, PipelineResult)
        if result1.detections and result2.detections:
            # First detected object should maintain same track_id
            self.assertEqual(result1.detections[0].track_id, result2.detections[0].track_id)
            if result2.features:
                self.assertGreaterEqual(result2.features[0].track_age, 1)

        # Verify serialization
        d = result2.to_dict()
        self.assertIn("timestamp", d)
        self.assertIn("num_detections", d)
        self.assertIn("num_texts", d)
        self.assertIn("fps", d)
        self.assertIn("latency_ms", d)
        self.assertIn("detections", d)
        self.assertIn("features", d)
        self.assertIn("texts", d)


if __name__ == "__main__":
    unittest.main()
