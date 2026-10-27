"""
Integration tests for AURA Milestone 1 Vision Pipeline.
Validates the complete flow from Frame generation to ObjectDetector inference and structured output.
"""

import unittest
import numpy as np
import cv2
from vision.camera import Frame
from vision.detector import ObjectDetector, Detection, draw_detections
from config.config import AuraConfig


class TestIntegrationMilestone1(unittest.TestCase):
    def test_milestone_one_end_to_end_pipeline(self):
        """
        Validates the full Milestone 1 data flow:
        1. Frame initialization
        2. Model inference via ObjectDetector
        3. Structured Detection list verification
        4. Serialized JSON-compatible dict verification (as required by SSD section 4.2)
        5. Annotation rendering with HUD / bounding boxes
        """
        config = AuraConfig()

        # 1. Initialize detector with default config
        detector = ObjectDetector(
            model_name=config.vision.model_name,
            confidence_threshold=config.vision.confidence_threshold,
        )

        # 2. Create synthetic test frame
        test_img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.circle(test_img, (320, 240), 100, (255, 255, 255), -1)
        frame = Frame(image=test_img, timestamp=1234567.89, source_id="integration_test")

        # 3. Detect objects
        detections = detector.detect(frame)
        self.assertIsInstance(detections, list)

        # 4. Create synthetic detection to verify contract with downstream modules
        sample_det = Detection(
            class_id=0,
            class_name="person",
            confidence=0.88,
            bbox=[50.0, 60.0, 200.0, 400.0],
        )
        detections.append(sample_det)

        # Verify structured dict adheres to SSD specification
        ssd_dict = sample_det.to_dict()
        self.assertIn("class_id", ssd_dict)
        self.assertIn("class_name", ssd_dict)
        self.assertIn("confidence", ssd_dict)
        self.assertIn("bbox", ssd_dict)
        self.assertIn("track_id", ssd_dict)

        # 5. Verify annotation works seamlessly
        annotated = detector.annotate(frame, detections)
        self.assertEqual(annotated.shape, frame.shape)
        self.assertEqual(annotated.dtype, np.uint8)


if __name__ == "__main__":
    unittest.main()
