"""
Unit and integration tests for AURA VisionPipeline and DatasetCollector modules.
"""

import os
import tempfile
import unittest
import numpy as np

from vision.camera import Frame
from vision.detector import Detection
from vision.features import FeatureBuilder
from vision.pipeline import VisionPipeline, PipelineResult
from vision.dataset_collector import DatasetCollector, _HAS_PANDAS
from config.config import AuraConfig


class TestVisionPipeline(unittest.TestCase):
    def setUp(self):
        self.config = AuraConfig()
        self.pipeline = VisionPipeline(config=self.config)

    def test_pipeline_process_frame(self):
        """Verify VisionPipeline processes a single frame and outputs PipelineResult."""
        test_img = np.zeros((480, 640, 3), dtype=np.uint8)
        frame = Frame(image=test_img, timestamp=100.0, source_id="pipeline_test")

        result = self.pipeline.process_frame(frame, extract_features=True, annotate=True)

        self.assertIsInstance(result, PipelineResult)
        self.assertIsInstance(result.detections, list)
        self.assertIsInstance(result.features, list)
        self.assertIsNotNone(result.annotated_frame)
        self.assertEqual(result.annotated_frame.shape, test_img.shape)
        self.assertGreaterEqual(result.latency_ms, 0.0)

        # Test to_dict serialization
        d = result.to_dict()
        self.assertIn("timestamp", d)
        self.assertIn("num_detections", d)
        self.assertIn("fps", d)
        self.assertIn("latency_ms", d)
        self.assertIn("detections", d)
        self.assertIn("features", d)

    def test_dataset_collector_and_export(self):
        """Verify DatasetCollector accumulates features and exports to Records, CSV, and NumPy."""
        collector = DatasetCollector()
        builder = FeatureBuilder()

        img = np.full((100, 100, 3), 150, dtype=np.uint8)
        det1 = Detection(class_id=0, class_name="person", confidence=0.9, bbox=[10, 10, 80, 80])
        det2 = Detection(class_id=63, class_name="laptop", confidence=0.8, bbox=[20, 20, 90, 90])

        feat1 = builder.extract(img, det1)
        feat2 = builder.extract(img, det2)

        collector.add_sample(feat1, is_reliable=1)
        collector.add_sample(feat2, is_reliable=0)

        self.assertEqual(collector.count, 2)

        # Records Export
        recs = collector.to_records()
        self.assertEqual(len(recs), 2)
        self.assertEqual(recs[0]["class_id"], 0)
        self.assertEqual(recs[1]["class_id"], 63)

        # DataFrame Export (if pandas is installed)
        if _HAS_PANDAS:
            df = collector.to_dataframe()
            self.assertEqual(len(df), 2)
            self.assertIn("is_reliable", df.columns)
            self.assertIn("blur_score", df.columns)

        # Array Export
        X, y = collector.to_arrays()
        self.assertEqual(X.shape, (2, 14))
        self.assertEqual(len(y), 2)
        self.assertEqual(list(y), [1, 0])

        # CSV Export in temp directory
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "test_features.csv")
            collector.to_csv(csv_path)
            self.assertTrue(os.path.exists(csv_path))
            self.assertGreater(os.path.getsize(csv_path), 0)

            jsonl_path = os.path.join(tmpdir, "test_features.jsonl")
            collector.to_jsonl(jsonl_path)
            self.assertTrue(os.path.exists(jsonl_path))
            self.assertGreater(os.path.getsize(jsonl_path), 0)

    def test_generate_mock_dataset(self):
        """Verify mock dataset generator produces expected samples for the ANN team."""
        collector = DatasetCollector.generate_mock_dataset(num_samples=50, seed=123)
        self.assertEqual(collector.count, 50)

        X, y = collector.to_arrays(include_class_onehot=False)
        self.assertEqual(X.shape, (50, 14))
        self.assertEqual(len(y), 50)
        self.assertFalse(np.isnan(X).any())
        self.assertTrue(set(y).issubset({0, 1}))


if __name__ == "__main__":
    unittest.main()
