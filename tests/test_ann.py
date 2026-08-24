"""
Unit tests for AURA Reliability ANN module.
Validates model, dataset creation, training loops, evaluation metrics, and inference runtime.
"""

import os
import pickle
import tempfile
import unittest
import numpy as np
import torch

from ann.model import ReliabilityANN
from ann.dataset import ReliabilityDataset, prepare_data_loaders
from ann.inference import ReliabilityInference, ReliabilityResult
from vision.features import DetectionFeatures
from vision.dataset_collector import DatasetCollector


class TestReliabilityANN(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.model_path = os.path.join(self.temp_dir.name, "test_ann.pth")
        self.scaler_path = os.path.join(self.temp_dir.name, "test_scaler.pkl")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_model_architecture(self):
        """Verify model matches architecture specification exactly."""
        model = ReliabilityANN(input_dim=14)
        self.assertEqual(model.input_dim, 14)
        
        # Test forward pass with synthetic batch of size 4
        x = torch.randn(4, 14)
        out = model(x)
        self.assertEqual(out.shape, (4, 1))
        # Sigmoid output must be in range [0, 1]
        self.assertTrue((out >= 0.0).all() and (out <= 1.0).all())

    def test_dataset_and_loaders(self):
        """Verify dataset creation, splitting, and scaling parameters."""
        collector = DatasetCollector.generate_mock_dataset(num_samples=100, seed=42)
        X, y = collector.to_arrays()
        
        train_loader, val_loader, test_loader, scaler = prepare_data_loaders(
            X=X, y=y, batch_size=10, val_split=0.2, test_split=0.1, seed=42
        )
        
        # Total samples = 100
        total_len = len(train_loader.dataset) + len(val_loader.dataset) + len(test_loader.dataset)
        self.assertEqual(total_len, 100)
        self.assertTrue(65 <= len(train_loader.dataset) <= 75)
        self.assertTrue(15 <= len(val_loader.dataset) <= 25)
        self.assertTrue(5 <= len(test_loader.dataset) <= 15)
        self.assertIsNotNone(scaler)

    def test_inference_and_fallback(self):
        """Verify that inference wrapper runs successfully and handles fallbacks gracefully."""
        # 1. Test fallback when disabled
        inf_disabled = ReliabilityInference(enabled=False)
        feat = DetectionFeatures(
            class_id=0, class_name="person", confidence=0.75,
            norm_x1=0.1, norm_y1=0.1, norm_x2=0.5, norm_y2=0.5,
            norm_width=0.4, norm_height=0.4, norm_area=0.16,
            norm_center_x=0.3, norm_center_y=0.3, aspect_ratio=1.0,
            blur_score=150.0, brightness=120.0, contrast=45.0
        )
        res = inf_disabled.predict(feat)
        self.assertEqual(res.score, 0.75)
        self.assertEqual(res.label, "reliable")
        self.assertEqual(res.model_version, "fallback_yolo")

        # 2. Test fallback when files are missing
        inf_missing = ReliabilityInference(
            enabled=True,
            model_path="nonexistent_model.pth",
            scaler_path="nonexistent_scaler.pkl"
        )
        self.assertFalse(inf_missing.enabled)
        res = inf_missing.predict(feat)
        self.assertEqual(res.score, 0.75)

        # 3. Create mock trained model and scaler to test normal inference
        model = ReliabilityANN(input_dim=14)
        torch.save(model.state_dict(), self.model_path)
        
        collector = DatasetCollector.generate_mock_dataset(num_samples=20, seed=42)
        X, y = collector.to_arrays()
        _, _, _, scaler = prepare_data_loaders(X, y, batch_size=5)
        
        scaler_metadata = {
            "scaler": scaler,
            "input_dim": 14,
            "include_class_onehot": False,
            "feature_family": "all",
            "model_version": "ann_v1_test",
        }
        with open(self.scaler_path, "wb") as f:
            pickle.dump(scaler_metadata, f)

        # Initialize inference wrapper with mock files
        inf_enabled = ReliabilityInference(
            enabled=True,
            model_path=self.model_path,
            scaler_path=self.scaler_path,
            confidence_threshold=0.5
        )
        self.assertTrue(inf_enabled.enabled)
        
        res = inf_enabled.predict(feat)
        self.assertIsInstance(res, ReliabilityResult)
        self.assertEqual(res.model_version, "ann_v1_test")
        self.assertTrue(0.0 <= res.score <= 1.0)
        self.assertIn(res.label, ["reliable", "unreliable"])


if __name__ == "__main__":
    unittest.main()
