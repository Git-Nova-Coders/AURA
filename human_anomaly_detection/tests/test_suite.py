import unittest
import numpy as np
import cv2
import os
import torch

import src.config as config
from src.preprocessing import preprocess_image, augment_image
from src.model import AnomalyCNN
from src.anomaly_detector import AnomalyDetector
from src.person_detector import PersonDetector

class TestUnitPreprocessing(unittest.TestCase):
    def test_config_loading(self):
        self.assertEqual(config.IMAGE_WIDTH, 224)
        self.assertEqual(config.IMAGE_HEIGHT, 224)
        self.assertEqual(config.TARGET_SIZE, (224, 224))
        self.assertGreater(config.BATCH_SIZE, 0)

    def test_preprocessing_deterministic(self):
        # Create synthetic 300x400 BGR image
        dummy_img = np.full((300, 400, 3), 120, dtype=np.uint8)
        processed = preprocess_image(dummy_img, is_training=False)
        
        self.assertEqual(processed.shape, (224, 224, 3))
        self.assertEqual(processed.dtype, np.float32)
        self.assertTrue(0.0 <= processed.min() <= processed.max() <= 1.0)

    def test_invalid_image_handling(self):
        with self.assertRaises(ValueError):
            preprocess_image(None)
        with self.assertRaises(ValueError):
            preprocess_image(np.array([]))

    def test_augmentation(self):
        dummy_img = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        aug = augment_image(dummy_img)
        self.assertEqual(aug.shape, dummy_img.shape)


class TestUnitModelLoading(unittest.TestCase):
    def test_model_instantiation(self):
        model = AnomalyCNN()
        self.assertIsInstance(model, torch.nn.Module)
        
        # Test forward pass with dummy tensor
        dummy_tensor = torch.zeros((2, 224, 224, 3), dtype=torch.float32)
        out = model(dummy_tensor)
        self.assertEqual(out.shape, (2, 1))

    def test_anomaly_detector_loading(self):
        model_path = os.path.join(config.MODELS_DIR, "anomaly_model.pth")
        if os.path.exists(model_path):
            detector = AnomalyDetector(model_path)
            self.assertIsNotNone(detector.model)


class TestModelRobustness(unittest.TestCase):
    """
    Tests model behavior on varied image inputs:
    Different resolutions, lighting variation, blur, partial crops.
    """
    @classmethod
    def setUpClass(cls):
        model_path = os.path.join(config.MODELS_DIR, "anomaly_model.pth")
        cls.detector = AnomalyDetector(model_path)

    def test_different_resolutions(self):
        resolutions = [(64, 64), (480, 640), (1080, 1920)]
        for h, w in resolutions:
            img = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
            result = self.detector.predict(img)
            self.assertIn(result.get("label"), ["Normal", "Anomaly"])
            self.assertTrue(0.0 <= result.get("confidence") <= 1.0)

    def test_lighting_variation(self):
        # Very dark image
        dark_img = np.full((224, 224, 3), 10, dtype=np.uint8)
        res_dark = self.detector.predict(dark_img)
        self.assertIn("label", res_dark)

        # Very bright image
        bright_img = np.full((224, 224, 3), 245, dtype=np.uint8)
        res_bright = self.detector.predict(bright_img)
        self.assertIn("label", res_bright)

    def test_blur_handling(self):
        img = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        blurred = cv2.GaussianBlur(img, (15, 15), 0)
        res_blurred = self.detector.predict(blurred)
        self.assertIn("label", res_blurred)

    def test_partial_crop(self):
        # Narrow crop (e.g. upper body or leg)
        narrow_crop = np.random.randint(0, 256, (180, 50, 3), dtype=np.uint8)
        res_narrow = self.detector.predict(narrow_crop)
        self.assertIn("label", res_narrow)

    def test_invalid_input_graceful_fail(self):
        res = self.detector.predict(None)
        self.assertIn("error", res)
        res_empty = self.detector.predict(np.zeros((0, 0, 3), dtype=np.uint8))
        self.assertIn("error", res_empty)


class TestIntegrationPipeline(unittest.TestCase):
    """
    Integration test:
    Frame -> Person Detector -> Crop -> Anomaly Detector -> Prediction
    """
    @classmethod
    def setUpClass(cls):
        cls.person_det = PersonDetector(confidence_threshold=0.5)
        cls.anomaly_det = AnomalyDetector("models/anomaly_model.pth")

    def test_frame_pipeline(self):
        test_path = os.path.join("dataset", "test", "normal", "norm_test_00000.jpg")
        if os.path.exists(test_path):
            frame = cv2.imread(test_path)
        else:
            frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

        person_results = self.person_det.detect_and_crop(frame)
        self.assertIsInstance(person_results, list)

        if len(person_results) > 0:
            for item in person_results:
                self.assertIn("crop", item)
                self.assertIn("box", item)
                pred = self.anomaly_det.predict(item["crop"])
                self.assertIn(pred.get("label"), ["Normal", "Anomaly"])
                self.assertTrue(0.0 <= pred.get("confidence") <= 1.0)
        else:
            # Fallback path test
            pred = self.anomaly_det.predict(frame)
            self.assertIn(pred.get("label"), ["Normal", "Anomaly"])

if __name__ == "__main__":
    unittest.main()
