"""
Unit tests for AURA SAHI (Slicing Aided Hyper Inference) Subsystem.
Tests slice window calculation, tile generation, NMS box fusion, and end-to-end sliced inference.
"""

import unittest
import numpy as np

from config.config import SAHIConfig
from vision.detector import Detection, ObjectDetector
from vision.camera import Frame
from vision.sahi import (
    SliceWindow,
    generate_slice_grid,
    compute_iou,
    apply_nms_merging,
    SlicedInferenceEngine,
)


class TestSAHISubsystem(unittest.TestCase):
    def test_slice_window_properties_and_crop(self):
        """Verify SliceWindow geometry and image cropping."""
        window = SliceWindow(x1=100, y1=50, x2=300, y2=250)
        self.assertEqual(window.width, 200)
        self.assertEqual(window.height, 200)
        self.assertEqual(window.area, 40000)

        # Test cropping on a 480x640 image
        img = np.ones((480, 640, 3), dtype=np.uint8) * 128
        patch = window.crop(img)
        self.assertEqual(patch.shape, (200, 200, 3))

    def test_generate_slice_grid_full_coverage(self):
        """Verify slice grid generates tiles covering the entire image with overlap."""
        w, h = 640, 480
        slice_w, slice_h = 320, 320
        overlap = 0.20

        grid = generate_slice_grid(
            image_width=w,
            image_height=h,
            slice_width=slice_w,
            slice_height=slice_h,
            overlap_width_ratio=overlap,
            overlap_height_ratio=overlap,
        )

        self.assertGreater(len(grid), 1)

        # Verify all window coordinates are strictly within bounds
        for win in grid:
            self.assertTrue(0 <= win.x1 < win.x2 <= w)
            self.assertTrue(0 <= win.y1 < win.y2 <= h)
            self.assertEqual(win.width, slice_w)
            self.assertEqual(win.height, slice_h)

        # Verify full horizontal and vertical extent is covered
        min_x = min(win.x1 for win in grid)
        max_x = max(win.x2 for win in grid)
        min_y = min(win.y1 for win in grid)
        max_y = max(win.y2 for win in grid)

        self.assertEqual(min_x, 0)
        self.assertEqual(max_x, w)
        self.assertEqual(min_y, 0)
        self.assertEqual(max_y, h)

    def test_generate_slice_grid_small_image_edge_case(self):
        """Verify slice generator handles images smaller than slice dimensions."""
        grid = generate_slice_grid(image_width=200, image_height=150, slice_width=320, slice_height=320)
        self.assertEqual(len(grid), 1)
        self.assertEqual(grid[0].x1, 0)
        self.assertEqual(grid[0].y1, 0)
        self.assertEqual(grid[0].x2, 200)
        self.assertEqual(grid[0].y2, 150)

    def test_apply_nms_merging_deduplication(self):
        """Verify overlapping duplicate detections are fused correctly."""
        # Two overlapping 'pen' boxes from adjacent slices
        det1 = Detection(class_id=5, class_name="pen", confidence=0.90, bbox=[100.0, 100.0, 150.0, 120.0])
        det2 = Detection(class_id=5, class_name="pen", confidence=0.80, bbox=[102.0, 98.0, 152.0, 118.0])
        # A disjoint 'notebook' box
        det3 = Detection(class_id=2, class_name="notebook", confidence=0.85, bbox=[300.0, 200.0, 500.0, 400.0])

        fused = apply_nms_merging([det1, det2, det3], iou_threshold=0.5, match_class=True)
        self.assertEqual(len(fused), 2)

        # First box should be the fused pen with peak confidence
        pen_det = next(d for d in fused if d.class_name == "pen")
        self.assertEqual(pen_det.confidence, 0.90)
        # Bbox should be weighted average between det1 (weight 0.9) and det2 (weight 0.8)
        self.assertAlmostEqual(pen_det.bbox[0], (100.0 * 0.9 + 102.0 * 0.8) / 1.7, places=1)

    def test_sahi_engine_end_to_end(self):
        """Verify SlicedInferenceEngine integrates with ObjectDetector and returns valid detections."""
        config = SAHIConfig(
            enabled=True,
            slice_width=320,
            slice_height=320,
            overlap_width_ratio=0.2,
            include_full_frame=True,
        )
        detector = ObjectDetector(model_name="yolo11n.pt", confidence_threshold=0.25, sahi_config=config)
        self.assertIsNotNone(detector._sahi_engine)

        # Create a synthetic image with a clear shape
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        frame = Frame(image=img, timestamp=0.0, source_id="test_sahi")

        # Run detect (which delegates to SAHI engine)
        detections = detector.detect(frame)
        self.assertIsInstance(detections, list)

        # Test enable_sahi and disable_sahi runtime toggles
        detector.disable_sahi()
        self.assertIsNone(detector._sahi_engine)
        detector.enable_sahi(config)
        self.assertIsNotNone(detector._sahi_engine)


if __name__ == "__main__":
    unittest.main()
