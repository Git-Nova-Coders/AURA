"""
Unit tests for AURA Camera Adapter and Frame data structures.
"""

import time
import unittest
import numpy as np
from vision.camera import Frame, CameraAdapter, CameraError, CameraNotFoundError


class TestCameraAdapter(unittest.TestCase):
    def test_frame_properties(self):
        """Verify Frame dataclass properties and dimensions."""
        sample_img = np.zeros((480, 640, 3), dtype=np.uint8)
        ts = time.time()
        frame = Frame(image=sample_img, timestamp=ts, source_id="test_cam")

        self.assertEqual(frame.height, 480)
        self.assertEqual(frame.width, 640)
        self.assertEqual(frame.shape, (480, 640, 3))
        self.assertEqual(frame.source_id, "test_cam")
        self.assertEqual(frame.timestamp, ts)

    def test_camera_adapter_init(self):
        """Verify CameraAdapter initial configuration."""
        cam = CameraAdapter(source=0, width=1280, height=720, fps=60)
        self.assertEqual(cam.source, 0)
        self.assertEqual(cam.requested_width, 1280)
        self.assertEqual(cam.requested_height, 720)
        self.assertEqual(cam.requested_fps, 60)
        self.assertFalse(cam.is_opened)
        self.assertEqual(cam.source_id, "webcam_0")

    def test_camera_adapter_invalid_source(self):
        """Verify CameraAdapter raises CameraNotFoundError for non-existent video file/source."""
        cam = CameraAdapter(source="non_existent_file_xyz_12345.mp4")
        with self.assertRaises(CameraNotFoundError):
            cam.open()

    def test_camera_read_before_open(self):
        """Verify read() raises CameraError if called before open()."""
        cam = CameraAdapter(source=0)
        with self.assertRaises(CameraError):
            cam.read()

    def test_camera_context_manager_invalid(self):
        """Verify context manager properly handles open failure and releases."""
        with self.assertRaises(CameraNotFoundError):
            with CameraAdapter(source="non_existent_path.mp4") as cam:
                cam.read()


if __name__ == "__main__":
    unittest.main()
