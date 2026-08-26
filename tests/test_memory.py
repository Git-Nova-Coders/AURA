"""
Unit tests for AURA Episodic & Spatial Memory Subsystem.
"""

import os
import time
import shutil
import tempfile
import unittest

from config.config import MemoryConfig
from brain.memory import EpisodicMemory, EpisodicEvent
from brain.context import ContextManager
from vision.detector import Detection
from ocr.engine import TextDetection


class TestEpisodicMemory(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_memory.db")
        self.config = MemoryConfig(enabled=True, db_path=self.db_path, snapshot_interval_seconds=0.01)
        self.memory = EpisodicMemory(config=self.config)

    def tearDown(self):
        self.memory.close()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_record_and_find_last_seen(self):
        """Verify recording entities and recalling last known location."""
        ctx_mgr = ContextManager()
        det = Detection(class_id=2, class_name="notebook", confidence=0.92, bbox=[100, 100, 300, 300])
        td = TextDetection(text="AURA Architecture", confidence=0.95, bbox=[120, 120, 250, 180])
        
        scene = ctx_mgr.update([det], text_detections=[td], object_texts={0: [td]}, frame_shape=(480, 640))
        
        count = self.memory.record_scene(scene, force=True)
        self.assertEqual(count, 1)

        event = self.memory.find_last_seen("notebook")
        self.assertIsNotNone(event)
        self.assertEqual(event.class_name, "notebook")
        self.assertIn("AURA Architecture", event.associated_text)

    def test_query_spatial_and_temporal_memory(self):
        """Verify natural-language spatial and temporal memory query responses."""
        ctx_mgr = ContextManager()
        det = Detection(class_id=0, class_name="person", confidence=0.95, bbox=[0, 0, 200, 400])
        scene = ctx_mgr.update([det], frame_shape=(480, 640))
        self.memory.record_scene(scene, force=True)

        spatial_resp = self.memory.query_spatial_memory("person")
        self.assertIn("person", spatial_resp.lower())
        self.assertIn("left", spatial_resp.lower())

        temporal_resp = self.memory.query_temporal_memory("person")
        self.assertIn("person", temporal_resp.lower())
        self.assertIn("ago", temporal_resp.lower())


if __name__ == "__main__":
    unittest.main()
