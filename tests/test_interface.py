"""
AURA Interface Module Tests (Milestone 9)
Unit tests for AuraBridge, REST API endpoints, and WebSocket serialization.
"""

import os
import sys
import time
import json
import unittest
from unittest.mock import patch, MagicMock, PropertyMock
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestTelemetrySnapshot(unittest.TestCase):
    """Tests for TelemetrySnapshot data structure."""

    def test_default_values(self):
        from interface.bridge import TelemetrySnapshot
        t = TelemetrySnapshot()
        self.assertEqual(t.fps, 0.0)
        self.assertEqual(t.inference_latency_ms, 0.0)
        self.assertEqual(t.frame_count, 0)
        self.assertEqual(t.active_tracks, 0)
        self.assertEqual(t.detection_count, 0)
        self.assertFalse(t.sahi_enabled)
        self.assertTrue(t.tracking_enabled)
        self.assertFalse(t.gestures_enabled)
        self.assertEqual(t.target_filter_mode, "ALL")

    def test_to_dict(self):
        from interface.bridge import TelemetrySnapshot
        t = TelemetrySnapshot(
            fps=29.5,
            inference_latency_ms=42.3,
            frame_count=100,
            active_tracks=3,
            detection_count=5,
            ocr_text_count=2,
            sahi_enabled=True,
            tracking_enabled=True,
            gestures_enabled=True,
            target_filter_mode="OBJECTS_ONLY",
            ann_version="ann_v1",
            memory_enabled=True,
            rag_enabled=True,
        )
        d = t.to_dict()
        self.assertEqual(d["fps"], 29.5)
        self.assertEqual(d["inference_latency_ms"], 42.3)
        self.assertEqual(d["frame_count"], 100)
        self.assertEqual(d["active_tracks"], 3)
        self.assertEqual(d["detection_count"], 5)
        self.assertEqual(d["ocr_text_count"], 2)
        self.assertTrue(d["sahi_enabled"])
        self.assertTrue(d["tracking_enabled"])
        self.assertTrue(d["gestures_enabled"])
        self.assertEqual(d["target_filter_mode"], "OBJECTS_ONLY")
        self.assertEqual(d["ann_version"], "ann_v1")
        self.assertTrue(d["memory_enabled"])
        self.assertTrue(d["rag_enabled"])
        self.assertIn("timestamp", d)

    def test_to_dict_rounding(self):
        from interface.bridge import TelemetrySnapshot
        t = TelemetrySnapshot(fps=29.456789, inference_latency_ms=42.345678)
        d = t.to_dict()
        self.assertEqual(d["fps"], 29.5)
        self.assertEqual(d["inference_latency_ms"], 42.3)


class TestAPIModels(unittest.TestCase):
    """Tests for Pydantic request models in the API module."""

    def test_chat_request_validation(self):
        from interface.api import ChatRequest
        req = ChatRequest(query="What do you see?")
        self.assertEqual(req.query, "What do you see?")

    def test_chat_request_empty_fails(self):
        from interface.api import ChatRequest
        with self.assertRaises(Exception):
            ChatRequest(query="")

    def test_rag_search_request(self):
        from interface.api import RAGSearchRequest
        req = RAGSearchRequest(query="laptop manual", top_k=5)
        self.assertEqual(req.query, "laptop manual")
        self.assertEqual(req.top_k, 5)

    def test_rag_search_default_top_k(self):
        from interface.api import RAGSearchRequest
        req = RAGSearchRequest(query="safety guidelines")
        self.assertEqual(req.top_k, 3)

    def test_memory_search_request(self):
        from interface.api import MemorySearchRequest
        req = MemorySearchRequest(query="notebook")
        self.assertEqual(req.query, "notebook")

    def test_config_update_request(self):
        from interface.api import ConfigUpdateRequest
        req = ConfigUpdateRequest(sahi_enabled=True, tracking_enabled=False)
        self.assertTrue(req.sahi_enabled)
        self.assertFalse(req.tracking_enabled)

    def test_config_update_partial(self):
        from interface.api import ConfigUpdateRequest
        req = ConfigUpdateRequest(sahi_enabled=True)
        self.assertTrue(req.sahi_enabled)
        self.assertIsNone(req.tracking_enabled)


class TestAPIBridgeIntegration(unittest.TestCase):
    """Tests for API module bridge getter."""

    def test_set_bridge(self):
        from interface.api import set_bridge, _get_bridge
        mock_bridge = MagicMock()
        set_bridge(mock_bridge)
        result = _get_bridge()
        self.assertEqual(result, mock_bridge)
        # Reset
        set_bridge(None)

    def test_get_bridge_raises_when_none(self):
        from interface.api import set_bridge, _get_bridge
        from fastapi import HTTPException
        set_bridge(None)
        with self.assertRaises(HTTPException):
            _get_bridge()


class TestWSModuleSetup(unittest.TestCase):
    """Tests for WebSocket module setup."""

    def test_set_bridge(self):
        from interface import ws
        mock_bridge = MagicMock()
        ws.set_bridge(mock_bridge)
        self.assertEqual(ws._bridge, mock_bridge)
        ws.set_bridge(None)

    def test_clients_set_initialized(self):
        from interface.ws import _clients
        self.assertIsInstance(_clients, set)


class TestServerAppFactory(unittest.TestCase):
    """Tests for FastAPI app creation."""

    def test_create_app_no_bridge(self):
        from interface.server import create_app
        app = create_app(bridge=None)
        self.assertIsNotNone(app)
        self.assertEqual(app.title, "AURA Dashboard")

    def test_create_app_with_bridge(self):
        from interface.server import create_app
        mock_bridge = MagicMock()
        app = create_app(bridge=mock_bridge)
        self.assertIsNotNone(app)

    def test_app_has_api_routes(self):
        from interface.server import create_app
        app = create_app()
        openapi_paths = list(app.openapi()["paths"].keys())
        self.assertIn("/api/status", openapi_paths)
        self.assertIn("/api/scene", openapi_paths)
        self.assertIn("/api/chat", openapi_paths)
        self.assertIn("/api/telemetry", openapi_paths)

    def test_app_has_websocket_route(self):
        from interface.server import create_app
        app = create_app()
        all_routes = []
        for r in app.routes:
            if hasattr(r, "path"):
                all_routes.append(r)
            elif hasattr(r, "routes"):
                all_routes.extend(r.routes)
        ws_routes = [r for r in all_routes if getattr(r, "path", None) == "/ws"]
        self.assertTrue(len(ws_routes) > 0)


class TestFastAPIEndpoints(unittest.TestCase):
    """Integration tests for REST API endpoints using TestClient."""

    @classmethod
    def setUpClass(cls):
        try:
            from fastapi.testclient import TestClient
            from interface.server import create_app
            from interface.api import set_bridge

            # Create a mock bridge
            cls.mock_bridge = MagicMock()
            cls.mock_bridge.get_status.return_value = {
                "version": "0.9.0",
                "pipeline_running": True,
                "source": "synthetic",
                "tracking_enabled": True,
                "sahi_enabled": False,
                "ocr_enabled": True,
                "rag_enabled": True,
                "memory_enabled": True,
                "ann_version": "ann_v1",
                "rag_document_count": 2,
            }
            cls.mock_bridge.get_telemetry.return_value = {
                "fps": 30.0,
                "inference_latency_ms": 25.0,
                "frame_count": 100,
                "active_tracks": 3,
                "detection_count": 5,
                "ocr_text_count": 1,
                "sahi_enabled": False,
                "tracking_enabled": True,
                "ann_version": "ann_v1",
                "memory_enabled": True,
                "rag_enabled": True,
                "timestamp": time.time(),
            }
            cls.mock_bridge.get_scene.return_value = {
                "timestamp": time.time(),
                "frame_index": 100,
                "entity_count": 2,
                "entities": [],
                "texts": [],
                "relations": [],
                "summary": "I see 1 person, 1 laptop.",
                "frame_shape": [480, 640],
            }
            cls.mock_bridge.get_detections.return_value = []
            cls.mock_bridge.send_chat.return_value = {
                "query": "what do you see?",
                "intent": "scene_summary",
                "response_text": "I see 1 person.",
                "sources": ["context_manager"],
            }
            cls.mock_bridge.search_rag.return_value = {
                "query": "laptop manual",
                "count": 1,
                "documents": [],
                "scores": [],
            }
            cls.mock_bridge.search_memory.return_value = {
                "found": True,
                "description": "Notebook was last seen in center.",
                "event": None,
            }
            cls.mock_bridge.get_memory_history.return_value = []
            cls.mock_bridge.get_rag_documents.return_value = []
            cls.mock_bridge.detector = MagicMock()
            cls.mock_bridge.detector.sahi_config = MagicMock()
            cls.mock_bridge.detector.sahi_config.enabled = False
            cls.mock_bridge._tracking_enabled = True

            set_bridge(cls.mock_bridge)
            app = create_app(bridge=cls.mock_bridge)
            cls.client = TestClient(app)
            cls.has_testclient = True
        except ImportError:
            cls.has_testclient = False

    def setUp(self):
        if not self.has_testclient:
            self.skipTest("FastAPI TestClient not available")

    def test_get_status(self):
        resp = self.client.get("/api/status")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["version"], "0.9.0")
        self.assertTrue(data["pipeline_running"])

    def test_get_telemetry(self):
        resp = self.client.get("/api/telemetry")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("fps", data)
        self.assertIn("inference_latency_ms", data)

    def test_get_scene(self):
        resp = self.client.get("/api/scene")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("entities", data)
        self.assertIn("summary", data)

    def test_post_chat(self):
        resp = self.client.post("/api/chat", json={"query": "what do you see?"})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("response_text", data)

    def test_post_chat_empty_query(self):
        resp = self.client.post("/api/chat", json={"query": ""})
        self.assertEqual(resp.status_code, 422)

    def test_post_rag_search(self):
        resp = self.client.post("/api/rag/search", json={"query": "laptop manual"})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("documents", data)

    def test_post_memory_search(self):
        resp = self.client.post("/api/memory/search", json={"query": "notebook"})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("found", data)
        self.assertIn("description", data)

    def test_get_memory_history(self):
        resp = self.client.get("/api/memory/history?limit=10")
        self.assertEqual(resp.status_code, 200)

    def test_get_rag_documents(self):
        resp = self.client.get("/api/rag/documents")
        self.assertEqual(resp.status_code, 200)

    def test_get_config(self):
        resp = self.client.get("/api/config")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("sahi_enabled", data)
        self.assertIn("tracking_enabled", data)

    def test_get_detections(self):
        resp = self.client.get("/api/detections")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("detections", data)


class TestBridgePerceptionControls(unittest.TestCase):
    """Tests for Gesture Master Armed Switch, OCR Cache Flushing, and Target Filter Modes."""

    @patch("interface.bridge.GestureActionController")
    @patch("interface.bridge.ObjectDetector")
    @patch("interface.bridge.ReliabilityInference")
    @patch("interface.bridge.RAGEngine")
    @patch("interface.bridge.EpisodicMemory")
    def setUp(self, mock_mem, mock_rag, mock_ann, mock_det, mock_gest):
        from interface.bridge import AuraBridge
        self.bridge = AuraBridge(
            source="synthetic",
            enable_ocr=True,
            enable_gestures=False,
            enable_rag=False,
            enable_memory=False,
        )

    def test_gestures_toggle_and_set(self):
        self.assertFalse(self.bridge._gestures_enabled)
        res = self.bridge.toggle_gestures()
        self.assertTrue(res)
        self.assertTrue(self.bridge._gestures_enabled)

        res2 = self.bridge.set_gestures(False)
        self.assertFalse(res2)
        self.assertFalse(self.bridge._gestures_enabled)

    def test_ocr_disabled_flushes_cache(self):
        from ocr.engine import TextDetection
        self.bridge._last_ocr_texts = [
            TextDetection(text="SAMPLE", confidence=0.9, bbox=[10, 10, 50, 50])
        ]
        self.assertTrue(self.bridge._ocr_enabled)
        self.assertEqual(len(self.bridge._last_ocr_texts), 1)

        # Toggle OCR OFF -> cache must be immediately emptied!
        res = self.bridge.toggle_ocr()
        self.assertFalse(res)
        self.assertFalse(self.bridge._ocr_enabled)
        self.assertEqual(len(self.bridge._last_ocr_texts), 0)

    def test_target_filter_objects_only(self):
        from vision.detector import Detection
        from interface.bridge import TargetFilterMode
        
        self.bridge.set_target_filter_mode("OBJECTS_ONLY")
        self.assertEqual(self.bridge._target_filter_mode, TargetFilterMode.OBJECTS_ONLY)

        raw_dets = [
            Detection(class_id=0, class_name="person", confidence=0.9, bbox=[0, 0, 10, 10]),
            Detection(class_id=1, class_name="human face", confidence=0.9, bbox=[0, 0, 10, 10]),
            Detection(class_id=2, class_name="human hand", confidence=0.9, bbox=[0, 0, 10, 10]),
            Detection(class_id=3, class_name="open palm", confidence=0.9, bbox=[0, 0, 10, 10]),
            Detection(class_id=4, class_name="pointing hand", confidence=0.9, bbox=[0, 0, 10, 10]),
            Detection(class_id=5, class_name="thumbs up", confidence=0.9, bbox=[0, 0, 10, 10]),
            Detection(class_id=6, class_name="laptop", confidence=0.95, bbox=[10, 10, 50, 50]),
            Detection(class_id=7, class_name="cup", confidence=0.85, bbox=[20, 20, 30, 30]),
            Detection(class_id=8, class_name="water bottle", confidence=0.85, bbox=[20, 20, 30, 30]),
        ]

        filtered = self.bridge._apply_target_filter(raw_dets)
        self.assertEqual(len(filtered), 3)
        names = [d.class_name for d in filtered]
        self.assertIn("laptop", names)
        self.assertIn("cup", names)
        self.assertIn("water bottle", names)
        self.assertNotIn("person", names)
        self.assertNotIn("human face", names)
        self.assertNotIn("human hand", names)
        self.assertNotIn("open palm", names)
        self.assertNotIn("pointing hand", names)
        self.assertNotIn("thumbs up", names)

    def test_target_filter_humans_only(self):
        from vision.detector import Detection
        from interface.bridge import TargetFilterMode

        self.bridge.set_target_filter_mode("HUMANS_ONLY")
        self.assertEqual(self.bridge._target_filter_mode, TargetFilterMode.HUMANS_ONLY)

        raw_dets = [
            Detection(class_id=0, class_name="person", confidence=0.9, bbox=[0, 0, 10, 10]),
            Detection(class_id=1, class_name="human face", confidence=0.9, bbox=[0, 0, 10, 10]),
            Detection(class_id=2, class_name="human hand", confidence=0.9, bbox=[0, 0, 10, 10]),
            Detection(class_id=3, class_name="laptop", confidence=0.95, bbox=[10, 10, 50, 50]),
            Detection(class_id=4, class_name="cup", confidence=0.85, bbox=[20, 20, 30, 30]),
        ]

        filtered = self.bridge._apply_target_filter(raw_dets)
        self.assertEqual(len(filtered), 3)
        names = [d.class_name for d in filtered]
        self.assertIn("person", names)
        self.assertIn("human face", names)
        self.assertIn("human hand", names)
        self.assertNotIn("laptop", names)
        self.assertNotIn("cup", names)

    def test_target_filter_off(self):
        from vision.detector import Detection
        from interface.bridge import TargetFilterMode

        self.bridge.set_target_filter_mode("OFF")
        self.assertEqual(self.bridge._target_filter_mode, TargetFilterMode.OFF)

        raw_dets = [
            Detection(class_id=0, class_name="person", confidence=0.9, bbox=[0, 0, 10, 10]),
            Detection(class_id=1, class_name="laptop", confidence=0.95, bbox=[10, 10, 50, 50]),
        ]

        filtered = self.bridge._apply_target_filter(raw_dets)
        self.assertEqual(len(filtered), 0)

    def test_cycle_target_filter(self):
        from interface.bridge import TargetFilterMode
        self.bridge.set_target_filter_mode("ALL")
        
        mode2 = self.bridge.cycle_target_filter_mode()
        self.assertEqual(mode2, "OBJECTS_ONLY")
        
        mode3 = self.bridge.cycle_target_filter_mode()
        self.assertEqual(mode3, "HUMANS_ONLY")
        
        mode4 = self.bridge.cycle_target_filter_mode()
        self.assertEqual(mode4, "OFF")

        mode5 = self.bridge.cycle_target_filter_mode()
        self.assertEqual(mode5, "ALL")

    def test_inspect_target_glasses_grounded_description(self):
        """Verify inspect_target for glasses yields proper optical eyewear description without person hijacking."""
        result = self.bridge.inspect_target("glasses")
        self.assertEqual(result["target"], "glasses")
        resp_text = result["response"]["response_text"].lower()
        self.assertIn("glasses", resp_text)
        self.assertNotIn("living being", resp_text)
        self.assertNotIn("dynamic posture", resp_text)


class TestInterfacePackageInit(unittest.TestCase):
    """Tests for the interface package init."""

    def test_version(self):
        from interface import __version__
        self.assertEqual(__version__, "0.9.0")


if __name__ == "__main__":
    unittest.main()
