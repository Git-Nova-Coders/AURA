"""
Unit tests for AURA Brain, Context Manager, Intent Classifier, and Conversation Engine (Milestone 6).
"""

import os
import sys
import unittest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vision.detector import Detection
from ocr.engine import TextDetection
from brain.context import ContextManager, ObjectEntity, SceneContext, SpatialRelation
from brain.intent import IntentClassifier, IntentType, ParsedQuery
from brain.conversation import ConversationEngine, ConversationResponse
from knowledge.retriever import KnowledgeRetriever


class TestContextManager(unittest.TestCase):
    def test_object_entity_properties(self):
        """Verify ObjectEntity calculations and text representation."""
        entity = ObjectEntity(
            entity_id="det_laptop_0",
            track_id=1,
            class_id=63,
            class_name="laptop",
            confidence=0.92,
            bbox=[100.0, 150.0, 300.0, 350.0],
            spatial_pos="center",
            reliability_score=0.96,
            reliability_label="reliable",
        )

        self.assertEqual(entity.width, 200.0)
        self.assertEqual(entity.height, 200.0)
        self.assertEqual(entity.area, 40000.0)
        self.assertEqual(entity.center, (200.0, 250.0))
        self.assertIn("laptop (Track #1) located in the center", entity.describe())

    def test_spatial_position_grid(self):
        """Verify 3x3 spatial grid mapping."""
        cm = ContextManager()
        # Frame: 640x480
        # Left-Top box: [20, 20, 100, 100]
        pos_tl = cm.compute_spatial_position([20, 20, 100, 100], img_w=640, img_h=480)
        self.assertEqual(pos_tl, "top-left")

        # Center box: [250, 180, 390, 300]
        pos_c = cm.compute_spatial_position([250, 180, 390, 300], img_w=640, img_h=480)
        self.assertEqual(pos_c, "center")

        # Right-Bottom box: [500, 350, 600, 450]
        pos_rb = cm.compute_spatial_position([500, 350, 600, 450], img_w=640, img_h=480)
        self.assertEqual(pos_rb, "bottom-right")

    def test_spatial_relations_calculation(self):
        """Verify relative positioning detection between objects."""
        cm = ContextManager()
        e1 = ObjectEntity("e1", 1, 63, "laptop", 0.9, [50, 200, 200, 350], "left")
        e2 = ObjectEntity("e2", 2, 64, "mouse", 0.85, [400, 200, 500, 300], "right")

        relations = cm.compute_spatial_relations([e1, e2])
        self.assertEqual(len(relations), 1)
        self.assertEqual(relations[0].relation, "left_of")
        self.assertEqual(relations[0].subject_name, "laptop")
        self.assertEqual(relations[0].target_name, "mouse")
        self.assertIn("The laptop is to the left of the mouse.", relations[0].to_sentence())

    def test_reference_resolution(self):
        """Verify reference resolution for pronouns, track numbers, and spatial descriptions."""
        cm = ContextManager()
        dets = [
            Detection(class_id=63, class_name="laptop", confidence=0.95, bbox=[50, 100, 250, 350], track_id=1),
            Detection(class_id=41, class_name="cup", confidence=0.88, bbox=[450, 100, 550, 250], track_id=2),
        ]
        cm.update(dets, frame_shape=(480, 640))

        # 1. Resolve by track ID
        resolved_track = cm.resolve_reference("tell me about track 2")
        self.assertIsNotNone(resolved_track)
        self.assertEqual(resolved_track.class_name, "cup")

        # 2. Resolve by spatial description
        resolved_left = cm.resolve_reference("what is the object on the left?")
        self.assertIsNotNone(resolved_left)
        self.assertEqual(resolved_left.class_name, "laptop")

        # 3. Resolve by pronoun 'it'
        resolved_it = cm.resolve_reference("what can I use it for?")
        self.assertIsNotNone(resolved_it)


class TestIntentClassifier(unittest.TestCase):
    def setUp(self):
        self.classifier = IntentClassifier()

    def test_intent_parsing(self):
        """Verify intent classifier matches user question categories."""
        # Scene summary
        p1 = self.classifier.classify("What do you see in front of me?")
        self.assertEqual(p1.intent, IntentType.SCENE_SUMMARY)

        # Object info
        p2 = self.classifier.classify("Tell me about the laptop")
        self.assertEqual(p2.intent, IntentType.OBJECT_INFO)
        self.assertEqual(p2.target_object, "laptop")

        # Object location
        p3 = self.classifier.classify("Where is my phone located?")
        self.assertEqual(p3.intent, IntentType.OBJECT_LOCATION)
        self.assertEqual(p3.target_object, "phone")

        # Object count
        p4 = self.classifier.classify("How many cups are there?")
        self.assertEqual(p4.intent, IntentType.OBJECT_COUNT)
        self.assertEqual(p4.target_object, "cup")

        # OCR read
        p5 = self.classifier.classify("Read the text on the book")
        self.assertEqual(p5.intent, IntentType.OCR_READ)

        # Reliability check
        p6 = self.classifier.classify("Is the detection of the laptop reliable?")
        self.assertEqual(p6.intent, IntentType.RELIABILITY_CHECK)


class TestConversationEngine(unittest.TestCase):
    def setUp(self):
        self.context = ContextManager()
        self.knowledge = KnowledgeRetriever(enable_curated=True, enable_wikipedia=False)
        self.engine = ConversationEngine(
            context_manager=self.context,
            knowledge_retriever=self.knowledge,
        )

    def test_scene_summary_response(self):
        """Verify scene summary response formatting."""
        dets = [
            Detection(class_id=63, class_name="laptop", confidence=0.92, bbox=[100, 150, 300, 350], track_id=1),
            Detection(class_id=67, class_name="cell phone", confidence=0.85, bbox=[400, 200, 500, 380], track_id=2),
        ]
        self.context.update(dets, frame_shape=(480, 640))

        resp = self.engine.respond("What do you see?")
        self.assertEqual(resp.intent, IntentType.SCENE_SUMMARY)
        self.assertIn("laptop", resp.response_text)
        self.assertIn("cell phone", resp.response_text)

    def test_object_info_grounded_response(self):
        """Verify object info response incorporates curated knowledge facts."""
        dets = [
            Detection(class_id=63, class_name="laptop", confidence=0.95, bbox=[100, 150, 300, 350], track_id=1),
        ]
        self.context.update(dets, frame_shape=(480, 640))

        resp = self.engine.respond("Tell me about the laptop")
        self.assertEqual(resp.intent, IntentType.OBJECT_INFO)
        self.assertIn("laptop", resp.response_text.lower())
        self.assertIn("computing", resp.response_text.lower())
        self.assertIsNotNone(resp.knowledge_item)

    def test_glasses_inspection_not_hijacked_by_person(self):
        """Verify inspecting glasses gives eyeglasses description even if person is in scene center."""
        dets = [
            Detection(class_id=0, class_name="person", confidence=0.90, bbox=[200, 100, 440, 400], track_id=1),
            Detection(class_id=18, class_name="glasses", confidence=0.85, bbox=[300, 120, 360, 160], track_id=2),
        ]
        self.context.update(dets, frame_shape=(480, 640))

        # Query glasses specifically
        resp = self.engine.respond("Describe glasses in detail")
        self.assertEqual(resp.intent, IntentType.OBJECT_INFO)
        self.assertIn("glasses", resp.response_text.lower())
        self.assertNotIn("bipedal", resp.response_text.lower())
        self.assertNotIn("living being", resp.response_text.lower())
        self.assertIsNotNone(resp.knowledge_item)
        self.assertEqual(resp.knowledge_item.title, "Eyeglasses / Optical Glasses")

    def test_object_location_response(self):
        """Verify spatial location reporting."""
        dets = [
            Detection(class_id=41, class_name="cup", confidence=0.89, bbox=[50, 100, 150, 250], track_id=3),
        ]
        self.context.update(dets, frame_shape=(480, 640))

        resp = self.engine.respond("Where is the cup?")
        self.assertEqual(resp.intent, IntentType.OBJECT_LOCATION)
        self.assertIn("cup", resp.response_text)
        self.assertIn("left", resp.response_text)

    def test_ocr_read_response(self):
        """Verify OCR text reading from scene and entity."""
        dets = [
            Detection(class_id=73, class_name="book", confidence=0.90, bbox=[100, 100, 300, 300], track_id=1),
        ]
        texts = [
            TextDetection(text="AURA Architecture", confidence=0.98, bbox=[120, 150, 280, 200]),
        ]
        object_texts = {1: texts}
        self.context.update(dets, text_detections=texts, object_texts=object_texts, frame_shape=(480, 640))

        resp = self.engine.respond("Read what is written on the book")
        self.assertEqual(resp.intent, IntentType.OCR_READ)
        self.assertIn("AURA Architecture", resp.response_text)


if __name__ == "__main__":
    unittest.main()
