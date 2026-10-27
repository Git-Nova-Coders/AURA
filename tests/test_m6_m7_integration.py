"""
End-to-End Integration tests for AURA Milestones 6 & 7:
Perception -> Tracking -> OCR -> ANN -> Context -> Knowledge -> Conversational Reasoning -> Voice
"""

import time
import unittest
import numpy as np

from vision.camera import Frame
from vision.detector import ObjectDetector, Detection
from vision.tracker import ObjectTracker
from vision.features import FeatureBuilder
from ocr.engine import OCREngine, TextDetection
from ann.inference import ReliabilityInference
from knowledge.retriever import KnowledgeRetriever
from brain.context import ContextManager
from brain.conversation import ConversationEngine, IntentType
from voice.engine import VoiceAssistant
from voice.tts import TextToSpeech


class TestMilestone6And7Integration(unittest.TestCase):
    def setUp(self):
        # 1. Perception & Vision Components
        self.detector = ObjectDetector(model_name="yolo11n.pt", confidence_threshold=0.25)
        self.tracker = ObjectTracker(max_age=30, min_hits=1, iou_threshold=0.3)
        self.feature_builder = FeatureBuilder()
        self.reliability_ann = ReliabilityInference(enabled=False)  # Fallback mode for deterministic testing

        # 2. Intelligence & Knowledge Components (Milestone 6)
        self.context_manager = ContextManager()
        self.knowledge_retriever = KnowledgeRetriever(enable_curated=True, enable_wikipedia=False)
        self.conversation_engine = ConversationEngine(
            context_manager=self.context_manager,
            knowledge_retriever=self.knowledge_retriever,
        )

        # 3. Voice Components (Milestone 7)
        self.tts = TextToSpeech(enabled=False)
        self.voice_assistant = VoiceAssistant(
            conversation_engine=self.conversation_engine,
            tts=self.tts,
            enable_tts=False,
            enable_stt=False,
        )

    def tearDown(self):
        self.voice_assistant.shutdown()

    def test_full_m6_m7_end_to_end_multimodal_flow(self):
        """
        Verify complete multimodal pipeline from visual input to conversational reasoning and voice.
        """
        # Step 1: Simulate video frames with synthetic objects
        blank_img = np.zeros((480, 640, 3), dtype=np.uint8)
        frame1 = Frame(image=blank_img, timestamp=time.time(), source_id="synthetic")

        # Injected detections: Laptop on left, Notebook in center, Cup on right
        mock_dets = [
            Detection(class_id=63, class_name="laptop", confidence=0.94, bbox=[50.0, 100.0, 250.0, 350.0]),
            Detection(class_id=73, class_name="notebook", confidence=0.88, bbox=[280.0, 150.0, 380.0, 300.0]),
            Detection(class_id=41, class_name="cup", confidence=0.82, bbox=[450.0, 200.0, 550.0, 380.0]),
        ]

        # Step 2: Multi-Object Tracking
        tracked_dets = self.tracker.update(mock_dets)
        self.assertEqual(len(tracked_dets), 3)
        for d in tracked_dets:
            self.assertIsNotNone(d.track_id)

        # Step 3: Feature Extraction & Reliability Estimation
        tracks_map = {t.track_id: t for t in self.tracker.all_tracks}
        features = self.feature_builder.extract_all(frame1, tracked_dets, tracks_map=tracks_map)
        self.assertEqual(len(features), 3)
        for i, feat in enumerate(features):
            rel = self.reliability_ann.predict(feat)
            tracked_dets[i].reliability_score = rel.score
            tracked_dets[i].reliability_label = rel.label

        # Step 4: OCR Text Extraction
        ocr_texts = [
            TextDetection(text="AURA Architecture Guide", confidence=0.95, bbox=[290.0, 180.0, 370.0, 220.0]),
        ]
        # Match OCR to notebook
        notebook_det = next(d for d in tracked_dets if d.class_name == "notebook")
        object_texts = {notebook_det.track_id: ocr_texts}

        # Step 5: Update Context Manager
        scene = self.context_manager.update(
            detections=tracked_dets,
            text_detections=ocr_texts,
            object_texts=object_texts,
            frame_shape=(480, 640),
        )
        self.assertEqual(scene.num_entities, 3)
        self.assertTrue(len(scene.spatial_relations) > 0)

        # Step 6: Test Conversational Queries (Milestone 6)

        # Query 1: Scene Summary
        r1 = self.voice_assistant.process_text_query("What do you see in front of me?", speak_output=False)
        self.assertEqual(r1.intent, IntentType.SCENE_SUMMARY)
        self.assertIn("laptop", r1.response_text)
        self.assertIn("cup", r1.response_text)
        self.assertIn("notebook", r1.response_text)

        # Query 2: Object Information with Grounded Curated Knowledge
        r2 = self.voice_assistant.process_text_query("Tell me about the laptop", speak_output=False)
        self.assertEqual(r2.intent, IntentType.OBJECT_INFO)
        self.assertIn("laptop", r2.response_text.lower())
        self.assertIn("portable", r2.response_text.lower())
        self.assertIsNotNone(r2.knowledge_item)
        self.assertEqual(r2.knowledge_item.source, "local_curated")

        # Query 3: Object Location
        r3 = self.voice_assistant.process_text_query("Where is the cup located?", speak_output=False)
        self.assertEqual(r3.intent, IntentType.OBJECT_LOCATION)
        self.assertIn("cup", r3.response_text)
        self.assertIn("right", r3.response_text)

        # Query 4: OCR Reading
        r4 = self.voice_assistant.process_text_query("Read the text on the notebook", speak_output=False)
        self.assertEqual(r4.intent, IntentType.OCR_READ)
        self.assertIn("AURA Architecture Guide", r4.response_text)

        # Query 5: Reliability Inspection
        r5 = self.voice_assistant.process_text_query("Is the detection of the laptop reliable?", speak_output=False)
        self.assertEqual(r5.intent, IntentType.RELIABILITY_CHECK)
        self.assertIn("laptop", r5.response_text)

        # Step 7: Verify Conversation Memory (Milestone 7)
        turns = self.context_manager.get_recent_turns(10)
        self.assertEqual(len(turns), 10)  # 5 user questions + 5 assistant answers
        self.assertEqual(turns[0]["role"], "user")
        self.assertEqual(turns[1]["role"], "assistant")


if __name__ == "__main__":
    unittest.main()
