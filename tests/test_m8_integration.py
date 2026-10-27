"""
End-to-End Integration tests for AURA Milestone 8:
Multimodal Visual Perception -> Context Manager -> Episodic Memory -> RAG Engine -> Conversational LLM Reasoning.
"""

import os
import shutil
import tempfile
import unittest

from config.config import AuraConfig
from vision.detector import Detection
from ocr.engine import TextDetection
from brain.context import ContextManager
from brain.memory import EpisodicMemory
from brain.conversation import ConversationEngine
from brain.intent import IntentType
from knowledge.retriever import KnowledgeRetriever
from knowledge.rag import RAGEngine


class TestMilestone8Integration(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "m8_memory.db")

        self.context_mgr = ContextManager()
        self.knowledge = KnowledgeRetriever(enable_curated=True, enable_wikipedia=False)
        self.memory = EpisodicMemory(config=AuraConfig().memory)
        self.memory.db_path = self.db_path
        self.memory._init_db()

        self.rag = RAGEngine(config=AuraConfig().rag)
        self.rag.initialize()

        self.engine = ConversationEngine(
            context_manager=self.context_mgr,
            knowledge_retriever=self.knowledge,
            rag_engine=self.rag,
            memory=self.memory,
        )

    def tearDown(self):
        self.memory.close()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_full_m8_memory_and_rag_flow(self):
        """Verify memory recall and document RAG integration."""
        # 1. Update scene with a notebook and record to memory
        det = Detection(class_id=2, class_name="notebook", confidence=0.88, bbox=[300, 200, 500, 400])
        td = TextDetection(text="AURA Architecture Guide", confidence=0.96, bbox=[320, 220, 480, 260])
        scene = self.context_mgr.update([det], text_detections=[td], object_texts={0: [td]}, frame_shape=(480, 640))
        self.memory.record_scene(scene, force=True)

        # 2. Clear active scene to simulate object moved/removed
        self.context_mgr.update([], frame_shape=(480, 640))

        # 3. Ask spatial memory question: "Where did I leave my notebook?"
        mem_resp = self.engine.respond("Where did I leave my notebook?")
        self.assertEqual(mem_resp.intent, IntentType.MEMORY_SPATIAL)
        self.assertIn("notebook", mem_resp.response_text.lower())
        self.assertIn("center", mem_resp.response_text.lower())
        self.assertIn("aura architecture guide", mem_resp.response_text.lower())

        # 4. Ask RAG question: "How to turn on the laptop workstation?"
        rag_resp = self.engine.respond("How to turn on the laptop power?")
        self.assertEqual(rag_resp.intent, IntentType.DOCUMENT_RAG)
        self.assertIn("power", rag_resp.response_text.lower())


if __name__ == "__main__":
    unittest.main()
