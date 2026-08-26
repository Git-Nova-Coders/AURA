"""
Unit tests for AURA RAGEngine.
"""

import os
import unittest

from config.config import RAGConfig
from knowledge.rag import RAGEngine, RAGResult
from vision.detector import Detection


class TestRAGEngine(unittest.TestCase):
    def setUp(self):
        self.config = RAGConfig(
            enabled=True,
            docs_directory="data/manuals",
            similarity_threshold=0.15,
        )
        self.rag = RAGEngine(config=self.config)
        self.rag.initialize()

    def test_rag_query_manual(self):
        """Verify RAG retrieves laptop workstation guide."""
        res = self.rag.query("How to turn on the laptop power")
        self.assertTrue(res.has_results)
        self.assertIn("Aura Laptop Manual", res.top_document.title)
        self.assertIn("power", res.synthesized_context.lower())

    def test_rag_retrieve_for_detection(self):
        """Verify RAG retrieves manual when given a Detection object."""
        det = Detection(class_id=1, class_name="laptop", confidence=0.9, bbox=[0, 0, 100, 100])
        res = self.rag.retrieve_for_detection(det, user_question="maintenance and care")
        self.assertTrue(res.has_results)
        self.assertIn("laptop", res.synthesized_context.lower())


if __name__ == "__main__":
    unittest.main()
