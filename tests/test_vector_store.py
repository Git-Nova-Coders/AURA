"""
Unit tests for AURA VectorStore and text chunking / cosine similarity retrieval.
"""

import os
import shutil
import tempfile
import unittest

from knowledge.vector_store import VectorStore, VectorDocument


class TestVectorStore(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.store = VectorStore(chunk_size=50, chunk_overlap=10)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_add_text_and_chunking(self):
        """Verify text chunking and vector index creation."""
        text = "The quick brown fox jumps over the lazy dog. " * 20
        docs = self.store.add_text("Fox Story", text, category="stories")

        self.assertGreater(len(docs), 1)
        self.assertEqual(self.store.count, len(docs))
        self.assertIn("fox", self.store.vocab)

    def test_cosine_similarity_search(self):
        """Verify semantic retrieval ranks relevant documents highest."""
        self.store.add_text(
            "Laptop Manual",
            "Press the power button on the top right to turn on the laptop workstation.",
            category="manuals",
        )
        self.store.add_text(
            "Coffee Machine Guide",
            "Fill water tank and insert coffee capsule to brew espresso.",
            category="manuals",
        )

        results = self.store.search("how to turn on laptop", top_k=1)
        self.assertEqual(len(results), 1)
        top_doc, score = results[0]
        self.assertEqual(top_doc.title, "Laptop Manual")
        self.assertGreater(score, 0.20)

    def test_save_and_load_persistence(self):
        """Verify disk JSON serialization and deserialization."""
        self.store.add_text("Safety Guide", "Always wear safety glasses in the workshop.", category="safety")
        save_path = os.path.join(self.temp_dir, "vector_index.json")
        self.store.save(save_path)

        self.assertTrue(os.path.exists(save_path))

        new_store = VectorStore()
        success = new_store.load(save_path)
        self.assertTrue(success)
        self.assertEqual(new_store.count, self.store.count)

        res = new_store.search("safety glasses", top_k=1)
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0][0].title, "Safety Guide")


if __name__ == "__main__":
    unittest.main()
