"""
Unit tests for AURA Knowledge Retrieval Engine (Milestone 6).
"""

import time
import unittest
from unittest.mock import patch, MagicMock

from vision.detector import Detection
from ocr.engine import TextDetection
from knowledge.sources import (
    KnowledgeItem,
    CuratedKnowledgeSource,
    WikipediaKnowledgeSource,
)
from knowledge.cache import KnowledgeCache
from knowledge.retriever import KnowledgeRetriever


class TestKnowledgeSources(unittest.TestCase):
    def test_knowledge_item_schema(self):
        """Verify KnowledgeItem properties and serialization."""
        item = KnowledgeItem(
            entity_name="laptop",
            title="Laptop Computer",
            category="Electronics",
            summary="A portable personal computer.",
            details={"ram": "16GB"},
            source="local_curated",
            confidence=1.0,
            url="https://example.com/laptop",
        )

        d = item.to_dict()
        self.assertEqual(d["entity_name"], "laptop")
        self.assertEqual(d["title"], "Laptop Computer")
        self.assertEqual(d["category"], "Electronics")
        self.assertEqual(d["source"], "local_curated")
        self.assertEqual(d["confidence"], 1.0)
        self.assertEqual(d["url"], "https://example.com/laptop")
        self.assertIn("Laptop Computer: A portable personal computer.", item.format_short_text())

    def test_curated_knowledge_source_exact_match(self):
        """Verify CuratedKnowledgeSource provides facts for standard COCO objects."""
        source = CuratedKnowledgeSource()

        laptop_info = source.lookup("laptop")
        self.assertIsNotNone(laptop_info)
        self.assertEqual(laptop_info.title, "Laptop Computer")
        self.assertEqual(laptop_info.source, "local_curated")
        self.assertIn("portable", laptop_info.summary.lower())

        cup_info = source.lookup("cup")
        self.assertIsNotNone(cup_info)
        self.assertEqual(cup_info.title, "Cup / Mug")

        person_info = source.lookup("person")
        self.assertIsNotNone(person_info)
        self.assertEqual(person_info.title, "Human / Person")

    def test_curated_knowledge_source_aliases(self):
        """Verify synonyms and aliases correctly map to canonical keys."""
        source = CuratedKnowledgeSource()

        # "computer" -> laptop
        comp = source.lookup("computer")
        self.assertIsNotNone(comp)
        self.assertEqual(comp.title, "Laptop Computer")

        # "mug" -> cup
        mug = source.lookup("mug")
        self.assertIsNotNone(mug)
        self.assertEqual(mug.title, "Cup / Mug")

        # "smartphone" -> cell phone
        phone = source.lookup("smartphone")
        self.assertIsNotNone(phone)
        self.assertEqual(phone.title, "Smartphone / Cell Phone")

    def test_curated_knowledge_source_missing(self):
        """Verify unknown entity returns None without raising an exception."""
        source = CuratedKnowledgeSource()
        self.assertIsNone(source.lookup("non_existent_alien_device_999"))
        self.assertIsNone(source.lookup(""))

    @patch("urllib.request.urlopen")
    def test_wikipedia_knowledge_source_mock(self, mock_urlopen):
        """Verify WikipediaKnowledgeSource parses standard REST API response."""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.read.return_value = b"""{
            "title": "Quantum Computing",
            "description": "Subfield of computer science",
            "extract": "Quantum computing is a rapidly-emerging technology.",
            "content_urls": {"desktop": {"page": "https://en.wikipedia.org/wiki/Quantum_computing"}}
        }"""
        mock_urlopen.return_value.__enter__.return_value = mock_response

        wiki = WikipediaKnowledgeSource(enabled=True)
        item = wiki.lookup("Quantum Computing")

        self.assertIsNotNone(item)
        self.assertEqual(item.title, "Quantum Computing")
        self.assertEqual(item.source, "wikipedia")
        self.assertIn("rapidly-emerging", item.summary)
        self.assertEqual(item.url, "https://en.wikipedia.org/wiki/Quantum_computing")

    def test_wikipedia_knowledge_source_disabled_or_error(self):
        """Verify disabled source returns None immediately."""
        wiki = WikipediaKnowledgeSource(enabled=False)
        self.assertIsNone(wiki.lookup("Quantum Computing"))


class TestKnowledgeCache(unittest.TestCase):
    def test_cache_set_get_contains(self):
        """Verify basic cache operations."""
        cache = KnowledgeCache(max_size=3, ttl_seconds=10.0)

        cache.set("apple", "A fruit")
        self.assertTrue(cache.contains("apple"))
        self.assertEqual(cache.get("apple"), "A fruit")
        self.assertIsNone(cache.get("banana"))

    def test_cache_lru_eviction(self):
        """Verify oldest unaccessed item is evicted when capacity is reached."""
        cache = KnowledgeCache(max_size=2, ttl_seconds=10.0)

        cache.set("k1", "val1")
        cache.set("k2", "val2")
        self.assertEqual(cache.size, 2)

        # Access k1 to make it most recently used
        _ = cache.get("k1")

        # Insert k3 -> should evict k2
        cache.set("k3", "val3")
        self.assertEqual(cache.size, 2)
        self.assertTrue(cache.contains("k1"))
        self.assertTrue(cache.contains("k3"))
        self.assertFalse(cache.contains("k2"))

    def test_cache_ttl_expiration(self):
        """Verify items expire after TTL."""
        cache = KnowledgeCache(max_size=5, ttl_seconds=0.05)
        cache.set("quick", "expires_fast")
        self.assertEqual(cache.get("quick"), "expires_fast")

        time.sleep(0.06)
        self.assertIsNone(cache.get("quick"))


class TestKnowledgeRetriever(unittest.TestCase):
    def test_retriever_for_entity_and_caching(self):
        """Verify retriever queries sources and populates cache."""
        retriever = KnowledgeRetriever(enable_curated=True, enable_wikipedia=False)

        # First retrieval -> from curated DB
        item1 = retriever.retrieve("laptop")
        self.assertIsNotNone(item1)
        self.assertEqual(item1.title, "Laptop Computer")

        # Second retrieval -> from cache
        self.assertTrue(retriever.cache.contains("laptop"))
        item2 = retriever.retrieve("laptop")
        self.assertEqual(item1, item2)

    def test_retriever_for_detection(self):
        """Verify retriever works directly with Detection object."""
        retriever = KnowledgeRetriever(enable_curated=True, enable_wikipedia=False)
        det = Detection(class_id=67, class_name="cell phone", confidence=0.9, bbox=[0, 0, 50, 100])

        item = retriever.retrieve_for_detection(det)
        self.assertIsNotNone(item)
        self.assertEqual(item.title, "Smartphone / Cell Phone")

    def test_retriever_for_text_ocr(self):
        """Verify entity resolution on OCR text snippets."""
        retriever = KnowledgeRetriever(enable_curated=True, enable_wikipedia=False)
        td = TextDetection(text="AURA Notebook Guide", confidence=0.95, bbox=[10, 10, 100, 50])

        item = retriever.retrieve_for_text(td)
        self.assertIsNotNone(item)
        self.assertEqual(item.title, "Notebook / Notepad")


if __name__ == "__main__":
    unittest.main()
