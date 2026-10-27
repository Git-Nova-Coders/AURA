"""
AURA Knowledge Retriever Module (Milestone 6)
Coordinates local curated repositories, online Wikipedia queries, OCR entity resolution,
and in-memory caching to provide grounded facts for detected objects and user questions.
"""

import logging
from typing import List, Optional, Dict, Any, Union
from vision.detector import Detection
from ocr.engine import TextDetection
from .sources import KnowledgeItem, KnowledgeSource, CuratedKnowledgeSource, WikipediaKnowledgeSource
from .cache import KnowledgeCache

logger = logging.getLogger(__name__)


class KnowledgeRetriever:
    """
    Main entry point for knowledge retrieval in AURA.
    Provides sub-millisecond local caching, curated knowledge, and online fallback.
    """

    def __init__(
        self,
        enable_curated: bool = True,
        enable_wikipedia: bool = True,
        cache_size: int = 256,
        cache_ttl_seconds: float = 3600.0,
        wikipedia_timeout: float = 3.0,
        custom_sources: Optional[List[KnowledgeSource]] = None,
    ):
        self.cache = KnowledgeCache(max_size=cache_size, ttl_seconds=cache_ttl_seconds)
        self.sources: List[KnowledgeSource] = []

        # 1. Primary fast offline curated source
        if enable_curated:
            self.curated_source = CuratedKnowledgeSource()
            self.sources.append(self.curated_source)
        else:
            self.curated_source = None

        # 2. Secondary online Wikipedia source
        if enable_wikipedia:
            self.wikipedia_source = WikipediaKnowledgeSource(timeout_seconds=wikipedia_timeout)
            self.sources.append(self.wikipedia_source)
        else:
            self.wikipedia_source = None

        # 3. Custom/plugin sources
        if custom_sources:
            self.sources.extend(custom_sources)

    def retrieve(self, entity_name: str, use_cache: bool = True) -> Optional[KnowledgeItem]:
        """
        Retrieves factual knowledge for a single entity or query term.
        
        Args:
            entity_name: Object class or concept name (e.g. 'laptop', 'coffee mug').
            use_cache: Whether to use cached results.
            
        Returns:
            Optional[KnowledgeItem]: Structured knowledge if found, else None.
        """
        if not entity_name or not entity_name.strip():
            return None

        clean_name = entity_name.strip()

        # Check Cache
        if use_cache:
            cached_item = self.cache.get(clean_name)
            if cached_item is not None:
                return cached_item

        # Query Sources in priority order
        for source in self.sources:
            try:
                item = source.lookup(clean_name)
                if item is not None:
                    if use_cache:
                        self.cache.set(clean_name, item)
                    return item
            except Exception as e:
                logger.warning(f"Error querying knowledge source {source.__class__.__name__}: {e}")

        return None

    def retrieve_for_detection(self, detection: Detection) -> Optional[KnowledgeItem]:
        """
        Retrieves knowledge for a given vision Detection object.
        """
        return self.retrieve(detection.class_name)

    def retrieve_for_detections(
        self,
        detections: List[Detection],
    ) -> Dict[str, KnowledgeItem]:
        """
        Retrieves knowledge for a list of detections, deduplicating by class name.
        Returns a dictionary mapping class_name -> KnowledgeItem.
        """
        results: Dict[str, KnowledgeItem] = {}
        for det in detections:
            if det.class_name not in results:
                item = self.retrieve_for_detection(det)
                if item:
                    results[det.class_name] = item
        return results

    def retrieve_for_text(self, text: Union[str, TextDetection]) -> Optional[KnowledgeItem]:
        """
        Performs entity resolution on OCR text to find relevant facts.
        """
        query_text = text.text if isinstance(text, TextDetection) else text
        if not query_text:
            return None

        # Try exact text lookup
        item = self.retrieve(query_text)
        if item:
            return item

        # Try individual meaningful words if longer phrase
        words = [w for w in query_text.split() if len(w) > 3]
        for w in words:
            item = self.retrieve(w)
            if item:
                return item

        return None

    def clear_cache(self) -> None:
        """Clears knowledge cache."""
        self.cache.clear()
