"""
AURA Retrieval-Augmented Generation (RAG) Engine (Milestone 8)
Links live visual perception with semantic document knowledge (manuals, guidelines, data sheets).
"""

import os
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any

from config.config import RAGConfig
from vision.detector import Detection
from .vector_store import VectorStore, VectorDocument

logger = logging.getLogger(__name__)


@dataclass
class RAGResult:
    """Encapsulates the retrieved document chunks and context for a query."""
    query: str
    documents: List[VectorDocument] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)

    @property
    def has_results(self) -> bool:
        return len(self.documents) > 0

    @property
    def top_document(self) -> Optional[VectorDocument]:
        return self.documents[0] if self.documents else None

    @property
    def synthesized_context(self) -> str:
        """Formats the retrieved document chunks into clean reference text for LLM / Reasoning."""
        if not self.documents:
            return "No relevant documentation found."

        parts = []
        for i, (doc, score) in enumerate(zip(self.documents, self.scores)):
            src = doc.metadata.get("source_file", doc.title)
            parts.append(f"[Document Excerpt {i+1}: {doc.title} (Relevance: {score:.2f})]\n{doc.content}")
        return "\n\n".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "count": len(self.documents),
            "documents": [d.to_dict() for d in self.documents],
            "scores": [round(s, 4) for s in self.scores],
            "synthesized_context": self.synthesized_context,
        }


class RAGEngine:
    """
    Retrieval-Augmented Generation Engine for AURA.
    Manages document ingestion, vector storage, and visual-context grounded document retrieval.
    """

    def __init__(
        self,
        config: Optional[RAGConfig] = None,
        vector_store: Optional[VectorStore] = None,
    ):
        self.config = config or RAGConfig()
        self.vector_store = vector_store or VectorStore(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )
        self._is_initialized = False

    @property
    def is_available(self) -> bool:
        return self.config.enabled

    def initialize(self) -> bool:
        """
        Initializes vector store: loads existing index if present,
        otherwise ingests documents from the configured directory.
        """
        if not self.config.enabled:
            logger.info("RAG Engine is disabled in configuration.")
            return False

        # 1. Try loading cached vector index
        if self.config.vector_index_path and os.path.exists(self.config.vector_index_path):
            success = self.vector_store.load(self.config.vector_index_path)
            if success and self.vector_store.count > 0:
                self._is_initialized = True
                logger.info(f"RAG Engine initialized with {self.vector_store.count} cached document vectors.")
                return True

        # 2. Ingest documents directory if available
        if self.config.docs_directory and os.path.exists(self.config.docs_directory):
            logger.info(f"Ingesting RAG documents from '{self.config.docs_directory}'...")
            chunks_count = self.vector_store.ingest_directory(self.config.docs_directory)
            if chunks_count > 0 and self.config.vector_index_path:
                self.vector_store.save(self.config.vector_index_path)
            self._is_initialized = True
            logger.info(f"RAG Engine ingested {chunks_count} chunks from '{self.config.docs_directory}'.")
            return True

        logger.info("RAG Engine initialized with empty vector index.")
        self._is_initialized = True
        return True

    def query(self, query_text: str, top_k: Optional[int] = None) -> RAGResult:
        """
        Retrieves relevant document chunks for a natural language user query.
        """
        if not self.config.enabled or not query_text:
            return RAGResult(query=query_text)

        if not self._is_initialized:
            self.initialize()

        k = top_k or self.config.top_k
        scored = self.vector_store.search(
            query=query_text,
            top_k=k,
            min_similarity=self.config.similarity_threshold,
        )

        docs = [item[0] for item in scored]
        scores = [item[1] for item in scored]

        return RAGResult(query=query_text, documents=docs, scores=scores)

    def retrieve_for_detection(
        self,
        detection: Detection,
        user_question: Optional[str] = None,
    ) -> RAGResult:
        """
        Retrieves relevant manuals/documentation for a specific detected object.
        """
        class_name = detection.class_name
        search_query = f"{class_name} {user_question}" if user_question else f"operating instructions manual {class_name}"
        return self.query(search_query)

    def retrieve_for_scene(
        self,
        scene_summary: str,
        entities: List[str],
    ) -> RAGResult:
        """
        Retrieves documentation relevant to the current visual scene composition.
        """
        query_terms = f"{scene_summary} {' '.join(entities)}"
        return self.query(query_terms)
