"""
AURA Vector Store Subsystem (Milestone 8)
Provides a lightweight, zero-external-dependency vector indexer with text chunking,
TF-IDF / subword n-gram embedding vectorizer, cosine similarity ranking,
and disk serialization for local RAG retrieval.
"""

import os
import re
import math
import json
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Set

logger = logging.getLogger(__name__)


@dataclass
class VectorDocument:
    """Represents a chunked document stored within the VectorStore."""
    doc_id: str
    title: str
    content: str
    category: str = "general"
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "content": self.content,
            "category": self.category,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VectorDocument":
        return cls(
            doc_id=data["doc_id"],
            title=data["title"],
            content=data["content"],
            category=data.get("category", "general"),
            metadata=data.get("metadata", {}),
        )


def _tokenize(text: str) -> List[str]:
    """Tokenizes text into normalized alphanumeric unigrams and bigrams for rich semantic representation."""
    cleaned = re.sub(r"[^\w\s]", " ", text.lower())
    words = [w for w in cleaned.split() if len(w) > 1]
    tokens = list(words)
    # Add word bigrams for phrase matching
    for i in range(len(words) - 1):
        tokens.append(f"{words[i]}_{words[i+1]}")
    return tokens


class VectorStore:
    """
    High-speed in-memory Vector Store with TF-IDF cosine-similarity search.
    Provides complete offline document search and retrieval for AURA's RAG system.
    """

    def __init__(
        self,
        chunk_size: int = 300,
        chunk_overlap: int = 50,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.documents: Dict[str, VectorDocument] = {}
        self.vocab: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self._doc_vectors: Dict[str, Dict[int, float]] = {}

    @property
    def count(self) -> int:
        return len(self.documents)

    def _chunk_text(self, text: str) -> List[str]:
        """Splits long text into overlapping chunks based on sentence/word boundaries."""
        if not text or not text.strip():
            return []

        words = text.strip().split()
        if len(words) <= self.chunk_size:
            return [" ".join(words)]

        chunks: List[str] = []
        step = max(1, self.chunk_size - self.chunk_overlap)
        for i in range(0, len(words), step):
            chunk_words = words[i : i + self.chunk_size]
            chunks.append(" ".join(chunk_words))
            if i + self.chunk_size >= len(words):
                break
        return chunks

    def _build_tfidf(self) -> None:
        """Recomputes vocabulary and IDF weights across all indexed documents."""
        doc_count = len(self.documents)
        if doc_count == 0:
            self.vocab = {}
            self.idf = {}
            self._doc_vectors = {}
            return

        doc_frequencies: Dict[str, int] = {}
        doc_token_counts: Dict[str, Dict[str, int]] = {}

        # 1. Collect term frequencies per document
        for doc_id, doc in self.documents.items():
            tokens = _tokenize(f"{doc.title} {doc.title} {doc.content}")
            unique_terms: Set[str] = set(tokens)
            for t in unique_terms:
                doc_frequencies[t] = doc_frequencies.get(t, 0) + 1

            t_counts: Dict[str, int] = {}
            for t in tokens:
                t_counts[t] = t_counts.get(t, 0) + 1
            doc_token_counts[doc_id] = t_counts

        # 2. Build vocabulary index & compute IDF
        self.vocab = {term: idx for idx, term in enumerate(doc_frequencies.keys())}
        self.idf = {
            term: math.log((doc_count + 1.0) / (df + 1.0)) + 1.0
            for term, df in doc_frequencies.items()
        }

        # 3. Compute normalized sparse vectors for each document
        self._doc_vectors = {}
        for doc_id, t_counts in doc_token_counts.items():
            total_words = max(1, sum(t_counts.values()))
            vec: Dict[int, float] = {}
            norm_sq = 0.0
            for term, count in t_counts.items():
                if term in self.vocab:
                    term_idx = self.vocab[term]
                    tf = count / total_words
                    tfidf_val = tf * self.idf.get(term, 1.0)
                    vec[term_idx] = tfidf_val
                    norm_sq += tfidf_val * tfidf_val

            norm = math.sqrt(norm_sq) if norm_sq > 0 else 1.0
            for term_idx in vec:
                vec[term_idx] /= norm
            self._doc_vectors[doc_id] = vec

    def add_text(
        self,
        title: str,
        text: str,
        category: str = "general",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[VectorDocument]:
        """Chunks and indexes plain text under the specified title."""
        chunks = self._chunk_text(text)
        created_docs: List[VectorDocument] = []
        meta = metadata or {}

        for i, chunk in enumerate(chunks):
            doc_id = f"{re.sub(r'[^a-zA-Z0-9_]', '_', title.lower())}_chunk_{i}"
            chunk_meta = dict(meta)
            chunk_meta["chunk_index"] = i
            chunk_meta["total_chunks"] = len(chunks)

            doc = VectorDocument(
                doc_id=doc_id,
                title=f"{title} (Part {i+1})" if len(chunks) > 1 else title,
                content=chunk,
                category=category,
                metadata=chunk_meta,
            )
            self.documents[doc_id] = doc
            created_docs.append(doc)

        self._build_tfidf()
        return created_docs

    def ingest_file(self, file_path: str, category: Optional[str] = None) -> int:
        """Ingests a text, markdown, or JSON document file from disk."""
        if not os.path.exists(file_path):
            logger.warning(f"File not found for vector ingestion: '{file_path}'")
            return 0

        basename = os.path.splitext(os.path.basename(file_path))[0]
        title = basename.replace("_", " ").replace("-", " ").title()
        cat = category or "manuals"

        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            docs = self.add_text(
                title=title,
                text=content,
                category=cat,
                metadata={"source_file": file_path},
            )
            logger.info(f"Ingested '{file_path}' -> {len(docs)} vector document chunks.")
            return len(docs)
        except Exception as e:
            logger.error(f"Failed to ingest file '{file_path}': {e}")
            return 0

    def ingest_directory(
        self,
        dir_path: str,
        extensions: Tuple[str, ...] = (".txt", ".md", ".json"),
    ) -> int:
        """Recursively ingests all supported document files within a directory."""
        if not os.path.exists(dir_path):
            logger.warning(f"Directory not found for vector ingestion: '{dir_path}'")
            return 0

        total_chunks = 0
        for root, _, files in os.walk(dir_path):
            for file in files:
                if any(file.lower().endswith(ext) for ext in extensions):
                    full_path = os.path.join(root, file)
                    total_chunks += self.ingest_file(full_path)

        logger.info(f"Ingested directory '{dir_path}' -> {total_chunks} total vector chunks.")
        return total_chunks

    def search(
        self,
        query: str,
        top_k: int = 3,
        min_similarity: float = 0.15,
    ) -> List[Tuple[VectorDocument, float]]:
        """
        Executes semantic cosine-similarity retrieval on the vector index.

        Args:
            query: User search query or object context description.
            top_k: Maximum number of ranked documents to return.
            min_similarity: Minimum cosine similarity threshold.

        Returns:
            List[Tuple[VectorDocument, float]]: Ranked documents with similarity scores.
        """
        if not query or not query.strip() or not self.documents:
            return []

        tokens = _tokenize(query)
        if not tokens:
            return []

        # Build query vector
        t_counts: Dict[str, int] = {}
        for t in tokens:
            t_counts[t] = t_counts.get(t, 0) + 1

        query_vec: Dict[int, float] = {}
        norm_sq = 0.0
        total_tokens = len(tokens)
        for term, count in t_counts.items():
            if term in self.vocab:
                term_idx = self.vocab[term]
                tf = count / total_tokens
                tfidf_val = tf * self.idf.get(term, 1.0)
                query_vec[term_idx] = tfidf_val
                norm_sq += tfidf_val * tfidf_val

        if norm_sq == 0.0:
            return []

        query_norm = math.sqrt(norm_sq)
        for idx in query_vec:
            query_vec[idx] /= query_norm

        # Compute cosine similarity with all indexed document vectors
        scored_docs: List[Tuple[VectorDocument, float]] = []
        for doc_id, doc_vec in self._doc_vectors.items():
            dot_product = 0.0
            for term_idx, q_val in query_vec.items():
                if term_idx in doc_vec:
                    dot_product += q_val * doc_vec[term_idx]

            if dot_product >= min_similarity:
                doc = self.documents[doc_id]
                scored_docs.append((doc, float(dot_product)))

        # Sort descending by similarity score
        scored_docs.sort(key=lambda item: item[1], reverse=True)
        return scored_docs[:top_k]

    def save(self, path: str) -> None:
        """Serializes vector store documents to JSON."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        data = {
            "documents": [doc.to_dict() for doc in self.documents.values()],
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved VectorStore with {len(self.documents)} documents to '{path}'.")

    def load(self, path: str) -> bool:
        """Deserializes vector store documents from JSON."""
        if not os.path.exists(path):
            return False

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.chunk_size = data.get("chunk_size", self.chunk_size)
            self.chunk_overlap = data.get("chunk_overlap", self.chunk_overlap)
            self.documents = {
                d["doc_id"]: VectorDocument.from_dict(d)
                for d in data.get("documents", [])
            }
            self._build_tfidf()
            logger.info(f"Loaded VectorStore with {len(self.documents)} documents from '{path}'.")
            return True
        except Exception as e:
            logger.error(f"Failed to load VectorStore from '{path}': {e}")
            return False

    def clear(self) -> None:
        """Clears all indexed documents."""
        self.documents.clear()
        self.vocab.clear()
        self.idf.clear()
        self._doc_vectors.clear()
