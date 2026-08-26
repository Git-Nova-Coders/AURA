"""
AURA Knowledge Package (Milestone 6)
Provides object factual information retrieval, online encyclopedia lookups, and caching.
"""

from .sources import (
    KnowledgeItem,
    KnowledgeSource,
    CuratedKnowledgeSource,
    WikipediaKnowledgeSource,
)
from .cache import KnowledgeCache
from .retriever import KnowledgeRetriever

__all__ = [
    "KnowledgeItem",
    "KnowledgeSource",
    "CuratedKnowledgeSource",
    "WikipediaKnowledgeSource",
    "KnowledgeCache",
    "KnowledgeRetriever",
]
