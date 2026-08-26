"""
AURA Knowledge Cache Module
Provides thread-safe in-memory caching with TTL (Time-To-Live) and LRU eviction
to deliver sub-millisecond retrieval latency for recurring knowledge queries.
"""

import time
import threading
from collections import OrderedDict
from typing import Any, Optional, Dict


class KnowledgeCache:
    """
    Thread-safe in-memory LRU cache with configurable capacity and TTL expiration.
    """

    def __init__(self, max_size: int = 256, ttl_seconds: float = 3600.0):
        self.max_size = max(1, max_size)
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._lock = threading.Lock()

    def _normalize_key(self, key: str) -> str:
        return key.strip().lower()

    def get(self, key: str) -> Optional[Any]:
        """
        Retrieves a cached value if present and not expired.
        Moves the accessed item to the end (most recently used).
        """
        norm_key = self._normalize_key(key)
        with self._lock:
            if norm_key not in self._cache:
                return None

            entry = self._cache[norm_key]
            # Check expiration
            if time.time() - entry["timestamp"] > self.ttl_seconds:
                del self._cache[norm_key]
                return None

            # Mark as recently used
            self._cache.move_to_end(norm_key)
            return entry["value"]

    def set(self, key: str, value: Any) -> None:
        """
        Inserts or updates a cache entry. Evicts the oldest item if max_size is exceeded.
        """
        norm_key = self._normalize_key(key)
        with self._lock:
            if norm_key in self._cache:
                self._cache.move_to_end(norm_key)
            elif len(self._cache) >= self.max_size:
                self._cache.popitem(last=False)  # Evict least recently used

            self._cache[norm_key] = {
                "value": value,
                "timestamp": time.time(),
            }

    def contains(self, key: str) -> bool:
        """Checks whether an unexpired key exists in the cache."""
        return self.get(key) is not None

    def clear(self) -> None:
        """Clears all cached entries."""
        with self._lock:
            self._cache.clear()

    @property
    def size(self) -> int:
        """Returns the current number of cached items."""
        with self._lock:
            return len(self._cache)
