"""
AURA Persistent Episodic & Spatial Memory Subsystem (Milestone 8)
Maintains historical, temporal, and spatial records of all observed visual entities,
supporting queries such as "Where did I last see my notebook?" or "When was a person last in the room?".
"""

import os
import time
import json
import sqlite3
import logging
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime

from config.config import MemoryConfig
from .context import SceneContext, ObjectEntity

logger = logging.getLogger(__name__)


@dataclass
class EpisodicEvent:
    """Represents an observed historical entity event in episodic memory."""
    id: Optional[int]
    timestamp: float
    class_name: str
    track_id: Optional[int]
    confidence: float
    bbox: List[float]
    spatial_region: str
    associated_text: Optional[str] = None
    reliability_label: Optional[str] = None
    event_type: str = "observed"  # 'appeared', 'observed', 'moved', 'disappeared'

    @property
    def formatted_time(self) -> str:
        return datetime.fromtimestamp(self.timestamp).strftime("%H:%M:%S")

    def time_ago_str(self, current_time: Optional[float] = None) -> str:
        now = current_time or time.time()
        diff_sec = max(0, int(now - self.timestamp))
        if diff_sec < 60:
            return f"{diff_sec} seconds ago"
        elif diff_sec < 3600:
            return f"{diff_sec // 60} minutes ago"
        else:
            return f"{diff_sec // 3600} hours ago"

    def describe(self) -> str:
        text_str = f" with text '{self.associated_text}'" if self.associated_text else ""
        return (
            f"The {self.class_name} was observed in the {self.spatial_region} region "
            f"at {self.formatted_time} ({self.time_ago_str()}){text_str}."
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": round(self.timestamp, 2),
            "formatted_time": self.formatted_time,
            "class_name": self.class_name,
            "track_id": self.track_id,
            "confidence": round(self.confidence, 4),
            "bbox": self.bbox,
            "spatial_region": self.spatial_region,
            "associated_text": self.associated_text,
            "reliability_label": self.reliability_label,
            "event_type": self.event_type,
        }


class EpisodicMemory:
    """
    Persistent SQLite-backed Episodic Memory store for AURA.
    Records spatial and temporal events to enable long-term visual recall.
    """

    def __init__(self, config: Optional[MemoryConfig] = None):
        self.config = config or MemoryConfig()
        self.db_path = self.config.db_path
        self._last_snapshot_time = 0.0
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    def _init_db(self) -> None:
        """Initializes SQLite tables for episodic entity events."""
        if not self.config.enabled:
            return

        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row

        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS episodic_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    class_name TEXT NOT NULL,
                    track_id INTEGER,
                    confidence REAL NOT NULL,
                    bbox_json TEXT NOT NULL,
                    spatial_region TEXT NOT NULL,
                    associated_text TEXT,
                    reliability_label TEXT,
                    event_type TEXT NOT NULL
                )
                """
            )
            self._conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_class_time 
                ON episodic_events(class_name, timestamp DESC)
                """
            )

    def record_scene(self, scene_context: SceneContext, force: bool = False) -> int:
        """
        Records the active entities from the current SceneContext into persistent memory.
        Throttled by snapshot_interval_seconds to avoid excessive DB writes.
        """
        if not self.config.enabled or not self._conn:
            return 0

        now = time.time()
        if not force and (now - self._last_snapshot_time < self.config.snapshot_interval_seconds):
            return 0

        self._last_snapshot_time = now
        records_added = 0

        with self._conn:
            for entity in scene_context.entities:
                texts = [t.text for t in entity.texts] if entity.texts else []
                text_snippet = ", ".join(texts) if texts else None

                self._conn.execute(
                    """
                    INSERT INTO episodic_events (
                        timestamp, class_name, track_id, confidence,
                        bbox_json, spatial_region, associated_text,
                        reliability_label, event_type
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        now,
                        entity.class_name.lower(),
                        entity.track_id,
                        entity.confidence,
                        json.dumps(entity.bbox),
                        entity.spatial_region,
                        text_snippet,
                        entity.reliability_label,
                        "observed",
                    ),
                )
                records_added += 1

        return records_added

    def find_last_seen(self, class_name: str) -> Optional[EpisodicEvent]:
        """
        Finds the most recent recorded observation for a specific entity class.
        """
        if not self.config.enabled or not self._conn:
            return None

        clean_name = class_name.lower().strip()
        cursor = self._conn.cursor()
        cursor.execute(
            """
            SELECT * FROM episodic_events 
            WHERE class_name = ? OR class_name LIKE ?
            ORDER BY timestamp DESC LIMIT 1
            """,
            (clean_name, f"%{clean_name}%"),
        )
        row = cursor.fetchone()
        if not row:
            return None

        return EpisodicEvent(
            id=row["id"],
            timestamp=row["timestamp"],
            class_name=row["class_name"],
            track_id=row["track_id"],
            confidence=row["confidence"],
            bbox=json.loads(row["bbox_json"]),
            spatial_region=row["spatial_region"],
            associated_text=row["associated_text"],
            reliability_label=row["reliability_label"],
            event_type=row["event_type"],
        )

    def get_history(self, class_name: str, limit: int = 5) -> List[EpisodicEvent]:
        """
        Retrieves recent historical observations for a specific object class.
        """
        if not self.config.enabled or not self._conn:
            return []

        clean_name = class_name.lower().strip()
        cursor = self._conn.cursor()
        cursor.execute(
            """
            SELECT * FROM episodic_events 
            WHERE class_name = ? OR class_name LIKE ?
            ORDER BY timestamp DESC LIMIT ?
            """,
            (clean_name, f"%{clean_name}%", limit),
        )
        rows = cursor.fetchall()
        return [
            EpisodicEvent(
                id=r["id"],
                timestamp=r["timestamp"],
                class_name=r["class_name"],
                track_id=r["track_id"],
                confidence=r["confidence"],
                bbox=json.loads(r["bbox_json"]),
                spatial_region=r["spatial_region"],
                associated_text=r["associated_text"],
                reliability_label=r["reliability_label"],
                event_type=r["event_type"],
            )
            for r in rows
        ]

    def query_spatial_memory(self, target_name: str) -> str:
        """
        Answers "Where did I leave my <object>?" grounded in historical visual observations.
        """
        event = self.find_last_seen(target_name)
        if not event:
            return f"I have no recorded visual memory of seeing a {target_name}."

        return (
            f"I last saw your {event.class_name} in the {event.spatial_region} region "
            f"at {event.formatted_time} ({event.time_ago_str()})."
            + (f" It had visible text: '{event.associated_text}'." if event.associated_text else "")
        )

    def query_temporal_memory(self, target_name: str) -> str:
        """
        Answers "When was <object> last seen?" with exact timestamp and time delta.
        """
        event = self.find_last_seen(target_name)
        if not event:
            return f"I have no record of observing a {target_name}."

        return f"A {event.class_name} was last recorded {event.time_ago_str()} (at {event.formatted_time})."

    def clear(self) -> None:
        """Purges all stored episodic memory events."""
        if self._conn:
            with self._conn:
                self._conn.execute("DELETE FROM episodic_events")

    def close(self) -> None:
        """Closes the SQLite connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
