"""
AURA Context Manager Module (Milestone 6)
Maintains real-time scene state, object entity histories, spatial relationships,
and pronoun/reference resolution ('it', 'that object', 'the cup on the left').
"""

import time
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any, Union
from vision.detector import Detection
from ocr.engine import TextDetection

logger = logging.getLogger(__name__)


@dataclass
class ObjectEntity:
    """
    Rich state representation of an observed object over time.
    """
    entity_id: str
    track_id: Optional[int]
    class_id: int
    class_name: str
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2]
    spatial_pos: str  # e.g., "center", "left", "right", "top-left", etc.
    reliability_score: Optional[float] = None
    reliability_label: Optional[str] = None
    texts: List[TextDetection] = field(default_factory=list)
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    frames_seen: int = 1
    is_active: bool = True

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.bbox[0] + self.bbox[2]) / 2.0, (self.bbox[1] + self.bbox[3]) / 2.0)

    @property
    def width(self) -> float:
        return max(0.0, self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> float:
        return max(0.0, self.bbox[3] - self.bbox[1])

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def text_content(self) -> str:
        """Concatenates all recognized text on this object."""
        return " ".join(t.text for t in self.texts) if self.texts else ""

    def to_dict(self) -> Dict[str, Any]:
        """Serializes ObjectEntity to dictionary."""
        return {
            "entity_id": self.entity_id,
            "track_id": self.track_id,
            "class_name": self.class_name,
            "confidence": round(self.confidence, 3),
            "bbox": [round(c, 1) for c in self.bbox],
            "spatial_pos": self.spatial_pos,
            "reliability_score": round(self.reliability_score, 3) if self.reliability_score is not None else None,
            "reliability_label": self.reliability_label,
            "texts": [t.to_dict() for t in self.texts],
            "frames_seen": self.frames_seen,
            "is_active": self.is_active,
        }

    def describe(self) -> str:
        """Returns a natural language description of this entity."""
        rel_str = f" ({self.reliability_label} reliability)" if self.reliability_label else ""
        text_str = f" with visible text '{self.text_content}'" if self.text_content else ""
        track_str = f" (Track #{self.track_id})" if self.track_id is not None else ""
        return f"{self.class_name}{track_str} located in the {self.spatial_pos}{rel_str}{text_str}"


@dataclass
class SpatialRelation:
    """Represents a spatial relationship between two object entities."""
    subject_name: str
    subject_id: str
    relation: str  # "left_of", "right_of", "above", "below", "near"
    target_name: str
    target_id: str

    def to_sentence(self) -> str:
        rel_map = {
            "left_of": "to the left of",
            "right_of": "to the right of",
            "above": "above",
            "below": "below",
            "near": "near",
        }
        human_rel = rel_map.get(self.relation, self.relation)
        return f"The {self.subject_name} is {human_rel} the {self.target_name}."


@dataclass
class SceneContext:
    """
    Consolidated contextual snapshot of the visual scene for reasoning modules.
    """
    timestamp: float
    frame_index: int
    entities: List[ObjectEntity]
    all_texts: List[TextDetection] = field(default_factory=list)
    spatial_relations: List[SpatialRelation] = field(default_factory=list)
    frame_shape: Tuple[int, int] = (480, 640)

    @property
    def num_entities(self) -> int:
        return len(self.entities)

    @property
    def object_counts(self) -> Dict[str, int]:
        """Returns counts for each object class present."""
        counts: Dict[str, int] = {}
        for e in self.entities:
            counts[e.class_name] = counts.get(e.class_name, 0) + 1
        return counts

    def summary(self) -> str:
        """Generates a concise natural-language summary of the visible scene."""
        if not self.entities:
            if self.all_texts:
                texts_str = ", ".join(f'"{t.text}"' for t in self.all_texts[:5])
                return f"I don't see distinct objects, but I detect visible text: {texts_str}."
            return "The scene is currently empty; no recognizable objects detected."

        counts = self.object_counts
        items = [f"{count} {name}{'s' if count > 1 and not name.endswith('s') else ''}" for name, count in counts.items()]
        items_str = ", ".join(items)

        text_extra = ""
        if self.all_texts:
            text_snippets = [f'"{t.text}"' for t in self.all_texts[:4]]
            text_extra = f" I also read text: {', '.join(text_snippets)}."

        relations_str = ""
        if self.spatial_relations:
            relations_str = f" {' '.join(r.to_sentence() for r in self.spatial_relations[:3])}"

        return f"I see {items_str}.{text_extra}{relations_str}"


class ContextManager:
    """
    Manages temporal scene history, entity tracking, spatial relations,
    conversation memory, and reference resolution.
    """

    def __init__(
        self,
        max_entity_history: int = 50,
        max_conversation_turns: int = 20,
        inactive_timeout_seconds: float = 5.0,
    ):
        self.max_entity_history = max_entity_history
        self.max_conversation_turns = max_conversation_turns
        self.inactive_timeout = inactive_timeout_seconds

        self.entities: Dict[str, ObjectEntity] = {}
        self.latest_scene: Optional[SceneContext] = None
        self.conversation_history: List[Dict[str, str]] = []
        self._last_referenced_entity: Optional[ObjectEntity] = None
        self._frame_count = 0

    def compute_spatial_position(self, bbox: List[float], img_w: int, img_h: int) -> str:
        """Calculates human-readable 3x3 grid spatial position from bbox."""
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0

        # Horizontal
        if cx < img_w * 0.33:
            h_pos = "left"
        elif cx > img_w * 0.66:
            h_pos = "right"
        else:
            h_pos = "center"

        # Vertical
        if cy < img_h * 0.33:
            v_pos = "top"
        elif cy > img_h * 0.66:
            v_pos = "bottom"
        else:
            v_pos = "middle"

        if h_pos == "center" and v_pos == "middle":
            return "center"
        if v_pos == "middle":
            return h_pos
        if h_pos == "center":
            return v_pos
        return f"{v_pos}-{h_pos}"

    def compute_spatial_relations(self, entities: List[ObjectEntity]) -> List[SpatialRelation]:
        """Calculates pair-wise spatial relationships between observed entities."""
        relations: List[SpatialRelation] = []
        n = len(entities)
        if n < 2:
            return relations

        for i in range(n):
            for j in range(i + 1, n):
                e1, e2 = entities[i], entities[j]
                c1, c2 = e1.center, e2.center
                dx = c2[0] - c1[0]
                dy = c2[1] - c1[1]

                # Horizontal dominance
                if abs(dx) > abs(dy) * 1.2 and abs(dx) > 60:
                    if dx > 0:
                        relations.append(SpatialRelation(e1.class_name, e1.entity_id, "left_of", e2.class_name, e2.entity_id))
                    else:
                        relations.append(SpatialRelation(e1.class_name, e1.entity_id, "right_of", e2.class_name, e2.entity_id))
                # Vertical dominance
                elif abs(dy) > abs(dx) * 1.2 and abs(dy) > 50:
                    if dy > 0:
                        relations.append(SpatialRelation(e1.class_name, e1.entity_id, "above", e2.class_name, e2.entity_id))
                    else:
                        relations.append(SpatialRelation(e1.class_name, e1.entity_id, "below", e2.class_name, e2.entity_id))

        return relations

    def update(
        self,
        detections: List[Detection],
        text_detections: Optional[List[TextDetection]] = None,
        object_texts: Optional[Dict[int, List[TextDetection]]] = None,
        frame_shape: Tuple[int, int] = (480, 640),
    ) -> SceneContext:
        """
        Updates the context state with new perception inputs for the current frame.
        """
        self._frame_count += 1
        now = time.time()
        img_h, img_w = frame_shape[:2]
        current_entities: List[ObjectEntity] = []

        # Mark all entities temporarily inactive
        for e in self.entities.values():
            if now - e.last_seen > self.inactive_timeout:
                e.is_active = False

        for idx, det in enumerate(detections):
            entity_key = f"track_{det.track_id}" if det.track_id is not None else f"det_{det.class_name}_{idx}"
            spatial_pos = self.compute_spatial_position(det.bbox, img_w, img_h)

            matched_texts = []
            if object_texts:
                det_key = det.track_id if det.track_id is not None else idx
                matched_texts = object_texts.get(det_key, [])

            if entity_key in self.entities:
                # Update existing entity
                entity = self.entities[entity_key]
                entity.confidence = det.confidence
                entity.bbox = det.bbox
                entity.spatial_pos = spatial_pos
                entity.reliability_score = det.reliability_score
                entity.reliability_label = det.reliability_label
                if matched_texts:
                    entity.texts = matched_texts
                entity.last_seen = now
                entity.frames_seen += 1
                entity.is_active = True
            else:
                # Create new entity
                entity = ObjectEntity(
                    entity_id=entity_key,
                    track_id=det.track_id,
                    class_id=det.class_id,
                    class_name=det.class_name,
                    confidence=det.confidence,
                    bbox=det.bbox,
                    spatial_pos=spatial_pos,
                    reliability_score=det.reliability_score,
                    reliability_label=det.reliability_label,
                    texts=matched_texts,
                    first_seen=now,
                    last_seen=now,
                    frames_seen=1,
                    is_active=True,
                )
                self.entities[entity_key] = entity

            current_entities.append(entity)

        # Prune old inactive entities if buffer exceeds max size
        if len(self.entities) > self.max_entity_history:
            sorted_keys = sorted(self.entities.keys(), key=lambda k: self.entities[k].last_seen)
            for k in sorted_keys[: len(self.entities) - self.max_entity_history]:
                del self.entities[k]

        # Compute spatial relationships
        relations = self.compute_spatial_relations(current_entities)

        # Build SceneContext
        self.latest_scene = SceneContext(
            timestamp=now,
            frame_index=self._frame_count,
            entities=current_entities,
            all_texts=text_detections or [],
            spatial_relations=relations,
            frame_shape=frame_shape,
        )

        return self.latest_scene

    def resolve_reference(self, query: str) -> Optional[ObjectEntity]:
        """
        Resolves ambiguous pronouns and natural phrases to a specific ObjectEntity.
        Handles 'it', 'that', 'the cup on the left', 'track #1', 'the laptop', etc.
        """
        if not self.latest_scene or not self.latest_scene.entities:
            return self._last_referenced_entity

        q = query.lower().strip()
        entities = self.latest_scene.entities

        # 1. Direct Track ID match: "track 2", "#2", "object 2"
        for e in entities:
            if e.track_id is not None:
                if f"track {e.track_id}" in q or f"#{e.track_id}" in q or f"object {e.track_id}" in q:
                    self._last_referenced_entity = e
                    return e

        # 2. Specific spatial reference: "on the left", "in the center", "on the right"
        spatial_keywords = ["left", "right", "center", "top", "bottom", "middle"]
        matched_spatial = [kw for kw in spatial_keywords if kw in q]

        # 3. Class name + spatial reference match: "the cup on the left"
        for e in entities:
            if e.class_name.lower() in q:
                if matched_spatial:
                    for sp in matched_spatial:
                        if sp in e.spatial_pos:
                            self._last_referenced_entity = e
                            return e
                else:
                    self._last_referenced_entity = e
                    return e

        # 4. Spatial-only reference: "the one on the right"
        if matched_spatial:
            for sp in matched_spatial:
                for e in entities:
                    if sp in e.spatial_pos:
                        self._last_referenced_entity = e
                        return e

        # 5. Pronoun reference: "it", "that", "this", "that object"
        pronoun_words = ["it", "this", "that", "the object", "that object", "this object"]
        if any(p in q for p in pronoun_words):
            if self._last_referenced_entity and self._last_referenced_entity.is_active:
                return self._last_referenced_entity
            # Default to dominant / center entity
            center_entities = [e for e in entities if "center" in e.spatial_pos]
            chosen = center_entities[0] if center_entities else entities[0]
            self._last_referenced_entity = chosen
            return chosen

        return None

    def add_conversation_turn(self, role: str, text: str) -> None:
        """Records a user query or assistant response into conversation memory."""
        self.conversation_history.append({
            "role": role,
            "text": text,
            "timestamp": time.time(),
        })
        if len(self.conversation_history) > self.max_conversation_turns:
            self.conversation_history.pop(0)

    def get_recent_turns(self, n: int = 5) -> List[Dict[str, str]]:
        """Returns the most recent n turns of conversation history."""
        return self.conversation_history[-n:]

    def clear(self) -> None:
        """Resets all context and history."""
        self.entities.clear()
        self.latest_scene = None
        self.conversation_history.clear()
        self._last_referenced_entity = None
        self._frame_count = 0
