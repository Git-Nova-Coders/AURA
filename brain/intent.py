"""
AURA Intent Classification Module (Milestone 6)
Parses natural-language user queries into structured intents and targets
(e.g., scene summary, object lookup, spatial locate, text reading, count).
"""

import re
from enum import Enum
from dataclasses import dataclass
from typing import Optional, List, Dict, Any


class IntentType(Enum):
    SCENE_SUMMARY = "scene_summary"
    OBJECT_INFO = "object_info"
    OBJECT_LOCATION = "object_location"
    OBJECT_COUNT = "object_count"
    OCR_READ = "ocr_read"
    RELIABILITY_CHECK = "reliability_check"
    GENERAL_QA = "general_qa"
    VOICE_CONTROL = "voice_control"


@dataclass
class ParsedQuery:
    """Structured representation of a parsed natural-language user question."""
    raw_query: str
    intent: IntentType
    target_object: Optional[str] = None
    spatial_keyword: Optional[str] = None
    track_id: Optional[int] = None
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw_query": self.raw_query,
            "intent": self.intent.value,
            "target_object": self.target_object,
            "spatial_keyword": self.spatial_keyword,
            "track_id": self.track_id,
            "confidence": round(self.confidence, 2),
        }


class IntentClassifier:
    """
    Fast rule- and pattern-based intent classifier for real-time assistant queries.
    """

    KNOWN_OBJECTS = [
        "person", "laptop", "cell phone", "phone", "mouse", "keyboard", "book",
        "bottle", "cup", "chair", "tv", "monitor", "backpack", "pen", "notebook",
        "clock", "vase", "potted plant", "plant", "dining table", "table", "desk",
        "it", "that", "this", "object"
    ]

    def classify(self, query: str) -> ParsedQuery:
        """
        Parses a query into an IntentType, target entity, and parameters.
        """
        q = query.lower().strip()

        # Extract track id if present (#1, track 2, object 3)
        track_match = re.search(r"(?:track|#|object)\s*(\d+)", q)
        track_id = int(track_match.group(1)) if track_match else None

        # Extract spatial keywords
        spatial_kws = ["left", "right", "center", "middle", "top", "bottom"]
        spatial_keyword = next((kw for kw in spatial_kws if re.search(r"\b" + kw + r"\b", q)), None)

        # Extract target object (sort longer names first to match 'cell phone' before 'phone')
        target_object = None
        if "people" in q:
            target_object = "person"
        else:
            for obj in sorted(self.KNOWN_OBJECTS, key=len, reverse=True):
                pattern = r"\b" + re.escape(obj) + r"(?:s|es)?\b"
                if re.search(pattern, q):
                    target_object = obj
                    break

        # 1. Voice control intent
        control_phrases = ["stop listening", "mute", "unmute", "clear memory", "toggle tracking"]
        if any(cp in q for cp in control_phrases):
            return ParsedQuery(
                raw_query=query,
                intent=IntentType.VOICE_CONTROL,
                target_object=target_object,
                spatial_keyword=spatial_keyword,
                track_id=track_id,
            )

        # 2. OCR / Reading intent
        ocr_phrases = [
            "read", "what does the text say", "read text", "read the words",
            "what is written", "read the book", "read the title", "transcribe"
        ]
        if any(op in q for op in ocr_phrases):
            return ParsedQuery(
                raw_query=query,
                intent=IntentType.OCR_READ,
                target_object=target_object,
                spatial_keyword=spatial_keyword,
                track_id=track_id,
            )

        # 3. Object Count intent
        count_phrases = ["how many", "count the", "count how many", "number of"]
        if any(cp in q for cp in count_phrases):
            return ParsedQuery(
                raw_query=query,
                intent=IntentType.OBJECT_COUNT,
                target_object=target_object,
                spatial_keyword=spatial_keyword,
                track_id=track_id,
            )

        # 4. Object Location intent
        location_phrases = ["where is", "where are", "locate", "find the", "which side is"]
        if any(lp in q for lp in location_phrases):
            return ParsedQuery(
                raw_query=query,
                intent=IntentType.OBJECT_LOCATION,
                target_object=target_object,
                spatial_keyword=spatial_keyword,
                track_id=track_id,
            )

        # 5. Reliability Check intent
        rel_phrases = ["reliable", "reliability", "how confident", "is that certain", "accuracy"]
        if any(rp in q for rp in rel_phrases):
            return ParsedQuery(
                raw_query=query,
                intent=IntentType.RELIABILITY_CHECK,
                target_object=target_object,
                spatial_keyword=spatial_keyword,
                track_id=track_id,
            )

        # 6. Scene Summary intent
        summary_phrases = [
            "what do you see", "describe the scene", "what is in front of me",
            "what is visible", "give me a summary", "what's here", "look around",
            "overview"
        ]
        if any(sp in q for sp in summary_phrases):
            return ParsedQuery(
                raw_query=query,
                intent=IntentType.SCENE_SUMMARY,
                target_object=target_object,
                spatial_keyword=spatial_keyword,
                track_id=track_id,
            )

        # 7. Object Info intent
        info_phrases = [
            "what is this", "what is that", "tell me about", "what's that",
            "explain the", "information about", "info on", "details about", "what kind of"
        ]
        if any(ip in q for ip in info_phrases) or (target_object is not None and len(q.split()) <= 4):
            return ParsedQuery(
                raw_query=query,
                intent=IntentType.OBJECT_INFO,
                target_object=target_object,
                spatial_keyword=spatial_keyword,
                track_id=track_id,
            )

        # 8. Default to General Visual QA
        return ParsedQuery(
            raw_query=query,
            intent=IntentType.GENERAL_QA,
            target_object=target_object,
            spatial_keyword=spatial_keyword,
            track_id=track_id,
            confidence=0.8,
        )
