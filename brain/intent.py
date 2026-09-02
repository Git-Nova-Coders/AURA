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
    MEMORY_SPATIAL = "memory_spatial"
    MEMORY_TEMPORAL = "memory_temporal"
    DOCUMENT_RAG = "document_rag"
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
        "person", "face", "human face", "hand", "human hand", "open palm", "pointing hand", "fist", "thumbs up",
        "laptop", "cell phone", "phone", "smartphone", "mouse", "computer mouse", "keyboard", "book",
        "bottle", "cup", "chair", "tv", "monitor", "backpack", "pen", "pencil", "notebook",
        "clock", "vase", "potted plant", "plant", "dining table", "table", "desk",
        "headphones", "headset", "glasses", "eyeglasses", "eyeglass", "sunglasses", "spectacles",
        "water bottle", "handbag", "wrist watch", "watch", "smartwatch", "it", "that", "this", "object"
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

        # Extract target object (sort longer names first to match 'water bottle' before 'bottle')
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
        control_phrases = ["stop listening", "mute", "unmute", "clear memory", "toggle tracking", "toggle sahi"]
        if any(cp in q for cp in control_phrases):
            return ParsedQuery(
                raw_query=query,
                intent=IntentType.VOICE_CONTROL,
                target_object=target_object,
                spatial_keyword=spatial_keyword,
                track_id=track_id,
            )

        # 2. Episodic Memory Spatial intent ("where was", "where did i leave", "where did i put")
        mem_spatial_phrases = [
            "where was", "where did i leave", "where did i put", "where did i place",
            "last seen location", "where was my", "where was the"
        ]
        if any(msp in q for msp in mem_spatial_phrases):
            return ParsedQuery(
                raw_query=query,
                intent=IntentType.MEMORY_SPATIAL,
                target_object=target_object,
                spatial_keyword=spatial_keyword,
                track_id=track_id,
            )

        # 3. Episodic Memory Temporal intent ("when was", "when did", "how long ago")
        mem_temporal_phrases = [
            "when was", "when did", "last time i saw", "how long ago was", "when did someone", "when did a"
        ]
        if any(mtp in q for mtp in mem_temporal_phrases):
            return ParsedQuery(
                raw_query=query,
                intent=IntentType.MEMORY_TEMPORAL,
                target_object=target_object,
                spatial_keyword=spatial_keyword,
                track_id=track_id,
            )

        # 4. Document RAG intent ("how to use", "manual for", "instructions for", "safety guidelines")
        rag_phrases = [
            "how to use", "how do i use", "how do i operate", "user manual", "manual for",
            "safety instructions", "instructions for", "operating instructions", "guidelines for",
            "how to turn on", "how to clean", "ergonomic"
        ]
        if any(rp in q for rp in rag_phrases):
            return ParsedQuery(
                raw_query=query,
                intent=IntentType.DOCUMENT_RAG,
                target_object=target_object,
                spatial_keyword=spatial_keyword,
                track_id=track_id,
            )

        # 5. OCR / Reading intent
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

        # 6. Object Count intent
        count_phrases = ["how many", "count the", "count how many", "number of"]
        if any(cp in q for cp in count_phrases):
            return ParsedQuery(
                raw_query=query,
                intent=IntentType.OBJECT_COUNT,
                target_object=target_object,
                spatial_keyword=spatial_keyword,
                track_id=track_id,
            )

        # 7. Object Location intent
        location_phrases = ["where is", "where are", "locate", "find the", "which side is"]
        if any(lp in q for lp in location_phrases):
            return ParsedQuery(
                raw_query=query,
                intent=IntentType.OBJECT_LOCATION,
                target_object=target_object,
                spatial_keyword=spatial_keyword,
                track_id=track_id,
            )

        # 8. Reliability Check intent
        rel_phrases = ["reliable", "reliability", "how confident", "is that certain", "accuracy"]
        if any(rp in q for rp in rel_phrases):
            return ParsedQuery(
                raw_query=query,
                intent=IntentType.RELIABILITY_CHECK,
                target_object=target_object,
                spatial_keyword=spatial_keyword,
                track_id=track_id,
            )

        # 9. Scene Summary intent
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

        # 10. Object Info / Inspection intent
        info_phrases = [
            "inspect", "inspect this", "inspect the", "what is this", "what is that", "what is",
            "tell me about", "tell me about the", "what's that", "explain the", "explain this", "explain",
            "information about", "info on", "details about", "what kind of",
            "breakdown of", "profile of", "describe the", "describe this", "describe", "what are"
        ]
        if any(ip in q for ip in info_phrases) or (target_object is not None and len(q.split()) <= 4):
            return ParsedQuery(
                raw_query=query,
                intent=IntentType.OBJECT_INFO,
                target_object=target_object,
                spatial_keyword=spatial_keyword,
                track_id=track_id,
            )

        # 11. Default to General Visual QA
        return ParsedQuery(
            raw_query=query,
            intent=IntentType.GENERAL_QA,
            target_object=target_object,
            spatial_keyword=spatial_keyword,
            track_id=track_id,
            confidence=0.8,
        )
