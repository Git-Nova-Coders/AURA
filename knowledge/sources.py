"""
AURA Knowledge Sources Module
Defines the standard KnowledgeItem schema, abstract KnowledgeSource interface,
CuratedKnowledgeSource (offline encyclopedic facts), and WikipediaKnowledgeSource (online REST lookup).
"""

import json
import urllib.parse
import urllib.request
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


@dataclass
class KnowledgeItem:
    """
    Structured representation of retrieved factual information about an object or concept.
    """
    entity_name: str
    title: str
    category: str
    summary: str
    details: Dict[str, Any] = field(default_factory=dict)
    source: str = "local_curated"  # "local_curated", "wikipedia", "ocr_match"
    confidence: float = 1.0
    url: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serializes knowledge item to a dictionary."""
        return {
            "entity_name": self.entity_name,
            "title": self.title,
            "category": self.category,
            "summary": self.summary,
            "details": self.details,
            "source": self.source,
            "confidence": round(self.confidence, 3),
            "url": self.url,
        }

    def format_short_text(self) -> str:
        """Returns a concise one-sentence description suitable for TTS or HUD."""
        return f"{self.title}: {self.summary}"


class KnowledgeSource(ABC):
    """Abstract base class for all knowledge providers."""

    @abstractmethod
    def lookup(self, entity_name: str) -> Optional[KnowledgeItem]:
        """Looks up knowledge for the given entity name."""
        pass


class CuratedKnowledgeSource(KnowledgeSource):
    """
    Built-in offline knowledge repository containing structured facts,
    categories, technical specs, and practical context for common detectable objects.
    """

    CURATED_DB: Dict[str, Dict[str, Any]] = {
        "person": {
            "title": "Human / Person",
            "category": "Living Being / Human",
            "summary": "A human individual exhibiting dynamic posture, movement, and interaction with the environment.",
            "details": {
                "attributes": ["Bipedal", "Social", "Tool user"],
                "interaction": "Can receive voice assistance, visual guidance, and status alerts.",
            },
        },
        "laptop": {
            "title": "Laptop Computer",
            "category": "Electronics / Computing Device",
            "summary": "A portable personal computer with an integrated screen, keyboard, and trackpad designed for mobile computing.",
            "details": {
                "primary_components": ["Display panel", "Keyboard", "Touchpad", "Motherboard", "Battery"],
                "typical_usage": "Software development, data processing, communication, and multimedia consumption.",
                "care": "Keep vents unobstructed to maintain cooling; avoid liquid spills near keyboard.",
            },
        },
        "cell phone": {
            "title": "Smartphone / Cell Phone",
            "category": "Electronics / Mobile Device",
            "summary": "A handheld mobile device combining cellular telephony with modern computing, camera, and internet capabilities.",
            "details": {
                "features": ["Capacitive touchscreen", "Wireless networking", "High-resolution camera sensors"],
                "typical_usage": "Communication, media, navigation, and mobile applications.",
            },
        },
        "phone": {
            "title": "Smartphone / Phone",
            "category": "Electronics / Mobile Device",
            "summary": "A handheld mobile communication and computing device.",
            "details": {
                "typical_usage": "Voice calls, messaging, internet browsing, and productivity.",
            },
        },
        "mouse": {
            "title": "Computer Mouse",
            "category": "Electronics / Computer Peripheral",
            "summary": "A hand-held pointing device that detects two-dimensional motion relative to a surface.",
            "details": {
                "sensor_types": ["Optical LED", "Laser sensor"],
                "interface": ["USB wired", "Bluetooth wireless", "2.4 GHz RF dongle"],
                "function": "Translates physical hand motion into cursor coordinates on a display.",
            },
        },
        "keyboard": {
            "title": "Computer Keyboard",
            "category": "Electronics / Input Peripheral",
            "summary": "A typewriter-style device using an arrangement of buttons or keys to act as mechanical levers or electronic switches.",
            "details": {
                "switch_types": ["Mechanical", "Membrane", "Scissor-switch", "Capacitive"],
                "layouts": ["QWERTY", "AZERTY", "Dvorak", "ANSI", "ISO"],
                "usage": "Text entry, hotkey navigation, and system command execution.",
            },
        },
        "book": {
            "title": "Book / Printed Media",
            "category": "Media / Documentation",
            "summary": "A medium for recording information in the form of writing or images, typically composed of bound paper pages.",
            "details": {
                "formats": ["Hardcover", "Paperback", "Journal", "Textbook"],
                "ocr_support": "Visible text on the cover or pages can be extracted by AURA OCR engine.",
            },
        },
        "bottle": {
            "title": "Bottle / Drink Container",
            "category": "Kitchenware / Container",
            "summary": "A narrow-necked container made of glass, plastic, or metal, used for storing and dispensing liquids.",
            "details": {
                "materials": ["PET Plastic", "Stainless Steel", "Borosilicate Glass", "Aluminium"],
                "common_contents": ["Water", "Beverages", "Cleaning solutions", "Condiments"],
            },
        },
        "cup": {
            "title": "Cup / Mug",
            "category": "Kitchenware / Tableware",
            "summary": "An open-top container used to hold liquids for pouring or drinking, often with a handle.",
            "details": {
                "materials": ["Ceramic", "Glass", "Stainless Steel", "Paper/Plastic"],
                "typical_use": "Hot or cold beverage consumption like coffee, tea, or water.",
            },
        },
        "chair": {
            "title": "Chair / Seating",
            "category": "Furniture",
            "summary": "A piece of furniture designed to accommodate one seated person, typically consisting of a seat, backrest, and legs.",
            "details": {
                "types": ["Office/Ergonomic chair", "Dining chair", "Armchair", "Stool"],
                "ergonomics": "Proper lumbar support promotes healthy posture during extended work sessions.",
            },
        },
        "tv": {
            "title": "Television / Display Monitor",
            "category": "Electronics / Visual Display",
            "summary": "An electronic device with a display screen used for viewing broadcast signals, streaming video, or computer output.",
            "details": {
                "display_tech": ["OLED", "QLED", "IPS LCD", "Mini-LED"],
                "resolutions": ["1080p Full HD", "4K UHD (3840x2160)", "8K UHD"],
            },
        },
        "backpack": {
            "title": "Backpack",
            "category": "Accessories / Luggage",
            "summary": "A fabric sack carried on one's back and secured with two straps that go over the shoulders.",
            "details": {
                "use_cases": ["Carrying laptops, books, personal gear, and travel essentials."],
            },
        },
        "pen": {
            "title": "Pen / Writing Instrument",
            "category": "Stationery / Office Supply",
            "summary": "A common writing instrument used to apply ink to a surface, usually paper, for writing or drawing.",
            "details": {
                "types": ["Ballpoint", "Gel pen", "Fountain pen", "Rollerball"],
            },
        },
        "notebook": {
            "title": "Notebook / Notepad",
            "category": "Stationery / Office Supply",
            "summary": "A book or binder of paper pages, often ruled, used for recording notes, memoranda, or sketches.",
            "details": {
                "bindings": ["Spiral", "Stitched", "Perfect bound", "Loose-leaf"],
                "ocr_support": "Handwritten or printed text can be analyzed using AURA OCR.",
            },
        },
        "clock": {
            "title": "Clock / Timepiece",
            "category": "Household / Instrument",
            "summary": "An instrument used to measure, keep, and indicate time.",
            "details": {
                "display": ["Analog dials", "Digital LED/LCD displays"],
            },
        },
        "vase": {
            "title": "Vase / Decorative Container",
            "category": "Decor / Household",
            "summary": "An open container often used to hold cut flowers or decorative arrangements.",
            "details": {
                "materials": ["Porcelain", "Ceramic", "Glass", "Clay"],
            },
        },
        "potted plant": {
            "title": "Potted Plant / Indoor Plant",
            "category": "Living Organism / Flora",
            "summary": "A plant grown in a container or pot, commonly used for interior decoration and air quality improvement.",
            "details": {
                "care": ["Adequate sunlight", "Regular watering", "Well-draining soil"],
            },
        },
        "dining table": {
            "title": "Dining Table / Desk",
            "category": "Furniture / Workspace",
            "summary": "A flat horizontal surface raised on legs used for dining, writing, or placing computer equipment.",
            "details": {
                "workspace": "Acts as the central spatial surface for object spatial relation queries.",
            },
        },
        "glasses": {
            "title": "Eyeglasses / Optical Glasses",
            "category": "Personal Wear / Eyewear & Optics",
            "summary": "An optical frame bearing lenses worn in front of the eyes for vision correction, reading clarity, blue-light filtering, or physical eye protection.",
            "details": {
                "types": ["Prescription eyeglasses", "Reading spectacles", "Blue-light filtering glasses", "Safety glasses"],
                "primary_components": ["Optical lenses", "Frames", "Nose pads", "Bridge", "Temples / arms"],
                "typical_usage": "Refractive error correction (myopia, hyperopia, astigmatism), reading assistance, and digital screen eye strain reduction.",
                "care": "Clean using microfiber cloth and lens cleaner spray; store in protective rigid case.",
            },
        },
        "sunglasses": {
            "title": "Sunglasses / Tinted Eyewear",
            "category": "Personal Wear / Eye & Sun Protection",
            "summary": "Protective eyewear featuring darkened or polarized lenses designed to shield the eyes from bright sunlight, glare, and harmful UV radiation.",
            "details": {
                "features": ["UV400 protection", "Polarized glare reduction", "Anti-reflective coating"],
                "typical_usage": "Outdoor vision comfort, driving safety, and ocular protection against ultraviolet rays.",
            },
        },
        "headphones": {
            "title": "Headphones / Audio Headset",
            "category": "Electronics / Personal Audio",
            "summary": "A pair of small acoustic transducer drivers worn on or over the ears, allowing private listening to audio, media, and voice communications.",
            "details": {
                "form_factors": ["Over-ear (Circumaural)", "On-ear (Supra-aural)", "In-ear (Earbuds)"],
                "features": ["Active Noise Cancellation (ANC)", "Bluetooth wireless connectivity", "Integrated microphone array"],
                "typical_usage": "High-fidelity media playback, video conferencing, audio monitoring, and hands-free calling.",
            },
        },
        "wrist watch": {
            "title": "Wrist Watch / Timepiece",
            "category": "Personal Accessories / Chronometer",
            "summary": "A wearable precision timepiece mounted on a strap around the wrist, providing continuous timekeeping, scheduling, and biometric health metrics.",
            "details": {
                "types": ["Smartwatch", "Quartz analog", "Mechanical automatic"],
                "features": ["Timekeeping", "Heart rate monitor", "Step counting", "Notifications"],
            },
        },
        "handbag": {
            "title": "Handbag / Personal Carry",
            "category": "Accessories / Personal Carry",
            "summary": "A handled bag or purse used to hold, organize, and transport personal items such as wallets, keys, mobile devices, and documents.",
            "details": {
                "styles": ["Tote bag", "Shoulder bag", "Crossbody", "Clutch"],
                "typical_usage": "Everyday carry, commuting, and secure personal item storage.",
            },
        },
        "human face": {
            "title": "Human Face / Biometric Profile",
            "category": "Biometrics / Facial Anatomy",
            "summary": "The facial region of a person comprising eyes, nose, and mouth, tracked for head pose, expression, and gaze direction.",
            "details": {
                "features": ["Eyes", "Nose", "Mouth", "Eyebrows", "Jawline"],
                "analysis": "Head pose estimation, facial expression recognition, and visual attention tracking.",
            },
        },
        "human hand": {
            "title": "Human Hand / 21-Keypoint Skeleton",
            "category": "Biometrics / Hand Articulation",
            "summary": "A prehensile, multi-fingered extremity located at the end of the human forearm, capable of fine motor manipulation, tool use, and 3D spatial gesture communication.",
            "details": {
                "structure": ["Thumb", "Index finger", "Middle finger", "Ring finger", "Pinky finger", "Palm / Wrist"],
                "gesture_tracking": "Tracked via MediaPipe 21 3D landmarks for real-time raycasting, air-pinch inspection, and tactical mode switching.",
            },
        },
        "water bottle": {
            "title": "Water Bottle / Hydration Container",
            "category": "Kitchenware / Hydration",
            "summary": "A portable container designed to hold water and beverages for hydration on the go.",
            "details": {
                "materials": ["Insulated stainless steel", "BPA-free plastic", "Borosilicate glass"],
                "typical_usage": "Hydration, fitness, travel, and personal beverage transport.",
            },
        },
    }

    # Synonym aliases mapping query terms to canonical database keys
    ALIASES: Dict[str, str] = {
        "smart phone": "cell phone",
        "smartphone": "cell phone",
        "mobile": "cell phone",
        "mobile phone": "cell phone",
        "pc": "laptop",
        "computer": "laptop",
        "screen": "tv",
        "monitor": "tv",
        "mug": "cup",
        "teacup": "cup",
        "flask": "bottle",
        "desk": "dining table",
        "table": "dining table",
        "plant": "potted plant",
        "flowerpot": "potted plant",
        "human": "person",
        "man": "person",
        "woman": "person",
        "kid": "person",
        "eyeglass": "glasses",
        "eyeglasses": "glasses",
        "spectacles": "glasses",
        "spectacle": "glasses",
        "reading glasses": "glasses",
        "sun glasses": "sunglasses",
        "shades": "sunglasses",
        "headset": "headphones",
        "earphones": "headphones",
        "earbuds": "headphones",
        "watch": "wrist watch",
        "smartwatch": "wrist watch",
        "smart watch": "wrist watch",
        "purse": "handbag",
        "bag": "handbag",
        "tote": "handbag",
        "face": "human face",
        "hand": "human hand",
        "computer mouse": "mouse",
    }

    def __init__(self, custom_db: Optional[Dict[str, Dict[str, Any]]] = None):
        self.db = dict(self.CURATED_DB)
        if custom_db:
            self.db.update(custom_db)

    def lookup(self, entity_name: str) -> Optional[KnowledgeItem]:
        """
        Looks up knowledge for the given entity in the curated local repository.
        """
        if not entity_name:
            return None

        clean_name = entity_name.strip().lower()
        # Check alias
        canonical_key = self.ALIASES.get(clean_name, clean_name)

        # 1. Exact match
        if canonical_key in self.db:
            data = self.db[canonical_key]
            return KnowledgeItem(
                entity_name=entity_name,
                title=data["title"],
                category=data["category"],
                summary=data["summary"],
                details=data.get("details", {}),
                source="local_curated",
                confidence=1.0,
            )

        # 2. Substring matching (prioritize longer matching keys first)
        for key in sorted(self.db.keys(), key=len, reverse=True):
            if key == canonical_key or key in canonical_key or canonical_key in key:
                data = self.db[key]
                return KnowledgeItem(
                    entity_name=entity_name,
                    title=data["title"],
                    category=data["category"],
                    summary=data["summary"],
                    details=data.get("details", {}),
                    source="local_curated",
                    confidence=0.85 if key != canonical_key else 1.0,
                )

        return None


class WikipediaKnowledgeSource(KnowledgeSource):
    """
    Online knowledge source fetching concise encyclopedia summaries from the Wikipedia REST API.
    Designed with timeout safeguards and error resilience.
    """

    API_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/"

    def __init__(self, timeout_seconds: float = 3.0, enabled: bool = True):
        self.timeout_seconds = timeout_seconds
        self.enabled = enabled

    def lookup(self, entity_name: str) -> Optional[KnowledgeItem]:
        """
        Queries the Wikipedia REST API for the entity summary.
        """
        if not self.enabled or not entity_name:
            return None

        clean_name = entity_name.strip()
        encoded_title = urllib.parse.quote(clean_name)
        url = f"{self.API_URL}{encoded_title}"

        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "AURA-Visual-Assistant/0.6 (academic_research_assistant)"},
            )
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as response:
                if response.status != 200:
                    return None
                data = json.loads(response.read().decode("utf-8"))

            title = data.get("title", clean_name)
            extract = data.get("extract", "")
            description = data.get("description", "Wikipedia Article")
            page_url = data.get("content_urls", {}).get("desktop", {}).get("page", None)

            if not extract:
                return None

            return KnowledgeItem(
                entity_name=entity_name,
                title=title,
                category=description,
                summary=extract,
                details={"source_type": "online_encyclopedia"},
                source="wikipedia",
                confidence=0.90,
                url=page_url,
            )

        except Exception as e:
            logger.debug(f"Wikipedia lookup for '{entity_name}' failed or timed out: {e}")
            return None
