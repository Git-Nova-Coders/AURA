"""
AURA Optical Character Recognition (OCR) Module (Milestone 5)
Extracts visible text, coordinates, and confidence scores from frames and detected objects.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union, Tuple
import numpy as np
import cv2

try:
    import easyocr
    _HAS_EASYOCR = True
except ImportError:
    _HAS_EASYOCR = False

from vision.camera import Frame
from vision.detector import Detection

logger = logging.getLogger(__name__)


@dataclass
class TextDetection:
    """
    Structured representation of a recognized text region.
    """
    text: str
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2] in pixel coordinates
    polygon: Optional[List[List[float]]] = None  # 4 corner points [[x, y], ...]

    @property
    def x1(self) -> float:
        return self.bbox[0]

    @property
    def y1(self) -> float:
        return self.bbox[1]

    @property
    def x2(self) -> float:
        return self.bbox[2]

    @property
    def y2(self) -> float:
        return self.bbox[3]

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def center(self) -> List[float]:
        return [(self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0]

    def to_dict(self) -> Dict[str, Any]:
        """Serializes text detection to JSON-compatible dictionary."""
        return {
            "text": self.text,
            "confidence": round(float(self.confidence), 4),
            "bbox": [round(float(c), 2) for c in self.bbox],
            "polygon": self.polygon,
        }


class OCREngine:
    """
    OCR Engine using EasyOCR with graceful fallback and crop-level text extraction.
    """

    def __init__(
        self,
        languages: Optional[List[str]] = None,
        confidence_threshold: float = 0.3,
        gpu: bool = False,
    ):
        self.languages = languages or ["en"]
        self.confidence_threshold = confidence_threshold
        self.gpu = gpu
        self._reader: Optional[Any] = None
        self._initialized = False

        self._lazy_init()

    def _lazy_init(self) -> None:
        """Initializes the EasyOCR reader model."""
        if not _HAS_EASYOCR:
            logger.warning("EasyOCR is not installed. OCR will operate in fallback mode.")
            return

        try:
            logger.info(f"Initializing EasyOCR reader for languages: {self.languages} (GPU={self.gpu})...")
            self._reader = easyocr.Reader(self.languages, gpu=self.gpu, verbose=False)
            self._initialized = True
            logger.info("EasyOCR reader initialized successfully.")
        except Exception as e:
            logger.warning(f"Failed to initialize EasyOCR reader: {e}. Fallback mode active.")
            self._reader = None
            self._initialized = False

    @property
    def is_available(self) -> bool:
        return self._initialized and self._reader is not None

    def extract_text(
        self,
        image_or_frame: Union[np.ndarray, Frame],
        conf_threshold: Optional[float] = None,
    ) -> List[TextDetection]:
        """
        Extracts all visible text instances from an image.
        
        Args:
            image_or_frame: BGR numpy image or Frame dataclass.
            conf_threshold: Minimum confidence threshold. Defaults to self.confidence_threshold.
            
        Returns:
            List[TextDetection]: List of recognized text detections.
        """
        threshold = conf_threshold if conf_threshold is not None else self.confidence_threshold

        if isinstance(image_or_frame, Frame):
            image = image_or_frame.image
        else:
            image = image_or_frame

        if image is None or image.size == 0 or not self.is_available:
            return []

        try:
            # EasyOCR expects RGB or BGR numpy array
            results = self._reader.readtext(image)
            text_detections: List[TextDetection] = []

            for bbox_polygon, text, conf in results:
                if float(conf) < threshold:
                    continue

                clean_text = text.strip()
                if not clean_text:
                    continue

                # EasyOCR polygon: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
                poly = [[float(pt[0]), float(pt[1])] for pt in bbox_polygon]
                xs = [pt[0] for pt in poly]
                ys = [pt[1] for pt in poly]
                x1, x2 = min(xs), max(xs)
                y1, y2 = min(ys), max(ys)

                text_detections.append(
                    TextDetection(
                        text=clean_text,
                        confidence=float(conf),
                        bbox=[x1, y1, x2, y2],
                        polygon=poly,
                    )
                )

            return text_detections

        except Exception as e:
            logger.error(f"Error during OCR extraction: {e}")
            return []

    def extract_text_for_detections(
        self,
        image: np.ndarray,
        detections: List[Detection],
        conf_threshold: Optional[float] = None,
    ) -> Dict[int, List[TextDetection]]:
        """
        Associates extracted text with corresponding detected objects.
        Returns a mapping of detection track_id (or index) to texts inside that object.
        """
        if not detections or image is None or image.size == 0:
            return {}

        all_texts = self.extract_text(image, conf_threshold=conf_threshold)
        if not all_texts:
            return {}

        mapping: Dict[int, List[TextDetection]] = {}

        for idx, det in enumerate(detections):
            key = det.track_id if det.track_id is not None else idx
            matching_texts = []

            dx1, dy1, dx2, dy2 = det.bbox
            for t in all_texts:
                tcx, tcy = t.center
                # Check if text center is within object bounding box
                if dx1 <= tcx <= dx2 and dy1 <= tcy <= dy2:
                    matching_texts.append(t)

            if matching_texts:
                mapping[key] = matching_texts

        return mapping


def draw_text_annotations(
    image: np.ndarray,
    text_detections: List[TextDetection],
    box_color: Tuple[int, int, int] = (255, 180, 0),  # Amber/Cyan text badge
    text_color: Tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """
    Renders text bounding boxes and transcriptions onto an image.
    """
    annotated = image.copy()

    for td in text_detections:
        x1, y1, x2, y2 = [int(round(c)) for c in td.bbox]

        # Draw bounding rectangle or polygon
        if td.polygon and len(td.polygon) >= 4:
            pts = np.array(td.polygon, np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated, [pts], isClosed=True, color=box_color, thickness=2)
        else:
            cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)

        # Label banner
        label = f'"{td.text}" ({int(td.confidence * 100)}%)'
        (w, h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)

        badge_y1 = max(0, y1 - h - 6)
        badge_y2 = y1
        badge_x2 = min(annotated.shape[1], x1 + w + 8)

        # Draw translucent background banner
        sub_img = annotated[badge_y1:badge_y2, x1:badge_x2]
        if sub_img.shape[0] > 0 and sub_img.shape[1] > 0:
            white_rect = np.full(sub_img.shape, box_color, dtype=np.uint8)
            res = cv2.addWeighted(sub_img, 0.3, white_rect, 0.7, 1.0)
            annotated[badge_y1:badge_y2, x1:badge_x2] = res

        cv2.putText(
            annotated,
            label,
            (x1 + 4, max(h + 2, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            label,
            (x1 + 4, max(h + 2, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            text_color,
            1,
            cv2.LINE_AA,
        )

    return annotated
