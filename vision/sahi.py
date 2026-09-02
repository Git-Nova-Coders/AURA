"""
AURA SAHI (Slicing Aided Hyper Inference) Subsystem
Enhances detection accuracy and precision for small, distant, and fine-grained objects
by dividing high-resolution frames into overlapping tiles, running inference, translating
coordinates, and merging bounding boxes with Class-Aware NMS box fusion.
"""

import logging
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any, TYPE_CHECKING
import cv2
import numpy as np

from config.config import SAHIConfig
from .detector import Detection

if TYPE_CHECKING:
    from .detector import ObjectDetector

logger = logging.getLogger(__name__)


@dataclass
class SliceWindow:
    """
    Defines a rectangular slice window over a full image frame.
    """
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        return self.width * self.height

    def crop(self, image: np.ndarray) -> np.ndarray:
        """Extracts the slice patch from the source image."""
        return image[self.y1:self.y2, self.x1:self.x2]

    def to_dict(self) -> Dict[str, int]:
        return {"x1": self.x1, "y1": self.y1, "x2": self.x2, "y2": self.y2}


def generate_slice_grid(
    image_width: int,
    image_height: int,
    slice_width: int = 320,
    slice_height: int = 320,
    overlap_width_ratio: float = 0.2,
    overlap_height_ratio: float = 0.2,
) -> List[SliceWindow]:
    """
    Computes overlapping slice windows covering the entire image dimension.

    Args:
        image_width: Width of the full image in pixels.
        image_height: Height of the full image in pixels.
        slice_width: Width of each slice window.
        slice_height: Height of each slice window.
        overlap_width_ratio: Fractional horizontal overlap between adjacent tiles in [0.0, 0.8].
        overlap_height_ratio: Fractional vertical overlap between adjacent tiles in [0.0, 0.8].

    Returns:
        List[SliceWindow]: List of tile boundaries covering the full frame.
    """
    if image_width <= 0 or image_height <= 0:
        return []

    # Handle image smaller than requested slice dimensions
    slice_w = min(slice_width, image_width)
    slice_h = min(slice_height, image_height)

    # Compute step strides
    overlap_w = max(0.0, min(0.8, overlap_width_ratio))
    overlap_h = max(0.0, min(0.8, overlap_height_ratio))

    step_x = max(1, int(slice_w * (1.0 - overlap_w)))
    step_y = max(1, int(slice_h * (1.0 - overlap_h)))

    x_starts: List[int] = []
    y_starts: List[int] = []

    curr_x = 0
    while curr_x + slice_w < image_width:
        x_starts.append(curr_x)
        curr_x += step_x
    x_starts.append(max(0, image_width - slice_w))
    # Deduplicate while preserving order
    x_starts = sorted(list(set(x_starts)))

    curr_y = 0
    while curr_y + slice_h < image_height:
        y_starts.append(curr_y)
        curr_y += step_y
    y_starts.append(max(0, image_height - slice_h))
    # Deduplicate while preserving order
    y_starts = sorted(list(set(y_starts)))

    windows: List[SliceWindow] = []
    for y1 in y_starts:
        for x1 in x_starts:
            windows.append(
                SliceWindow(
                    x1=x1,
                    y1=y1,
                    x2=min(image_width, x1 + slice_w),
                    y2=min(image_height, y1 + slice_h),
                )
            )

    return windows


def compute_iou(box1: List[float], box2: List[float]) -> float:
    """Computes Intersection-over-Union (IoU) between two bounding boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_w = max(0.0, x2 - x1)
    intersection_h = max(0.0, y2 - y1)
    intersection_area = intersection_w * intersection_h

    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union_area = area1 + area2 - intersection_area

    if union_area <= 1e-6:
        return 0.0

    return float(intersection_area / union_area)


def compute_ios(box1: List[float], box2: List[float]) -> float:
    """Computes Intersection-over-Smaller (IoS) containment metric."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_w = max(0.0, x2 - x1)
    intersection_h = max(0.0, y2 - y1)
    intersection_area = intersection_w * intersection_h

    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    min_area = min(area1, area2)

    if min_area <= 1e-6:
        return 0.0

    return float(intersection_area / min_area)


def apply_nms_merging(
    detections: List[Detection],
    iou_threshold: float = 0.5,
    match_class: bool = True,
) -> List[Detection]:
    """
    Performs Class-Aware Non-Maximum Suppression (NMS) and weighted box merging across
    overlapping slice predictions.

    Args:
        detections: List of raw candidate Detection objects.
        iou_threshold: IoU overlap threshold to consider two boxes as duplicates.
        match_class: If True, only suppresses boxes of the exact same class.

    Returns:
        List[Detection]: Deduplicated and fused list of final detections.
    """
    if not detections:
        return []

    # Sort detections by confidence descending
    sorted_dets = sorted(detections, key=lambda d: d.confidence, reverse=True)
    kept_dets: List[Detection] = []

    while sorted_dets:
        best_det = sorted_dets.pop(0)
        remaining: List[Detection] = []

        # Accumulate overlapping matching detections for weighted coordinate refinement
        overlapping: List[Detection] = [best_det]
        current_union = list(best_det.bbox)

        for other_det in sorted_dets:
            if match_class and (best_det.class_id != other_det.class_id):
                remaining.append(other_det)
                continue

            iou = compute_iou(current_union, other_det.bbox)
            ios = compute_ios(current_union, other_det.bbox)
            if iou >= iou_threshold or ios >= 0.45:
                overlapping.append(other_det)
                current_union = [
                    min(current_union[0], other_det.bbox[0]),
                    min(current_union[1], other_det.bbox[1]),
                    max(current_union[2], other_det.bbox[2]),
                    max(current_union[3], other_det.bbox[3]),
                ]
            else:
                remaining.append(other_det)

        # Intelligent box merging: preserves outer boundary when containment occurs
        if len(overlapping) > 1:
            areas = [max(1.0, (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1])) for d in overlapping]
            max_area = max(areas)
            min_area = min(areas)

            # If containment or disparate scales, use union boundary so macro objects aren't shrunk
            if (max_area / min_area) > 1.6:
                fused_x1 = min(d.bbox[0] for d in overlapping)
                fused_y1 = min(d.bbox[1] for d in overlapping)
                fused_x2 = max(d.bbox[2] for d in overlapping)
                fused_y2 = max(d.bbox[3] for d in overlapping)
            else:
                total_weight = sum(d.confidence for d in overlapping)
                fused_x1 = sum(d.bbox[0] * d.confidence for d in overlapping) / total_weight
                fused_y1 = sum(d.bbox[1] * d.confidence for d in overlapping) / total_weight
                fused_x2 = sum(d.bbox[2] * d.confidence for d in overlapping) / total_weight
                fused_y2 = sum(d.bbox[3] * d.confidence for d in overlapping) / total_weight

            fused_det = Detection(
                class_id=best_det.class_id,
                class_name=best_det.class_name,
                confidence=best_det.confidence,  # Retain peak confidence
                bbox=[fused_x1, fused_y1, fused_x2, fused_y2],
                track_id=best_det.track_id,
                reliability_score=best_det.reliability_score,
                reliability_label=best_det.reliability_label,
            )
            kept_dets.append(fused_det)
        else:
            kept_dets.append(best_det)

        sorted_dets = remaining

    return kept_dets


class SlicedInferenceEngine:
    """
    High-Performance Slicing Aided Hyper Inference (SAHI) Engine for AURA.
    Extracts image slices, executes high-resolution inference, shifts coordinates,
    and fuses bounding boxes into high-precision detections.
    """

    def __init__(self, config: Optional[SAHIConfig] = None):
        self.config = config or SAHIConfig()

    def slice_and_detect(
        self,
        detector: "ObjectDetector",
        image: np.ndarray,
        conf_threshold: Optional[float] = None,
    ) -> List[Detection]:
        """
        Executes SAHI sliced inference on the provided image using the detector.

        Args:
            detector: ObjectDetector instance.
            image: Full-resolution numpy BGR frame [H, W, 3].
            conf_threshold: Optional confidence threshold override.

        Returns:
            List[Detection]: Fused list of high-precision object detections.
        """
        if image is None or image.size == 0:
            return []

        h, w = image.shape[:2]
        all_detections: List[Detection] = []
        full_frame_dets: List[Detection] = []

        # 1. Full-frame global glance (captures large/contextual objects)
        if self.config.include_full_frame:
            # Disable recursive SAHI during the full frame detect call
            full_frame_dets = detector._detect_single_frame(image, conf_threshold=conf_threshold)
            all_detections.extend(full_frame_dets)

        # 2. Generate Slice Grid
        slice_windows = generate_slice_grid(
            image_width=w,
            image_height=h,
            slice_width=self.config.slice_width,
            slice_height=self.config.slice_height,
            overlap_width_ratio=self.config.overlap_width_ratio,
            overlap_height_ratio=self.config.overlap_height_ratio,
        )

        # 3. Perform inference on each high-resolution slice
        for window in slice_windows:
            slice_img = window.crop(image)
            if slice_img.size == 0:
                continue

            slice_dets = detector._detect_single_frame(slice_img, conf_threshold=conf_threshold)

            # Project coordinates back to global image space
            for det in slice_dets:
                global_bbox = [
                    det.bbox[0] + window.x1,
                    det.bbox[1] + window.y1,
                    det.bbox[2] + window.x1,
                    det.bbox[3] + window.y1,
                ]
                # Clamp within frame boundaries
                global_bbox = [
                    max(0.0, min(float(w), global_bbox[0])),
                    max(0.0, min(float(h), global_bbox[1])),
                    max(0.0, min(float(w), global_bbox[2])),
                    max(0.0, min(float(h), global_bbox[3])),
                ]

                # Suppress redundant slice fragments of macro objects already captured in full frame
                if full_frame_dets and det.class_name.lower() in ("person", "couch", "bed", "dining table"):
                    is_sub_fragment = False
                    for full_det in full_frame_dets:
                        if full_det.class_name.lower() == det.class_name.lower():
                            if compute_ios(full_det.bbox, global_bbox) >= 0.40:
                                is_sub_fragment = True
                                break
                    if is_sub_fragment:
                        continue

                all_detections.append(
                    Detection(
                        class_id=det.class_id,
                        class_name=det.class_name,
                        confidence=det.confidence,
                        bbox=global_bbox,
                    )
                )

        # 4. Class-Aware NMS & Spatial Box Fusion
        fused_detections = apply_nms_merging(
            all_detections,
            iou_threshold=self.config.nms_threshold,
            match_class=True,
        )

        # 5. Apply geometric filters on final fused results
        if detector.enable_geometric_filter:
            fused_detections = detector._filter_geometric_anomalies(fused_detections, w, h)

        return fused_detections
