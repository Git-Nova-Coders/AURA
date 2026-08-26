"""
AURA Object Tracking Subsystem (Milestone 5)
Associates detections across consecutive frames to maintain persistent track IDs and motion trajectories.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import numpy as np

from .detector import Detection

logger = logging.getLogger(__name__)


def compute_iou(box1: List[float], box2: List[float]) -> float:
    """
    Computes Intersection-over-Union (IoU) between two bounding boxes in [x1, y1, x2, y2] format.
    """
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


@dataclass
class TrackedObject:
    """
    State of a single persistent tracked object across time.
    """
    track_id: int
    class_id: int
    class_name: str
    bbox: List[float]
    confidence: float
    history: List[Tuple[float, float]] = field(default_factory=list)  # (cx, cy) history
    age: int = 1                     # Total frames since first detection
    hits: int = 1                    # Number of frames successfully matched
    time_since_update: int = 0       # Frames elapsed since last detection match
    velocity: Tuple[float, float] = (0.0, 0.0)  # (dx, dy) in pixels per frame

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.bbox[0] + self.bbox[2]) / 2.0, (self.bbox[1] + self.bbox[3]) / 2.0)

    @property
    def temporal_persistence(self) -> float:
        """Ratio of frames detected over total track lifespan in range (0.0, 1.0]."""
        return float(self.hits / max(self.age, 1))

    @property
    def motion_speed(self) -> float:
        """Euclidean speed in pixels per frame."""
        return float(np.sqrt(self.velocity[0] ** 2 + self.velocity[1] ** 2))

    def update(self, detection: Detection) -> None:
        """Updates the track state with a newly matched Detection."""
        old_cx, old_cy = self.center
        self.bbox = [float(c) for c in detection.bbox]
        self.confidence = float(detection.confidence)
        new_cx, new_cy = self.center

        self.velocity = (new_cx - old_cx, new_cy - old_cy)
        self.history.append((new_cx, new_cy))
        if len(self.history) > 30:  # Keep last 30 trajectory points
            self.history.pop(0)

        self.hits += 1
        self.age += 1
        self.time_since_update = 0

    def mark_missed(self) -> None:
        """Advances track age when no detection matched in the current frame."""
        self.age += 1
        self.time_since_update += 1

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the track state to a dictionary."""
        return {
            "track_id": self.track_id,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "bbox": [round(float(c), 2) for c in self.bbox],
            "confidence": round(float(self.confidence), 4),
            "age": self.age,
            "hits": self.hits,
            "time_since_update": self.time_since_update,
            "temporal_persistence": round(self.temporal_persistence, 4),
            "motion_speed": round(self.motion_speed, 2),
            "velocity": [round(self.velocity[0], 2), round(self.velocity[1], 2)],
        }


class ObjectTracker:
    """
    Multi-Object Tracker based on Spatial IoU and Centroid Association.
    Assigns persistent track_id integers across consecutive frames.
    """

    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 1,
        iou_threshold: float = 0.3,
    ):
        """
        Args:
            max_age: Number of consecutive missed frames before a track is deleted.
            min_hits: Minimum detections before a track is considered confirmed.
            iou_threshold: Minimum IoU required to associate a detection with an existing track.
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self._tracks: List[TrackedObject] = []
        self._next_id: int = 1

    @property
    def active_tracks(self) -> List[TrackedObject]:
        """Returns currently active and confirmed tracks."""
        return [t for t in self._tracks if t.time_since_update == 0 and t.hits >= self.min_hits]

    @property
    def all_tracks(self) -> List[TrackedObject]:
        return list(self._tracks)

    def reset(self) -> None:
        """Clears all tracks and resets the ID counter."""
        self._tracks.clear()
        self._next_id = 1

    def update(self, detections: List[Detection]) -> List[Detection]:
        """
        Updates the tracker with detections from the current frame.
        Enriches and returns the Detection list with assigned track_id attributes.
        
        Args:
            detections: List of Detection objects from current frame.
            
        Returns:
            List[Detection]: Detection objects with updated `track_id`.
        """
        if not detections:
            # Advance all existing tracks as missed
            for track in self._tracks:
                track.mark_missed()
            # Prune dead tracks
            self._tracks = [t for t in self._tracks if t.time_since_update <= self.max_age]
            return []

        # If no tracks currently exist, initialize all detections as new tracks
        if not self._tracks:
            for det in detections:
                track = TrackedObject(
                    track_id=self._next_id,
                    class_id=det.class_id,
                    class_name=det.class_name,
                    bbox=list(det.bbox),
                    confidence=float(det.confidence),
                    history=[det.center],
                    age=1,
                    hits=1,
                    time_since_update=0,
                )
                self._tracks.append(track)
                det.track_id = self._next_id
                self._next_id += 1
            return detections

        # Build IoU Cost Matrix: [num_tracks, num_detections]
        num_tracks = len(self._tracks)
        num_dets = len(detections)
        iou_matrix = np.zeros((num_tracks, num_dets), dtype=np.float32)

        for t_idx, track in enumerate(self._tracks):
            for d_idx, det in enumerate(detections):
                # Prefer same class association
                if track.class_id == det.class_id:
                    iou_matrix[t_idx, d_idx] = compute_iou(track.bbox, det.bbox)
                else:
                    # Allow cross-class matching only if high overlap (> 0.6)
                    raw_iou = compute_iou(track.bbox, det.bbox)
                    iou_matrix[t_idx, d_idx] = raw_iou * 0.5 if raw_iou > 0.6 else 0.0

        # Greedy bipartite matching
        matched_tracks = set()
        matched_dets = set()

        if num_tracks > 0 and num_dets > 0:
            # Flatten indices sorted by descending IoU
            sorted_indices = np.dstack(
                np.unravel_index(np.argsort(-iou_matrix, axis=None), iou_matrix.shape)
            )[0]

            for t_idx, d_idx in sorted_indices:
                iou_val = iou_matrix[t_idx, d_idx]
                if iou_val < self.iou_threshold:
                    break
                if t_idx in matched_tracks or d_idx in matched_dets:
                    continue

                # Match found
                matched_tracks.add(t_idx)
                matched_dets.add(d_idx)
                self._tracks[t_idx].update(detections[d_idx])
                detections[d_idx].track_id = self._tracks[t_idx].track_id

        # Update unmatched tracks
        for t_idx, track in enumerate(self._tracks):
            if t_idx not in matched_tracks:
                track.mark_missed()

        # Initialize unmatched detections as new tracks
        for d_idx, det in enumerate(detections):
            if d_idx not in matched_dets:
                new_track = TrackedObject(
                    track_id=self._next_id,
                    class_id=det.class_id,
                    class_name=det.class_name,
                    bbox=list(det.bbox),
                    confidence=float(det.confidence),
                    history=[det.center],
                    age=1,
                    hits=1,
                    time_since_update=0,
                )
                self._tracks.append(new_track)
                det.track_id = self._next_id
                self._next_id += 1

        # Prune expired tracks
        self._tracks = [t for t in self._tracks if t.time_since_update <= self.max_age]

        return detections
