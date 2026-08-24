"""
AURA Dataset Collector Module
Collects, aggregates, and exports DetectionFeatures into training datasets
(CSV, JSONL, Pandas DataFrame, NumPy matrices) for the Reliability ANN (Milestone 3).
"""

import os
import csv
import json
import logging
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
import numpy as np

try:
    import pandas as pd
    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False

from .features import DetectionFeatures

logger = logging.getLogger(__name__)


@dataclass
class LabeledFeatureSample:
    """A feature sample bundled with an optional ground-truth reliability label."""
    features: DetectionFeatures
    is_reliable: Optional[int] = None  # 1 = reliable, 0 = unreliable, None = unlabeled


class DatasetCollector:
    """
    Accumulates extracted visual features and exports them to structured datasets
    for training, validation, and benchmarking of the Reliability ANN.
    """

    def __init__(self):
        self._samples: List[LabeledFeatureSample] = []

    def add_sample(
        self,
        features: DetectionFeatures,
        is_reliable: Optional[int] = None,
    ) -> None:
        """Adds a single feature sample to the collection."""
        self._samples.append(LabeledFeatureSample(features=features, is_reliable=is_reliable))

    def add_samples(
        self,
        features_list: List[DetectionFeatures],
        labels: Optional[List[Optional[int]]] = None,
    ) -> None:
        """Adds a batch of feature samples."""
        if labels is None:
            labels = [None] * len(features_list)
        for feat, lab in zip(features_list, labels):
            self.add_sample(feat, lab)

    @property
    def count(self) -> int:
        """Returns the number of collected samples."""
        return len(self._samples)

    def clear(self) -> None:
        """Clears all collected samples."""
        self._samples.clear()

    def to_records(
        self,
        include_class_onehot: bool = False,
        num_classes: int = 80,
    ) -> List[Dict[str, Any]]:
        """Converts the dataset to a list of flat dictionaries."""
        records: List[Dict[str, Any]] = []
        for sample in self._samples:
            f = sample.features
            rec: Dict[str, Any] = {
                "class_id": f.class_id,
                "class_name": f.class_name,
                "confidence": f.confidence,
                "norm_x1": f.norm_x1,
                "norm_y1": f.norm_y1,
                "norm_x2": f.norm_x2,
                "norm_y2": f.norm_y2,
                "norm_width": f.norm_width,
                "norm_height": f.norm_height,
                "norm_area": f.norm_area,
                "norm_center_x": f.norm_center_x,
                "norm_center_y": f.norm_center_y,
                "aspect_ratio": f.aspect_ratio,
                "blur_score": f.blur_score,
                "brightness": f.brightness,
                "contrast": f.contrast,
                "track_id": f.track_id,
                "is_reliable": sample.is_reliable,
            }
            if include_class_onehot:
                for i in range(num_classes):
                    rec[f"class_{i}"] = 1.0 if f.class_id == i else 0.0
            records.append(rec)
        return records

    def to_dataframe(
        self,
        include_class_onehot: bool = False,
        num_classes: int = 80,
    ) -> Any:
        """
        Converts the collected dataset into a Pandas DataFrame if pandas is installed.
        
        Returns:
            pd.DataFrame: Tabular feature dataset.
        """
        if not _HAS_PANDAS:
            raise ImportError("Pandas is not installed. Use to_records(), to_csv(), or to_arrays() instead.")

        records = self.to_records(include_class_onehot=include_class_onehot, num_classes=num_classes)
        if not records:
            col_names = DetectionFeatures.feature_names(
                include_class_onehot=include_class_onehot,
                num_classes=num_classes,
            ) + ["class_id", "class_name", "is_reliable"]
            return pd.DataFrame(columns=col_names)

        return pd.DataFrame(records)

    def to_arrays(
        self,
        include_class_onehot: bool = False,
        num_classes: int = 80,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Exports feature vectors as NumPy array X and optional label array y.
        
        Returns:
            Tuple[np.ndarray, Optional[np.ndarray]]: Feature matrix X and labels y.
        """
        if not self._samples:
            num_feat = 14 + (num_classes if include_class_onehot else 0)
            return np.empty((0, num_feat), dtype=np.float32), None

        x_list = [
            s.features.to_vector(include_class_onehot=include_class_onehot, num_classes=num_classes)
            for s in self._samples
        ]
        X = np.stack(x_list, axis=0)

        has_labels = any(s.is_reliable is not None for s in self._samples)
        y = None
        if has_labels:
            y = np.array(
                [s.is_reliable if s.is_reliable is not None else -1 for s in self._samples],
                dtype=np.int64,
            )

        return X, y

    def to_csv(
        self,
        filepath: str,
        include_class_onehot: bool = False,
        num_classes: int = 80,
    ) -> str:
        """Exports the dataset to a CSV file."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        records = self.to_records(include_class_onehot=include_class_onehot, num_classes=num_classes)

        if not records:
            col_names = DetectionFeatures.feature_names(
                include_class_onehot=include_class_onehot,
                num_classes=num_classes,
            ) + ["class_id", "class_name", "is_reliable"]
            with open(filepath, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(col_names)
            return filepath

        fieldnames = list(records[0].keys())
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)

        logger.info(f"Exported {len(records)} feature records to '{filepath}'")
        return filepath

    def to_jsonl(self, filepath: str) -> str:
        """Exports the dataset to a JSON Lines file."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            for s in self._samples:
                item = s.features.to_dict()
                item["is_reliable"] = s.is_reliable
                f.write(json.dumps(item) + "\n")
        logger.info(f"Exported {len(self._samples)} JSONL records to '{filepath}'")
        return filepath

    @classmethod
    def generate_mock_dataset(cls, num_samples: int = 200, seed: int = 42) -> "DatasetCollector":
        """
        Generates a synthetic dataset of DetectionFeatures and labels for the ANN team (Member 2).
        Simulates realistic detection geometry, quality indicators, and reliability ground truth.
        """
        np.random.seed(seed)
        collector = cls()

        class_names_map = {0: "person", 56: "chair", 60: "dining table", 63: "laptop", 39: "bottle", 67: "cell phone"}
        class_ids = list(class_names_map.keys())

        for _ in range(num_samples):
            cid = int(np.random.choice(class_ids))
            cname = class_names_map[cid]

            # High confidence typically correlates with higher reliability
            conf = float(np.clip(np.random.beta(5, 2), 0.25, 0.99))
            
            # Geometry
            w = float(np.clip(np.random.beta(2, 5), 0.05, 0.8))
            h = float(np.clip(np.random.beta(3, 4), 0.05, 0.9))
            x1 = float(np.random.uniform(0.0, max(0.0, 1.0 - w)))
            y1 = float(np.random.uniform(0.0, max(0.0, 1.0 - h)))
            x2 = min(1.0, x1 + w)
            y2 = min(1.0, y1 + h)
            area = w * h
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            aspect = w / max(h, 1e-6)

            # Quality metrics
            blur = float(np.random.exponential(150.0) + 20.0)
            bright = float(np.random.normal(120.0, 35.0))
            contrast = float(np.random.normal(45.0, 15.0))

            # Reliability label: higher conf + sharp image + reasonable size -> reliable
            reliability_prob = 0.5 * conf + 0.3 * min(1.0, blur / 300.0) + 0.2 * min(1.0, area / 0.3)
            is_rel = 1 if np.random.rand() < reliability_prob else 0

            feat = DetectionFeatures(
                class_id=cid,
                class_name=cname,
                confidence=conf,
                norm_x1=x1,
                norm_y1=y1,
                norm_x2=x2,
                norm_y2=y2,
                norm_width=w,
                norm_height=h,
                norm_area=area,
                norm_center_x=cx,
                norm_center_y=cy,
                aspect_ratio=aspect,
                blur_score=blur,
                brightness=bright,
                contrast=contrast,
            )
            collector.add_sample(feat, is_reliable=is_rel)

        return collector
