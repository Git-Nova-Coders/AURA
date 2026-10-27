"""
AURA Reliability ANN Dataset Module
Handles loading, splitting, scaling, and preparing PyTorch DataLoaders.
Ensures feature scaling is serialized and applied consistently during training and inference.
"""

import os
import pickle
import logging
from typing import Tuple, List, Optional, Dict, Any
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class ReliabilityDataset(Dataset):
    """PyTorch Dataset for object detection reliability estimation."""
    
    def __init__(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """
        Args:
            X (np.ndarray): Numerical features of shape (N, input_dim)
            y (np.ndarray, optional): Binary reliability labels of shape (N,)
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1) if y is not None else None

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx], torch.tensor([])


def load_dataset_from_csv(
    filepath: str,
    include_class_onehot: bool = False,
    num_classes: int = 80,
    feature_family: str = "all",  # "all" (14 features), "geometry" (11 features), "quality" (3 features)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads features and reliability labels from a CSV dataset.
    
    Args:
        filepath (str): Path to CSV dataset.
        include_class_onehot (bool): Whether to load one-hot class encodings.
        num_classes (int): Number of classes if one-hot encoding is used.
        feature_family (str): Feature subset: 'all' (geometry + quality), 'geometry' (only first 11), 'quality' (only blur/brightness/contrast).
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: Feature matrix X and label vector y.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset CSV not found at '{filepath}'")

    df = pd.read_csv(filepath)
    if "is_reliable" not in df.columns:
        raise ValueError(f"CSV dataset must contain 'is_reliable' label column. Found: {df.columns.tolist()}")

    # Determine base features
    geometry_cols = [
        "confidence", "norm_x1", "norm_y1", "norm_x2", "norm_y2",
        "norm_width", "norm_height", "norm_area", "norm_center_x",
        "norm_center_y", "aspect_ratio"
    ]
    quality_cols = ["blur_score", "brightness", "contrast"]

    if feature_family == "all":
        feature_cols = geometry_cols + quality_cols
    elif feature_family == "geometry":
        feature_cols = geometry_cols
    elif feature_family == "quality":
        feature_cols = quality_cols
    else:
        raise ValueError(f"Unknown feature_family '{feature_family}'. Select from 'all', 'geometry', 'quality'")

    # Optional class one-hot encoding columns
    if include_class_onehot:
        onehot_cols = [f"class_{i}" for i in range(num_classes)]
        # Check if they exist in CSV, if not build them from class_id
        missing_onehot = [col for col in onehot_cols if col not in df.columns]
        if missing_onehot:
            if "class_id" in df.columns:
                logger.info("One-hot columns missing in CSV. Generating from 'class_id' column...")
                for i in range(num_classes):
                    df[f"class_{i}"] = (df["class_id"] == i).astype(float)
            else:
                raise ValueError("Cannot construct one-hot encodings because both 'class_*' and 'class_id' are missing.")
        feature_cols = feature_cols + onehot_cols

    # Verify all features exist
    missing_features = [col for col in feature_cols if col not in df.columns]
    if missing_features:
        raise ValueError(f"Missing required feature columns in CSV: {missing_features}")

    # Handle missing labels or drop NaNs
    clean_df = df.dropna(subset=feature_cols + ["is_reliable"])
    if len(clean_df) < len(df):
        logger.warning(f"Dropped {len(df) - len(clean_df)} rows with NaN values.")

    X = clean_df[feature_cols].values.astype(np.float32)
    y = clean_df["is_reliable"].values.astype(np.int64)

    return X, y


def prepare_data_loaders(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 32,
    val_split: float = 0.2,
    test_split: float = 0.1,
    seed: int = 42,
    scaler: Optional[StandardScaler] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, StandardScaler]:
    """
    Splits features and labels, fits/applies StandardScaler, and returns PyTorch DataLoaders.
    
    Args:
        X (np.ndarray): Complete feature matrix.
        y (np.ndarray): Complete label array.
        batch_size (int): DataLoader batch size.
        val_split (float): Validation split fraction.
        test_split (float): Test split fraction.
        seed (int): Random seed for reproducibility.
        scaler (StandardScaler, optional): Pre-fitted scaler to apply. If None, fits a new scaler on training data.
        
    Returns:
        Tuple[DataLoader, DataLoader, DataLoader, StandardScaler]:
            train_loader, val_loader, test_loader, fitted_scaler
    """
    total_val_test = val_split + test_split
    
    # First split: Train vs Val+Test
    X_train, X_val_test, y_train, y_val_test = train_test_split(
        X, y, test_size=total_val_test, random_state=seed, stratify=y
    )
    
    # Second split: Val vs Test
    test_ratio_of_val_test = test_split / total_val_test
    X_val, X_test, y_val, y_test = train_test_split(
        X_val_test, y_val_test, test_size=test_ratio_of_val_test, random_state=seed, stratify=y_val_test
    )

    # Scaling
    if scaler is None:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
    else:
        X_train_scaled = scaler.transform(X_train)

    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Build datasets
    train_dataset = ReliabilityDataset(X_train_scaled, y_train)
    val_dataset = ReliabilityDataset(X_val_scaled, y_val)
    test_dataset = ReliabilityDataset(X_test_scaled, y_test)

    # Build loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    logger.info(
        f"Data partitioning complete: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}"
    )

    return train_loader, val_loader, test_loader, scaler
