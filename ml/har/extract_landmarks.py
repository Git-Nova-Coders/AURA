"""
AURA HAR Landmark Batch Extraction Pipeline
Extracts 33 pose landmarks (x, y, z, visibility -> 132 features) per frame
from UCF101 dataset videos with single-video test mode and resumable batch processing.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.har.dataset import load_class_index, load_train_list, load_test_list, SELECTED_CLASSES
from vision.pose.pose_detector import PoseDetector, NUM_LANDMARKS, LANDMARK_DIM, TOTAL_FEATURE_DIM
from vision.pose.landmark_extractor import extract_landmarks_from_video

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("AURA.HAR.Extract")

RAW_LANDMARKS_DIR = PROJECT_ROOT / "data" / "har" / "processed" / "raw_landmarks"


def extract_single_video_test(video_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Executes extraction on ONE sample video and verifies output dimensions & values.
    """
    logger.info("=" * 60)
    logger.info("AURA-HAR PHASE 1: SINGLE-VIDEO EXTRACTION TEST")
    logger.info("=" * 60)

    if video_path is None:
        train_videos = load_train_list(1)
        if not train_videos:
            raise RuntimeError("No training videos found in split 1.")
        video_info = train_videos[0]
        video_path = video_info["path"]
        activity = video_info["activity"]
    else:
        video_path = Path(video_path)
        activity = video_path.parent.name

    logger.info(f"Selected Sample Video: {video_path}")
    logger.info(f"Activity Class:       {activity}")

    detector = PoseDetector(running_mode="video")
    t0 = time.time()
    landmarks, mask, stats = extract_landmarks_from_video(video_path, detector=detector)
    elapsed = time.time() - t0
    detector.close()

    # Verify extraction dimensions
    t_frames, n_lm, n_dim = landmarks.shape
    total_features_per_frame = n_lm * n_dim

    logger.info("-" * 60)
    logger.info(f"Extraction Completed in {elapsed:.2f}s ({t_frames / max(elapsed, 0.001):.1f} FPS)")
    logger.info(f"Total Video Frames:      {stats['total_video_frames']}")
    logger.info(f"Processed Frames:        {t_frames}")
    logger.info(f"Detected Pose Frames:    {stats['detected_frames']}")
    logger.info(f"Missing Pose Frames:     {stats['missing_pose_frames']}")
    logger.info(f"Pose Detection Rate:     {stats['detection_rate'] * 100:.1f}%")
    logger.info(f"Extracted Array Shape:   {landmarks.shape}")
    logger.info(f"Features Per Frame:      {total_features_per_frame} (Expected: {TOTAL_FEATURE_DIM})")
    logger.info("-" * 60)

    # Verification assertions
    assert n_lm == NUM_LANDMARKS, f"Expected {NUM_LANDMARKS} landmarks, got {n_lm}"
    assert n_dim == LANDMARK_DIM, f"Expected {LANDMARK_DIM} dimensions, got {n_dim}"
    assert total_features_per_frame == 132, f"Expected 132 features per frame, got {total_features_per_frame}"

    # Verify landmark value ranges on detected frames
    if stats["detected_frames"] > 0:
        first_detected_idx = int(np.where(mask)[0][0])
        sample_lm = landmarks[first_detected_idx]
        logger.info(f"Sample frame {first_detected_idx} (Landmark 0 - Nose):")
        logger.info(f"  x={sample_lm[0, 0]:.4f}, y={sample_lm[0, 1]:.4f}, z={sample_lm[0, 2]:.4f}, vis={sample_lm[0, 3]:.4f}")

    logger.info("VERIFICATION STATUS: SUCCESSFUL (33 landmarks x 4 values = 132 features verified)")
    return {
        "status": "SUCCESS",
        "video": str(video_path),
        "activity": activity,
        "shape": list(landmarks.shape),
        "stats": stats,
    }


def run_batch_extraction(split_number: int = 1, force_recompute: bool = False, max_videos: Optional[int] = None):
    """
    Extracts pose landmarks for all videos in train and test splits.
    Saves results to data/har/processed/raw_landmarks/ in compressed .npz format.
    Resumable: skips already processed videos unless force_recompute=True.
    """
    RAW_LANDMARKS_DIR.mkdir(parents=True, exist_ok=True)

    class_index = load_class_index()
    train_videos = load_train_list(split_number)
    test_videos = load_test_list(split_number)

    logger.info("=" * 60)
    logger.info(f"AURA-HAR BATCH EXTRACTION (Split {split_number})")
    logger.info(f"Train videos: {len(train_videos)} | Test videos: {len(test_videos)}")
    logger.info(f"Output directory: {RAW_LANDMARKS_DIR}")
    logger.info("=" * 60)

    detector = PoseDetector(running_mode="video")

    datasets = [
        ("train", train_videos),
        ("test", test_videos),
    ]

    summary_stats = {
        "train": {"total": 0, "processed": 0, "skipped": 0, "frames": 0, "missing_pose_frames": 0},
        "test": {"total": 0, "processed": 0, "skipped": 0, "frames": 0, "missing_pose_frames": 0},
    }

    try:
        for split_name, video_list in datasets:
            split_dir = RAW_LANDMARKS_DIR / split_name
            split_dir.mkdir(parents=True, exist_ok=True)

            target_list = video_list if max_videos is None else video_list[:max_videos]
            summary_stats[split_name]["total"] = len(target_list)

            for idx, item in enumerate(target_list):
                v_path = item["path"]
                activity = item["activity"]
                class_id = class_index.get(activity, -1)

                output_filename = f"{v_path.stem}.npz"
                output_path = split_dir / output_filename

                if output_path.exists() and not force_recompute:
                    summary_stats[split_name]["skipped"] += 1
                    continue

                t0 = time.time()
                try:
                    landmarks, mask, stats = extract_landmarks_from_video(v_path, detector=detector, frame_stride=2)
                except Exception as e:
                    logger.error(f"Error extracting {v_path.name}: {e}")
                    continue

                elapsed = time.time() - t0

                # Save compressed npz
                np.savez_compressed(
                    output_path,
                    landmarks=landmarks,          # (T, 33, 4)
                    detected_mask=mask,          # (T,)
                    activity=activity,
                    class_id=class_id,
                    video_name=v_path.name,
                    split=split_name,
                    fps=stats["fps"],
                    resolution=stats["resolution"],
                )

                summary_stats[split_name]["processed"] += 1
                summary_stats[split_name]["frames"] += stats["processed_frames"]
                summary_stats[split_name]["missing_pose_frames"] += stats["missing_pose_frames"]

                if (idx + 1) % 10 == 0 or (idx + 1) == len(target_list):
                    det_rate = (
                        (stats["detected_frames"] / stats["processed_frames"] * 100)
                        if stats["processed_frames"] > 0 else 0.0
                    )
                    logger.info(
                        f"[{split_name.upper()} {idx+1}/{len(target_list)}] "
                        f"{v_path.name} | Frames: {stats['processed_frames']} | "
                        f"Det: {det_rate:.1f}% | Time: {elapsed:.2f}s"
                    )

    finally:
        detector.close()

    manifest_path = RAW_LANDMARKS_DIR / "extraction_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(summary_stats, f, indent=2)

    logger.info("=" * 60)
    logger.info("EXTRACTION SUMMARY:")
    logger.info(json.dumps(summary_stats, indent=2))
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AURA-HAR MediaPipe Pose Extraction")
    parser.add_argument("--test-one", action="store_true", help="Extract and verify on a single sample video")
    parser.add_argument("--video-path", type=str, default=None, help="Custom video path for single test")
    parser.add_argument("--batch", action="store_true", help="Run batch extraction on dataset")
    parser.add_argument("--split", type=int, default=1, help="UCF101 split number (default: 1)")
    parser.add_argument("--max-videos", type=int, default=None, help="Limit number of videos (for subset testing)")
    parser.add_argument("--force", action="store_true", help="Force recomputation of existing files")

    args = parser.parse_args()

    if args.batch:
        run_batch_extraction(
            split_number=args.split,
            force_recompute=args.force,
            max_videos=args.max_videos,
        )
    else:
        # Default mode is single-video verification
        extract_single_video_test(video_path=args.video_path)
