from pathlib import Path


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = PROJECT_ROOT / "data" / "har" / "raw" / "UCF-101"
SPLIT_DIR = PROJECT_ROOT / "data" / "har" / "splits"


# ============================================================
# ACTIVITIES WE ARE USING
# ============================================================

SELECTED_CLASSES = {
    "BodyWeightSquats",
    "JumpingJack",
    "JumpRope",
    "Lunges",
    "Punch",
    "PushUps",
    "TaiChi",
    "WallPushups",
}


# ============================================================
# LOAD CLASS INDEX
# ============================================================

def load_class_index():
    """
    Reads classInd.txt.

    Returns:
        Dictionary mapping activity name -> class ID
    """

    class_file = SPLIT_DIR / "classInd.txt"

    class_index = {}

    with open(class_file, "r") as file:
        for line in file:
            parts = line.strip().split()

            if len(parts) >= 2:
                class_id = int(parts[0])
                class_name = parts[1]

                class_index[class_name] = class_id

    return class_index


# ============================================================
# LOAD TRAINING LIST
# ============================================================

def load_train_list(split_number=1):
    """
    Loads one official UCF101 training split.

    Example:
        split_number=1
        -> trainlist01.txt
    """

    split_file = SPLIT_DIR / f"trainlist0{split_number}.txt"

    videos = []

    with open(split_file, "r") as file:
        for line in file:

            line = line.strip()

            if not line:
                continue

            # trainlist files contain:
            # ClassName/video_name.avi class_id

            parts = line.split()

            video_relative_path = parts[0]

            activity = video_relative_path.split("/")[0]

            if activity in SELECTED_CLASSES:

                video_path = DATASET_DIR / video_relative_path

                videos.append({
                    "path": video_path,
                    "activity": activity
                })

    return videos


# ============================================================
# LOAD TEST LIST
# ============================================================

def load_test_list(split_number=1):
    """
    Loads one official UCF101 test split.
    """

    split_file = SPLIT_DIR / f"testlist0{split_number}.txt"

    videos = []

    with open(split_file, "r") as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            video_relative_path = line

            activity = video_relative_path.split("/")[0]

            if activity in SELECTED_CLASSES:

                video_path = DATASET_DIR / video_relative_path

                videos.append({
                    "path": video_path,
                    "activity": activity
                })

    return videos


# ============================================================
# CHECK DATASET
# ============================================================

def check_dataset(videos):
    """
    Checks whether all referenced videos exist.
    """

    missing = []

    for item in videos:

        if not item["path"].exists():
            missing.append(item["path"])

    print(f"Total videos: {len(videos)}")
    print(f"Missing videos: {len(missing)}")

    if missing:

        print("\nFirst missing files:")

        for path in missing[:10]:
            print(path)

    else:
        print("All videos found successfully!")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("AURA-HAR DATASET CHECK")
    print("=" * 60)

    print("\nDataset directory:")
    print(DATASET_DIR)

    print("\nSplit directory:")
    print(SPLIT_DIR)

    print("\nSelected activities:")

    for activity in sorted(SELECTED_CLASSES):
        print(" -", activity)

    # Load class mapping
    class_index = load_class_index()

    print("\nClass IDs:")

    for activity in sorted(SELECTED_CLASSES):

        if activity in class_index:
            print(
                f" - {activity}: "
                f"{class_index[activity]}"
            )
        else:
            print(
                f" - {activity}: NOT FOUND"
            )

    # Load split 1
    train_videos = load_train_list(1)
    test_videos = load_test_list(1)

    print("\n" + "=" * 60)
    print("TRAINING DATA")
    print("=" * 60)

    check_dataset(train_videos)

    print("\n" + "=" * 60)
    print("TESTING DATA")
    print("=" * 60)

    check_dataset(test_videos)

    print("\n" + "=" * 60)
    print("EXAMPLE TRAINING VIDEOS")
    print("=" * 60)

    for item in train_videos[:10]:

        print(
            f"{item['activity']:20s} "
            f"{item['path'].name}"
        )