"""
AURA — Model Asset Downloader
Pre-downloads required YOLO model weights if they are not already present.
Called by setup.bat during first-time setup.

Strategy:
  1. If the model file already exists in models/, skip.
  2. Otherwise, use Ultralytics' built-in auto-download by instantiating the model.
     This fetches the correct weights from the official Ultralytics CDN.
"""

import os
import sys

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "models")

# ── Required model assets ────────────────────────────────────────────────────
REQUIRED_MODELS = [
    "yoloe-26s-seg.pt",
]


def ensure_model(filename: str, models_dir: str) -> bool:
    """Ensures a model file is available, downloading via Ultralytics if needed."""
    dest_path = os.path.join(models_dir, filename)

    if os.path.exists(dest_path):
        size_mb = os.path.getsize(dest_path) / (1024 * 1024)
        print(f"  OK  {filename} already exists ({size_mb:.1f} MB) - skipping.")
        return True

    print(f"  Downloading {filename} via Ultralytics auto-download...")
    print(f"    (This may take a few minutes on first run)")
    try:
        # Ultralytics auto-downloads the model from its CDN on first use.
        # We instantiate the model in the models/ directory context to cache it there.
        from ultralytics import YOLO
        
        # Ultralytics downloads to the CWD or a cache; we pre-set the path
        os.makedirs(models_dir, exist_ok=True)
        original_cwd = os.getcwd()
        os.chdir(models_dir)
        try:
            model = YOLO(filename)
            del model  # Free memory immediately
        finally:
            os.chdir(original_cwd)

        # Verify the file was created
        if os.path.exists(dest_path):
            size_mb = os.path.getsize(dest_path) / (1024 * 1024)
            print(f"  OK  {filename} downloaded successfully ({size_mb:.1f} MB).")
            return True
        else:
            # Ultralytics may have cached it elsewhere; check common locations
            home_cache = os.path.join(os.path.expanduser("~"), ".ultralytics", filename)
            if os.path.exists(home_cache):
                import shutil
                shutil.copy2(home_cache, dest_path)
                size_mb = os.path.getsize(dest_path) / (1024 * 1024)
                print(f"  OK  {filename} copied from cache ({size_mb:.1f} MB).")
                return True
            print(f"  NOTE: {filename} will be auto-downloaded on first run of AURA.")
            return True  # Not fatal — Ultralytics will handle it

    except ImportError:
        print(f"  NOTE: ultralytics package not yet installed.")
        print(f"    {filename} will be auto-downloaded when you first run AURA.")
        return True  # Not fatal
    except Exception as e:
        print(f"  WARNING: Could not pre-download {filename}: {e}")
        print(f"    The model will be auto-downloaded on first run of AURA.")
        return True  # Not fatal — Ultralytics auto-downloads


def main():
    models_dir = os.path.normpath(MODELS_DIR)
    print(f"\n  Model directory: {models_dir}")
    os.makedirs(models_dir, exist_ok=True)

    for filename in REQUIRED_MODELS:
        ensure_model(filename, models_dir)

    print("\n  Model setup complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
