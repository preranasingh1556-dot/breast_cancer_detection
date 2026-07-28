"""
Data Preparation Script
-----------------------
The IDC dataset from Kaggle has this structure:
    IDC_regular_ps50_idx5/
        <patient_id>/
            0/   ← non-cancerous patches
            1/   ← cancerous patches

This script flattens it into ImageFolder format:
    data/processed/
        0/   ← all non-cancerous images
        1/   ← all cancerous images

STEPS BEFORE RUNNING:
1. Download dataset from Kaggle:
   https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images
2. Extract it so you have:
   breast_cancer_detection/
       IDC_regular_ps50_idx5/   ← extracted Kaggle folder
3. Run:  python prepare_data.py
"""

import os
import shutil
from tqdm import tqdm

RAW_DIR      = "IDC_regular_ps50_idx5"
OUTPUT_DIR   = "data/processed"
MAX_PER_CLASS = 10_000   # use None to copy all images (can be slow)


def prepare():
    if not os.path.exists(RAW_DIR):
        print(f"ERROR: '{RAW_DIR}' folder not found.")
        print("Please download the Kaggle dataset first (see instructions above).")
        return

    # Create output class folders
    for cls in ["0", "1"]:
        os.makedirs(os.path.join(OUTPUT_DIR, cls), exist_ok=True)

    counts = {"0": 0, "1": 0}

    patient_ids = os.listdir(RAW_DIR)
    print(f"Found {len(patient_ids)} patient folders. Copying images...")

    for pid in tqdm(patient_ids, desc="Processing patients"):
        pid_path = os.path.join(RAW_DIR, pid)
        if not os.path.isdir(pid_path):
            continue

        for cls in ["0", "1"]:
            cls_path = os.path.join(pid_path, cls)
            if not os.path.isdir(cls_path):
                continue

            for fname in os.listdir(cls_path):
                if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    continue

                if MAX_PER_CLASS and counts[cls] >= MAX_PER_CLASS:
                    continue

                src = os.path.join(cls_path, fname)
                # Prefix with patient id to avoid name collisions
                dst = os.path.join(OUTPUT_DIR, cls, f"{pid}_{fname}")
                shutil.copy2(src, dst)
                counts[cls] += 1

    print(f"\nDone!")
    print(f"  Non-cancerous (class 0): {counts['0']} images")
    print(f"  Cancerous     (class 1): {counts['1']} images")
    print(f"  Saved to: {OUTPUT_DIR}")
    print(f"\nNow run:  python train.py")


if __name__ == "__main__":
    prepare()
