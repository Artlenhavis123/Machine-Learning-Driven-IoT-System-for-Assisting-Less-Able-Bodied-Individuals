import os
import shutil
import random
from pathlib import Path
from collections import defaultdict

# Config
SOURCE_DIR = Path("datasets/frames")
DEST_DIR = Path("datasets/split_frames")
CLASS_NAMES = ["Fall", "Idle", "Pre_Fall", "Sitting", "Lying"]
SPLIT_RATIOS = {"train": 0.7, "val": 0.15, "test": 0.15}
SEED = 42
random.seed(SEED)

# check the dest directory
if not DEST_DIR.exists():
    DEST_DIR.mkdir(parents=True, exist_ok=True)
else:
    for item in DEST_DIR.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()

# Setup destination directories
for split in SPLIT_RATIOS:
    for class_name in CLASS_NAMES:
        split_path = DEST_DIR / split / class_name
        split_path.mkdir(parents=True, exist_ok=True)

# Collect and shuffle images by class
all_images = defaultdict(list)

for class_name in CLASS_NAMES:
    class_dir = SOURCE_DIR / class_name
    for img_file in class_dir.glob("*.[jp][pn]g"):
        all_images[class_name].append(img_file)

    random.shuffle(all_images[class_name])

# Split and copy
for class_name, files in all_images.items():
    n_total = len(files)
    n_train = int(n_total * SPLIT_RATIOS["train"])
    n_val = int(n_total * SPLIT_RATIOS["val"])

    splits = {
        "train": files[:n_train],
        "val": files[n_train:n_train + n_val],
        "test": files[n_train + n_val:]
    }

    for split_name, split_files in splits.items():
        for src_file in split_files:
            dst = DEST_DIR / split_name / class_name / src_file.name
            shutil.copy2(src_file, dst)

print("✅ Multi-class dataset split complete.")