import os
import shutil
import random
from pathlib import Path

# Define paths
SOURCE_DIR = Path("datasets/frames")
DEST_DIR = Path("datasets/split_frames")
CLASSES = ["Fall", "No_Fall"]
SPLIT_RATIO = [0.7, 0.15, 0.15]  # train, val, test
random.seed(42)

# Ensure target dirs exist
for split in ["train", "val", "test"]:
    for cls in CLASSES:
        (DEST_DIR / split / cls).mkdir(parents=True, exist_ok=True)

# Group files by sequence prefix
def group_by_sequence(image_dir):
    groups = {}
    for file in image_dir.glob("*.jpg"):
        prefix = "_".join(file.stem.split("_")[:-1])  # strip frame index
        groups.setdefault(prefix, []).append(file)
    return list(groups.values())

for cls in CLASSES:
    class_path = SOURCE_DIR / cls
    grouped_sequences = group_by_sequence(class_path)
    random.shuffle(grouped_sequences)

    total = len(grouped_sequences)
    train_end = int(SPLIT_RATIO[0] * total)
    val_end = train_end + int(SPLIT_RATIO[1] * total)

    split_map = {
        "train": grouped_sequences[:train_end],
        "val": grouped_sequences[train_end:val_end],
        "test": grouped_sequences[val_end:]
    }

    for split, groups in split_map.items():
        for group in groups:
            for file in group:
                dest = DEST_DIR / split / cls / file.name
                shutil.copy2(file, dest)

print("✅ Dataset split by sequence complete.")

