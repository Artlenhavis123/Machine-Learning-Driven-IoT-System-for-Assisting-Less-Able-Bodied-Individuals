import os
import shutil

SOURCE_DIR = "datasets/URFD"
TARGET_DIR = "datasets/frames"
FALL_DIR = os.path.join(TARGET_DIR, "Fall")
NO_FALL_DIR = os.path.join(TARGET_DIR, "No_Fall")
FRAME_INTERVAL = 2  # Extract every 2nd frame

# Create output folders
os.makedirs(FALL_DIR, exist_ok=True)
os.makedirs(NO_FALL_DIR, exist_ok=True)

# Process each sequence
for folder in os.listdir(SOURCE_DIR):
    sequence_path = os.path.join(SOURCE_DIR, folder)
    if not os.path.isdir(sequence_path):
        continue

    # Decide label based on folder name
    label = "Fall" if folder.startswith("fall") else "No_Fall"
    target_subdir = FALL_DIR if label == "Fall" else NO_FALL_DIR

    images = sorted([f for f in os.listdir(sequence_path) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    for i, image_file in enumerate(images):
        if i % FRAME_INTERVAL != 0:
            continue
        src_path = os.path.join(sequence_path, image_file)
        prefix = folder.replace("-cam0-rgb", "").replace("-cam1-rgb", "")
        dest_name = f"{prefix}_{i:04d}.jpg"
        dest_path = os.path.join(target_subdir, dest_name)
        shutil.copy(src_path, dest_path)

    print(f"📁 {folder} → {label} — {len(images)} frames found, extracting every {FRAME_INTERVAL}th frame.")

print("✅ Frame extraction complete.")

