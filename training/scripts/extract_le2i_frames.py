import cv2
import os
from pathlib import Path

# Source video folders
source_dirs = {
    "Fall": "datasets/LE2I/raw_videos/Fall",
    "No_Fall": "datasets/LE2I/raw_videos/No_Fall"
}

# Destination frame output
output_base = Path("datasets/merged")

# Frame extraction rate (1/10 second = 10 FPS)
frame_interval = 0.1  # in seconds

# Loop through each class folder
for label, folder in source_dirs.items():
    folder_path = Path(folder)
    output_dir = output_base / label
    output_dir.mkdir(parents=True, exist_ok=True)

    for video_file in folder_path.glob("*.avi"):
        cap = cv2.VideoCapture(str(video_file))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_step = int(fps * frame_interval)

        frame_idx = 0
        saved_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_step == 0:
                filename = f"{video_file.stem}_frame{frame_idx}.jpg"
                save_path = output_dir / filename
                cv2.imwrite(str(save_path), frame)
                saved_idx += 1

            frame_idx += 1

        cap.release()
        print(f"✅ Extracted {saved_idx} frames from {video_file.name} → {label}/")

print("✅ All Le2i videos processed.")