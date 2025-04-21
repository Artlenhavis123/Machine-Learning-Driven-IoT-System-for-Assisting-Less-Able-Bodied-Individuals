import os
import shutil
from pathlib import Path

SOURCE_DIR = Path("datasets/UPFD")
DEST_DIR = Path("datasets/frames")
FALL_DIR = DEST_DIR / "Fall"
NO_FALL_DIR = DEST_DIR / "No_Fall"

# Ensure destination folders exist
FALL_DIR.mkdir(parents=True, exist_ok=True)
NO_FALL_DIR.mkdir(parents=True, exist_ok=True)

def classify_and_copy():
    for folder in SOURCE_DIR.iterdir():
        if not folder.is_dir():
            continue

        folder_name = folder.name.lower()

        if "fall" in folder_name and "no fall" not in folder_name:
            label = "Fall"
            target_dir = FALL_DIR
        elif "adl" in folder_name or "no fall" in folder_name:
            label = "No_Fall"
            target_dir = NO_FALL_DIR
        else:
            print(f"⚠️ Skipping unclassified folder: {folder.name}")
            continue

        print(f"📂 Copying from: {folder.name} → {label}")

        for i, file in enumerate(sorted(folder.glob("*.[jp][pn]g"))):
            if i % 2 != 0:
                continue  # ✅ Sample every 2nd frame
            new_name = f"{folder.name.replace(' ', '_')}_{file.name}"
            dest = target_dir / new_name
            shutil.copy2(file, dest)


    print("✅ All UP-Fall frames imported and sorted.")

if __name__ == "__main__":
    classify_and_copy()

