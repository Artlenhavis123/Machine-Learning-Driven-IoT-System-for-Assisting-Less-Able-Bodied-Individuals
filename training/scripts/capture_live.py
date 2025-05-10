import cv2
import os
import time
import numpy as np

# Settings
LABELS = ["Idle", "Fall", "Lying", "Pre_Fall", "Sitting"]
SAVE_DIR = "datasets/frames/"
camera_sources = [
    # List Camera sources here
]

# Create label folders
for label in LABELS:
    label_dir = os.path.join(SAVE_DIR, label)
    os.makedirs(label_dir, exist_ok=True)

# Initialise cameras
caps = [cv2.VideoCapture(src) for src in camera_sources]

# FPS tracking per camera
fps = [0] * len(caps)
frame_counts = [0] * len(caps)
start_times = [time.time()] * len(caps)

print("\nPress keys to label and save from ALL cameras:")
for idx, label in enumerate(LABELS):
    print(f"'{idx}' - {label}")
print("'q' - Quit\n")

saving_fall = False
save_start_time = None
save_duration = 5  # seconds

while True:
    frames = []
    for i, cap in enumerate(caps):
        ret, frame = cap.read()
        if not ret:
            frame = np.zeros((240, 320, 3), dtype=np.uint8)
            cv2.putText(frame, "Camera Disconnected", (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        frame = cv2.resize(frame, (320, 240))

        frame_counts[i] += 1
        elapsed = time.time() - start_times[i]
        if elapsed >= 1.0:
            fps[i] = frame_counts[i] / elapsed
            frame_counts[i] = 0
            start_times[i] = time.time()

        cv2.putText(frame, f"FPS: {fps[i]:.1f}", (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        frames.append(frame)

    combined_frame = np.hstack(frames)
    cv2.imshow("Multi-Camera Capture", combined_frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif chr(key).isdigit():
        idx = int(chr(key))
        if 0 <= idx < len(LABELS):
            label = LABELS[idx]
            timestamp = int(time.time() * 1000)

            if label == "Fall":
                print("⚡ Fall detected: Saving next 5 seconds of frames...")
                saving_fall = True
                save_start_time = time.time()
            else:
                # Save immediately for other labels
                for i, frame in enumerate(frames):
                    filename = os.path.join(SAVE_DIR, label, f"{label}_cam{i}_{timestamp}.jpg")
                    cv2.imwrite(filename, frame)
                    print(f"Saved {filename}")

    # Auto-saving frames for Fall sequence
    if saving_fall:
        elapsed_save = time.time() - save_start_time
        if elapsed_save <= save_duration:
            current_label = "Pre_Fall" if elapsed_save <= 2 else "Fall"
            timestamp = int(time.time() * 1000)
            for i, frame in enumerate(frames):
                filename = os.path.join(SAVE_DIR, current_label, f"{current_label}_cam{i}_{timestamp}.jpg")
                cv2.imwrite(filename, frame)
                print(f"[{current_label}] Saved {filename}")
        else:
            saving_fall = False
            print("✅ Fall sequence capture complete!")

# Cleanup
for cap in caps:
    cap.release()
cv2.destroyAllWindows()
