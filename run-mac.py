import cv2
import time
from src.inferenceMac import TFLiteModel
from src.preprocess import preprocess_frame
from src.alert import trigger_alert

# Settings
MODEL_PATH = "models/fall_model.tflite"
LABELS = ["Fall", "Idle", "Lying", "Pre_Fall", "Sitting"]
CONFIDENCE_THRESHOLD = 0.7

# Initialise model
model = TFLiteModel(MODEL_PATH)

# Initialise webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

# Force webcam resolution (lower for faster FPS)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

# FPS tracking variables
fps = 0
frame_count = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Preprocess frame
    input_data = preprocess_frame(frame)

    # Inference
    prediction, confidence = model.predict(input_data)
    label = LABELS[prediction]

    # Prepare text
    if confidence > CONFIDENCE_THRESHOLD:
        display_text = f"{label}: {confidence * 100:.1f}%"
        if label == "Fall":
            trigger_alert()
    else:
        display_text = "Unknown"

    # Calculate FPS
    frame_count += 1
    elapsed_time = time.time() - start_time
    if elapsed_time > 1:
        fps = frame_count / elapsed_time
        frame_count = 0
        start_time = time.time()

    # Annotate frame
    cv2.putText(frame, display_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Display frame
    cv2.imshow("Fall Detection", frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()

