import cv2
import time
from collections import deque
from src.inferenceMac import TFLiteModel
from src.preprocess import preprocess_frame
from src.alert import trigger_alert

# Settings
MODEL_PATH = "models/fall_model.tflite"
LABELS = ["Fall", "Idle", "Lying", "Pre_Fall", "Sitting"]
CONFIDENCE_THRESHOLD = 0.3 
BUFFER_SIZE = 15
FALL_TRIGGER_COUNT = 3
PRE_FALL_TRIGGER_COUNT = 3

# Initialize model
model = TFLiteModel(MODEL_PATH)

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

# Set webcam resolution for performance
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

# Initialize frame buffer for temporal logic
prediction_buffer = deque(maxlen=BUFFER_SIZE)

# FPS tracking
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

    # Predict
    prediction, confidence, probs = model.predict(input_data)
    label = LABELS[prediction]

    # Print class confidence
    for i, prob in enumerate(probs[0]):
        print(f"{LABELS[i]}: {prob:.2f}")



    # Temporal buffer logic
    if confidence > CONFIDENCE_THRESHOLD:
        prediction_buffer.append(label)
    else:
        prediction_buffer.append("Unknown")

    pre_fall_count = prediction_buffer.count("Pre_Fall")
    fall_count = prediction_buffer.count("Fall")

    if pre_fall_count >= PRE_FALL_TRIGGER_COUNT and fall_count >= FALL_TRIGGER_COUNT:
        display_text = "🚨 Fall Detected"
        trigger_alert()
        prediction_buffer.clear()  # Avoid repeat alerts
    elif confidence > CONFIDENCE_THRESHOLD:
        display_text = f"{label}: {confidence * 100:.1f}%"
    else:
        display_text = "Unknown"
    
    # FPS tracking
    frame_count += 1
    elapsed = time.time() - start_time
    if elapsed > 1:
        fps = frame_count / elapsed
        frame_count = 0
        start_time = time.time()

    # Draw output
    cv2.putText(frame, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.imshow("Fall Detection", frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()