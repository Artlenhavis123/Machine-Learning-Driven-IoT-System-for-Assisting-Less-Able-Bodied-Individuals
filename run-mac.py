import cv2
import time
import tkinter as tk
from tkinter import ttk, messagebox
from collections import deque
from threading import Thread
from PIL import Image, ImageTk
from src.inferenceMac import TFLiteModel
from src.preprocess import preprocess_frame
from src.alert import trigger_alert
from dotenv import load_dotenv
import os

load_dotenv()

# Settings
MODEL_PATH = "models/fall_model.tflite"
LABELS = ["Fall", "Idle", "Lying", "Pre_Fall", "Sitting"]
CONFIDENCE_THRESHOLD = 0.3
BUFFER_SIZE = 15
FALL_TRIGGER_COUNT = 3
PRE_FALL_TRIGGER_COUNT = 3

# Model
model = TFLiteModel(MODEL_PATH)
prediction_buffer = deque(maxlen=BUFFER_SIZE)

# Webcam
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    raise IOError("Cannot open webcam")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# UI setup
root = tk.Tk()
root.title("Fall Detection System")
root.geometry("1000x600")
root.rowconfigure(0, weight=1)
root.columnconfigure(1, weight=1)

frame_ui = ttk.Frame(root, padding=10)
frame_ui.grid(row=0, column=0, sticky="ns")

# Input fields
email_label = ttk.Label(frame_ui, text="Alert Email:")
email_label.grid(row=0, column=0, sticky="e")
email_entry = ttk.Entry(frame_ui, width=30)
email_entry.grid(row=0, column=1)
email_entry.insert(0, os.getenv("EMAIL_RECIPIENT", ""))

sms_label = ttk.Label(frame_ui, text="SMS Number:")
sms_label.grid(row=1, column=0, sticky="e")
sms_entry = ttk.Entry(frame_ui, width=30)
sms_entry.grid(row=1, column=1)
sms_entry.insert(0, os.getenv("SMS_RECIPIENT", ""))

status_label = ttk.Label(frame_ui, text="Status:", font=("Arial", 12, "bold"))
status_label.grid(row=2, column=0, columnspan=2, pady=(10, 0))

status_text = tk.StringVar()
status_text.set("Waiting for prediction...")
status_display = ttk.Label(frame_ui, textvariable=status_text, foreground="blue")
status_display.grid(row=3, column=0, columnspan=2)

fps_text = tk.StringVar()
fps_text.set("FPS: 0.00")
fps_display = ttk.Label(frame_ui, textvariable=fps_text)
fps_display.grid(row=4, column=0, columnspan=2, pady=(5, 0))

# Canvas for video (dynamically resize)
canvas_frame = ttk.Frame(root)
canvas_frame.grid(row=0, column=1, sticky="nsew")
canvas_frame.columnconfigure(0, weight=1)
canvas_frame.rowconfigure(0, weight=1)
canvas = tk.Canvas(canvas_frame, bg="black")
canvas.grid(row=0, column=0, sticky="nsew")

photo = None  # Reference to prevent garbage collection

# Video rendering function
def show_video():
    global photo
    fps = 0
    frame_count = 0
    start_time = time.time()
    fall_detected = False

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        input_data = preprocess_frame(frame)
        prediction, confidence, probs = model.predict(input_data)
        label = LABELS[prediction]

        if confidence > CONFIDENCE_THRESHOLD:
            prediction_buffer.append(label)
        else:
            prediction_buffer.append("Unknown")

        pre_fall_count = prediction_buffer.count("Pre_Fall")
        fall_count = prediction_buffer.count("Fall")

        if pre_fall_count >= PRE_FALL_TRIGGER_COUNT and fall_count >= FALL_TRIGGER_COUNT:
            status_text.set("🚨 Fall Detected")
            status_display.config(foreground="red")
            canvas.config(bg="red")
            trigger_alert()
            prediction_buffer.clear()
            fall_detected = True
        elif confidence > CONFIDENCE_THRESHOLD:
            status_text.set(f"{label}: {confidence * 100:.1f}%")
            status_display.config(foreground="green")
            canvas.config(bg="black")
            fall_detected = False
        else:
            status_text.set("Unknown")
            status_display.config(foreground="gray")
            canvas.config(bg="black")
            fall_detected = False

        # FPS calculation
        frame_count += 1
        elapsed = time.time() - start_time
        if elapsed > 1:
            fps = frame_count / elapsed
            frame_count = 0
            start_time = time.time()
        fps_text.set(f"FPS: {fps:.2f}")

        # Resize frame to canvas size
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        resized_frame = cv2.resize(frame, (canvas_width, canvas_height))

        # Convert and show frame
        display_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        display_image = Image.fromarray(display_frame)
        photo = ImageTk.PhotoImage(image=display_image)
        canvas.create_image(0, 0, image=photo, anchor=tk.NW)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    root.quit()

# Launch video thread
video_thread = Thread(target=show_video)
video_thread.daemon = True
video_thread.start()

root.mainloop()