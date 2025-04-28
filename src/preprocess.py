import cv2
import numpy as np

def preprocess_frame(frame):
    img_height, img_width = 224, 224
    frame = cv2.resize(frame, (img_width, img_height))
    frame = frame.astype(np.float32) / 255.0
    frame = np.expand_dims(frame, axis=0)
    return frame
