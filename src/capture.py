import cv2

def capture_frame():
    cap = cv2.VideoCapture(0)  # 0 = default USB webcam
    if not cap.isOpened():
        raise IOError("Cannot open webcam")
    ret, frame = cap.read()
    cap.release()
    return frame

