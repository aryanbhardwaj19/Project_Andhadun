import cv2
import os

def test_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open video device")
    
    ret, frame = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError("Could not read frame from video device")
    
    save_path = "edge-assist/tests/cam_test.jpg"
    cv2.imwrite(save_path, frame)
    cap.release()
    
    if os.path.exists(save_path):
        print(f"SUCCESS: Frame saved to {save_path}")
    else:
        raise FileNotFoundError("Failed to save frame")

if __name__ == "__main__":
    test_camera()
