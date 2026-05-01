import urllib.request
import os

def download_model():
    url = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
    target_dir = "edge_assist/models"
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    filepath = os.path.join(target_dir, "face_detector.tflite")
    if not os.path.exists(filepath):
        print(f"Downloading face detector model...")
        urllib.request.urlretrieve(url, filepath)
        print("Done.")

if __name__ == "__main__":
    download_model()
