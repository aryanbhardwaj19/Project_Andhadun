import cv2
import os
import sys
import numpy as np

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from edge_assist.emotion_classifier import EmotionClassifier
from edge_assist.face_detector import FaceDetector

def run_suite():
    classifier = EmotionClassifier(confidence_threshold=0.5) # Lowered for testing variety
    detector = FaceDetector()
    test_dir = "edge_assist/tests/data/emotions/test_suite"
    
    # Expected emotions based on file names
    expected_map = {
        "angry": "angry",
        "disgust": "disgust",
        "fear": "fear",
        "happy": "happy",
        "sad": "sad",
        "surprise": "surprise",
        "neutral": "neutral"
    }
    
    files = [f for f in os.listdir(test_dir) if f.startswith("emotion_")]
    
    print(f"\nRunning Emotion Detection Suite on {len(files)} images...")
    print("-" * 50)
    
    passed = 0
    total = 0
    
    results = []
    
    for f in files:
        # Extract expected emotion from filename (e.g., emotion_angry_123.png)
        parts = f.split('_')
        expected = parts[1]
        
        img_path = os.path.join(test_dir, f)
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"  [ERROR] Could not read {f}")
            continue
            
        # Detect face first (as the classifier expects a face ROI)
        faces = detector.detect(frame)
        if not faces:
            print(f"  [FAILED] {f} -> No face detected")
            results.append((f, expected, "No Face", 0.0))
            continue
            
        x, y, w, h = faces[0]
        roi = frame[y:y+h, x:x+w]
        
        # Classify emotion
        (label, conf), lat = classifier.classify(roi)
        
        status = "SUCCESS" if label == expected else "FAILED"
        if status == "SUCCESS": passed += 1
        total += 1
        
        print(f"  {status}: {f} -> Expected {expected}, Got {label} ({conf:.2f})")
        results.append((f, expected, label, conf))

    print("-" * 50)
    print(f"Final Result: {passed}/{total} passed.")
    
if __name__ == "__main__":
    run_test_suite = True
    if run_test_suite:
        run_suite()
