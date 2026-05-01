import cv2
import numpy as np
from fer.fer import FER
import time
import os

class EmotionClassifier:
    def __init__(self, confidence_threshold=0.65):
        """
        Initialize FER Emotion Classifier.
        use_tflite=True: Uses quantized TFLite model (22MB).
        mtcnn=False: Skips heavy MTCNN detection.
        """
        # Note: The library might still import tensorflow, but we use the TFLite interpreter.
        self.detector = FER(mtcnn=False, use_tflite=True)
        self.confidence_threshold = confidence_threshold
        # The TFLite model in fer 22.5.1 uses 64x64 or 48x48 depending on version.
        # detect_emotions handles resizing internally.

    def classify(self, face_roi):
        """
        Accepts a face ROI (can be 48x48 grayscale as per requirement).
        Returns: (emotion_label, confidence) tuple and latency in ms.
        """
        if face_roi is None or face_roi.size == 0:
            return ("unknown", 0.0), 0.0

        # FER library internally expects BGR to convert to grayscale, 
        # or it can handle grayscale if we are careful. 
        # To be safe and reuse the library's preprocessing, we convert to BGR.
        if len(face_roi.shape) == 2:
            bgr_roi = cv2.cvtColor(face_roi, cv2.COLOR_GRAY2BGR)
        else:
            bgr_roi = face_roi
            
        h, w = bgr_roi.shape[:2]
        
        start_time = time.time()
        # Passing face_rectangles=[(0, 0, w, h)] forces classification of the entire ROI.
        results = self.detector.detect_emotions(bgr_roi, face_rectangles=[(0, 0, w, h)])
        latency = (time.time() - start_time) * 1000

        if not results:
            return ("neutral", 0.0), latency

        emotions_dict = results[0]['emotions']
        # The library returns rounded values (e.g. 0.95), we find the max.
        top_emotion = max(emotions_dict, key=emotions_dict.get)
        confidence = emotions_dict[top_emotion]

        if confidence < self.confidence_threshold:
            return ("uncertain", confidence), latency
            
        return (top_emotion, confidence), latency

if __name__ == "__main__":
    # Functional check
    print("Initializing Emotion Classifier...")
    classifier = EmotionClassifier()
    
    # Test 1: Black ROI
    print("Test 1: Blank 48x48 grayscale ROI")
    test_roi = np.zeros((48, 48), dtype=np.uint8)
    (emotion, conf), lat = classifier.classify(test_roi)
    print(f"  Result: {emotion} ({conf:.2f}), Latency: {lat:.2f}ms")
    
    # Test 2: Random ROI
    print("Test 2: Random 48x48 grayscale ROI")
    test_roi = np.random.randint(0, 255, (48, 48), dtype=np.uint8)
    (emotion, conf), lat = classifier.classify(test_roi)
    print(f"  Result: {emotion} ({conf:.2f}), Latency: {lat:.2f}ms")
    
    assert lat < 100, f"Latency {lat:.2f}ms exceeds 100ms limit!"
    print("Functional test passed.")
