import cv2
import os
import time
import numpy as np
from edge_assist.face_detector import FaceDetector
from edge_assist.emotion_classifier import EmotionClassifier
from edge_assist.role_detector import RoleDetector
from edge_assist.fusion import DecisionFusion, FeedbackEvent
from edge_assist.haptic import HapticSimulator

class EdgeAIPipeline:
    def __init__(self):
        self.face_detector = FaceDetector()
        self.emotion_classifier = EmotionClassifier(confidence_threshold=0.6)
        self.role_detector = RoleDetector()
        self.fusion = DecisionFusion(buffer_size=5, cooldown_seconds=5.0)
        self.haptic = HapticSimulator()
        
    def process_frame(self, frame):
        h, w = frame.shape[:2]
        
        # 1. Role Detection (every 5 frames or so, but here every frame for simplicity)
        roles = self.role_detector.detect(frame)
        
        # 2. Face & Emotion Detection
        faces = self.face_detector.detect(frame)
        
        fused_event = FeedbackEvent.NO_ALERT
        
        for (fx, fy, fw, fh) in faces:
            # Face ROI
            face_roi = frame[fy:fy+fh, fx:fx+fw]
            if face_roi.size == 0: continue
            
            # Emotion
            (emotion, conf), _ = self.emotion_classifier.classify(face_roi)
            
            # Fusion logic
            face_area_ratio = (fw * fh) / (w * h)
            fused_event = self.fusion.update(emotion, face_area_ratio)
            
            # Visualize
            cv2.rectangle(frame, (fx, fy), (fx+fw, fy+fh), (0, 255, 0), 2)
            cv2.putText(frame, f"{emotion} ({conf:.2f})", (fx, fy-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
        # Draw Roles
        for r in roles:
            rx, ry, rw, rh = r['box']
            cv2.rectangle(frame, (rx, ry), (rx+rw, ry+rh), (255, 0, 0), 2)
            cv2.putText(frame, f"Role: {r['role']}", (rx, ry+rh+20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
        # Trigger Haptic if fused event fires
        if fused_event != FeedbackEvent.NO_ALERT:
            pattern = 'double-short' if fused_event == FeedbackEvent.POSITIVE else 'long'
            if fused_event == FeedbackEvent.NEUTRAL: pattern = 'single-short'
            
            self.haptic.vibrate(pattern)
            cv2.putText(frame, f"ALERT: {fused_event.value.upper()}", (w//2 - 100, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            
        return frame

def run_demo():
    pipeline = EdgeAIPipeline()
    test_suite_dir = "edge_assist/tests/data/emotions/test_suite"
    output_dir = "edge_assist/tests/demo_output"
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    
    # We'll use the generated emotion images to simulate a stream
    files = [f for f in os.listdir(test_suite_dir) if f.startswith("emotion_")]
    
    print(f"Starting Demo Pipeline on {len(files)} test images...")
    
    for i, f in enumerate(files):
        img_path = os.path.join(test_suite_dir, f)
        frame = cv2.imread(img_path)
        if frame is None: continue
        
        # Simulate multiple frames for fusion to kick in (since fusion needs 5 frames)
        for _ in range(5):
            processed = pipeline.process_frame(frame.copy())
            
        save_path = os.path.join(output_dir, f"demo_{i}.jpg")
        cv2.imwrite(save_path, processed)
        print(f"  Processed {f} -> saved to {save_path}")

if __name__ == "__main__":
    run_demo()
