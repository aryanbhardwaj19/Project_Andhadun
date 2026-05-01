import cv2
import time
from edge_assist.face_detector import FaceDetector
from edge_assist.emotion_classifier import EmotionClassifier

def live_emotion_test():
    # Initialize detectors
    face_detector = FaceDetector()
    emotion_classifier = EmotionClassifier(confidence_threshold=0.6)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Starting Live Emotion Detection Test.")
    print("Controls: Press 'q' to quit.")
    
    frame_count = 0
    faces = []
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
            
        start_time = time.time()
        
        # Detect faces every 2 frames to maintain high FPS
        if frame_count % 2 == 0:
            faces = face_detector.detect(frame)
            
        for (x, y, w, h) in faces:
            # Extract Face ROI
            face_roi = frame[y:y+h, x:x+w]
            if face_roi.size == 0:
                continue
                
            # Classify Emotion
            (label, conf), lat = emotion_classifier.classify(face_roi)
            
            # Draw Bounding Box and Label
            color = (0, 255, 0) # Green for face
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            display_text = f"{label} ({conf:.2f})"
            cv2.putText(frame, display_text, (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Performance Overlay
        frame_count += 1
        fps = 1.0 / (time.time() - start_time)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imshow('Live Emotion Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    live_emotion_test()
