import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import os

class FaceDetector:
    def __init__(self, min_detection_confidence=0.5, model_path=None):
        """
        Initialize MediaPipe Face Detection using Tasks API.
        """
        if model_path is None:
            # Default path relative to this file
            base_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(base_dir, "models", "face_detector.tflite")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}. Run download_models.py first.")

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceDetectorOptions(
            base_options=base_options,
            min_detection_confidence=min_detection_confidence
        )
        self.detector = vision.FaceDetector.create_from_options(options)

    def detect(self, image):
        """
        Detect faces in an image.
        Returns: List of bounding boxes [xmin, ymin, width, height] sorted by area (largest first).
        """
        if image is None:
            return []

        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create MediaPipe Image object
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        
        # Detect faces
        detection_result = self.detector.detect(mp_image)

        bboxes = []
        if detection_result.detections:
            h, w, _ = image.shape
            for detection in detection_result.detections:
                bbox = detection.bounding_box
                # Tasks API returns absolute coordinates directly or relative? 
                # Let's check: bounding_box is in pixels for detect(), relative for detect_async() usually.
                # Actually for vision.FaceDetector.detect(), it's in pixels.
                xmin = int(bbox.origin_x)
                ymin = int(bbox.origin_y)
                width = int(bbox.width)
                height = int(bbox.height)
                
                if width > 0 and height > 0:
                    bboxes.append([xmin, ymin, width, height])

        # Sort by area (width * height) descending - proximity sort
        bboxes.sort(key=lambda x: x[2] * x[3], reverse=True)
        return bboxes

if __name__ == "__main__":
    # Quick functional test with frame skipping
    cap = cv2.VideoCapture(0)
    detector = FaceDetector()
    
    print("Starting Face Detection Test (Tasks API + Frame-Skip). Press 'q' to quit.")
    frame_count = 0
    faces = []
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        start_time = time.time()
        
        # Process every 2nd frame
        if frame_count % 2 == 0:
            faces = detector.detect(frame)
        
        frame_count += 1
        duration = time.time() - start_time
        fps = 1.0 / duration if duration > 0 else 0

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"Face: {w}x{h}", (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.putText(frame, f"FPS (Process/Skip): {fps:.2f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imshow('Face Detection P1', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
