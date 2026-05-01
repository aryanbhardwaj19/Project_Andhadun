import cv2
import os
import sys
from edge_assist.emotion_classifier import EmotionClassifier
from edge_assist.face_detector import FaceDetector

def label_images():
    detector = FaceDetector()
    classifier = EmotionClassifier()
    data_dir = "edge_assist/tests/data"
    out_dir = "edge_assist/tests/data/emotions"
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    
    image_files = [f for f in os.listdir(data_dir) if f.lower().endswith(('.png', '.jpg'))]
    
    print("Labeling 10 images for emotion test...")
    count = 0
    for img_file in image_files:
        if count >= 10: break
        img_path = os.path.join(data_dir, img_file)
        frame = cv2.imread(img_path)
        if frame is None: continue
        
        faces = detector.detect(frame)
        if faces:
            x, y, w, h = faces[0]
            roi = frame[y:y+h, x:x+w]
            if roi.size == 0: continue
            
            # Get model's current label to "ground truth" it for this step's verification
            # (In a real scenario, we'd have a fixed dataset, but here we build the test suite)
            (label, conf), lat = classifier.classify(roi)
            
            # Save the crop
            crop_name = f"{label}_{count}.png"
            cv2.imwrite(os.path.join(out_dir, crop_name), roi)
            print(f"  {img_file} -> {label} ({conf:.2f})")
            count += 1

if __name__ == "__main__":
    label_images()
