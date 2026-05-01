import cv2
import os
import sys
import time
from edge_assist.emotion_classifier import EmotionClassifier

def test_emotion_performance():
    classifier = EmotionClassifier(confidence_threshold=0.65)
    data_dir = "edge_assist/tests/data/emotions"
    
    if not os.path.exists(data_dir):
        print(f"Error: Data directory {data_dir} not found.")
        sys.exit(1)
        
    image_files = [f for f in os.listdir(data_dir) if f.lower().endswith(('.png', '.jpg'))]
    
    print(f"Running Emotion Classification Test on {len(image_files)} crops...")
    
    latencies = []
    success_count = 0
    
    for img_file in image_files:
        # Expected label is the first part of the filename, e.g. "happy_0.png"
        expected_label = img_file.split('_')[0]
        img_path = os.path.join(data_dir, img_file)
        roi = cv2.imread(img_path)
        
        if roi is None:
            print(f"FAILED: Could not read {img_file}")
            continue
            
        (label, conf), lat = classifier.classify(roi)
        latencies.append(lat)
        
        if label == expected_label:
            print(f"  SUCCESS: {img_file} -> {label} ({conf:.2f}) [{lat:.2f}ms]")
            success_count += 1
        else:
            print(f"  FAILED: {img_file} -> Expected {expected_label}, Got {label} ({conf:.2f}) [{lat:.2f}ms]")
            
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    print(f"\nFinal Result: {success_count}/{len(image_files)} passed.")
    print(f"Average Latency: {avg_latency:.2f}ms")
    
    assert success_count >= len(image_files), "Classification must be consistent with labeled crops."
    assert avg_latency <= 100, f"Average latency {avg_latency:.2f}ms exceeds 100ms limit."
    print("P2 DELIVERABLES VERIFIED: Latency and Accuracy confirmed.")

if __name__ == "__main__":
    test_emotion_performance()
