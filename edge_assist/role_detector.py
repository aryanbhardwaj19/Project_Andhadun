import cv2
import numpy as np
import onnxruntime as ort
import os
import time

class RoleDetector:
    def __init__(self, model_path=None, confidence_threshold=0.4, nms_threshold=0.45):
        """
        Initialize YOLOv5n Role Detector.
        """
        if model_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(base_dir, "models", "yolov5n.onnx")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")

        # Initialize ONNX Runtime
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.conf_threshold = confidence_threshold
        self.nms_threshold = nms_threshold

    def detect(self, image):
        """
        Detects people and classifies roles.
        Returns: List of {'box': [x, y, w, h], 'role': str}
        """
        h_img, w_img = image.shape[:2]
        
        # 1. Preprocess (320x320)
        blob = cv2.resize(image, (320, 320))
        blob = cv2.cvtColor(blob, cv2.COLOR_BGR2RGB)
        blob = blob.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))
        blob = np.expand_dims(blob, 0)

        # 2. Inference
        outputs = self.session.run(None, {self.input_name: blob})
        predictions = outputs[0][0]

        # 3. Filter class=0 (person) and apply NMS
        boxes = []
        confidences = []
        # Ultralytics ONNX output is [1, 84, 2100]
        # Transpose to [2100, 84] for easier iteration
        predictions = predictions.transpose()
        
        for pred in predictions:
            # Class probabilities start at index 4
            class_probs = pred[4:]
            conf = np.max(class_probs)
            if conf > self.conf_threshold:
                class_id = np.argmax(class_probs)
                if class_id == 0: # Person
                    x_c, y_c, w, h = pred[:4]
                    x = (x_c - w/2) * w_img / 320.0
                    y = (y_c - h/2) * h_img / 320.0
                    boxes.append([int(x), int(y), int(w * w_img / 320.0), int(h * h_img / 320.0)])
                    confidences.append(float(conf))

        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.nms_threshold)
        
        results = []
        if len(indices) > 0:
            for i in indices.flatten():
                box = boxes[i]
                role = self._classify_role(image, box)
                results.append({'box': box, 'role': role})
        
        return results

    def _classify_role(self, image, box):
        """
        Extract torso ROI and use k-means on HSV to find dominant color.
        """
        x, y, w, h = box
        h_img, w_img = image.shape[:2]
        
        # Torso ROI: middle vertical third of person bbox
        # Approximately from 20% to 50% of person height
        ty1 = max(0, y + int(h * 0.2))
        ty2 = min(h_img, y + int(h * 0.5))
        tx1 = max(0, x + int(w * 0.2))
        tx2 = min(w_img, x + int(w * 0.8))
        
        if ty2 <= ty1 or tx2 <= tx1:
            return "unknown"
            
        torso_roi = image[ty1:ty2, tx1:tx2]
        if torso_roi.size == 0:
            return "unknown"
            
        # k-means k=3 on HSV
        hsv_roi = cv2.cvtColor(torso_roi, cv2.COLOR_BGR2HSV)
        pixels = hsv_roi.reshape((-1, 3)).astype(np.float32)
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(pixels, 3, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        counts = np.bincount(labels.flatten())
        dom_hsv = centers[np.argmax(counts)]
        hue, sat, val = dom_hsv
        
        # Hue-based lookup (OpenCV H: 0-179)
        # red=delivery (0-10 or 160-179)
        # navy=security (100-130)
        # white=medical (S<40, V>200)
        
        if sat < 40 and val > 200:
            return "medical"
        if (hue < 10 or hue > 165) and sat > 50:
            return "delivery"
        if 100 < hue < 135 and sat > 50:
            return "security"
            
        return "unknown"

if __name__ == "__main__":
    # Quick functional test
    cap = cv2.VideoCapture(0)
    detector = RoleDetector()
    frame_count = 0
    roles = []
    
    print("Starting Role Detection Test. Runs every 5th frame. Press 'q' to quit.")
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
        
        start_time = time.time()
        if frame_count % 5 == 0:
            roles = detector.detect(frame)
        
        frame_count += 1
        fps = 1.0 / (time.time() - start_time)
        
        for r in roles:
            x, y, w, h = r['box']
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, f"Role: {r['role']}", (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        cv2.putText(frame, f"FPS: {fps:.2f} (Inference: {frame_count % 5 == 0})", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imshow('Role Detection P3', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        
    cap.release()
    cv2.destroyAllWindows()
