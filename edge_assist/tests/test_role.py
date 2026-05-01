import cv2
import os
import sys

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from edge_assist.role_detector import RoleDetector

def test_role_detection():
    detector = RoleDetector()
    data_dir = "edge_assist/tests/data/roles"
    
    if not os.path.exists(data_dir):
        print(f"Error: Data directory {data_dir} not found.")
        sys.exit(1)
        
    image_files = [f for f in os.listdir(data_dir) if f.lower().endswith(('.png', '.jpg'))]
    
    print(f"Running Role Detection Test on {len(image_files)} images...")
    
    success_count = 0
    for img_file in image_files:
        # Expected role from filename: role_delivery_1.png -> delivery
        expected_role = img_file.split('_')[1]
        img_path = os.path.join(data_dir, img_file)
        image = cv2.imread(img_path)
        
        if image is None:
            print(f"FAILED: Could not read {img_file}")
            continue
            
        results = detector.detect(image)
        
        # Check if any detected person has the expected role
        found_expected = False
        for r in results:
            if r['role'] == expected_role:
                found_expected = True
                break
        
        if found_expected:
            print(f"  SUCCESS: {img_file} -> Found {expected_role}")
            success_count += 1
        else:
            detected_roles = [r['role'] for r in results]
            print(f"  FAILED: {img_file} -> Expected {expected_role}, Got {detected_roles}")
            
    print(f"\nFinal Result: {success_count}/{len(image_files)} passed.")
    assert success_count >= 4, "Role detection accuracy too low (expected at least 4/6 for P3 threshold)."
    if success_count == len(image_files):
        print("P3 DELIVERABLES VERIFIED: All roles detected correctly.")

if __name__ == "__main__":
    test_role_detection()
