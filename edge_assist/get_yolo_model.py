import os
import requests

def download_yolo():
    # URL for YOLOv5n ONNX (pre-exported or from a reliable source)
    # Using a common hub URL or similar. 
    # Alternatively, use ultralytics to export it once.
    target_path = "edge_assist/models/yolov5n.onnx"
    os.makedirs("edge_assist/models", exist_ok=True)
    
    if not os.path.exists(target_path):
        print("Exporting YOLOv5n to ONNX 320x320...")
        try:
            from ultralytics import YOLO
            model = YOLO("yolov5n.pt")
            model.export(format="onnx", imgsz=320)
            # Ultralytics exports to yolov5n.onnx in the current dir or same dir as .pt
            if os.path.exists("yolov5n.onnx"):
                os.rename("yolov5n.onnx", target_path)
            elif os.path.exists("edge_assist/models/yolov5n.onnx"):
                pass
            else:
                # Find where it went
                print("Checking for exported file...")
                import glob
                files = glob.glob("**/yolov5n.onnx", recursive=True)
                if files:
                    os.rename(files[0], target_path)
        except Exception as e:
            print(f"Export failed: {e}")
            # Fallback download if export fails
            url = "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.onnx"
            print(f"Downloading from {url}...")
            r = requests.get(url, allow_redirects=True)
            with open(target_path, "wb") as f:
                f.write(r.content)
    
    if os.path.exists(target_path):
        print(f"Model ready at {target_path}")
    else:
        print("Failed to acquire model.")

if __name__ == "__main__":
    download_yolo()
