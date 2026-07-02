# pedestrian_detection_carla.py

import os
import json
import cv2
from pathlib import Path
from ultralytics import YOLO
from collections import defaultdict

# ------------------------ SETTINGS ------------------------
DATA_ROOT = r"C:\Users\Sumangowda\Desktop\CVIP_project\pedestrian_intent_prediction\data\Town01\Town01\generated"
IMG_DIR = os.path.join(DATA_ROOT, "images_rgb")
OUT_DIR = "pedestrian_detections_carla"
IMAGE_SAVE_DIR = os.path.join(OUT_DIR, "images")
JSON_SAVE_DIR = os.path.join(OUT_DIR, "jsons")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(IMAGE_SAVE_DIR, exist_ok=True)
os.makedirs(JSON_SAVE_DIR, exist_ok=True)

CONF_THRESHOLD = 0.4
TARGET_CLASSES = ['person']

# ------------------------ HELPER FUNCTIONS ------------------------
def enhance_image(image):
    """Enhance input image for better detection"""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    lab = cv2.merge((cl, a, b))
    image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    gamma = 1.2
    table = np.array([(i / 255.0) ** (1.0 / gamma) * 255 for i in range(256)]).astype("uint8")
    image = cv2.LUT(image, table)
    sharpen = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    image = cv2.filter2D(image, -1, sharpen)
    return image

def draw_boxes(image, detections):
    """Draw bounding boxes on image"""
    for det in detections:
        label = det['label']
        conf = det['confidence']
        bbox = det['bbox']
        x1, y1, x2, y2 = bbox
        color = (0, 255, 0)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, f"{label} {conf:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return image

# ------------------------ MAIN PIPELINE ------------------------
def main():
    # Load the pre-trained YOLOv8 model
    model = YOLO("yolov8n.pt")
    model.conf = CONF_THRESHOLD

    image_paths = list(Path(IMG_DIR).rglob("*.png"))
    print(f"[INFO] Found {len(image_paths)} images.")

    all_detections = {}
    pedestrian_detections = {}
    total_pedestrians = 0

    for idx, img_path in enumerate(image_paths):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[WARNING] Could not read image: {img_path}")
            continue

        img_enhanced = enhance_image(img)
        results = model.predict(img_enhanced, save=False, verbose=False)[0]

        frame_name = Path(img_path).stem
        frame_detections = []
        frame_pedestrians = []

        for box in results.boxes:
            cls_id = int(box.cls)
            label = model.names[cls_id]
            conf = float(box.conf)
            bbox = box.xyxy.cpu().numpy().astype(int).tolist()[0]

            if label in TARGET_CLASSES:
                detection_entry = {
                    "label": label,
                    "confidence": round(conf, 3),
                    "bbox": bbox
                }
                frame_detections.append(detection_entry)
                frame_pedestrians.append({
                    "confidence": round(conf, 3),
                    "bbox": bbox
                })

        # Save detections
        all_detections[frame_name] = frame_detections
        pedestrian_detections[frame_name] = frame_pedestrians
        total_pedestrians += len(frame_pedestrians)

        # Draw and save image with pedestrian boxes
        img_with_boxes = draw_boxes(img.copy(), frame_pedestrians)
        save_path = os.path.join(IMAGE_SAVE_DIR, f"{frame_name}_pedestrians.png")
        cv2.imwrite(save_path, img_with_boxes)

        print(f"[{idx+1}/{len(image_paths)}] Processed: {frame_name} | Pedestrians Detected: {len(frame_pedestrians)}")

    # Save JSON outputs
    with open(os.path.join(JSON_SAVE_DIR, "all_detections.json"), "w") as f:
        json.dump(all_detections, f, indent=2)

    with open(os.path.join(JSON_SAVE_DIR, "pedestrian_only.json"), "w") as f:
        json.dump(pedestrian_detections, f, indent=2)

    # Final summary
    print("\n[INFO] Pedestrian detection complete.")
    print(f"[INFO] Total frames processed: {len(image_paths)}")
    print(f"[INFO] Total pedestrians detected across all frames: {total_pedestrians}")
    print(f"[INFO] Results saved in '{OUT_DIR}' folder.")

# ------------------------ RUN ------------------------
if __name__ == "__main__":
    main()