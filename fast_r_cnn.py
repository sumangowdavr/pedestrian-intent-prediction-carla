# detection_fasterrcnn.py

import cv2
import numpy as np
import os
import json
import random
import csv
from pathlib import Path
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.transforms import functional as F
from collections import defaultdict
import matplotlib.pyplot as plt
from torchvision.ops import nms

# ------------------------ SETTINGS ------------------------
IMG_DIR = r"C:\Users\Sumangowda\Desktop\CVIP_project\pedestrian_intent_prediction\data\Town01\Town01\generated\images_rgb"
OUT_DIR = "detections_fasterrcnn"
CROP_DIR = os.path.join(OUT_DIR, "crops")
IMAGE_SAVE_DIR = os.path.join(OUT_DIR, "images")
HEATMAP_PATH = os.path.join(OUT_DIR, "pedestrian_heatmap.png")

CONF_THRESHOLD = 0.7  # Higher confidence for clean detection
IOU_THRESHOLD_NMS = 0.5  # NMS to remove overlapping boxes
TARGET_CLASSES = ['person', 'car']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(CROP_DIR, exist_ok=True)
os.makedirs(IMAGE_SAVE_DIR, exist_ok=True)

USE_IMAGE_ENHANCEMENT = True
USE_TEST_AUGMENT = False  # NO augmentation while detecting
DRAW_EDGE_OVERLAY = False  # NO edge overlay
PRINT_FRAME_PROGRESS = True

# ------------------------ IMAGE ENHANCEMENT ------------------------
def enhance_image(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    lab = cv2.merge((cl, a, b))
    image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return image.astype(np.uint8)

# ------------------------ MODEL INITIALIZATION ------------------------
def get_model(num_classes):
    weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1
    model = fasterrcnn_resnet50_fpn(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    return model.to(DEVICE)

# ------------------------ DETECTION FUNCTION ------------------------
def detect_objects(model, image):
    model.eval()
    with torch.no_grad():
        image_tensor = F.to_tensor(image).to(DEVICE)
        predictions = model([image_tensor])[0]
        boxes = predictions['boxes']
        scores = predictions['scores']
        labels = predictions['labels']

        # Apply confidence filter
        keep = scores >= CONF_THRESHOLD
        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]

        # Apply NMS to remove duplicates
        keep_idx = nms(boxes, scores, IOU_THRESHOLD_NMS)
        boxes = boxes[keep_idx]
        scores = scores[keep_idx]
        labels = labels[keep_idx]

        detections = []
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i].cpu().numpy().astype(int)
            label_idx = labels[i].item() - 1
            if 0 <= label_idx < len(TARGET_CLASSES):
                label = TARGET_CLASSES[label_idx]
                detections.append({
                    "label": label,
                    "confidence": float(scores[i]),
                    "bbox": [x1, y1, x2, y2]
                })
        return detections

# ------------------------ MAIN ------------------------
def main():
    model = get_model(len(TARGET_CLASSES) + 1)
    model.eval()

    image_paths = sorted(list(Path(IMG_DIR).rglob("*.png")))[:2000]  # Limit to 2000 frames
    all_detections = {}
    pedestrian_detections = {}
    summary_data = []
    log_lines = []
    heatmap_accumulator = None

    print(f"[INFO] Starting detection on {len(image_paths)} images...")

    for idx, img_path in enumerate(image_paths):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[WARNING] Could not read {img_path}")
            continue

        if USE_IMAGE_ENHANCEMENT:
            img = enhance_image(img)

        detections = detect_objects(model, img)
        frame_name = Path(img_path).stem
        has_person = False
        person_count = 0
        class_counts = defaultdict(int)

        for i, det in enumerate(detections):
            label = det["label"]
            conf = det["confidence"]
            x1, y1, x2, y2 = det["bbox"]
            class_counts[label] += 1
            if label == "person":
                has_person = True
                person_count += 1
                if heatmap_accumulator is None:
                    heatmap_accumulator = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
                heatmap_accumulator[y1:y2, x1:x2] += 1.0

            crop = img[y1:y2, x1:x2]
            cv2.imwrite(os.path.join(CROP_DIR, f"{label}_{frame_name}_{i}.png"), crop)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(img, f"{label} {conf:.2f}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        all_detections[frame_name] = detections
        if has_person:
            pedestrian_detections[frame_name] = [det for det in detections if det["label"] == "person"]

        save_path = os.path.join(IMAGE_SAVE_DIR, f"{frame_name}_detected.png")
        cv2.imwrite(save_path, img)

        summary_data.append({"frame": frame_name, "total": len(detections), "person": person_count, "classes": dict(class_counts)})
        log_lines.append(f"{frame_name}: total={len(detections)}, person={person_count}, classes={dict(class_counts)}")

        if PRINT_FRAME_PROGRESS:
            print(f"[{idx+1}/{len(image_paths)}] Processed {frame_name}: {len(detections)} detections, {person_count} pedestrians.")

    with open(os.path.join(OUT_DIR, "detections.json"), "w") as f:
        json.dump(all_detections, f, indent=2)
    with open(os.path.join(OUT_DIR, "pedestrian_only.json"), "w") as f:
        json.dump(pedestrian_detections, f, indent=2)
    with open(os.path.join(OUT_DIR, "detection_log.txt"), "w") as f:
        f.write("\n".join(log_lines))
    with open(os.path.join(OUT_DIR, "detection_summary.csv"), "w", newline='') as csvfile:
        fieldnames = ["frame", "total", "person", "classes"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_data:
            writer.writerow(row)

    if heatmap_accumulator is not None:
        heatmap_norm = (heatmap_accumulator / np.max(heatmap_accumulator) * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
        cv2.imwrite(HEATMAP_PATH, heatmap_color)

    print("[INFO] Detection pipeline finished successfully.")

if __name__ == "__main__":
    main()
