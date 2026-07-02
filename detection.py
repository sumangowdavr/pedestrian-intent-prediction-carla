#!/usr/bin/env python3
"""
detection.py

1) Enhance synthetic CARLA RGB frames (CLAHE, gamma, sharpen, bilateral).
2) Run YOLOv8l with test-time augment (TTA) for maximal pedestrian recall.
3) Save per-frame:
     - annotated image (green = YOLO box)
     - per-frame JSON with bbox + confidence
     - crops of each pedestrian
4) Emit a global summary at the end.
"""

import os
import cv2
import json
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from datetime import datetime
from tqdm import tqdm

# ------------------------ SETTINGS ------------------------
# Paths
IMG_DIR      = Path("data/Town01/Town01/generated/images_rgb")
OUT_DIR      = Path("detections")
CROP_DIR     = OUT_DIR / "crops"
JSON_DIR     = OUT_DIR / "json"
IMG_OUT_DIR  = OUT_DIR / "annotated"

# Model
# swap in your own fine-tuned weights if you have them
MODEL_WEIGHTS  = "yolov8l.pt"  
CONF_THRESH    = 0.25
IOU_THRESH     = 0.45
USE_TTA        = True           # test-time augment

# Pre/process toggles
USE_IMAGE_ENHANCE = True
USE_EDGE_OVERLAY  = False
PRINT_PROGRESS    = True

# Make output dirs
for d in (OUT_DIR, CROP_DIR, JSON_DIR, IMG_OUT_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ------------------------ IMAGE ENHANCEMENT ------------------------
def enhance_image(img: np.ndarray) -> np.ndarray:
    """Contrast‐limited AHE + gamma + sharpen + bilateral filter."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    cl = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(l)
    lab = cv2.merge((cl, a, b))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    # gamma correction
    gamma = 1.3
    invG = 1.0 / gamma
    table = np.array([((i / 255.0) ** invG) * 255 for i in np.arange(256)]).astype("uint8")
    img = cv2.LUT(img, table)
    # sharpen
    kernel = np.array([[ 0, -1,  0],
                       [-1,  5, -1],
                       [ 0, -1,  0]])
    img = cv2.filter2D(img, -1, kernel)
    # bilateral denoise
    img = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
    return img

def overlay_edges(img: np.ndarray) -> np.ndarray:
    """Simple Canny edge overlay."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edges_col = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(img, 0.8, edges_col, 0.2, 0)

# ------------------------ YOLO INIT ------------------------
model = YOLO(MODEL_WEIGHTS)
model.conf = CONF_THRESH
model.iou  = IOU_THRESH
model.to('cpu')  # force CPU for stability; remove to run on GPU

# ------------------------ MAIN DETECTION LOOP ------------------------
start_time = datetime.now()
stats = {"frames":0, "total_ped":0}

for img_path in tqdm(sorted(IMG_DIR.rglob("*.png")), disable=not PRINT_PROGRESS):
    frame = img_path.stem
    img = cv2.imread(str(img_path))
    if img is None:
        continue

    # preprocess
    proc = img.copy()
    if USE_IMAGE_ENHANCE:
        proc = enhance_image(proc)
    if USE_EDGE_OVERLAY:
        proc = overlay_edges(proc)

    # infer
    results = model.predict(proc, augment=USE_TTA, verbose=False)[0]

    # collect person boxes
    ped_boxes = []
    for box in results.boxes:
        if int(box.cls)==0:  # class 0 = person in COCO
            x1,y1,x2,y2 = box.xyxy.cpu().numpy()[0].astype(int)
            conf = float(box.conf)
            ped_boxes.append((x1,y1,x2,y2,conf))

    # draw & crop & JSON
    annotated = img.copy()
    out_json = {"frame":frame, "pedestrians":[], "count":len(ped_boxes)}

    for idx, (x1,y1,x2,y2,conf) in enumerate(ped_boxes):
        # draw
        cv2.rectangle(annotated, (x1,y1),(x2,y2),(0,255,0),2)
        cv2.putText(annotated, f"ped {conf:.2f}", (x1,y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1,cv2.LINE_AA)
        # crop
        crop = img[y1:y2, x1:x2]
        cv2.imwrite(str(CROP_DIR/f"{frame}_ped_{idx}.png"), crop)
        # JSON entry
        out_json["pedestrians"].append({
            "bbox":[int(x1),int(y1),int(x2),int(y2)],
            "confidence":round(conf,3)
        })

    # save annotated & json
    cv2.imwrite(str(IMG_OUT_DIR/f"{frame}_annotated.png"), annotated)
    with open(JSON_DIR/f"{frame}.json","w") as jf:
        json.dump(out_json, jf, indent=2)

    # update stats
    stats["frames"] += 1
    stats["total_ped"] += len(ped_boxes)

# ------------------------ SUMMARY ------------------------
elapsed = datetime.now() - start_time
print(f"\nProcessed {stats['frames']} frames in {elapsed}.")
print(f"-> Detected {stats['total_ped']} pedestrians total.")
if stats["frames"]:
    print(f"-> Avg {stats['total_ped']/stats['frames']:.2f} per frame.")
