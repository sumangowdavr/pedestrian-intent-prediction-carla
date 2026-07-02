# detection_gem.py (Enhanced for Grad Level Analysis - Basic Tracking)
import cv2
import numpy as np
import os
import json
import random
import csv
from pathlib import Path
from ultralytics import YOLO
from datetime import datetime
from collections import defaultdict
import matplotlib.pyplot as plt
from skimage import exposure

# ------------------------ SETTINGS ------------------------
IMG_DIR = r"C:\Users\Sumangowda\Desktop\CVIP_project\pedestrian_intent_prediction\data\Town01\Town01\generated\images_rgb"
OUT_DIR = "gem_detections"
CROP_DIR = os.path.join(OUT_DIR, "crops")
HEATMAP_PATH = os.path.join(OUT_DIR, "pedestrian_heatmap.png")
TRACKING_OUTPUT_PATH = os.path.join(OUT_DIR, "tracked_objects.json")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(CROP_DIR, exist_ok=True)

MODEL_PATH = "yolov8n.pt"
CONF_THRESHOLD = 0.25
TARGET_CLASSES = ['person', 'car', 'truck', 'bus', 'motorcycle', 'bicycle']

# Toggle options
USE_IMAGE_ENHANCEMENT = True
USE_TEST_AUGMENT = False
SAVE_PEDESTRIAN_IMAGES_ONLY = True
DRAW_EDGE_OVERLAY = False
PRINT_FRAME_PROGRESS = True
ENABLE_TRACKING = True  # Toggle object tracking

# Tracking parameters (simple IoU based)
IOU_THRESHOLD = 0.3
NEXT_ID = 0
TRACKED_OBJECTS = defaultdict(list)  # Store tracked objects and their trajectories

# Heatmap accumulator
heatmap_accumulator = None

# ------------------------ IMAGE ENHANCEMENT ------------------------
def enhance_image(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))  # Reduced clipLimit
    cl = clahe.apply(l)
    lab = cv2.merge((cl, a, b))
    enhanced_img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return enhanced_img


# ------------------------ AUGMENTATION ------------------------
def augment_image(image):
    alpha = 1.2 + 0.3 * random.random()
    beta = 15 + 20 * random.random()
    image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    noise = np.random.normal(0, 10, image.shape).astype(np.uint8)
    image = cv2.add(image, noise)
    if random.random() > 0.5:
        image = cv2.flip(image, 1)
    rows, cols = image.shape[:2]
    M = cv2.getRotationMatrix2D((cols/2, rows/2), angle=random.uniform(-5,5), scale=1.0)
    image = cv2.warpAffine(image, M, (cols, rows))
    return image

# ------------------------ EDGE OVERLAY ------------------------
def overlay_edges(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(image, 0.8, edge_colored, 0.2, 0)
    return overlay

# ------------------------ UTILITY FUNCTIONS FOR TRACKING ------------------------
def calculate_iou(box1, box2):
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)

    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

def assign_tracks(previous_tracks, current_detections, iou_threshold):
    assigned_tracks = {}
    unassigned_tracks = list(previous_tracks.keys())
    unassigned_detections = list(range(len(current_detections)))
    iou_matrix = np.zeros((len(previous_tracks), len(current_detections)), dtype=np.float32)

    for i, track_id in enumerate(previous_tracks):
        track_bbox = previous_tracks[track_id][-1]['bbox']
        for j, detection in enumerate(current_detections):
            det_bbox = detection['bbox']
            iou_matrix[i, j] = calculate_iou(track_bbox, det_bbox)

    # Linear assignment (simple greedy approach)
    while unassigned_tracks and unassigned_detections:
        best_match = None
        max_iou = iou_threshold
        best_track_idx = -1
        best_det_idx = -1

        for i_track, track_id in enumerate(unassigned_tracks):
            for i_det in unassigned_detections:
                if iou_matrix[list(previous_tracks.keys()).index(track_id), i_det] > max_iou:
                    max_iou = iou_matrix[list(previous_tracks.keys()).index(track_id), i_det]
                    best_match = (track_id, i_det)
                    best_track_idx = i_track
                    best_det_idx = unassigned_detections.index(i_det)

        if best_match:
            track_id, det_index = best_match
            assigned_tracks[track_id] = current_detections[det_index]
            unassigned_tracks.pop(best_track_idx)
            unassigned_detections.pop(best_det_idx)
        else:
            break

    return assigned_tracks, unassigned_tracks, [current_detections[i] for i in unassigned_detections]

# ------------------------ YOLO INIT ------------------------
model = YOLO(MODEL_PATH)
model.conf = CONF_THRESHOLD

# ------------------------ DETECTION PIPELINE ------------------------
image_paths = list(Path(IMG_DIR).rglob("*.png"))
# Limit to the first 2000 frames
image_paths = image_paths[:2000]
all_detections = {}
pedestrian_detections = {}
summary_data = []
log_lines = []
previous_frame_tracks = {}

print(" Starting detection and tracking on", len(image_paths), "images...")

for frame_idx, img_path in enumerate(image_paths):
    img = cv2.imread(str(img_path))
    if img is None:
        continue

    if USE_IMAGE_ENHANCEMENT:
        img = enhance_image(img)
    if USE_TEST_AUGMENT:
        img = augment_image(img)
    if DRAW_EDGE_OVERLAY:
        img = overlay_edges(img)

    results = model.predict(img, save=False, verbose=False)[0]
    frame_name = Path(img_path).stem
    current_detections = []
    has_person = False
    person_count = 0
    class_counts = defaultdict(int)

    for i, box in enumerate(results.boxes):
        cls_id = int(box.cls)
        label = model.names[cls_id]
        conf = float(box.conf)
        bbox = box.xyxy.cpu().numpy().astype(int).tolist()[0]

        if label in TARGET_CLASSES:
            current_detections.append({"label": label, "confidence": round(conf, 2), "bbox": bbox})
            class_counts[label] += 1
            if label == "person":
                has_person = True
                person_count += 1
                if heatmap_accumulator is None:
                    heatmap_accumulator = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
                x1, y1, x2, y2 = bbox
                heatmap_accumulator[y1:y2, x1:x2] += 1.0

            x1, y1, x2, y2 = bbox
            crop = img[y1:y2, x1:x2]
            cv2.imwrite(os.path.join(CROP_DIR, f"{label}_{frame_name}_{i}.png"), crop)

            cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(img, f"{label} {conf:.2f}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    all_detections[frame_name] = current_detections

    if has_person:
        pedestrian_detections[frame_name] = [det for det in current_detections if det["label"] == "person"]

    save_condition = not SAVE_PEDESTRIAN_IMAGES_ONLY or has_person
    if save_condition:
        cv2.imwrite(os.path.join(OUT_DIR, f"{frame_name}_detected.png"), img)

    summary_data.append({"frame": frame_name, "total": len(current_detections), "person": person_count, "classes": dict(class_counts)})
    log_lines.append(f"{frame_name}: total={len(current_detections)}, person={person_count}, classes={dict(class_counts)}")

    if PRINT_FRAME_PROGRESS:
        print(f" Processed {frame_idx+1}/{len(image_paths)}: {frame_name} - {len(current_detections)} objects, {person_count} persons")

    # ------------------------ OBJECT TRACKING ------------------------
    if ENABLE_TRACKING:
        assigned_tracks, unassigned_tracks, new_detections = assign_tracks(previous_frame_tracks, current_detections, IOU_THRESHOLD)

        # Update existing tracks
        for track_id, detection in assigned_tracks.items():
            TRACKED_OBJECTS[track_id].append({"frame": frame_name, **detection})

        # Initialize new tracks
        # Initialize new tracks
        for detection in new_detections:
            TRACKED_OBJECTS[NEXT_ID].append({"frame": frame_name, **detection})
            previous_frame_tracks[NEXT_ID] = [detection]
            NEXT_ID += 1


        else:
            previous_frame_tracks = {track_id: TRACKED_OBJECTS[track_id][-1:] for track_id in TRACKED_OBJECTS if TRACKED_OBJECTS[track_id][-1]['frame'] == frame_name}


# ------------------------ OUTPUT FILES ------------------------
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

# ------------------------ HEATMAP ------------------------
if heatmap_accumulator is not None:
    heatmap_norm = (heatmap_accumulator / np.max(heatmap_accumulator) * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
    cv2.imwrite(HEATMAP_PATH, heatmap_color)

# ------------------------ TRACKING OUTPUT ------------------------
if ENABLE_TRACKING:
    with open(TRACKING_OUTPUT_PATH, "w") as f:
        json.dump(TRACKED_OBJECTS, f, indent=2)
    print(f" Tracking information saved to: {TRACKING_OUTPUT_PATH}")

print(" Detection and tracking pipeline finished successfully.")

# ------------------------ GRAD LEVEL ANALYSIS (Further Analysis on TRACKED_OBJECTS) ------------------------
if ENABLE_TRACKING:
    print("\n--- Starting Graduate Level Analysis (Tracking) ---")

    # Example: Analyze trajectories of pedestrians
    pedestrian_trajectories = {
        track_id: trajectory for track_id, trajectory in TRACKED_OBJECTS.items() if trajectory[0]['label'] == 'person'
    }

    print(f"Found {len(pedestrian_trajectories)} tracked pedestrians.")

    for track_id, trajectory in pedestrian_trajectories.items():
        if len(trajectory) > 1:
            start_bbox = trajectory[0]['bbox']
            end_bbox = trajectory[-1]['bbox']
            print(f"Pedestrian Track ID: {track_id}, Length: {len(trajectory)} frames")
            print(f"  Start BBox: {start_bbox}, End BBox: {end_bbox}")
            # Further analysis: Calculate displacement, velocity, changes in direction, etc.

    # Example: Visualize a few pedestrian trajectories (requires matplotlib)
    if pedestrian_trajectories:
        num_to_visualize = min(5, len(pedestrian_trajectories))
        random_tracks = random.sample(list(pedestrian_trajectories.keys()), num_to_visualize)

        plt.figure(figsize=(10, 8))
        for track_id in random_tracks:
            trajectory = pedestrian_trajectories[track_id]
            x_coords = [ (bbox[0] + bbox[2]) / 2 for item in trajectory] # Center x
            y_coords = [ (bbox[1] + bbox[3]) / 2 for item in trajectory] # Center y
            plt.plot(x_coords, y_coords, label=f"Track {track_id}")

        plt.xlabel("Frame Number (Relative)")
        plt.ylabel("Image Y Coordinate (Pixel)")
        plt.title("Trajectories of Sampled Pedestrians")
        plt.legend()
        plt.savefig(os.path.join(OUT_DIR, "pedestrian_trajectories.png"))
        plt.close()
        print("Sample pedestrian trajectories visualized.")
    else:
        print("No pedestrian trajectories to visualize.")