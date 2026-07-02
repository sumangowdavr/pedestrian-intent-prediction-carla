import os
import cv2
import json
from pathlib import Path
from tqdm import tqdm

# -------------------- SETTINGS --------------------
RGB_IMG_DIR = r"C:/Users/Sumangowda/Desktop/CVIP_project/pedestrian_intent_prediction/data/Town01/Town01/generated/images_rgb"
SS_JSON_DIR = r"semantic_pedestrian_detections"
OUTPUT_DIR = r"rgb_with_ss_pedestrians"

# Create output folders
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Bounding box color and thickness
BOX_COLOR = (0, 255, 0)  # Green
THICKNESS = 2

# -------------------- MAIN SCRIPT --------------------

# Collect all RGB image paths
rgb_image_paths = list(Path(RGB_IMG_DIR).rglob("*.png"))

# Initialize pedestrian detection record
all_pedestrian_detections = {}

total_frames_processed = 0
total_pedestrians_detected = 0

print(f"[INFO] Starting annotation of {len(rgb_image_paths)} RGB images using SS detections...")

for rgb_path in tqdm(rgb_image_paths):
    frame_name = rgb_path.stem
    ss_json_path = Path(SS_JSON_DIR) / f"{frame_name}_detections.json"

    if not ss_json_path.exists():
        print(f"[WARNING] Missing SS detection JSON for {frame_name}. Skipping.")
        continue

    # Load RGB image
    rgb_img = cv2.imread(str(rgb_path))
    if rgb_img is None:
        print(f"[ERROR] Failed to load {rgb_path}. Skipping.")
        continue

    # Load SS detections
    with open(ss_json_path, 'r') as f:
        ss_data = json.load(f)

    pedestrians = ss_data.get("pedestrians", [])

    # Draw each pedestrian bounding box
    for ped in pedestrians:
        x1, y1, x2, y2 = ped["bbox"]
        cv2.rectangle(rgb_img, (x1, y1), (x2, y2), BOX_COLOR, THICKNESS)

    # Save annotated image
    save_path = Path(OUTPUT_DIR) / f"{frame_name}_annotated.png"
    cv2.imwrite(str(save_path), rgb_img)

    # Save detections
    all_pedestrian_detections[frame_name] = pedestrians

    total_frames_processed += 1
    total_pedestrians_detected += len(pedestrians)

# Save combined pedestrian_only.json
combined_json_path = Path(OUTPUT_DIR) / "pedestrian_only.json"
with open(combined_json_path, 'w') as f:
    json.dump(all_pedestrian_detections, f, indent=2)

print(f"[INFO] Annotation completed.")
print(f"[INFO] Total frames processed: {total_frames_processed}")
print(f"[INFO] Total pedestrians detected: {total_pedestrians_detected}")
print(f"[INFO] Outputs saved in {OUTPUT_DIR}")
