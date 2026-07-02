import cv2
import numpy as np
import os
import json
from pathlib import Path

# -------- SETTINGS --------
SEMANTIC_IMG_DIR = r"C:\Users\Sumangowda\Desktop\CVIP_project\pedestrian_intent_prediction\data\Town01\Town01\generated\images_ss"
OUTPUT_DIR = "semantic_pedestrian_detections"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Pedestrian color (Red) in HSV format
# Red has two ranges in HSV due to color wheel continuity.
PEDESTRIAN_HSV_RANGES = [
    ([0, 100, 100], [10, 255, 255]),
    ([160, 100, 100], [179, 255, 255])
]

# -------- FUNCTIONS --------
def get_pedestrian_mask(hsv_img):
    mask = np.zeros(hsv_img.shape[:2], dtype=np.uint8)
    for lower, upper in PEDESTRIAN_HSV_RANGES:
        lower = np.array(lower)
        upper = np.array(upper)
        temp_mask = cv2.inRange(hsv_img, lower, upper)
        mask = cv2.bitwise_or(mask, temp_mask)
    return mask

def clean_mask(mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask

def get_bounding_boxes(mask, min_area=150):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    detections = []
    for i in range(1, num_labels):  # skip background
        x, y, w, h, area = stats[i]
        if area >= min_area:
            detections.append({
                "bbox": [int(x), int(y), int(x + w), int(y + h)],
                "area": int(area),
                "centroid": [float(centroids[i][0]), float(centroids[i][1])]
            })
    return detections

def draw_detections(img, detections):
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, "person", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return img

# -------- MAIN PIPELINE --------
def main():
    img_paths = list(Path(SEMANTIC_IMG_DIR).rglob("*.png"))[:2000]
    print(f"[INFO] Starting pedestrian detection on {len(img_paths)} images.")

    for idx, img_path in enumerate(img_paths):
        img = cv2.imread(str(img_path))
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Generate mask
        ped_mask = get_pedestrian_mask(hsv_img)

        # Clean mask
        clean_ped_mask = clean_mask(ped_mask)

        # Get bounding boxes
        detections = get_bounding_boxes(clean_ped_mask)

        # Draw bounding boxes on image
        output_img = draw_detections(img.copy(), detections)

        # Save image with bounding boxes
        output_image_path = os.path.join(
            OUTPUT_DIR, f"{Path(img_path).stem}_detected.png")
        cv2.imwrite(output_image_path, output_img)

        # Save detections to JSON
        output_json_path = os.path.join(
            OUTPUT_DIR, f"{Path(img_path).stem}_detections.json")
        with open(output_json_path, 'w') as f:
            json.dump({"pedestrians": detections}, f, indent=4)

        # Progress logging
        if (idx + 1) % 100 == 0 or idx == len(img_paths) - 1:
            print(f"[INFO] Processed {idx + 1}/{len(img_paths)} images.")

    print("[INFO] Pedestrian detection completed successfully.")

if __name__ == "__main__":
    main()
