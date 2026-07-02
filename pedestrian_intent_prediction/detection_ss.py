import argparse
import cv2
import numpy as np
import os
import json
from pathlib import Path
import matplotlib.pyplot as plt
# import time  # Uncomment for performance profiling

# -------- SETTINGS --------

# paths to input and the output directries
parser = argparse.ArgumentParser(description="Semantic segmentation-based pedestrian detection")

parser.add_argument("--input_dir", 
                    type=str, 
                    default="./data/images_ss", 
                    help="Path to semantic segmented images")

parser.add_argument("--output_dir", 
                    type=str, 
                    default="semantic_pedestrian_detection", 
                    help="Path to save outputs")

args = parser.parse_args()

image_dir = args.input_dir
output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)

# swithes to save the mask and the visual plots
visual_img = True  
mask_img = True  

# -------- ALTERNATIVE SETTINGS (COMMENTED) --------
# UNCOMMENT THESE FOR DIFFERENT APPROACHES
# DEEP_LEARNING_MODEL = "pednet.h5"  # Path to hypothetical DL model
# USE_DEEP_LEARNING = False  # Flag to switch to DL approach
# ENABLE_MULTI_SCALE = False  # Enable multi-scale detection (slower)
# SCALES = [0.5, 1.0, 1.5]  # Scaling factors for multi-scale

# Pedestrian Color Configuration
ped_clr = (220, 20, 60)  # RGB for red pedestrian in CARLA
COLOR_TOLERANCE = 10  # Allow small deviation for better robustness

# Detection filter parameters
ASPECT_RATIO_THRESHOLD = 1.2   # height must be at least 1.2x width
MIN_AREA = 20                  # minimum pixel area
MIN_HEIGHT = 8                 # minimum pixel height


# -------- FUNCTIONS --------
# Earlier trial (failed): HSV based detection attempt
# Commented to document history
# def get_pedestrian_mask_hsv(img):
#     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#     lower_red1 = np.array([0, 70, 50])
#     upper_red1 = np.array([10, 255, 255])
#     lower_red2 = np.array([170, 70, 50])
#     upper_red2 = np.array([180, 255, 255])
#     mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
#     mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
#     return cv2.bitwise_or(mask1, mask2)
# -------- FUNCTIONS --------
def get_pedestrian_mask(img):
    """
    Create a binary mask for pedestrian pixels using color thresholding.
    """
    lower = np.array([ped_clr[2] - COLOR_TOLERANCE,
                      ped_clr[1] - COLOR_TOLERANCE,
                      ped_clr[0] - COLOR_TOLERANCE])
    upper = np.array([ped_clr[2] + COLOR_TOLERANCE,
                      ped_clr[1] + COLOR_TOLERANCE,
                      ped_clr[0] + COLOR_TOLERANCE])
    img_bgr = img
    mask = cv2.inRange(img_bgr, lower, upper)
    return mask

    # ALTERNATIVE APPROACH 1: HSV color space (sometimes better for lighting variations)
    # img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # lower_hsv = np.array([0, 100, 100])
    # upper_hsv = np.array([10, 255, 255])
    # mask_hsv = cv2.inRange(img_hsv, lower_hsv, upper_hsv)
    # mask = cv2.bitwise_or(mask, mask_hsv)


def clean_mask(mask):
    """
    Clean the binary mask using morphological operations.
    """
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        # ALTERNATIVE APPROACH 2: More aggressive cleaning for noisy environments
    # kernel_large = np.ones((5,5), np.uint8)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_large, iterations=1)
    # mask = cv2.erode(mask, kernel, iterations=1)

    return mask


def get_pedestrian_bboxes(mask):
    """
    Find pedestrian bounding boxes using connected components analysis.
    """
        # ALTERNATIVE APPROACH 3: Contour-based detection
    # contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # for cnt in contours:
    #     area = cv2.contourArea(cnt)
    #     if area >= MIN_AREA:
    #         x, y, w, h = cv2.boundingRect(cnt)
    #         aspect_ratio = h / float(w) if w > 0 else 0
    #         if aspect_ratio > ASPECT_RATIO_THRESHOLD:
    #             detections.append({
    #                 "bbox": [x, y, x+w, y+h],
    #                 "area": area,
    #                 "centroid": [x+w/2, y+h/2],
    #                 "aspect_ratio": round(aspect_ratio, 2)
    #             })

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    detections = []
    for i in range(1, num_labels):  # skip background
        x, y, w, h, area = stats[i]
        cx, cy = centroids[i]
        if area >= MIN_AREA and h >= MIN_HEIGHT:
            aspect_ratio = h / float(w) if w > 0 else 0
            if aspect_ratio > ASPECT_RATIO_THRESHOLD:
                detections.append({
                    "bbox": [int(x), int(y), int(x+w), int(y+h)],
                    "area": float(area),
                    "centroid": [float(cx), float(cy)],
                    "aspect_ratio": round(aspect_ratio, 2)
                })
    return detections


def draw_detections(img, detections, zoom_factor=2):
    """
    Draw bounding boxes and labels on the image with optional zoom for small pedestrians.
    """
    vis_img = img.copy()
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det["bbox"]
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"ped {i+1}"
        font_scale = 0.5 if det["area"] > 50 else 0.4
        thickness = 2 if det["area"] > 50 else 1
        cv2.putText(vis_img, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX,
                   font_scale, (0,255,0), thickness)
        if det["area"] < 20 and zoom_factor > 1:
            zoom_h = (y2 - y1) * zoom_factor
            zoom_w = (x2 - x1) * zoom_factor
            zoom_img = img[max(0,y1-zoom_h//2):min(img.shape[0],y2+zoom_h//2),
                           max(0,x1-zoom_w//2):min(img.shape[1],x2+zoom_w//2)]
            if zoom_img.size > 0:
                zoom_img = cv2.resize(zoom_img, (100,100))
                vis_img[20:120,20:120] = zoom_img
                cv2.line(vis_img, (70,120), ((x1+x2)//2, y1), (0,255,0), 1)
    return vis_img


def create_visualization(original, mask, cleaned_mask, detections):
    """
    Create a 4-panel visualization for debugging and analysis.
    """
    fig, axes = plt.subplots(2,2,figsize=(12,10))
    axes[0,0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    axes[0,0].set_title("Original Image")
    axes[0,1].imshow(mask, cmap='gray')
    axes[0,1].set_title("Initial Red Mask")
    axes[1,0].imshow(cleaned_mask, cmap='gray')
    axes[1,0].set_title("Cleaned Mask")
    detection_img = draw_detections(original, detections)
    axes[1,1].imshow(cv2.cvtColor(detection_img, cv2.COLOR_BGR2RGB))
    axes[1,1].set_title(f"Detections: {len(detections)} pedestrians")
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    return fig

# -------- ALTERNATIVE PROCESSING FUNCTIONS (COMMENTED) --------

# def load_deep_learning_model(model_path):
#     """Hypothetical function to load a DL model"""
#     print(f"[INFO] Loading deep learning model from {model_path}")
#     # model = tf.keras.models.load_model(model_path)
#     # return model
#     return None

# def detect_with_dl(model, img):
#     """Hypothetical DL-based detection"""
#     # img_preprocessed = preprocess_image_for_model(img)
#     # predictions = model.predict(img_preprocessed)
#     # return process_dl_predictions(predictions)
#     return []

# def multi_scale_detection(img):
#     """Detect pedestrians at multiple scales"""
#     all_detections = []
#     for scale in SCALES:
#         resized = cv2.resize(img, None, fx=scale, fy=scale)
#         mask = get_pedestrian_mask(resized)
#         clean_mask = clean_mask(mask)
#         detections = get_pedestrian_bboxes(clean_mask)
        
#         # Scale detections back to original size
#         for det in detections:
#             det["bbox"] = [int(x/scale) for x in det["bbox"]]
#             det["centroid"] = [x/scale for x in det["centroid"]]
#             det["area"] = det["area"] / (scale*scale)
#         all_detections.extend(detections)
    
#     # Merge overlapping detections
#     return merge_detections(all_detections)

def process_image(img_path):
    """
    Process a single image to detect pedestrians.
    """
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"[ERROR] Failed to load image: {img_path}")
        return None, None, None, None
    
        # ALTERNATIVE APPROACH 5: Deep learning path
    # if USE_DEEP_LEARNING:
    #     model = load_deep_learning_model(DEEP_LEARNING_MODEL)
    #     detections = detect_with_dl(model, img)
    #     return img, None, detections, None
    
    # ALTERNATIVE APPROACH 6: Multi-scale detection
    # if ENABLE_MULTI_SCALE:
    #     detections = multi_scale_detection(img)
    #     return img, None, detections, None
    

    ped_mask = get_pedestrian_mask(img)
    cleaned = clean_mask(ped_mask)
    detections = get_pedestrian_bboxes(cleaned)
    vis_fig = None
    if visual_img:
        vis_fig = create_visualization(img, ped_mask, cleaned, detections)
    return img, cleaned, detections, vis_fig


def plot_aggregate_statistics(all_results, output_dir):
    """Generate aggregate plots without modifying detection pipeline"""
    counts = [result['count'] for result in all_results]
    areas = [ped['area'] for result in all_results for ped in result['pedestrians']]
    aspect_ratios = [ped['aspect_ratio'] for result in all_results for ped in result['pedestrians']]
    analysis_dir = os.path.join(output_dir, "analysis")

    os.makedirs(analysis_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.hist(counts, bins=20, edgecolor='black')
    plt.title('Distribution of Pedestrian Counts per Frame')
    plt.xlabel('Number of Pedestrians')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(analysis_dir, 'count_distribution.png'))
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.hist(areas, bins=30, edgecolor='black', range=(0, 500))
    plt.title('Distribution of Pedestrian Bounding Box Areas')
    plt.xlabel('Area (pixels)')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(analysis_dir, 'area_distribution.png'))
    plt.close()



def create_heatmap(all_results, img_shape, output_dir):
    """Generate detection heatmap without modifying detection logic"""
    centroids_x = [ped['centroid'][0] for result in all_results for ped in result['pedestrians']]
    centroids_y = [ped['centroid'][1] for result in all_results for ped in result['pedestrians']]
    analysis_dir = os.path.join(output_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)
    plt.figure(figsize=(12, 8))
    plt.hexbin(centroids_x, centroids_y, gridsize=50, cmap='inferno', bins='log')
    plt.colorbar(label='Log10 of Detection Density')
    plt.title('Pedestrian Locations Heatmap')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.savefig(os.path.join(analysis_dir, 'location_heatmap.png'), dpi=300)
    plt.close()


def main():
    """Main execution function to process all images in directory."""
    input_dir = Path(image_dir)
    if not input_dir.exists():
        print(f"[ERROR] Input directory not found: {image_dir}")
        return
    img_paths = list(input_dir.rglob("*.png"))
    if not img_paths:
        print(f"[ERROR] No PNG images found in {image_dir}")
        return
    print(f"[INFO] Found {len(img_paths)} images to process.")
    total_pedestrians = 0
    all_detections = []
    img_size = None
    for idx, img_path in enumerate(img_paths):
        img, mask, detections, vis_fig = process_image(img_path)
        if img is None:
            continue
        if img_size is None:
            img_size = (img.shape[1], img.shape[0])
        total_pedestrians += len(detections)
        output_basename = Path(img_path).stem
        result_data = {
            "image_path": str(img_path),
            "pedestrians": detections,
            "count": len(detections),
            "image_size": {"width": img.shape[1], "height": img.shape[0]},
            "detection_quality": {
                "min_area": min([d['area'] for d in detections]) if detections else 0,
                "avg_aspect_ratio": np.mean([d['aspect_ratio'] for d in detections]) if detections else 0
            }
        }
        all_detections.append(result_data)
        # Save JSON
        output_json_path = os.path.join(output_dir, f"{output_basename}_detections.json")
        with open(output_json_path, 'w') as f:
            json.dump({"pedestrians": detections, "count": len(detections)}, f, indent=4)
        # Save visualization
        if visual_img and vis_fig:
            vis_path = os.path.join(output_dir, f"{output_basename}_visualization.png")
            vis_fig.savefig(vis_path)
            plt.close(vis_fig)
        # Save detection image
        detection_img = draw_detections(img, detections)
        detection_img_path = os.path.join(output_dir, f"{output_basename}_detection.png")
        cv2.imwrite(detection_img_path, detection_img)
        # Save mask if enabled
        if mask_img:
            mask_path = os.path.join(output_dir, f"{output_basename}_mask.png")
            cv2.imwrite(mask_path, mask)
        # Progress reporting
        if (idx+1) % 50 == 0 or idx == len(img_paths)-1:
            print(f"[INFO] Processed {idx+1}/{len(img_paths)} images. Total pedestrians: {total_pedestrians}")
    # --- AGGREGATE ANALYSIS SECTION ---
    if all_detections:
        analysis_dir = os.path.join(output_dir, "analysis")
        os.makedirs(analysis_dir, exist_ok=True)
        # 1. Save comprehensive ground truth
        with open(os.path.join(analysis_dir, "aggregated_ground_truth.json"), 'w') as f:
            json.dump({
                "total_frames": len(all_detections),
                "total_pedestrians": total_pedestrians,
                "avg_pedestrians_per_frame": total_pedestrians / len(all_detections),
                "detections": all_detections
            }, f, indent=4)
        # 2. Generate analysis plots
        plot_aggregate_statistics(all_detections, output_dir)
        create_heatmap(all_detections, img_size, output_dir)
    # Final report
    print(f"[INFO] Pedestrian detection completed. Total: {total_pedestrians}")
    if all_detections:
        print(f"[INFO] Analysis reports saved to: {os.path.join(output_dir, 'analysis')}.")

if __name__ == "__main__":
    main()
