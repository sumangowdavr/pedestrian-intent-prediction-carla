"""
Note: The detection results from this script (JSONs, visualization images, plots) 
will be used as the ground-truth to evaluate the RGB pedestrian detection models.
"""

"""
Pedestrian Detection in CARLA Semantic Segmentation Images
This script detects pedestrians in CARLA's semantic segmentation output by:
1. Color thresholding (red pixels)
2. Morphological cleaning
3. Connected component analysis
4. Geometric filtering (aspect ratio, area)

Alternative approaches are included but commented out for reference.
"""

import cv2
import numpy as np
import os
import json
from pathlib import Path
import matplotlib.pyplot as plt
# import time  # Uncomment for performance profiling

# -------- SETTINGS --------
SEMANTIC_IMG_DIR = r"C:\Users\Sumangowda\Desktop\CVIP_project\pedestrian_intent_prediction\data\Town01\Town01\generated\images_ss"
OUTPUT_DIR = "semantic_pedestrian_detection"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Visualization settings
VISUALIZE = True  # Set to True to generate visualization images
SAVE_MASKS = True  # Save intermediate mask images for debugging

# Pedestrian Color Configuration
PEDESTRIAN_COLOR = (220, 20, 60)  # RGB for red pedestrian in CARLA
COLOR_TOLERANCE = 10  # Allow small deviation for better robustness

# Detection filter parameters
ASPECT_RATIO_THRESHOLD = 1.2   # height must be at least 1.2x width
MIN_AREA = 20                  # minimum pixel area
MIN_HEIGHT = 8                 # minimum pixel height

# -------- ALTERNATIVE SETTINGS (COMMENTED) --------
# UNCOMMENT THESE FOR DIFFERENT APPROACHES
# DEEP_LEARNING_MODEL = "pednet.h5"  # Path to hypothetical DL model
# USE_DEEP_LEARNING = False  # Flag to switch to DL approach
# ENABLE_MULTI_SCALE = False  # Enable multi-scale detection (slower)
# SCALES = [0.5, 1.0, 1.5]  # Scaling factors for multi-scale

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

def get_pedestrian_mask(img):
    """
    Create a binary mask for pedestrian pixels using color thresholding.
    
    Args:
        img (numpy.ndarray): Input BGR image from CARLA semantic segmentation
        
    Returns:
        numpy.ndarray: Binary mask where pedestrian pixels are white (255)
    """
    # Standard approach - color thresholding
    lower = np.array([PEDESTRIAN_COLOR[2] - COLOR_TOLERANCE,
                      PEDESTRIAN_COLOR[1] - COLOR_TOLERANCE,
                      PEDESTRIAN_COLOR[0] - COLOR_TOLERANCE])
    upper = np.array([PEDESTRIAN_COLOR[2] + COLOR_TOLERANCE,
                      PEDESTRIAN_COLOR[1] + COLOR_TOLERANCE,
                      PEDESTRIAN_COLOR[0] + COLOR_TOLERANCE])
    
    img_bgr = img  # OpenCV loads as BGR
    mask = cv2.inRange(img_bgr, lower, upper)
    
    # ALTERNATIVE APPROACH 1: HSV color space (sometimes better for lighting variations)
    # img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # lower_hsv = np.array([0, 100, 100])
    # upper_hsv = np.array([10, 255, 255])
    # mask_hsv = cv2.inRange(img_hsv, lower_hsv, upper_hsv)
    # mask = cv2.bitwise_or(mask, mask_hsv)
    
    return mask

def clean_mask(mask):
    """
    Clean the binary mask using morphological operations.
    
    Args:
        mask (numpy.ndarray): Binary mask with potential noise
        
    Returns:
        numpy.ndarray: Cleaned binary mask
    """
    # Standard morphological cleaning
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
    
    Args:
        mask (numpy.ndarray): Cleaned binary mask
        
    Returns:
        list: List of dictionaries containing detection information
    """
    # Standard connected components approach
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
    
    return detections

def draw_detections(img, detections, zoom_factor=2):
    """
    Draw bounding boxes and labels on the image with optional zoom for small pedestrians.
    
    Args:
        img (numpy.ndarray): Original image
        detections (list): List of detection dictionaries
        zoom_factor (int): Zoom multiplier for small pedestrians
        
    Returns:
        numpy.ndarray: Image with visualizations
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
        
        # Enhanced visualization for small pedestrians
        if det["area"] < 20 and zoom_factor > 1:
            zoom_h = (y2 - y1) * zoom_factor
            zoom_w = (x2 - x1) * zoom_factor
            zoom_img = img[max(0,y1-zoom_h//2):min(img.shape[0],y2+zoom_h//2),
                           max(0,x1-zoom_w//2):min(img.shape[1],x2+zoom_w//2)]
            if zoom_img.size > 0:
                zoom_img = cv2.resize(zoom_img, (100,100))
                vis_img[20:120,20:120] = zoom_img
                cv2.line(vis_img, (70,120), ((x1+x2)//2, y1), (0,255,0), 1)
    
    # ALTERNATIVE APPROACH 4: Different visualization style
    # for det in detections:
    #     x1, y1, x2, y2 = det["bbox"]
    #     cv2.rectangle(vis_img, (x1, y1), (x2, y2), (255, 0, 0), 1)  # Blue boxes
    #     cv2.circle(vis_img, (int(det["centroid"][0]), int(det["centroid"][1])), 
    #               3, (0, 0, 255), -1)  # Red centroid
    
    return vis_img

def create_visualization(original, mask, cleaned_mask, detections):
    """
    Create a 4-panel visualization for debugging and analysis.
    
    Args:
        original (numpy.ndarray): Original input image
        mask (numpy.ndarray): Initial binary mask
        cleaned_mask (numpy.ndarray): Processed binary mask
        detections (list): List of detected pedestrians
        
    Returns:
        matplotlib.figure.Figure: Visualization figure
    """
    fig, axes = plt.subplots(2,2,figsize=(12,10))
    
    # Original image
    axes[0,0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    axes[0,0].set_title("Original Image")
    
    # Initial mask
    axes[0,1].imshow(mask, cmap='gray')
    axes[0,1].set_title("Initial Red Mask")
    
    # Cleaned mask
    axes[1,0].imshow(cleaned_mask, cmap='gray')
    axes[1,0].set_title("Cleaned Mask")
    
    # Detections
    detection_img = draw_detections(original, detections)
    axes[1,1].imshow(cv2.cvtColor(detection_img, cv2.COLOR_BGR2RGB))
    axes[1,1].set_title(f"Detections: {len(detections)} pedestrians")
    
    # Formatting
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
    
    Args:
        img_path (str/Path): Path to input image
        
    Returns:
        tuple: (original image, cleaned mask, detections, visualization figure)
    """
    # Load image
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
    
    # Standard processing pipeline
    ped_mask = get_pedestrian_mask(img)
    clean_ped_mask = clean_mask(ped_mask)
    detections = get_pedestrian_bboxes(clean_ped_mask)
    
    vis_fig = None
    if VISUALIZE:
        vis_fig = create_visualization(img, ped_mask, clean_ped_mask, detections)
    
    return img, clean_ped_mask, detections, vis_fig

def plot_aggregate_statistics(all_results, output_dir):
    """Generate aggregate plots without modifying detection pipeline"""
    # Extract data from all results
    counts = []
    areas = []
    aspect_ratios = []
    
    for result in all_results:
        counts.append(result["count"])
        for ped in result["pedestrians"]:
            areas.append(ped["area"])
            aspect_ratios.append(ped["aspect_ratio"])
    
    # Create output directory for analysis
    analysis_dir = os.path.join(output_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)
    
    # Plot 1: Detection count distribution
    plt.figure(figsize=(10, 6))
    plt.hist(counts, bins=20, color='skyblue', edgecolor='black')
    plt.title('Distribution of Pedestrian Counts per Frame')
    plt.xlabel('Number of Pedestrians')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(analysis_dir, 'count_distribution.png'))
    plt.close()
    
    # Plot 2: Area distribution
    plt.figure(figsize=(10, 6))
    plt.hist(areas, bins=30, color='salmon', edgecolor='black', range=(0, 500))
    plt.title('Distribution of Pedestrian Bounding Box Areas')
    plt.xlabel('Area (pixels)')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(analysis_dir, 'area_distribution.png'))
    plt.close()
    
    # Plot 3: Aspect ratio distribution
    plt.figure(figsize=(10, 6))
    plt.hist(aspect_ratios, bins=20, color='lightgreen', edgecolor='black')
    plt.title('Distribution of Aspect Ratios')
    plt.xlabel('Height/Width Ratio')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(analysis_dir, 'aspect_ratio_distribution.png'))
    plt.close()

def create_heatmap(all_results, img_shape, output_dir):
    """Generate detection heatmap without modifying detection logic"""
    heatmap = np.zeros(img_shape[:2], dtype=np.float32)
    
    for result in all_results:
        for ped in result["pedestrians"]:
            x1, y1, x2, y2 = ped["bbox"]
            heatmap[y1:y2, x1:x2] += 1
    
    plt.figure(figsize=(12, 8))
    plt.imshow(heatmap, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Detection Density')
    plt.title('Pedestrian Detection Heatmap')
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, "analysis", 'detection_heatmap.png'))
    plt.close()

def main():
    """Main execution function to process all images in directory."""
    img_paths = list(Path(SEMANTIC_IMG_DIR).rglob("*.png"))
    print(f"[INFO] Starting pedestrian detection on {len(img_paths)} images.")
    total_pedestrians = 0
    
    # New variables for enhanced analysis
    all_detections = []  # Stores all detection results for aggregate analysis
    img_size = None      # Stores image dimensions for heatmap
    
    # Performance tracking (uncomment if needed)
    # start_time = time.time()
    
    for idx, img_path in enumerate(img_paths):
        # Process each image
        img, mask, detections, vis_fig = process_image(img_path)
        if img is None:
            continue
        
        # Store image dimensions from first image
        if img_size is None:
            img_size = (img.shape[1], img.shape[0])  # (width, height)
        
        # Update statistics
        total_pedestrians += len(detections)
        output_basename = Path(img_path).stem
        
        # Enhanced output data structure
        result_data = {
            "image_path": str(img_path),
            "pedestrians": detections,
            "count": len(detections),
            "image_size": {"width": img.shape[1], "height": img.shape[0]},
            "detection_quality": {
                "min_area": min([d["area"] for d in detections]) if detections else 0,
                "avg_aspect_ratio": np.mean([d["aspect_ratio"] for d in detections]) if detections else 0
            }
        }
        all_detections.append(result_data)
        
        # Save results (original format maintained)
        output_json_path = os.path.join(OUTPUT_DIR, f"{output_basename}_detections.json")
        with open(output_json_path, 'w') as f:
            json.dump({"pedestrians": detections, "count": len(detections)}, f, indent=4)
        
        # Save visualizations if enabled
        if VISUALIZE and vis_fig:
            vis_path = os.path.join(OUTPUT_DIR, f"{output_basename}_visualization.png")
            vis_fig.savefig(vis_path)
            plt.close(vis_fig)
        
        # Save detection image
        detection_img = draw_detections(img, detections)
        detection_img_path = os.path.join(OUTPUT_DIR, f"{output_basename}_detection.png")
        cv2.imwrite(detection_img_path, detection_img)
        
        # Save mask if enabled
        if SAVE_MASKS:
            mask_path = os.path.join(OUTPUT_DIR, f"{output_basename}_mask.png")
            cv2.imwrite(mask_path, mask)
        
        # Progress reporting
        if (idx+1) % 50 == 0 or idx == len(img_paths)-1:
            print(f"[INFO] Processed {idx+1}/{len(img_paths)} images. Total pedestrians: {total_pedestrians}")
    
    # --- NEW AGGREGATE ANALYSIS SECTION ---
    if len(all_detections) > 0:
        analysis_dir = os.path.join(OUTPUT_DIR, "analysis")
        os.makedirs(analysis_dir, exist_ok=True)
        
        # 1. Save comprehensive ground truth
        with open(os.path.join(analysis_dir, "aggregated_ground_truth.json"), 'w') as f:
            json.dump({
                "total_frames": len(all_detections),
                "total_pedestrians": total_pedestrians,
                "avg_pedestrians_per_frame": total_pedestrians / len(all_detections),
                "detections": all_detections
            }, f, indent=4)
        
        # 2. Generate size distribution plot
        plt.figure(figsize=(10, 6))
        areas = [d["area"] for det in all_detections for d in det["pedestrians"]]
        plt.hist(areas, bins=30, range=(0, 500), color='skyblue', edgecolor='black')
        plt.xlabel("Bounding Box Area (pixels)")
        plt.ylabel("Frequency")
        plt.title("Pedestrian Size Distribution")
        plt.savefig(os.path.join(analysis_dir, "size_distribution.png"), dpi=300)
        plt.close()
        
        # 3. Generate location heatmap
        heatmap = np.zeros((img_size[1], img_size[0]), dtype=np.float32)
        for det in all_detections:
            for ped in det["pedestrians"]:
                x1, y1, x2, y2 = ped["bbox"]
                heatmap[y1:y2, x1:x2] += 1
        
        plt.figure(figsize=(12, 8))
        plt.imshow(heatmap, cmap='hot', interpolation='nearest')
        plt.colorbar(label='Detection Density')
        plt.title("Pedestrian Location Heatmap")
        plt.axis('off')
        plt.savefig(os.path.join(analysis_dir, "location_heatmap.png"), dpi=300)
        plt.close()
    
    # Final report
    # elapsed = time.time() - start_time
    # print(f"[INFO] Processing completed in {elapsed:.2f} seconds")
    print(f"[INFO] Pedestrian detection completed. Total: {total_pedestrians}")
    if len(all_detections) > 0:
        print(f"[INFO] Analysis reports saved to: {os.path.join(OUTPUT_DIR, 'analysis')}")


if __name__ == "__main__":
    main()