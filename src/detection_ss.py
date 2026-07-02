#!/usr/bin/env python3
"""
detection_ss.py

Stage 2: pedestrian detection from CARLA semantic-segmentation frames.
CARLA paints pedestrians a fixed red (RGB ~220,20,60), so a color threshold +
morphological cleanup + connected components recovers boxes that YOLO misses
(small / distant / occluded). Writes per-frame detection JSONs plus optional
mask / visualization images and aggregate analysis plots.
"""

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Pedestrian color (CARLA red) and detection thresholds
PED_CLR = (220, 20, 60)          # RGB of the CARLA pedestrian class
COLOR_TOLERANCE = 10             # allowed deviation per channel
ASPECT_RATIO_THRESHOLD = 1.2     # height must be >= 1.2x width
MIN_AREA = 20                    # minimum pixel area
MIN_HEIGHT = 8                   # minimum pixel height


def get_pedestrian_mask(img):
    """Binary mask of pedestrian-colored pixels via BGR color thresholding."""
    lower = np.array([PED_CLR[2] - COLOR_TOLERANCE,
                      PED_CLR[1] - COLOR_TOLERANCE,
                      PED_CLR[0] - COLOR_TOLERANCE])
    upper = np.array([PED_CLR[2] + COLOR_TOLERANCE,
                      PED_CLR[1] + COLOR_TOLERANCE,
                      PED_CLR[0] + COLOR_TOLERANCE])
    return cv2.inRange(img, lower, upper)


def clean_mask(mask):
    """Close small gaps then drop speckle noise."""
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask


def get_pedestrian_bboxes(mask):
    """Connected-components bboxes filtered by area, height and aspect ratio."""
    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    detections = []
    for i in range(1, num_labels):  # skip background
        x, y, w, h, area = stats[i]
        cx, cy = centroids[i]
        if area >= MIN_AREA and h >= MIN_HEIGHT:
            aspect_ratio = h / float(w) if w > 0 else 0
            if aspect_ratio > ASPECT_RATIO_THRESHOLD:
                detections.append({
                    "bbox": [int(x), int(y), int(x + w), int(y + h)],
                    "area": float(area),
                    "centroid": [float(cx), float(cy)],
                    "aspect_ratio": round(aspect_ratio, 2),
                })
    return detections


def draw_detections(img, detections, zoom_factor=2):
    """Draw bounding boxes; inset a zoom for very small pedestrians."""
    vis_img = img.copy()
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det["bbox"]
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        font_scale = 0.5 if det["area"] > 50 else 0.4
        thickness = 2 if det["area"] > 50 else 1
        cv2.putText(vis_img, f"ped {i+1}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (0, 255, 0), thickness)
        if det["area"] < 20 and zoom_factor > 1:
            zoom_h = (y2 - y1) * zoom_factor
            zoom_w = (x2 - x1) * zoom_factor
            zoom_img = img[max(0, y1 - zoom_h // 2):min(img.shape[0], y2 + zoom_h // 2),
                           max(0, x1 - zoom_w // 2):min(img.shape[1], x2 + zoom_w // 2)]
            if zoom_img.size > 0:
                zoom_img = cv2.resize(zoom_img, (100, 100))
                vis_img[20:120, 20:120] = zoom_img
                cv2.line(vis_img, (70, 120), ((x1 + x2) // 2, y1), (0, 255, 0), 1)
    return vis_img


def create_visualization(original, mask, cleaned_mask, detections):
    """4-panel figure: original, raw mask, cleaned mask, detections."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes[0, 0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("Original Image")
    axes[0, 1].imshow(mask, cmap="gray")
    axes[0, 1].set_title("Initial Red Mask")
    axes[1, 0].imshow(cleaned_mask, cmap="gray")
    axes[1, 0].set_title("Cleaned Mask")
    axes[1, 1].imshow(cv2.cvtColor(draw_detections(original, detections), cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title(f"Detections: {len(detections)} pedestrians")
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    return fig


def process_image(img_path, make_visual):
    """Detect pedestrians in a single SS frame."""
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"[ERROR] Failed to load image: {img_path}")
        return None, None, None, None
    ped_mask = get_pedestrian_mask(img)
    cleaned = clean_mask(ped_mask)
    detections = get_pedestrian_bboxes(cleaned)
    vis_fig = create_visualization(img, ped_mask, cleaned, detections) if make_visual else None
    return img, cleaned, detections, vis_fig


def plot_aggregate_statistics(all_results, output_dir):
    """Histograms of per-frame pedestrian counts and bbox areas."""
    counts = [r["count"] for r in all_results]
    areas = [ped["area"] for r in all_results for ped in r["pedestrians"]]
    analysis_dir = os.path.join(output_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.hist(counts, bins=20, edgecolor="black")
    plt.title("Distribution of Pedestrian Counts per Frame")
    plt.xlabel("Number of Pedestrians")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(analysis_dir, "count_distribution.png"))
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.hist(areas, bins=30, edgecolor="black", range=(0, 500))
    plt.title("Distribution of Pedestrian Bounding Box Areas")
    plt.xlabel("Area (pixels)")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(analysis_dir, "area_distribution.png"))
    plt.close()


def create_heatmap(all_results, output_dir):
    """Hexbin heatmap of pedestrian centroid locations."""
    centroids_x = [ped["centroid"][0] for r in all_results for ped in r["pedestrians"]]
    centroids_y = [ped["centroid"][1] for r in all_results for ped in r["pedestrians"]]
    analysis_dir = os.path.join(output_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)
    plt.figure(figsize=(12, 8))
    plt.hexbin(centroids_x, centroids_y, gridsize=50, cmap="inferno", bins="log")
    plt.colorbar(label="Log10 of Detection Density")
    plt.title("Pedestrian Locations Heatmap")
    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    plt.savefig(os.path.join(analysis_dir, "location_heatmap.png"), dpi=300)
    plt.close()


def parse_args():
    ap = argparse.ArgumentParser(description="Semantic-segmentation pedestrian detection")
    ap.add_argument("--input_dir", type=str, default="data/images_ss",
                    help="Path to semantic-segmentation frames")
    ap.add_argument("--output_dir", type=str, default="semantic_pedestrian_detection",
                    help="Path to save outputs")
    ap.add_argument("--no_visual", action="store_true", help="Skip per-frame visualizations")
    ap.add_argument("--no_mask", action="store_true", help="Skip saving mask images")
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    input_dir = Path(args.input_dir)
    img_paths = sorted(input_dir.rglob("*.png")) if input_dir.exists() else []
    if not img_paths:
        print(f"[ERROR] No PNG images found in {args.input_dir}")
        return
    print(f"[INFO] Found {len(img_paths)} images to process.")

    total_pedestrians = 0
    all_detections = []
    for idx, img_path in enumerate(img_paths):
        img, mask, detections, vis_fig = process_image(img_path, not args.no_visual)
        if img is None:
            continue
        total_pedestrians += len(detections)
        stem = img_path.stem
        all_detections.append({
            "image_path": str(img_path),
            "pedestrians": detections,
            "count": len(detections),
            "image_size": {"width": img.shape[1], "height": img.shape[0]},
        })

        (Path(args.output_dir) / f"{stem}_detections.json").write_text(
            json.dumps({"pedestrians": detections, "count": len(detections)}, indent=4)
        )
        if vis_fig is not None:
            vis_fig.savefig(os.path.join(args.output_dir, f"{stem}_visualization.png"))
            plt.close(vis_fig)
        cv2.imwrite(os.path.join(args.output_dir, f"{stem}_detection.png"),
                    draw_detections(img, detections))
        if not args.no_mask:
            cv2.imwrite(os.path.join(args.output_dir, f"{stem}_mask.png"), mask)

        if (idx + 1) % 50 == 0 or idx == len(img_paths) - 1:
            print(f"[INFO] Processed {idx+1}/{len(img_paths)} images. "
                  f"Total pedestrians: {total_pedestrians}")

    if all_detections:
        analysis_dir = os.path.join(args.output_dir, "analysis")
        os.makedirs(analysis_dir, exist_ok=True)
        with open(os.path.join(analysis_dir, "aggregated_ground_truth.json"), "w") as f:
            json.dump({
                "total_frames": len(all_detections),
                "total_pedestrians": total_pedestrians,
                "avg_pedestrians_per_frame": total_pedestrians / len(all_detections),
                "detections": all_detections,
            }, f, indent=4)
        plot_aggregate_statistics(all_detections, args.output_dir)
        create_heatmap(all_detections, args.output_dir)
        print(f"[INFO] Analysis reports saved to: {analysis_dir}")

    print(f"[INFO] Pedestrian detection completed. Total: {total_pedestrians}")


if __name__ == "__main__":
    main()
