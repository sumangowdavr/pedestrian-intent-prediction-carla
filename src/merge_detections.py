#!/usr/bin/env python3


#### IMPORTS ##############################################
import argparse
import json
import csv
import os
import time
from pathlib import Path

import cv2
from tqdm import tqdm

from common import compute_iou

#### CONFIGURATION #######################################
# defaults; override via CLI or env vars
DEFAULT_RGB_DIR = os.environ.get("RGB_DIR", "data/images_rgb")
IMG_RGB_DIR    = Path(DEFAULT_RGB_DIR)
YOLO_JSON_DIR  = Path("detections")
SS_JSON_DIR    = Path("semantic_pedestrian_detections")
ANNOTATED_DIR  = YOLO_JSON_DIR / "annotated"

OUT_JSON_DIR   = Path("merged_detections")
OUT_IMG_DIR    = Path("merged_images")
OUT_CROP_DIR   = Path("crops_missing")

IOU_THRESH     = 0.5

#### MAIN MERGE LOOP ####################################
def parse_args():
    ap = argparse.ArgumentParser(description="Merge YOLO + SS pedestrian detections")
    ap.add_argument("--rgb_dir", type=Path, default=IMG_RGB_DIR,
                    help="Directory with raw RGB frames")
    ap.add_argument("--yolo_dir", type=Path, default=YOLO_JSON_DIR,
                    help="Directory with YOLO detection JSONs")
    ap.add_argument("--ss_dir", type=Path, default=SS_JSON_DIR,
                    help="Directory with SS detection JSONs")
    ap.add_argument("--iou_thresh", type=float, default=IOU_THRESH,
                    help="IoU threshold for detection matching")
    return ap.parse_args()


def main():
    global IMG_RGB_DIR, YOLO_JSON_DIR, SS_JSON_DIR, ANNOTATED_DIR, IOU_THRESH
    args = parse_args()
    IMG_RGB_DIR   = args.rgb_dir
    YOLO_JSON_DIR = args.yolo_dir
    SS_JSON_DIR   = args.ss_dir
    ANNOTATED_DIR = YOLO_JSON_DIR / "annotated"
    IOU_THRESH    = args.iou_thresh

    for d in (OUT_JSON_DIR, OUT_IMG_DIR, OUT_CROP_DIR):
        d.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    summary = []

    # 1) gather all frame IDs from your YOLO-annotated PNGs
    frames = sorted({
        p.stem.replace("_annotated", "")
        for p in ANNOTATED_DIR.glob("*.png")
    })

    print(f"[INFO] Found {len(frames)} frames to merge. IOU_THRESH={IOU_THRESH}")
    for frame in tqdm(frames, desc="Merging frames"):
        # 2) load YOLO JSON for this frame
        yolo_json = YOLO_JSON_DIR / f"{frame}_detections.json"
        if yolo_json.exists():
            yobj      = json.loads(yolo_json.read_text())
            yolo_peds = yobj.get("pedestrians", [])
        else:
            yolo_peds = []

        # 3) map RGB suffix (_0/_1) → SS suffix (_10/_11)
        prefix, suffix = frame.rsplit("_", 1)
        sem_suffix     = str(int(suffix) + 10)
        ss_frame       = f"{prefix}_{sem_suffix}"
        ss_json_path   = SS_JSON_DIR / f"{ss_frame}_detections.json"

        if ss_json_path.exists():
            ss_peds = json.loads(ss_json_path.read_text()).get("pedestrians", [])
        else:
            ss_peds = []

        # 4) find any SS boxes that don't match a YOLO box
        yolo_boxes = [d["bbox"] for d in yolo_peds]
        missing = [
            d for d in ss_peds
            if all(compute_iou(d["bbox"], yb) < IOU_THRESH for yb in yolo_boxes)
        ]

        # 5) write out merged JSON
        merged = yolo_peds + missing
        out_json = OUT_JSON_DIR / f"{frame}.json"
        out_json.write_text(
            json.dumps({"pedestrians": merged, "count": len(merged)}, indent=2)
        )

        # 6) load the RGB image (fallback to annotated if raw missing)
        img_path = IMG_RGB_DIR / f"{frame}.png"
        img      = cv2.imread(str(img_path))
        if img is None:
            img = cv2.imread(str(ANNOTATED_DIR / f"{frame}_annotated.png"))
        if img is None:
            # give up on visuals for this frame
            summary.append({
                "frame": frame,
                "yolo":  len(yolo_peds),
                "ss_only": len(missing),
                "total": len(merged)
            })
            continue

        # 7) draw YOLO boxes (green) and SS misses (red)
        for d in yolo_peds:
            x1,y1,x2,y2 = d["bbox"]
            cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)

        for d in missing:
            x1,y1,x2,y2 = d["bbox"]
            # RED in BGR
            cv2.rectangle(img, (x1,y1), (x2,y2), (0,0,255), 2)

        # 8) save merged image
        cv2.imwrite(str(OUT_IMG_DIR / f"{frame}.png"), img)

        # 9) save crops of each SS-only miss
        for idx, d in enumerate(missing):
            x1,y1,x2,y2 = d["bbox"]
            crop = img[y1:y2, x1:x2]
            if crop.size:
                cv2.imwrite(str(OUT_CROP_DIR / f"{frame}_miss_{idx}.png"), crop)

        summary.append({
            "frame":   frame,
            "yolo":    len(yolo_peds),
            "ss_only": len(missing),
            "total":   len(merged)
        })

    # 10) write summary CSV
    with open("merged_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["frame","yolo","ss_only","total"])
        writer.writeheader()
        writer.writerows(summary)

    elapsed = time.time() - start_time
    print(f"\n Done! Processed {len(frames)} frames in {elapsed:.1f}s")
    print("→ See merged_summary.csv for per-frame counts")

if __name__ == "__main__":
    main()
