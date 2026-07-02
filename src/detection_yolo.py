#!/usr/bin/env python3
"""
detection_yolo.py

Stage 1 of the pipeline: run YOLOv8 on (optionally enhanced) RGB frames and
keep only the pedestrian (COCO class 0) boxes.

For each frame it writes:
  - <frame>.json          per-frame bbox + confidence
  - annotated/<frame>.png  RGB with green pedestrian boxes
  - crops/<frame>_ped_N.png  a crop per detected pedestrian
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import cv2
from tqdm import tqdm
from ultralytics import YOLO

from common import enhance_image


def parse_args():
    ap = argparse.ArgumentParser(description="YOLOv8 pedestrian detection on RGB frames")
    ap.add_argument("--img_dir", type=Path, default=Path("data/images_rgb"),
                    help="Directory with RGB frames")
    ap.add_argument("--out_dir", type=Path, default=Path("detections"),
                    help="Output directory")
    ap.add_argument("--weights", type=str, default="yolov8s.pt",
                    help="YOLO weights (downloaded separately)")
    ap.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    ap.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    ap.add_argument("--tta", action="store_true", help="Enable test-time augmentation")
    ap.add_argument("--no_enhance", action="store_true", help="Skip CLAHE/gamma/sharpen")
    return ap.parse_args()


def main():
    args = parse_args()
    crop_dir = args.out_dir / "crops"
    ann_dir = args.out_dir / "annotated"
    for d in (args.out_dir, crop_dir, ann_dir):
        d.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)
    model.to("cpu")  # force CPU for stability; remove to run on GPU

    start = datetime.now()
    frames, total = 0, 0
    for img_path in tqdm(sorted(args.img_dir.rglob("*.png"))):
        stem = img_path.stem
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        proc = img if args.no_enhance else enhance_image(img)
        results = model.predict(proc, conf=args.conf, iou=args.iou,
                                augment=args.tta, verbose=False)[0]

        peds = []
        for box in results.boxes:
            if int(box.cls) != 0:  # class 0 = person in COCO
                continue
            x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0].astype(int).tolist()
            peds.append({"bbox": [x1, y1, x2, y2], "confidence": round(float(box.conf), 3)})

        annotated = img.copy()
        for idx, d in enumerate(peds):
            x1, y1, x2, y2 = d["bbox"]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated, f"ped {d['confidence']:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            crop = img[y1:y2, x1:x2]
            if crop.size:
                cv2.imwrite(str(crop_dir / f"{stem}_ped_{idx}.png"), crop)

        cv2.imwrite(str(ann_dir / f"{stem}_annotated.png"), annotated)
        (args.out_dir / f"{stem}_detections.json").write_text(
            json.dumps({"pedestrians": peds, "count": len(peds)}, indent=2)
        )
        frames += 1
        total += len(peds)

    if frames:
        elapsed = datetime.now() - start
        print(f"\nProcessed {frames} frames in {elapsed}.")
        print(f"-> Detected {total} pedestrians total ({total / frames:.2f} per frame).")
    else:
        print(f"No PNG frames found under {args.img_dir}")


if __name__ == "__main__":
    main()
