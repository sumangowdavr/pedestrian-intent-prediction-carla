"""
visualize_intent.py

Overlay per-frame intent predictions on top of merged RGB frames.
Green box = 'moving', red box = 'stationary'.
"""

import argparse
import json
from pathlib import Path

import cv2
from tqdm import tqdm

COL_MOVING = (0, 255, 0)
COL_STATIONARY = (0, 0, 255)


def parse_args():
    ap = argparse.ArgumentParser(description="Visualize intent predictions on RGB frames")
    ap.add_argument("--img_dir", type=Path, default=Path("merged_images"),
                    help="Directory with merged RGB frames")
    ap.add_argument("--intent_dir", type=Path, default=Path("intent_predictions"),
                    help="Directory with per-frame intent JSONs")
    ap.add_argument("--out_dir", type=Path, default=Path("merged_visual_intent_final"),
                    help="Output directory for annotated frames")
    return ap.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(args.intent_dir.glob("*.json"))
    if not json_files:
        print(f"[WARN] No intent JSONs in {args.intent_dir}")
        return

    written = 0
    for js in tqdm(json_files, desc="Visualizing intents"):
        frame = js.stem
        img_path = args.img_dir / f"{frame}.png"
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        data = json.loads(js.read_text())
        for pred in data.get("predictions", []):
            x1, y1, x2, y2 = pred["bbox"]
            intent = pred["intent"]
            speed = pred.get("speed", 0.0)
            tid = pred.get("track_id", -1)
            color = COL_MOVING if intent == "moving" else COL_STATIONARY

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label = f"#{tid} {intent} {speed}px/f"
            cv2.putText(
                img, label, (x1, max(0, y2 + 15)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
                lineType=cv2.LINE_AA,
            )

        cv2.imwrite(str(args.out_dir / f"{frame}.png"), img)
        written += 1

    print(f"Wrote {written} annotated frames to {args.out_dir}/")


if __name__ == "__main__":
    main()
