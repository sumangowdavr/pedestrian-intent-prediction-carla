"""
generate_demo.py

Create a small self-contained demo dataset so the intent-prediction pipeline
can be run and visualized without needing CARLA or YOLO weights.

Outputs (under --out_dir, default ./demo):
  merged_images/<frame>.png       synthetic RGB frames
  merged_detections/<frame>.json  ground-truth bboxes per frame

Scene: a static road backdrop with textured noise (so Farneback flow is
meaningful) plus four "pedestrians":
  - #1 walks left -> right   (moving)
  - #2 walks top -> bottom   (moving)
  - #3 is stationary
  - #4 briefly stops, then starts moving
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


PED_COLOR = (60, 20, 220)   # BGR: CARLA-red-ish
PED_SIZE = (24, 60)         # width, height


def make_backdrop(h, w, rng):
    """Static-ish backdrop with light structured noise so Farneback flow has signal."""
    bg = np.full((h, w, 3), 90, dtype=np.uint8)
    # a "road" band
    cv2.rectangle(bg, (0, h // 2 - 40), (w, h // 2 + 40), (60, 60, 60), -1)
    # lane markings
    for x in range(0, w, 60):
        cv2.rectangle(bg, (x + 10, h // 2 - 2), (x + 40, h // 2 + 2), (240, 240, 240), -1)
    # a few "buildings"
    for cx in (80, 260, 460, 640):
        cv2.rectangle(bg, (cx, 40), (cx + 80, h // 2 - 45), (140, 130, 120), -1)
    # low-frequency noise so optical flow has texture
    noise = rng.integers(-8, 8, size=(h, w, 3)).astype(np.int16)
    bg = np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return bg


def draw_ped(img, x, y, color=PED_COLOR, size=PED_SIZE):
    w, h = size
    x1, y1 = int(x - w / 2), int(y - h / 2)
    x2, y2 = x1 + w, y1 + h
    cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
    # horizontal stripes so the interior has texture -> optical flow has signal
    for sy in range(y1 + 6, y2, 8):
        cv2.line(img, (x1 + 1, sy), (x2 - 1, sy), (255, 255, 255), 1)
    # vertical belt in the middle
    mid = (x1 + x2) // 2
    cv2.line(img, (mid, y1 + 2), (mid, y2 - 2), (30, 30, 30), 1)
    # "head" with contrast
    cv2.circle(img, (int(x), int(y1 - 5)), 7, color, -1)
    cv2.circle(img, (int(x), int(y1 - 5)), 7, (10, 10, 10), 1)
    return [max(x1, 0), max(y1 - 12, 0), x2, y2]


def frame_id(i):
    return f"{i:04d}_10"


REPO_ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser(description="Generate synthetic intent-prediction demo")
    ap.add_argument("--out_dir", type=Path, default=REPO_ROOT / "outputs/demo")
    ap.add_argument("--num_frames", type=int, default=30)
    ap.add_argument("--width", type=int, default=800)
    ap.add_argument("--height", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    img_dir = args.out_dir / "merged_images"
    json_dir = args.out_dir / "merged_detections"
    img_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)

    W, H = args.width, args.height
    backdrop = make_backdrop(H, W, rng)

    # regenerate a new noise every frame so the *background* has small residual flow
    # that our threshold must reject
    for i in range(args.num_frames):
        # small independent noise sample for each frame
        noise = rng.integers(-4, 4, size=(H, W, 3)).astype(np.int16)
        img = np.clip(backdrop.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        peds = []

        # Ped #1: walks left -> right along the road
        x1 = 40 + i * 10
        y1 = H // 2 + 10
        peds.append(draw_ped(img, x1, y1))

        # Ped #2: walks top -> bottom near center
        x2 = W // 2 + 20
        y2 = 60 + i * 6
        peds.append(draw_ped(img, x2, y2))

        # Ped #3: stationary sidewalk figure
        x3, y3 = W - 120, H // 2 - 70
        peds.append(draw_ped(img, x3, y3))

        # Ped #4: stationary for first half, then moves
        if i < args.num_frames // 2:
            x4, y4 = 200, H // 2 - 70
        else:
            x4 = 200 + (i - args.num_frames // 2) * 12
            y4 = H // 2 - 70
        peds.append(draw_ped(img, x4, y4))

        fid = frame_id(i)
        cv2.imwrite(str(img_dir / f"{fid}.png"), img)
        (json_dir / f"{fid}.json").write_text(json.dumps({
            "frame": fid,
            "pedestrians": [{"bbox": bb} for bb in peds],
            "count": len(peds),
        }, indent=2))

    print(f"Wrote {args.num_frames} demo frames to {img_dir}/")
    print(f"Wrote {args.num_frames} detection JSONs to {json_dir}/")


if __name__ == "__main__":
    main()
