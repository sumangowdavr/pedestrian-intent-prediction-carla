"""
intent_prediction.py

Tracker + speed based intent prediction.
Reads per-frame merged detection JSONs, matches boxes to existing tracks by
IoU, computes centroid pixel speed, and labels each detection as
'moving' or 'stationary'.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from common import compute_iou


def run(merged_dir: Path, out_dir: Path, iou_thresh: float, speed_thresh: float):
    out_dir.mkdir(parents=True, exist_ok=True)

    next_track_id = 0
    tracks = {}

    frames = sorted(p.stem for p in merged_dir.glob("*.json"))
    if not frames:
        print(f"[WARN] No JSON files in {merged_dir}")
        return

    for frame in tqdm(frames, desc="Tracking & classifying"):
        data = json.loads((merged_dir / f"{frame}.json").read_text())
        detections = data.get("pedestrians", [])
        bboxes = [d["bbox"] for d in detections]

        assigned = {}
        used_tracks = set()

        for det_idx, bb in enumerate(bboxes):
            best_iou = iou_thresh
            best_tid = None
            for tid, info in tracks.items():
                if info["last_frame"] == frame or tid in used_tracks:
                    continue
                i = compute_iou(bb, info["bbox"])
                if i > best_iou:
                    best_iou = i
                    best_tid = tid
            if best_tid is not None:
                assigned[det_idx] = best_tid
                used_tracks.add(best_tid)

        for det_idx, bb in enumerate(bboxes):
            if det_idx in assigned:
                continue
            assigned[det_idx] = next_track_id
            next_track_id += 1

        for det_idx, tid in assigned.items():
            bb = bboxes[det_idx]
            x1, y1, x2, y2 = bb
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            info = tracks.get(tid, {"history": []})
            info["bbox"] = bb
            info["last_frame"] = frame
            info["history"].append((frame, (cx, cy)))
            tracks[tid] = info

        predictions = []
        for det_idx, d in enumerate(detections):
            tid = assigned[det_idx]
            hist = tracks[tid]["history"]
            if len(hist) >= 2 and hist[-2][0] != hist[-1][0]:
                (_, (px_old, py_old)), (_, (px_new, py_new)) = hist[-2], hist[-1]
                speed = float(np.hypot(px_new - px_old, py_new - py_old))
            else:
                speed = 0.0

            intent = "moving" if speed > speed_thresh else "stationary"
            predictions.append({
                "track_id": int(tid),
                "bbox": d["bbox"],
                "speed": round(speed, 2),
                "intent": intent,
            })

        (out_dir / f"{frame}.json").write_text(
            json.dumps({"frame": frame, "predictions": predictions}, indent=2)
        )

    print(f"Wrote intent predictions for {len(frames)} frames to `{out_dir}/`")


def parse_args():
    p = argparse.ArgumentParser(description="Tracker + speed based intent prediction")
    p.add_argument("--merged_dir", type=Path, default=Path("merged_detections"),
                   help="Directory with per-frame merged detection JSONs")
    p.add_argument("--out_dir", type=Path, default=Path("intent_predictions"),
                   help="Output directory for per-frame intent JSONs")
    p.add_argument("--iou_thresh", type=float, default=0.3,
                   help="IoU threshold for track association")
    p.add_argument("--speed_thresh", type=float, default=1.0,
                   help="Pixel-speed threshold above which a track is called 'moving'")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.merged_dir, args.out_dir, args.iou_thresh, args.speed_thresh)
