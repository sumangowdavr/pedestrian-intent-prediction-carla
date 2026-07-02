"""
intent_prediction.py

Dense-optical-flow based intent prediction:
For each pedestrian bbox in frame t, compute mean magnitude of the
Farneback flow between t and t+1 inside the bbox, and classify
the pedestrian as 'moving' or 'stationary' vs a threshold.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def parse_args():
    ap = argparse.ArgumentParser(description="Flow-based intent prediction")
    ap.add_argument("--img_dir", type=Path, default=Path("merged_images"),
                    help="Directory of RGB frames named <frame>.png")
    ap.add_argument("--json_dir", type=Path, default=Path("merged_detections"),
                    help="Directory with per-frame detection JSONs")
    ap.add_argument("--out_json_dir", type=Path, default=Path("intent_with_flow"),
                    help="Output directory for per-frame intent JSONs")
    ap.add_argument("--viz_dir", type=Path, default=Path("merged_visual_flow_final"),
                    help="Output directory for annotated overlays")
    ap.add_argument("--flow_thresh", type=float, default=0.5,
                    help="Mean flow magnitude above which intent = 'moving'")
    ap.add_argument("--no_viz", action="store_true",
                    help="Disable writing annotated overlay images")
    return ap.parse_args()


def load_frame(img_dir: Path, frame_id: str):
    p = img_dir / f"{frame_id}.png"
    img = cv2.imread(str(p))
    if img is None:
        return None, None
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), img


def main():
    args = parse_args()
    args.out_json_dir.mkdir(exist_ok=True, parents=True)
    if not args.no_viz:
        args.viz_dir.mkdir(exist_ok=True, parents=True)

    frames = sorted(p.stem for p in args.img_dir.glob("*.png"))
    if len(frames) < 2:
        print(f"[WARN] Need at least 2 frames in {args.img_dir}; found {len(frames)}")
        return

    for idx in tqdm(range(len(frames) - 1), desc="Computing intents"):
        f0, f1 = frames[idx], frames[idx + 1]

        gray0, img0 = load_frame(args.img_dir, f0)
        gray1, _ = load_frame(args.img_dir, f1)
        if gray0 is None or gray1 is None:
            continue

        det_path = args.json_dir / f"{f0}.json"
        if not det_path.exists():
            continue
        anns = json.loads(det_path.read_text()).get("pedestrians", [])

        flow = cv2.calcOpticalFlowFarneback(
            gray0, gray1, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
        )
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=True)

        preds = []
        for det in anns:
            x1, y1, x2, y2 = map(int, det["bbox"])
            x1, y1 = max(0, x1), max(0, y1)
            x2 = min(mag.shape[1], x2)
            y2 = min(mag.shape[0], y2)
            if x2 <= x1 or y2 <= y1:
                continue

            avg_flow = float(np.mean(mag[y1:y2, x1:x2]))
            intent = "moving" if avg_flow > args.flow_thresh else "stationary"

            preds.append({
                "bbox": det["bbox"],
                "avg_flow": round(avg_flow, 3),
                "intent": intent,
            })

        (args.out_json_dir / f"{f0}.json").write_text(
            json.dumps({"frame": f0, "predictions": preds}, indent=2)
        )

        if not args.no_viz:
            vis = img0.copy()
            for p in preds:
                x1, y1, x2, y2 = map(int, p["bbox"])
                col = (0, 255, 0) if p["intent"] == "moving" else (0, 0, 255)
                cv2.rectangle(vis, (x1, y1), (x2, y2), col, 2)
                lbl = f"{p['intent']}:{p['avg_flow']:.2f}"
                cv2.putText(vis, lbl, (x1, max(0, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1, cv2.LINE_AA)
            cv2.imwrite(str(args.viz_dir / f"{f0}.png"), vis)

    print("Flow-based intent done.")


if __name__ == "__main__":
    main()
