"""
run_real_pipeline.py

Drive the pedestrian-intent pipeline on a contiguous slice of the CARLA
Town01 data the user placed in ../data/{images_rgb,images_ss}.
Produces:
  real_run/rgb/           the RGB frames used
  real_run/ss/            the paired SS frames used
  real_run/yolo/          per-frame YOLO detection JSONs (empty if YOLO skipped)
  real_run/ss_det/        per-frame SS detection JSONs
  real_run/merged_det/    merged detection JSONs
  real_run/merged_img/    RGB with YOLO=green + SS-only=red overlays
  real_run/intent_flow/   per-frame flow-based intent JSONs
  real_run/vis_flow/      RGB with intent overlays (green=moving, red=stationary)
  real_run/intent_tracker/ tracker-based intent JSONs
  real_run/vis_tracker/   tracker-based overlays
  real_run/results_summary.json  aggregate stats
"""

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def collect_slice(rgb_root: Path, ss_root: Path, start: int, count: int, suffix: str):
    """Return list of (rgb_path, ss_path) pairs for the contiguous slice."""
    pairs = []
    for i in range(start, start + count):
        rgb = rgb_root / f"{i:04d}_{suffix}.png"
        ss  = ss_root  / f"{i:04d}_{int(suffix) + 10}.png"
        if rgb.exists() and ss.exists():
            pairs.append((rgb, ss))
    return pairs


def enhance_image(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    cl = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    lab = cv2.merge((cl, a, b))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    invG = 1.0 / 1.3
    table = np.array([((i / 255.0) ** invG) * 255 for i in np.arange(256)]).astype("uint8")
    img = cv2.LUT(img, table)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    img = cv2.filter2D(img, -1, kernel)
    return img


def run_yolo(rgb_paths, out_dir: Path, weights: str, conf: float, imgsz: int):
    from ultralytics import YOLO
    print(f"[yolo] loading {weights}")
    model = YOLO(weights)
    model.to("cpu")
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    for p in tqdm(rgb_paths, desc="YOLO"):
        img = cv2.imread(str(p))
        if img is None:
            continue
        img_e = enhance_image(img)
        res = model.predict(img_e, imgsz=imgsz, conf=conf, verbose=False, augment=False)[0]
        peds = []
        for box in res.boxes:
            if int(box.cls) == 0:
                x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0].astype(int).tolist()
                cv = float(box.conf)
                peds.append({"bbox": [x1, y1, x2, y2], "confidence": round(cv, 3)})
        stem = p.stem  # e.g. "1539_0"
        (out_dir / f"{stem}_detections.json").write_text(
            json.dumps({"pedestrians": peds, "count": len(peds)}, indent=2)
        )
    print(f"[yolo] done in {time.time() - t0:.1f}s")


PED_BGR = np.array([60, 20, 220], dtype=np.uint8)  # BGR of CARLA pedestrian red
COLOR_TOL = 15


def ss_detect(ss_paths, out_dir: Path, min_area: int = 4, min_h: int = 3,
              aspect: float = 1.0, close_iter: int = 1):
    """Extract pedestrian bboxes from a CARLA SS frame.

    CARLA pedestrians are red (BGR ~60,20,220). Note: distant peds here can be
    only 1-2 pixels wide, so we skip MORPH_OPEN (which would erase them) and
    use very permissive area/height thresholds. A gentle CLOSE joins the head
    and body components.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in tqdm(ss_paths, desc="SS"):
        img = cv2.imread(str(p))
        if img is None:
            continue
        lower = np.clip(PED_BGR.astype(int) - COLOR_TOL, 0, 255).astype(np.uint8)
        upper = np.clip(PED_BGR.astype(int) + COLOR_TOL, 0, 255).astype(np.uint8)
        mask = cv2.inRange(img, lower, upper)
        if close_iter > 0:
            k = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=close_iter)
        n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        peds = []
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if area < min_area or h < min_h:
                continue
            ar = h / float(w) if w > 0 else 0
            if ar < aspect:
                continue
            peds.append({"bbox": [int(x), int(y), int(x + w), int(y + h)]})
        (out_dir / f"{p.stem}_detections.json").write_text(
            json.dumps({"pedestrians": peds, "count": len(peds)}, indent=2)
        )


def compute_iou(a, b):
    xA, yA = max(a[0], b[0]), max(a[1], b[1])
    xB, yB = min(a[2], b[2]), min(a[3], b[3])
    interW = max(0, xB - xA); interH = max(0, yB - yA)
    inter = interW * interH
    aA = (a[2] - a[0]) * (a[3] - a[1])
    aB = (b[2] - b[0]) * (b[3] - b[1])
    u = aA + aB - inter
    return inter / u if u > 0 else 0.0


def merge(rgb_paths, yolo_dir: Path, ss_dir: Path, out_json_dir: Path,
          out_img_dir: Path, iou_thresh: float):
    out_json_dir.mkdir(parents=True, exist_ok=True)
    out_img_dir.mkdir(parents=True, exist_ok=True)
    for p in tqdm(rgb_paths, desc="Merge"):
        stem = p.stem                        # 1539_0
        prefix, suf = stem.rsplit("_", 1)
        ss_stem = f"{prefix}_{int(suf) + 10}"

        yolo_peds = []
        yjp = yolo_dir / f"{stem}_detections.json"
        if yjp.exists():
            yolo_peds = json.loads(yjp.read_text()).get("pedestrians", [])

        ss_peds = []
        sjp = ss_dir / f"{ss_stem}_detections.json"
        if sjp.exists():
            ss_peds = json.loads(sjp.read_text()).get("pedestrians", [])

        yolo_boxes = [d["bbox"] for d in yolo_peds]
        ss_only = [
            d for d in ss_peds
            if all(compute_iou(d["bbox"], yb) < iou_thresh for yb in yolo_boxes)
        ]
        merged = yolo_peds + ss_only

        (out_json_dir / f"{stem}.json").write_text(
            json.dumps({"pedestrians": merged, "count": len(merged),
                        "n_yolo": len(yolo_peds), "n_ss_only": len(ss_only)}, indent=2)
        )

        img = cv2.imread(str(p))
        if img is None:
            continue
        for d in yolo_peds:
            x1, y1, x2, y2 = d["bbox"]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        for d in ss_only:
            x1, y1, x2, y2 = d["bbox"]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.imwrite(str(out_img_dir / f"{stem}.png"), img)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rgb_root", type=Path, default=Path("../data/images_rgb"))
    ap.add_argument("--ss_root",  type=Path, default=Path("../data/images_ss"))
    ap.add_argument("--out_root", type=Path, default=Path("real_run"))
    ap.add_argument("--start", type=int, default=1539)
    ap.add_argument("--count", type=int, default=65)
    ap.add_argument("--suffix", type=str, default="0")
    ap.add_argument("--yolo_weights", type=str, default="yolov8s.pt")
    ap.add_argument("--yolo_conf", type=float, default=0.25)
    ap.add_argument("--yolo_imgsz", type=int, default=640)
    ap.add_argument("--iou_thresh", type=float, default=0.5)
    ap.add_argument("--flow_thresh", type=float, default=0.5)
    ap.add_argument("--speed_thresh", type=float, default=2.0)
    ap.add_argument("--skip_yolo", action="store_true", help="Skip YOLO stage entirely")
    return ap.parse_args()


def main():
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)

    pairs = collect_slice(args.rgb_root, args.ss_root, args.start, args.count, args.suffix)
    if not pairs:
        print("No paired frames in the requested slice.")
        sys.exit(1)
    print(f"[data] using {len(pairs)} paired frames")

    rgb_paths = [p[0] for p in pairs]
    ss_paths  = [p[1] for p in pairs]

    # 1) YOLO on RGB
    yolo_dir = args.out_root / "yolo"
    if args.skip_yolo:
        yolo_dir.mkdir(parents=True, exist_ok=True)
    else:
        run_yolo(rgb_paths, yolo_dir, args.yolo_weights, args.yolo_conf, args.yolo_imgsz)

    # 2) SS-based detection
    ss_det_dir = args.out_root / "ss_det"
    ss_detect(ss_paths, ss_det_dir)

    # 3) Merge
    merged_json = args.out_root / "merged_det"
    merged_img  = args.out_root / "merged_img"
    merge(rgb_paths, yolo_dir, ss_det_dir, merged_json, merged_img, args.iou_thresh)

    # 4) Copy merged_img to a canonical merged_images so intent script can find them
    #    (the flow intent script wants merged_images with same names as merged JSONs)
    #    - reuse merged_img itself but reset filename mapping:
    #      merged_img filenames use RGB stem (e.g. 1539_0.png)

    # 5) Flow-based intent
    intent_flow = args.out_root / "intent_flow"
    vis_flow    = args.out_root / "vis_flow"
    subprocess.check_call([
        sys.executable, "intent_prediction.py",
        "--img_dir", str(merged_img),
        "--json_dir", str(merged_json),
        "--out_json_dir", str(intent_flow),
        "--viz_dir", str(vis_flow),
        "--flow_thresh", str(args.flow_thresh),
    ])

    # 6) Tracker-based intent (root script)
    intent_tr = args.out_root / "intent_tracker"
    subprocess.check_call([
        sys.executable, "../intent_prediction.py",
        "--merged_dir", str(merged_json),
        "--out_dir", str(intent_tr),
        "--iou_thresh", "0.3",
        "--speed_thresh", str(args.speed_thresh),
    ])

    # 7) Tracker-based viz
    vis_tr = args.out_root / "vis_tracker"
    subprocess.check_call([
        sys.executable, "visualize_intent.py",
        "--img_dir", str(merged_img),
        "--intent_dir", str(intent_tr),
        "--out_dir", str(vis_tr),
    ])

    # 8) Aggregate summary
    summary = summarize(args.out_root, len(pairs))
    (args.out_root / "results_summary.json").write_text(json.dumps(summary, indent=2))
    print("\n== RESULTS ==")
    for k, v in summary.items():
        print(f"  {k}: {v}")


def summarize(root: Path, n_frames: int):
    def load_all(d):
        return [json.loads(p.read_text()) for p in sorted(d.glob("*.json"))]

    yolo = load_all(root / "yolo") if (root / "yolo").exists() else []
    ss   = load_all(root / "ss_det")
    mer  = load_all(root / "merged_det")
    intf = load_all(root / "intent_flow")
    intt = load_all(root / "intent_tracker")

    def cnt(items, key="pedestrians"): return sum(len(x.get(key, [])) for x in items)

    return {
        "frames_paired": n_frames,
        "yolo_detections": cnt(yolo),
        "ss_detections": cnt(ss),
        "merged_detections": cnt(mer),
        "ss_only_added_by_merge": sum(x.get("n_ss_only", 0) for x in mer),
        "flow_moving": sum(sum(1 for p in x.get("predictions", []) if p["intent"] == "moving") for x in intf),
        "flow_stationary": sum(sum(1 for p in x.get("predictions", []) if p["intent"] == "stationary") for x in intf),
        "tracker_moving": sum(sum(1 for p in x.get("predictions", []) if p["intent"] == "moving") for x in intt),
        "tracker_stationary": sum(sum(1 for p in x.get("predictions", []) if p["intent"] == "stationary") for x in intt),
        "tracker_unique_track_ids": len({p["track_id"] for x in intt for p in x.get("predictions", [])}),
    }


if __name__ == "__main__":
    main()
