"""
evaluate_demo.py

Compare tracker-based and flow-based intent predictions against the known
ground truth of the synthetic demo:

  Ped index 0 (left->right walker) : moving in every frame
  Ped index 1 (top->bottom walker) : moving in every frame
  Ped index 2 (sidewalk stander)   : stationary in every frame
  Ped index 3 (delayed walker)     : stationary for first half, moving after

Because each frame's JSON preserves detection order, we can align ground
truth to prediction by index.
"""

import argparse
import json
from pathlib import Path


def gt_for_frame(idx: int, num_frames: int):
    """Return the four ground-truth intent labels for demo frame index idx."""
    half = num_frames // 2
    ped4 = "stationary" if idx < half else "moving"
    return ["moving", "moving", "stationary", ped4]


def eval_dir(pred_dir: Path, num_frames: int, key: str):
    files = sorted(pred_dir.glob("*.json"))
    correct = 0
    total = 0
    conf = {("moving", "moving"): 0, ("moving", "stationary"): 0,
            ("stationary", "moving"): 0, ("stationary", "stationary"): 0}
    for i, f in enumerate(files):
        data = json.loads(f.read_text())
        preds = data.get(key, [])
        gts = gt_for_frame(i, num_frames)
        for j, p in enumerate(preds):
            if j >= len(gts):
                break
            gt = gts[j]
            pr = p["intent"]
            conf[(gt, pr)] += 1
            total += 1
            if gt == pr:
                correct += 1
    acc = correct / total if total else 0.0
    return {"files": len(files), "total_preds": total, "correct": correct,
            "accuracy": round(acc, 4), "confusion": {f"gt={k[0]},pred={k[1]}": v for k, v in conf.items()}}


REPO_ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser(description="Evaluate demo intent predictions")
    ap.add_argument("--tracker_dir", type=Path,
                    default=REPO_ROOT / "outputs/demo/intent_predictions")
    ap.add_argument("--flow_dir", type=Path,
                    default=REPO_ROOT / "outputs/demo/intent_with_flow")
    ap.add_argument("--num_frames", type=int, default=30)
    ap.add_argument("--out", type=Path,
                    default=REPO_ROOT / "outputs/demo/results_summary.json")
    args = ap.parse_args()

    summary = {
        "tracker_based": eval_dir(args.tracker_dir, args.num_frames, "predictions"),
        "flow_based": eval_dir(args.flow_dir, args.num_frames, "predictions"),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))

    for name, s in summary.items():
        print(f"[{name}] frames={s['files']} preds={s['total_preds']} "
              f"correct={s['correct']} acc={s['accuracy']:.4f}")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
