# Synthetic Demo

The main pipeline needs CARLA RGB/SS frames and YOLO weights. This
self-contained demo runs end-to-end with only `opencv-python`, `numpy`,
and `tqdm` (plus optional PIL for the animated GIF).

Run everything from the repo root - the scripts figure out their own
default paths (`outputs/demo/*`).

## Quick start

```bash
# 1) Generate synthetic frames + ground-truth boxes
python src/generate_demo.py

# 2a) Flow-based intent (Farneback optical flow inside each bbox)
python src/intent_flow.py \
    --img_dir outputs/demo/merged_images \
    --json_dir outputs/demo/merged_detections \
    --out_json_dir outputs/demo/intent_with_flow \
    --viz_dir outputs/demo/merged_visual_flow_final \
    --flow_thresh 0.5

# 2b) Tracker-based intent (IoU association + pixel-speed)
python src/intent_tracker.py \
    --merged_dir outputs/demo/merged_detections \
    --out_dir outputs/demo/intent_predictions \
    --iou_thresh 0.3 --speed_thresh 2.0

python src/visualize_intent.py \
    --img_dir outputs/demo/merged_images \
    --intent_dir outputs/demo/intent_predictions \
    --out_dir outputs/demo/merged_visual_intent_final

# 3) Stitch into a video / GIF
python src/make_demo_video.py \
    --frames_dir outputs/demo/merged_visual_intent_final \
    --out_mp4 outputs/demo/intent_demo.mp4 \
    --out_gif outputs/demo/intent_demo.gif --fps 6

# 4) Evaluate vs known ground truth
python src/evaluate_demo.py
```

## Demo scene

Every frame contains four "pedestrian" rectangles:

| index | behavior                          | expected label            |
|-------|-----------------------------------|---------------------------|
| 0     | walks left -> right along road    | moving                    |
| 1     | walks top -> bottom               | moving                    |
| 2     | stands on sidewalk                | stationary                |
| 3     | stands for first half, then walks | stationary -> moving      |

## Latest run

Numbers from `outputs/demo/results_summary.json`:

| method          | frames | preds | correct | accuracy |
|-----------------|--------|-------|---------|----------|
| tracker-based   | 30     | 120   | 117     | 0.9750   |
| flow-based      | 29     | 116   | 116     | 1.0000   |

Tracker-based misses the first frame of each track (no prior centroid,
so speed = 0). Flow-based is 100% because texture inside each ped means
the Farneback flow reflects the actual motion.
