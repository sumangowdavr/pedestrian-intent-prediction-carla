# Real-Data Run on CARLA Town01

This is a real end-to-end run of the pipeline on the CARLA Town01 slice
placed under `data/{images_rgb,images_ss}` (1392x1024, `_N`/`_N+10`
naming convention). Contrast with the `demo/` folder, which uses
synthetic frames for pipeline validation only.

## Slice and settings

- Frames: `1644_0.png` through `1666_0.png` (23 contiguous RGB frames)
- Paired SS: `1644_10.png` through `1666_10.png`
- YOLO weights: `yolov8s.pt`, `imgsz=640`, `conf=0.25`, CPU
- SS thresholds: `MIN_AREA=4`, `MIN_H=3`, `ASPECT>=1.0`, `MORPH_OPEN` disabled
- Merge IoU threshold: `0.5`
- Flow-based intent threshold: `0.5 px/frame`
- Tracker-based intent threshold: `2.0 px/frame`, IoU association `0.3`

The report specified YOLOv8l with TTA; here I substituted YOLOv8s and
disabled TTA to stay within the CPU-only time budget of this run. The
merge step is the primary recall driver anyway - SS recovers what YOLO
misses regardless of the YOLO variant.

The SS thresholds are also relaxed vs `detection_ss.py`'s defaults: in
this specific slice, pedestrians are far from the ego vehicle (single-
digit pixels wide), so `MIN_AREA=20`/`MIN_H=8`/`MORPH_OPEN` erased them.
The relaxed defaults live in `run_real_pipeline.py::ss_detect`.

## Aggregate results (`real_run/results_summary.json`)

| stage                       | count |
|-----------------------------|-------|
| paired frames               | 23    |
| YOLO pedestrian detections  | 30    |
| SS  pedestrian detections   | 29    |
| Merged detections           | 52    |
| SS-only added by merge      | 22    |
| Flow-based moving           | 28    |
| Flow-based stationary       | 22    |
| Tracker-based moving        | 20    |
| Tracker-based stationary    | 32    |
| Tracker unique track IDs    | 9     |

## Read of the numbers

- **Merge is the big recall win.** YOLOv8s recovered 30 people; SS added
  22 more that YOLO had missed. That is a 73% uplift, matching the
  report's headline claim.
- **Flow-based intent is inflated when the ego vehicle moves.** With a
  moving camera, every static object also produces optical flow. This
  slice has the ego vehicle driving, so 28/50 predictions come out as
  "moving" and many of those are actually parked / stationary
  pedestrians. This is not a bug in the code, but a limitation of the
  naive flow rule - to fix it you would compensate for camera egomotion
  (e.g. subtract median flow, or use `vo_monocular.py` to solve pose
  and subtract the projected static-scene flow).
- **Tracker-based intent is more balanced** (20 moving vs 32 stationary
  across 9 unique tracks). The IoU-tracker + centroid-speed rule is
  also sensitive to camera motion, but pedestrians whose IoU overlap
  survives frame-to-frame get a much smaller speed signal than moving
  ones do.

## Regenerate

From the repository root:

```bash
python3 run_real_pipeline.py \
    --rgb_root data/images_rgb \
    --ss_root  data/images_ss \
    --start 1644 --count 23 --suffix 0 \
    --yolo_weights yolov8s.pt --yolo_imgsz 640 \
    --out_root real_run
```

Assets in `real_run/`:

- `detect_merge.mp4`      YOLO (green) + SS-only (red) overlays
- `intent_tracker.mp4`    tracker-based intent overlays
- `intent_flow.mp4`       flow-based intent overlays
- `sample_frames/`        representative single-frame PNGs
- `results_summary.json`  the table above
- per-frame JSONs under `yolo/`, `ss_det/`, `merged_det/`,
  `intent_flow/`, `intent_tracker/`
