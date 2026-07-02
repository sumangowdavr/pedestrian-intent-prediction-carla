# Pedestrian Intent Prediction on CARLA Town01

Vision-based pedestrian detection + short-term intent (moving /
stationary) on synthetic CARLA driving data. Combines a **YOLOv8** RGB
detector with a **semantic-segmentation** fallback to recover
small/occluded pedestrians, then classifies each track as moving or
stationary using either dense optical flow or IoU-based centroid speed.

Course project for CSE 573 - Computer Vision and Image Processing,
University at Buffalo, Spring 2025. Full write-up in
[`docs/CVIP_final_report.docx`](docs/CVIP_final_report.docx).

## Pipeline

```
   images_rgb/*.png                 images_ss/*.png
        |                                 |
        v                                 v
   src/detection_yolo.py             src/detection_ss.py
   (YOLOv8 + CLAHE + gamma)          (color mask + morphology)
        |                                 |
        +---------> src/merge_detections.py <---------+
                            |
                            v
              +-------------+-------------+
              |                           |
              v                           v
      src/intent_flow.py           src/intent_tracker.py
      (Farneback dense flow)       (IoU tracker + centroid speed)
              |                           |
              +--> src/visualize_intent.py --> annotated frames
```

Each stage writes per-frame JSON, so you can plug in your own detector
or intent classifier at any point.

## Repository layout

```
.
|-- README.md
|-- LICENSE                          # MIT
|-- requirements.txt
|-- src/                             # all pipeline code lives here
|   |-- detection_yolo.py            # Stage 1: YOLOv8 on RGB (CLAHE + sharpen)
|   |-- detection_ss.py              # Stage 2: pedestrian mask on semantic seg
|   |-- merge_detections.py          # Stage 3: IoU-based union of both streams
|   |-- intent_flow.py               # Stage 4a: Farneback flow inside each bbox
|   |-- intent_tracker.py            # Stage 4b: IoU tracker + pixel speed
|   |-- visualize_intent.py          # Overlay intent (green/red) on RGB
|   |-- vo_monocular.py              # Bonus: monocular visual odometry
|   |-- run_real_pipeline.py         # One-shot driver for a CARLA slice
|   |-- generate_demo.py             # Synthetic frames for smoke-testing
|   |-- evaluate_demo.py             # Score intent methods vs known GT
|   `-- make_demo_video.py           # Stitch frames -> mp4/gif
|-- docs/
|   |-- DEMO.md                      # Synthetic-demo run instructions
|   |-- REAL_DATA_RESULTS.md         # Real CARLA-run numbers and caveats
|   `-- CVIP_final_report.docx       # Full course report
`-- outputs/
    |-- demo/                        # Synthetic demo outputs (committed)
    `-- real_run/                    # Real CARLA-slice outputs (committed)
```

Not committed (either regenerable or too large): `data/`, `runs/`,
YOLO weights (`*.pt`), giant PNG intermediates under `outputs/*/`.

## Install

```bash
pip install -r requirements.txt
```

Weights are not committed. Download once from the Ultralytics releases,
then drop them anywhere on your `PATH` or pass `--yolo_weights <path>`
(default `yolov8s.pt`).

## Two ways to run

Both are driven from the repo root - the scripts figure out their own
default paths.

### 1) Synthetic demo (no data needed)

Renders a 30-frame scene with four pedestrians (two moving, one
stationary, one that starts stationary then walks) and runs the full
pipeline against it. Useful for smoke-testing the code and reproducing
the numbers below.

```bash
python src/generate_demo.py
python src/intent_flow.py \
    --img_dir outputs/demo/merged_images \
    --json_dir outputs/demo/merged_detections \
    --out_json_dir outputs/demo/intent_with_flow \
    --viz_dir outputs/demo/merged_visual_flow_final
python src/intent_tracker.py \
    --merged_dir outputs/demo/merged_detections \
    --out_dir outputs/demo/intent_predictions \
    --speed_thresh 2.0
python src/evaluate_demo.py
```

Latest run (`outputs/demo/results_summary.json`):

| method          | preds | correct | accuracy |
|-----------------|-------|---------|----------|
| tracker-based   |  120  |  117    |  0.9750  |
| flow-based      |  116  |  116    |  1.0000  |

Details and expected scene labels: [`docs/DEMO.md`](docs/DEMO.md).

### 2) Real CARLA data

Drop paired RGB + semantic-segmentation frames into
`data/images_rgb/` and `data/images_ss/` using the CARLA-KITTI naming
convention:

- `XXXX_0.png` (RGB) is paired with `XXXX_10.png` (SS)
- `XXXX_1.png` (RGB) is paired with `XXXX_11.png` (SS)

Then run the driver:

```bash
python src/run_real_pipeline.py --start 1644 --count 23 --suffix 0
```

Results from the committed run on 23 Town01 frames
(`outputs/real_run/results_summary.json`):

| stage                              | count |
|------------------------------------|-------|
| YOLO detections                    | 30    |
| SS detections                      | 29    |
| Merged detections                  | 52    |
| SS-only additions                  | 22    |
| Flow-based moving / stationary     | 28 / 22 |
| Tracker-based moving / stationary  | 20 / 32 |
| Tracker unique track IDs           | 9     |

Full commentary (including the ego-motion caveat): [`docs/REAL_DATA_RESULTS.md`](docs/REAL_DATA_RESULTS.md).

## Getting CARLA data

- **KITTI-CARLA** pre-recorded dataset (fastest way to get started with
  paired RGB + SS): https://github.com/jedeschaud/kitti_carla_simulator
- **CARLA-KITTI generator** if you want your own scenarios:
  https://github.com/fnozarian/CARLA-KITTI
- **CARLA simulator**, pre-built Linux release:
  https://github.com/carla-simulator/carla/releases

## Prior work

- **CARLA simulator**: Dosovitskiy et al., *CARLA: An Open Urban Driving
  Simulator*, CoRL 2017.
- **YOLOv8**: Ultralytics, https://github.com/ultralytics/ultralytics
- **Farneback dense flow**: Farneback, *Two-Frame Motion Estimation
  Based on Polynomial Expansion*, SCIA 2003.

Full reference list in the report.

## License

[MIT](LICENSE).
