# Pedestrian Intent Prediction on CARLA Town01

Vision-based pedestrian detection + short-term intent (moving /
stationary) on synthetic CARLA driving data. Combines a **YOLOv8** RGB
detector with a **semantic-segmentation** fallback to recover
small/occluded pedestrians, then classifies each track as moving or
stationary using either dense optical flow or IoU-based centroid speed.

Course project for CSE 573 — Computer Vision and Image Processing,
University at Buffalo, Spring 2025. Full write-up in
[`docs/CVIP_final_report.docx`](docs/CVIP_final_report.docx).

## Pipeline

```
   images_rgb/*.png      images_ss/*.png
        |                     |
        v                     v
   detection_yolo.py     detection_ss.py
   (YOLOv8 + CLAHE)      (color mask + morphology)
        |                     |
        +--------> merge_detections.py <--------+
                          |
                          v
              +-----------+-----------+
              |                       |
              v                       v
       intent_flow.py          intent_tracker.py
       (Farneback flow)        (IoU + centroid speed)
              |                       |
              +---------> visualize_intent.py
```

Each stage writes per-frame JSON so you can plug in your own detector or
intent classifier at any point.

## Repo layout

```
.
|-- README.md                  # you are here
|-- LICENSE                    # MIT
|-- detection_yolo.py          # Stage 1: YOLOv8 on RGB (with CLAHE + sharpen)
|-- detection_ss.py            # Stage 2: pedestrian mask on semantic seg
|-- merge_detections.py        # Stage 3: IoU-based union of the two streams
|-- intent_flow.py             # Stage 4a: Farneback flow inside each bbox
|-- intent_tracker.py          # Stage 4b: IoU tracker + pixel-speed rule
|-- visualize_intent.py        # Overlay intent (green/red) on RGB
|-- vo_monocular.py            # Bonus: monocular visual odometry
|-- run_real_pipeline.py       # One-shot driver for a CARLA slice
|-- generate_demo.py           # Synthetic frames for smoke-testing
|-- evaluate_demo.py           # Score intent methods vs known GT
|-- make_demo_video.py         # Stitch frames -> mp4/gif
|-- demo/                      # Synthetic demo outputs (committed)
|-- real_run/                  # Real CARLA-slice outputs (committed)
`-- docs/
    |-- DEMO.md                # How to run the synthetic demo
    |-- REAL_DATA_RESULTS.md   # Real-run numbers and caveats
    `-- CVIP_final_report.docx # Full course report
```

## Install

```bash
# core (needed by every stage)
pip install opencv-python numpy tqdm

# YOLO stage
pip install ultralytics torch --extra-index-url https://download.pytorch.org/whl/cpu

# optional: pandas for vo_monocular.py, imageio for animated GIFs
pip install pandas imageio pillow
```

Weights are not committed. Download once from the Ultralytics releases,
then drop them next to the scripts (default `yolov8s.pt`).

## Two ways to run

### 1) Synthetic demo (no data required)

Renders a 30-frame scene with four pedestrians (two moving, one
stationary, one that starts stationary then walks) and runs the full
pipeline against it. Useful for smoke-testing the code and reproducing
the numbers below.

```bash
python3 generate_demo.py --out_dir demo --num_frames 30
python3 intent_flow.py    --img_dir demo/merged_images \
                          --json_dir demo/merged_detections
python3 intent_tracker.py --merged_dir demo/merged_detections \
                          --out_dir demo/intent_predictions
python3 evaluate_demo.py  --num_frames 30
```

Latest run (see `demo/results_summary.json`):

| method          | preds | correct | accuracy |
|-----------------|-------|---------|----------|
| tracker-based   |  120  |  117    |  0.9750  |
| flow-based      |  116  |  116    |  1.0000  |

Details and expected scene labels in [`docs/DEMO.md`](docs/DEMO.md).

### 2) Real CARLA data

Drop paired RGB + semantic-segmentation frames into
`data/images_rgb/` and `data/images_ss/` using the filename convention

- `XXXX_0.png` (RGB) is paired with `XXXX_10.png` (SS)
- `XXXX_1.png` (RGB) is paired with `XXXX_11.png` (SS)

Then run the driver:

```bash
python3 run_real_pipeline.py \
    --rgb_root data/images_rgb \
    --ss_root  data/images_ss \
    --start 1644 --count 23 --suffix 0 \
    --yolo_weights yolov8s.pt --yolo_imgsz 640 \
    --out_root real_run
```

Results from the committed run on 23 frames of Town01
(see `real_run/results_summary.json`):

| stage                       | count |
|-----------------------------|-------|
| YOLO detections             | 30    |
| SS  detections              | 29    |
| Merged detections           | 52    |
| SS-only additions           | 22    |
| Flow-based moving/stationary  | 28 / 22 |
| Tracker-based moving/stationary | 20 / 32 |
| Tracker unique track IDs    | 9     |

Full commentary (including the ego-motion caveat) in
[`docs/REAL_DATA_RESULTS.md`](docs/REAL_DATA_RESULTS.md).

## Getting CARLA data

- **KITTI-CARLA** pre-recorded dataset (fastest way to get started with
  paired RGB + SS): https://github.com/jedeschaud/kitti_carla_simulator
- **CARLA-KITTI generator** if you want your own scenarios:
  https://github.com/fnozarian/CARLA-KITTI
- **CARLA** simulator, pre-built Linux release:
  https://github.com/carla-simulator/carla/releases

## Citing / prior work

The three ideas that shape this repo:

- **CARLA simulator**: Dosovitskiy et al., *CARLA: An Open Urban Driving
  Simulator*, CoRL 2017.
- **YOLOv8**: Ultralytics, https://github.com/ultralytics/ultralytics
- **Farneback dense flow**: Farneback, *Two-Frame Motion Estimation Based
  on Polynomial Expansion*, SCIA 2003.

Full reference list in the report.

## License

[MIT](LICENSE).
