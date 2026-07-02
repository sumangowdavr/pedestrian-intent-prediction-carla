# pedestrian_detection_finetuned_full.py

# ------------------------- IMPORTS -------------------------
import os
import json
import cv2
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict
from torchvision import models
import torchvision.transforms as T
from ultralytics import YOLO
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import time
from collections import deque
from PIL import Image

# ------------------------- SETTINGS -------------------------
DATA_ROOT = r"C:\Users\Sumangowda\Desktop\CVIP_project\pedestrian_intent_prediction\data\Town01\Town01\generated"
IMG_DIR = os.path.join(DATA_ROOT, "images_rgb")
ANNOTATION_DIR = os.path.join(DATA_ROOT, "labels_coco")
OUT_DIR = "enhanced_results"
IMAGE_SAVE_DIR = os.path.join(OUT_DIR, "images")
JSON_SAVE_DIR = os.path.join(OUT_DIR, "jsons")
MODEL_SAVE_PATH = os.path.join(OUT_DIR, "finetuned_yolov8.pt")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(IMAGE_SAVE_DIR, exist_ok=True)
os.makedirs(JSON_SAVE_DIR, exist_ok=True)

CONF_THRESHOLD = 0.4
BATCH_SIZE = 16
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
STEP_SIZE = 3
GAMMA = 0.1

COCO_CLASSES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'none', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
    'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'none', 'backpack', 'umbrella', 'none',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
    'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'none', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog',
    'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'none', 'dining table', 'toilet', 'none',
    'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]
PERSON_CLASS_ID = COCO_CLASSES.index('person')

# ------------------------- DATASET CLASS -------------------------
class CarlaDataset(Dataset):
    def __init__(self, img_dir, annotation_dir, transforms=None):
        self.img_dir = Path(img_dir)
        self.annotation_dir = Path(annotation_dir)
        self.img_paths = list(self.img_dir.rglob("*.png"))
        self.transforms = transforms

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        annotation_path = self.annotation_dir / f"{img_path.stem}.json"
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        boxes, labels = [], []

        try:
            with open(annotation_path, 'r') as f:
                annotations = json.load(f)
                for annotation in annotations:
                    bbox = annotation['bbox']
                    x_min, y_min, width, height = bbox
                    x_max = x_min + width
                    y_max = y_min + height
                    boxes.append([x_min / w, y_min / h, x_max / w, y_max / h])
                    labels.append(annotation['category_id'])
        except FileNotFoundError:
            print(f"Annotation file not found: {annotation_path}")

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)
        target = {"boxes": boxes, "labels": labels, "image_id": image_id, "area": area, "iscrowd": iscrowd}

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target
# ------------------------- DATA AUGMENTATION -------------------------
def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

# ------------------------- IMAGE ENHANCEMENT -------------------------
def enhance_image(image):
    """Enhance input image for better detection."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    lab = cv2.merge((cl, a, b))
    image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    gamma = 1.2
    table = np.array([(i / 255.0) ** (1.0 / gamma) * 255 for i in range(256)]).astype("uint8")
    image = cv2.LUT(image, table)

    sharpen = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    image = cv2.filter2D(image, -1, sharpen)
    return image

# ------------------------- MODEL LOADING -------------------------
def load_finetuned_yolo_model():
    """Load fine-tuned YOLOv8 model if exists, else fallback to default YOLOv8n."""
    if os.path.exists(MODEL_SAVE_PATH):
        print(f"[INFO] Loading fine-tuned YOLOv8 model from {MODEL_SAVE_PATH}")
        model = YOLO(MODEL_SAVE_PATH)
    else:
        print(f"[INFO] Fine-tuned model not found, loading default YOLOv8n.")
        model = YOLO("yolov8n.pt")
    model.conf = CONF_THRESHOLD
    return model

# ------------------------- DRAW BOUNDING BOXES -------------------------
def draw_boxes(image, detections):
    for det in detections:
        label = det['label']
        conf = det['confidence']
        bbox = det['bbox']
        x1, y1, x2, y2 = bbox
        color = (0, 255, 0) if label == "person" else (255, 0, 0)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, f"{label} {conf:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return image

# ------------------------- MAIN DETECTION PIPELINE -------------------------
def detect_and_save():
    model = load_finetuned_yolo_model()
    image_paths = list(Path(IMG_DIR).rglob("*.png"))
    print(f"[INFO] Found {len(image_paths)} images.")

    all_detections = {}
    pedestrian_detections = {}

    total_pedestrians = 0

    for idx, img_path in enumerate(image_paths):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[WARNING] Could not read image {img_path}")
            continue

        img_enhanced = enhance_image(img)
        results = model.predict(img_enhanced, save=False, verbose=False)[0]
        frame_name = Path(img_path).stem

        frame_detections = []
        frame_pedestrians = []

        for box in results.boxes:
            cls_id = int(box.cls)
            label = model.names[cls_id]
            conf = float(box.conf)
            bbox = box.xyxy.cpu().numpy().astype(int).tolist()[0]

            detection_entry = {
                "label": label,
                "confidence": round(conf, 3),
                "bbox": bbox
            }
            frame_detections.append(detection_entry)

            if label == "person":
                frame_pedestrians.append({
                    "confidence": round(conf, 3),
                    "bbox": bbox
                })

        all_detections[frame_name] = frame_detections
        if frame_pedestrians:
            pedestrian_detections[frame_name] = frame_pedestrians
            total_pedestrians += len(frame_pedestrians)

        # Draw and save image
        img_with_boxes = draw_boxes(img.copy(), frame_detections)
        save_path = os.path.join(IMAGE_SAVE_DIR, f"{frame_name}_detected.png")
        cv2.imwrite(save_path, img_with_boxes)

        print(f"[{idx + 1}/{len(image_paths)}] Processed: {frame_name} | Pedestrians: {len(frame_pedestrians)}")

    # Save JSON outputs
    with open(os.path.join(JSON_SAVE_DIR, "all_detections.json"), "w") as f:
        json.dump(all_detections, f, indent=2)

    with open(os.path.join(JSON_SAVE_DIR, "pedestrian_only.json"), "w") as f:
        json.dump(pedestrian_detections, f, indent=2)

    # Final summary
    print("\n[INFO] Detection complete.")
    print(f"[INFO] Total frames processed: {len(image_paths)}")
    print(f"[INFO] Total pedestrians detected: {total_pedestrians}")
    print(f"[INFO] Results saved under '{OUT_DIR}' folder.")
# ------------------------- TRAINING (OPTIONAL) -------------------------
# (These are kept but won't be executed unless you call them manually.)

def train_faster_rcnn():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"[INFO] Using device: {device}")

    dataset = CarlaDataset(IMG_DIR, ANNOTATION_DIR, transforms=get_transform(train=True))
    dataset_test = CarlaDataset(IMG_DIR, ANNOTATION_DIR, transforms=get_transform(train=False))

    indices = torch.randperm(len(dataset)).tolist()
    train_size = int(0.8 * len(dataset))
    dataset_train = torch.utils.data.Subset(dataset, indices[:train_size])
    dataset_val = torch.utils.data.Subset(dataset_test, indices[train_size:])

    data_loader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, collate_fn=lambda batch: tuple(zip(*batch)))
    data_loader_val = DataLoader(dataset_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, collate_fn=lambda batch: tuple(zip(*batch)))

    num_classes = len(COCO_CLASSES)
    model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    model.to(device)

    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LEARNING_RATE)
    lr_scheduler = StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

    for epoch in range(NUM_EPOCHS):
        model.train()
        for images, targets in DataLoader(dataset_train, batch_size=4, shuffle=True, collate_fn=lambda batch: tuple(zip(*batch))):
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        lr_scheduler.step()

    torch.save(model.state_dict(), os.path.join(OUT_DIR, "faster_rcnn_finetuned.pth"))
    print("[INFO] Faster R-CNN fine-tuning complete!")

def train_yolov8():
    model = YOLO("yolov8n.pt") # Pre-trained base model
    model.train(data=os.path.join(DATA_ROOT, 'dataset.yaml'),
                epochs=NUM_EPOCHS,
                batch=BATCH_SIZE,
                lr0=LEARNING_RATE,
                lrf=0.01,
                save=True,
                name='yolov8_finetuned')
    print("[INFO] YOLOv8 fine-tuning complete.")

# ------------------------- METRIC LOGGER -------------------------
class SmoothedValue:
    def __init__(self, window_size=20, fmt=None):
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt or "{median:.4f} ({global_avg:.4f})"

    def update(self, value):
        self.deque.append(value)
        self.total += value
        self.count += 1

    def median(self):
        return torch.tensor(list(self.deque)).median().item()

    def avg(self):
        return torch.tensor(list(self.deque)).mean().item()

    def global_avg(self):
        return self.total / self.count

    def __str__(self):
        return self.fmt.format(median=self.median(), avg=self.avg(), global_avg=self.global_avg())

class MetricLogger:
    def __init__(self, delimiter="  "):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            self.meters[k].update(v)

    def __str__(self):
        return self.delimiter.join(f"{k}: {v}" for k, v in self.meters.items())

# ------------------------- MAIN ENTRY -------------------------
if __name__ == "__main__":
    detect_and_save()  # Runs detection + saves outputs
    # To train models later separately:
    # train_faster_rcnn()
    # train_yolov8()
