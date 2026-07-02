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
