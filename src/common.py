"""Shared helpers used across the pipeline stages."""

import cv2
import numpy as np


def enhance_image(img: np.ndarray) -> np.ndarray:
    """Boost pedestrian recall on a BGR frame: CLAHE + gamma + sharpen."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    cl = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    lab = cv2.merge((cl, a, b))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    inv_gamma = 1.0 / 1.3
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(256)]).astype("uint8")
    img = cv2.LUT(img, table)

    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(img, -1, kernel)


def compute_iou(box_a, box_b) -> float:
    """Intersection-over-Union of two [x1, y1, x2, y2] boxes."""
    x_a, y_a = max(box_a[0], box_b[0]), max(box_a[1], box_b[1])
    x_b, y_b = min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
    inter = max(0, x_b - x_a) * max(0, y_b - y_a)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0
