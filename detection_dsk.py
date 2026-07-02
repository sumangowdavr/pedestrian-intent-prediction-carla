import cv2
import numpy as np
import torch
from ultralytics import YOLO
from torchvision.ops import nms
import os
from pathlib import Path
import json

class PedestrianDetector:
    def __init__(self):
        # Initialize models with proper CUDA handling
        self.model = YOLO('yolov8s.pt').cuda()  # Using small model for speed
        self.model.conf = 0.2  # Lower confidence threshold
        self.model.iou = 0.4   # Balanced IoU threshold

    def detect_pedestrians(self, img):
        """Detect pedestrians with proper bounding box handling"""
        results = self.model(img, verbose=False)[0]
        detections = []
        
        for box in results.boxes:
            if box.cls == 0:  # person class
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': float(box.conf)
                })
        return detections

def main():
    # Configuration
    INPUT_DIR = "path/to/your/carla/images"  # Update this path
    OUTPUT_DIR = "pedestrian_detections"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Initialize detector
    detector = PedestrianDetector()
    img_paths = list(Path(INPUT_DIR).rglob("*.png"))
    all_results = {}
    
    print(f"Processing {len(img_paths)} images...")
    
    for img_path in img_paths:
        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        # Detect pedestrians
        detections = detector.detect_pedestrians(img)
        frame_id = img_path.stem
        all_results[frame_id] = detections
        
        # Visualization
        vis_img = img.copy()
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(vis_img, f"{det['confidence']:.2f}", 
                       (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        
        # Save visualization
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{frame_id}_vis.jpg"), vis_img)
    
    # Save ALL results to JSON
    with open(os.path.join(OUTPUT_DIR, 'detections.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"Saved detections for {len(all_results)} frames")
    print(f"Results saved to: {os.path.abspath(OUTPUT_DIR)}")

if __name__ == "__main__":
    main()