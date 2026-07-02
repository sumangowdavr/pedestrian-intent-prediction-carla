import cv2
import numpy as np
import os
from pathlib import Path

# CARLA semantic segmentation class IDs
PEDESTRIAN_CLASS = 4
CAR_CLASS = 10
BACKGROUND_CLASS = 0

def generate_annotations(ss_dir, rgb_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    for ss_path in Path(ss_dir).glob('*.png'):
        # Load segmentation image
        ss_img = cv2.imread(str(ss_path), cv2.IMREAD_UNCHANGED)
        if ss_img is None:
            continue
            
        # Get corresponding RGB image
        rgb_path = Path(rgb_dir) / ss_path.name
        if not rgb_path.exists():
            continue
            
        # Initialize annotation file
        annotation_lines = []
        
        # Process pedestrians
        pedestrian_mask = (ss_img == PEDESTRIAN_CLASS)
        if np.any(pedestrian_mask):
            contours, _ = cv2.findContours(pedestrian_mask.astype(np.uint8), 
                                         cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                annotation_lines.append(f"{x} {y} {x+w} {y+h} person")
        
        # Process cars
        car_mask = (ss_img == CAR_CLASS)
        if np.any(car_mask):
            contours, _ = cv2.findContours(car_mask.astype(np.uint8), 
                                         cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                annotation_lines.append(f"{x} {y} {x+w} {y+h} car")
        
        # Save annotation file
        if annotation_lines:
            output_path = os.path.join(output_dir, ss_path.stem + '.txt')
            with open(output_path, 'w') as f:
                f.write('\n'.join(annotation_lines))
            
            # Also save the RGB image in our training directory
            cv2.imwrite(os.path.join(output_dir, ss_path.name), cv2.imread(str(rgb_path)))

# Usage
ss_dir = r"C:\Users\Sumangowda\Desktop\CVIP_project\pedestrian_intent_prediction\data\Town01\Town01\generated\images_ss"
rgb_dir = r"C:\Users\Sumangowda\Desktop\CVIP_project\pedestrian_intent_prediction\data\Town01\Town01\generated\images_rgb"
output_dir = "training_data"
generate_annotations(ss_dir, rgb_dir, output_dir)