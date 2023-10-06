import json
import os
import numpy as np

def convert_to_yolo_format(json_file_path, yolo_file_path):
    # Read JSON file
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    # Image dimensions (change these to your actual dimensions)
    img_width, img_height = 1920, 1080

    # Open YOLO format file for writing
    with open(yolo_file_path, 'w') as f:
        for box, class_id in zip(data['boxes'], data['class']):
            center_x, center_y, phi, theta, h, w = box[:6]

            # Convert angular dimensions to pixel dimensions
            w_erp = w / 360 * img_width
            h_erp = h / 180 * img_height

            # Calculate top-left corner coordinates
            x1, y1 = center_x - w_erp // 2, center_y - h_erp // 2

            # Calculate bottom-right corner coordinates
            x2, y2 = center_x + w_erp // 2, center_y + h_erp // 2

            # Convert to YOLO format (normalized x_center, y_center, width, height)
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2
            w = x2 - x1
            h = y2 - y1

            x_center /= img_width
            y_center /= img_height
            w /= img_width
            h /= img_height

            # Write to YOLO format file
            f.write(f"{class_id} {x_center} {y_center} {w} {h}\n")

# Directory containing JSON files
json_dir = "/home/mstveras/360-obj-det/annotations"

# Directory to save YOLO format files
yolo_dir = "/home/mstveras/360-obj-det/yolo_annotations"
os.makedirs(yolo_dir, exist_ok=True)

# Loop through all JSON files in the directory
for json_file in os.listdir(json_dir):
    if json_file.endswith(".json"):
        json_file_path = os.path.join(json_dir, json_file)
        yolo_file_path = os.path.join(yolo_dir, json_file.replace(".json", ".txt"))
        
        # Convert each JSON file to YOLO format
        convert_to_yolo_format(json_file_path, yolo_file_path)
