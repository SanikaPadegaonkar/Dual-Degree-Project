import numpy as np
import json
import cv2
from PIL import Image
from pathlib import Path
from skimage.draw import polygon
from ultralytics import YOLO
import os

# Class name to integer mapping
CLASS_MAP = {
    'background': 0,
    'nuclei_tumor': 1,
    'nuclei_lymphocyte': 2,
    'nuclei_plasma_cell': 3,
    'nuclei_histiocyte': 4,
    'nuclei_melanophage': 5,
    'nuclei_neutrophil': 6,
    'nuclei_stroma': 7,
    'nuclei_endothelium': 8,
    'nuclei_epithelium': 9,
    'nuclei_apoptosis': 10
}

# Function to create segmentation mask and class labels
def create_segmentation_mask_with_classes(geojson_path, image_size=(1024, 1024)):
    mask = np.zeros(image_size, dtype=np.uint16)
    bin_mask = np.zeros(image_size, dtype=np.uint16)
    class_list = []
    instance_sizes = {}  # instance ID -> area in pixels
    
    with open(geojson_path, 'r') as f:
        data = json.load(f)
    instance_idx = 1
    for feature in data['features']:
        if feature['geometry']['type'] == 'Polygon':
            polygon_coords = np.array(feature['geometry']['coordinates'][0])
            row_coords, col_coords = polygon(polygon_coords[:, 1], polygon_coords[:, 0], shape=image_size)
            mask[row_coords, col_coords] = instance_idx
            bin_mask[row_coords, col_coords] = 1

            nucleus_type = feature.get("properties", {}).get("classification", {}).get("name", "Unknown")
            class_id = CLASS_MAP.get(nucleus_type, -1)  # -1 if unknown
            class_list.append(class_id)

            # Calculate and store size (area)
            instance_sizes[instance_idx] = len(row_coords)

            instance_idx += 1
    return bin_mask, mask, class_list, instance_sizes

# Function to get YOLO boxes
def get_yolo_boxes(image_path, yolo, conf=0.35, max_det=2000):
    if '7128' in str(image_path) or '7129' in str(image_path) or '7130' in str(image_path):
        im = cv2.imread(image_path, cv2.IMREAD_COLOR)
        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        im = cv2.resize(im, (2048, 2048))
        im = Image.fromarray(im)
    else:
        im = cv2.imread(image_path, cv2.IMREAD_COLOR)
        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        im = Image.fromarray(im)

    true_w, true_h = Image.open(image_path).size
    w, h = im.size
    r = yolo.predict(im, conf=conf, max_det=max_det, imgsz=w)[0]
    b = r.boxes.xyxyn.cpu().numpy()
    b = b * true_w
    b = b.astype(int)

    if b.shape[0] > 0:
        b = np.hstack([np.ones((b.shape[0], 1)), b, np.ones((b.shape[0], 1))])
    return b

# YOLO model path and device
device = "cuda"  # or "cpu"
#yolo = YOLO("/workspace/gururaj_data/train_folds_PumaPannuke_MAR20/C/weights/best.pt").to(device)
yolo = YOLO("/workspace/gururaj_files/share_sanika/yolo_ckpts/A.pt").to(device)

# Example usage loop for one image
image_input_dir = '/workspace/HoVerNet_puma/PUMA/Images/01_training_dataset_tif_ROIs'
geojson_input_dir = '/workspace/HoVerNet_puma/PUMA/Annotations'#/01_training_dataset_geojson_nuclei'

for ip in os.listdir(image_input_dir):
    if not ip.endswith('.tif'):
        continue

    image_path = os.path.join(image_input_dir, ip)
    geojson_path = os.path.join(geojson_input_dir, Path(ip).stem + '_nuclei.geojson')
    stem = Path(ip).stem

    # Output paths
    box_path = f'/workspace/classification_pipeline/PumaDataset_yoloA/boxes/{stem}.npy'
    nuclei_path = f'/workspace/classification_pipeline/PumaDataset_yoloA/nuclei/{stem}.npy'
    class_path = f'/workspace/classification_pipeline/PumaDataset_yoloA/class/{stem}.npy'
    os.makedirs(os.path.dirname(box_path), exist_ok=True)
    os.makedirs(os.path.dirname(nuclei_path), exist_ok=True)
    os.makedirs(os.path.dirname(class_path), exist_ok=True)


    # Get YOLO predictions
    boxes = get_yolo_boxes(image_path, yolo)

    # Create instance segmentation mask and class labels
    bin_mask, mask, instance_classes, instance_sizes = create_segmentation_mask_with_classes(geojson_path)

    nuclei_list = []
    classes_in_boxes = []

    for box in boxes:
        _, x1, y1, x2, y2, _ = box.astype(int)
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, 1024), min(y2, 1024)

        if x2 > x1 and y2 > y1:
            crop = mask[y1:y2, x1:x2]
            bin_crop = bin_mask[y1:y2, x1:x2]
            # Create a new blank mask
            full_mask = np.zeros_like(bin_mask)
            # Paste the cropped region back into the corresponding location
            full_mask[y1:y2, x1:x2] = bin_crop
            unique_instances = np.unique(crop)
            unique_instances = [i for i in unique_instances if i != 0]
            
            #class_labels = [instance_classes[i - 1] for i in unique_instances if i != 0 and (i - 1) < len(instance_classes)]

            # If multiple instances, pick the one with the largest area
            if unique_instances:
                largest_instance = max(unique_instances, key=lambda i: instance_sizes.get(i, 0))
                class_id = instance_classes[largest_instance - 1] if (largest_instance - 1) < len(instance_classes) else -1
                classes_in_boxes.append([class_id])
            else:
                classes_in_boxes.append([0])

            #nuclei_list.append(bin_crop)
            nuclei_list.append(full_mask)
            #classes_in_boxes.append(class_labels)

    # Save data
    np.save(box_path, boxes)
    #np.save(nuclei_path, np.array(nuclei_list, dtype=object))
    np.save(nuclei_path, np.stack(nuclei_list))
    np.save(class_path, np.array(classes_in_boxes, dtype=object))
