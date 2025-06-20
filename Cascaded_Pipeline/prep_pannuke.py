import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import os
from skimage.draw import polygon
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage.measure import regionprops, label

# Class name to integer mapping for PanNuke
CLASS_MAP = {
    "Background": 0,
    "Neoplastic": 1,
    "Inflammatory": 2,
    "Connective": 3,
    "Dead": 4,
    "Epithelial": 5
    #"Background": 5
}

def remap_label(pred, by_size=False):
    pred_id = list(np.unique(pred))
    if 0 in pred_id:
        pred_id.remove(0)
    if len(pred_id) == 0:
        return pred
    if by_size:
        pred_size = [(pred == inst_id).sum() for inst_id in pred_id]
        pred_id = [x for x, _ in sorted(zip(pred_id, pred_size), key=lambda x: x[1], reverse=True)]
    new_pred = np.zeros(pred.shape, np.int32)
    for idx, inst_id in enumerate(pred_id):
        new_pred[pred == inst_id] = idx + 1
    return new_pred

def create_segmentation_mask_with_classes_from_pannuke(mask, image_size=(256, 256)):
    """
    Simulates the function used for PUMA, but works with PanNuke's mask format.

    Args:
        mask: A (H, W, 6) numpy array from PanNuke masks.npy
        image_size: Default (256, 256)

    Returns:
        bin_mask: binary mask (foreground vs background)
        inst_mask: unique instance labeled mask
        class_list: list of integer class ids in order of instance_id
        instance_sizes: dict of instance_id -> area in pixels
    """
    inst_mask = np.zeros(image_size, dtype=np.uint16)
    bin_mask = np.zeros(image_size, dtype=np.uint16)
    inv_bin_mask = np.ones(image_size, dtype=np.uint16)
    class_list = []
    instance_sizes = {}
    instance_idx = 1
    num_nuc = 0

    for channel in range(5):  # skip Background
        # copy value from new array if value is not equal 0
        nuclei_channel = mask[:, :, channel]
        layer_res = remap_label(nuclei_channel)
        # inst_map = np.where(mask[:,:,j] != 0, mask[:,:,j], inst_map)
        inst_mask = np.where(layer_res != 0, layer_res + num_nuc, inst_mask)
        bin_mask = np.where(layer_res != 0, inv_bin_mask, bin_mask)
        num_nuc = num_nuc + np.max(layer_res)
    inst_mask = remap_label(inst_mask)

    type_mask = np.zeros(image_size, dtype=np.uint16)
    for channel in range(5):
        layer_res = ((channel + 1) * np.clip(mask[:, :, channel], 0, 1)).astype(np.int32)
        type_mask = np.where(layer_res != 0, layer_res, type_mask)
    # Get unique instance indices (excluding background, typically indexed as 0)
    unique_instances = np.unique(inst_mask).astype(int)
    unique_instances = unique_instances[unique_instances > 0]  # Remove background index
    
    for instance_idx in unique_instances:
        # Get the corresponding class ID from type_mask
        class_mask = (inst_mask == instance_idx)
        class_id = np.max(type_mask[class_mask])  # Assuming class ID is consistent within the instance
    
        # Extract coordinates of pixels belonging to this instance
        coords = np.column_stack(np.where(class_mask))
    
        if len(coords) > 2:  # Polygon requires at least 3 points
            rr, cc = polygon(coords[:, 0], coords[:, 1])
            instance_sizes[instance_idx] = len(rr)  # Count number of pixels in the polygon
        else:
            instance_sizes[instance_idx] = len(coords)  # Fallback if fewer than 3 points
    
        class_list.append(class_id)
    return bin_mask, mask, class_list, instance_sizes

    """
        props = regionprops(labeled)

        for prop in props:
            coords = prop.coords
            inst_mask[coords[:, 0], coords[:, 1]] = instance_idx
            bin_mask[coords[:, 0], coords[:, 1]] = 1
            class_list.append(CHANNEL_CLASS_MAP[channel])
            instance_sizes[instance_idx] = len(coords)
            instance_idx += 1

    return bin_mask, inst_mask, class_list, instance_sizes
    """

# Function to get YOLO boxes
def get_yolo_boxes(image, yolo, conf=0.25, max_det=2000):
    """
    if '7128' in str(image_path) or '7129' in str(image_path) or '7130' in str(image_path):
        im = cv2.imread(image_path, cv2.IMREAD_COLOR)
        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        im = cv2.resize(im, (2048, 2048))
        im = Image.fromarray(im)
    else:
        #im = cv2.imread(image_path, cv2.IMREAD_COLOR)
    """
    # Debug 
    # print(image.shape)
    # Ensure image is uint8 for OpenCV
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
    im = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    im = Image.fromarray(im)
    true_w, true_h = im.size #Image.open(im).size
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
yolo = YOLO("/workspace/gururaj_data/train_folds_PumaPannuke_MAR20/C/weights/best.pt").to(device)

# Example usage loop for one image
image_input_dir = '/workspace/classification_pipeline/PanNuke/Fold 1/images/fold1/images.npy'
mask_input_dir = '/workspace/classification_pipeline/PanNuke/Fold 1/masks/fold1/masks.npy'

print("Loading large numpy files, this may take a while")
images = np.load(image_input_dir)
masks = np.load(mask_input_dir)

# Save directories
# img_save_dir = '/workspace/classification_pipeline/PanNukeDataset/images_png'
overlay_save_dir = '/workspace/classification_pipeline/PanNuke_vis/overlaid_png'
# os.makedirs(img_save_dir, exist_ok=True)
os.makedirs(overlay_save_dir, exist_ok=True)

# print("Process images")
for i in tqdm(range(len(images)), total=len(images)):
    img = images[i]
    mask = masks[i]
    # Debug
    # print(mask)
    stem = i
    # Output paths
    box_path = f'/workspace/classification_pipeline/PanNukeDataset/boxes/{stem}.npy'
    nuclei_path = f'/workspace/classification_pipeline/PanNukeDataset/nuclei/{stem}.npy'
    class_path = f'/workspace/classification_pipeline/PanNukeDataset/class/{stem}.npy'
    os.makedirs(os.path.dirname(box_path), exist_ok=True)
    os.makedirs(os.path.dirname(nuclei_path), exist_ok=True)
    os.makedirs(os.path.dirname(class_path), exist_ok=True)

    # Get YOLO predictions
    boxes = get_yolo_boxes(img, yolo)

    # Create instance segmentation mask and class labels
    bin_mask, mask, instance_classes, instance_sizes = create_segmentation_mask_with_classes_from_pannuke(mask)

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
                largest_instance = int(max(unique_instances, key=lambda i: instance_sizes.get(i, 0)))
                class_id = instance_classes[largest_instance - 1] if (largest_instance - 1) < len(instance_classes) else -1
                classes_in_boxes.append([class_id])
            else:
                classes_in_boxes.append([0])

            #nuclei_list.append(bin_crop)
            nuclei_list.append(full_mask)
            #classes_in_boxes.append(class_labels)

    # Debug
    print("Image dtype:", img.dtype, "min:", img.min(), "max:", img.max(), "shape:", img.shape)


    # Convert image to RGB and save it
    img_uint8 = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
    bin_mask_uint8 = (bin_mask * 255).astype(np.uint8) if bin_mask.max() <= 1.0 else bin_mask.astype(np.uint8)
    # img_path = os.path.join(img_save_dir, f"{stem}.png")
    # cv2.imwrite(img_path, cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR))
    
    # Create figure with two subplots: Original | Mask + Boxes
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # --- Panel 1: Original image ---
    axes[0].imshow(img_uint8)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # --- Panel 2: Binary mask ---
    axes[1].imshow(bin_mask_uint8, cmap='gray')
    axes[1].set_title("Original Mask")
    axes[1].axis('off')
    
    # --- Right: Mask + YOLO Boxes ---
    axes[2].imshow(bin_mask_uint8, cmap='gray')
    
    # Overlay binary mask in red
    # masked = np.zeros_like(bin_mask_uint8)
    # masked[..., 0] = bin_mask * 255  # Red channel
    # axes[1].imshow(masked, alpha=0.4)
    
    """
    # Draw YOLO bounding boxes
    for box in boxes:
        _, x1, y1, x2, y2, _ = box.astype(int)
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                 linewidth=1.5, edgecolor='lime', facecolor='none')
        axes[2].add_patch(rect)

    axes[2].set_title("Mask + YOLO Boxes")
    axes[2].axis('off')
    """
    labeled = label(bin_mask)
    h, w = img.shape[0], img.shape[1]
    for region in regionprops(labeled):
        minr, minc, maxr, maxc = region.bbox
        # x1, y1, x2, y2 = minc, minr, maxc, maxr
        # Convert to YOLO (center-x, center-y, width, height) normalized
        # Bounding box center, width, height (normalized)
        box_w = maxc - minc
        box_h = maxr - minr
        x1 = minc
        y1 = minr
        rect = patches.Rectangle((x1, y1), box_w, box_h,
                                 linewidth=1.5, edgecolor='lime', facecolor='none')
        axes[2].add_patch(rect)
    
    axes[2].set_title("Mask + Skimage Boxes")
    axes[2].axis('off')
    
    # Save the combined figure
    overlay_path = os.path.join(overlay_save_dir, f"{stem}.png")
    plt.tight_layout()
    plt.savefig(overlay_path, bbox_inches='tight', pad_inches=0)
    plt.close()

    # Save data
    # Debug 
    # print(f"Image {i}: YOLO boxes = {len(boxes)}, Nuclei = {len(nuclei_list)}")
    # Skip if no nuclei detected
    if len(nuclei_list) == 0:
        print(f"Image {i}: YOLO boxes = {len(boxes)}, Nuclei = {len(nuclei_list)}")
        print(f"Skipping Image {i}")
        continue
    np.save(box_path, boxes)
    #np.save(nuclei_path, np.array(nuclei_list, dtype=object))
    # Debug 
    # print(nuclei_list)
    np.save(nuclei_path, np.stack(nuclei_list))
    np.save(class_path, np.array(classes_in_boxes, dtype=object))