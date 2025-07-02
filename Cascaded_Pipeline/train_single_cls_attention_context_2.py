# train_single_cls.py

import os
import sys
import random
import glob
import argparse
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import segmentation_models_pytorch as smp
from tqdm import tqdm
import difflib
import logging
from datetime import datetime
import matplotlib.pyplot as plt
from metrics import process_one_mask_from_arrays, cell_type_detection_scores, compute_metrics_with_hierarchy, compute_metrics_with_hierarchy2, compute_metrics_with_hierarchy3
import csv
from typing import List
from scipy.ndimage import label as connected_components

# Class name to integer mapping
COARSE_CLASS_MAP = {
    'epithelial': 0, # Epithelial / Glandular
    'immune': 1, # Immune / Inflammatory
    'stromal': 2, # Stromal / Connective
    'other': 3,
}
# Class name to integer mapping
FINE_CLASS_MAP = {
    'normal_epithelial': 0, 
    'tumoral': 1, # Tumoral / Neoplastic
    'lymphocyte': 2, 
    'histiocyte': 3,
    'macrophage': 4,
    'neutrophil': 5,
    'plasma': 6,
    'melanophage': 7,
    'other_immune': 8,
    'muscle': 9,
    'fibroblast': 10,
    'stroma': 11,
    'endothelial': 12,
    'other_connective': 13,
    'necrotic': 14,
    'apoptosis': 15,
}

# Class name to integer mapping
PUMA_CLASS_MAP = {
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

COARSE_CLASS_MEMBERS = {
    'epithelial': ['epithelium', 'tumoral'],
    'immune': ['lymphocyte', 'plasma', 'histiocyte', 'melanophage', 'neutrophil'],
    'stromal': ['stroma', 'fibroblast', 'endothelial'],
    'other': ['apoptosis', 'necrotic'],
}

# Reverse the map for integer -> class string
PUMA_IDX_TO_NAME = {v: k for k, v in PUMA_CLASS_MAP.items()}

hierarchy_map = {
    0: [0, 1],             # Epithelial / Glandular: Normal epithi, Tumoral/Neoplastic
    1: [2, 3, 4, 5, 6, 7, 8],       # Immune / Inflammatory: Lymphocyte, Histiocyte, Macrophage, Neutrophil, Plasma, Melanophage, Other_immune
    2: [9, 10, 11, 12, 13],          # Stromal / Connective: Muscle, Fibroblast, Stroma, Endothelial, Other_connective
    3: [14, 15],                # Other: Necrotic, Apoptosis
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dims", type=int, required=True, help="Patch size (e.g. 32,48,64,96,128)")
    parser.add_argument("--depth", type=int, required=True, help="Encoder depth (e.g. 3 or 4)")
    parser.add_argument("--gpu", type=str, default="0", help="Which GPU to use, e.g. '0', '1'")
    parser.add_argument("--plots", action="store_true", help="Generate training/validation loss plots if set")
    parser.add_argument("--checkpoint_seg", type=str, default=None, help="Path to checkpoint to resume training from")
    parser.add_argument("--continue_seg", action="store_true", help="Continue segmentation training from checkpoint")
    parser.add_argument("--checkpoint_cls", type=str, default=None, help="Path to checkpoint to resume training from")
    parser.add_argument("--continue_cls", action="store_true", help="Continue classification training from checkpoint")
    return parser.parse_args()

# Define logging wrapper
logger = None
def log(msg):
    print(msg)
    if logger:
        logger.info(msg)

class HierarchicalCrossEntropyLoss(nn.Module):
    def __init__(self, hierarchy_map):
        """
        Args:
            hierarchy_map (dict): Maps super-class index to list of sub-class indices.
        """
        super(HierarchicalCrossEntropyLoss, self).__init__()
        self.hierarchy_map = hierarchy_map

        # Build reverse map: sub-class → super-class
        self.sub_to_super = {}
        for super_cls, sub_classes in hierarchy_map.items():
            for sub_cls in sub_classes:
                self.sub_to_super[sub_cls] = super_cls

    def forward(self, logits, remapped_targets, label_levels):
        """
        Compute the generalized hierarchical cross-entropy loss.

        Args:
            logits (Tensor): Raw model outputs (N, C) where C = num fine-grained classes.
            remapped_targets (Tensor): Class labels remapped to fine or coarse indices. Shape: (N,)
            label_levels (List[str]): 'fine' or 'coarse' for each sample.

        Returns:
            Tensor: Scalar loss value.
        """
        log_probs = F.log_softmax(logits, dim=1)
        batch_size = logits.shape[0]
        loss = 0.0

        for i in range(batch_size):
            target = remapped_targets[i]
            label_level = label_levels[i]
            log_p = log_probs[i]

            if label_level == 'fine':
                loss -= log_p[target]
            elif label_level == 'coarse':
                sub_classes = self.hierarchy_map[target]
                # log_sum = torch.logsumexp(log_p[sub_classes], dim=0) # mathematically incorrect, because log_p is already log-softmaxed.
                # loss -= log_sum
                loss -= torch.log(torch.exp(log_p[sub_classes]).sum() + 1e-8)
            else:
                raise ValueError(f"Unknown label level '{label_level}' at index {i}")

        return loss / batch_size

class FusionClassifier(nn.Module):
    def __init__(self, num_classes=16, hierarchy_map=None, pretrained=False):
        super().__init__()

        self.hierarchy_map = hierarchy_map
        self.num_classes = num_classes
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.attn_gate = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.encoder_tight = nn.Sequential(*list(models.resnet18(pretrained=pretrained).children())[:-2])
        self.encoder_context = nn.Sequential(*list(models.resnet18(pretrained=pretrained).children())[:-2])

        # Learned spatial attention on context features
        self.attention_conv = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1),
            nn.Sigmoid()  # output: [B,1,H,W] mask
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

        bottleneck_channels = 512
        fusion_dim = bottleneck_channels * 2

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.BatchNorm1d(fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(fusion_dim, bottleneck_channels),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck_channels, self.num_classes)
        )

    def forward(self, tight, context):
        f_tight = self.encoder_tight(tight)         # B x 512 x H x W
        f_context = self.encoder_context(context)   # B x 512 x H x W

        attn_mask = self.attention_conv(f_context)  # B x 1 x H x W
        #f_context_att = f_context * attn_mask + f_context       # apply spatial attention
        #f_context_att = self.alpha * f_context * attn_mask + (1 - self.alpha) * f_context
        f_context_att = self.attn_gate(f_context) * f_context * attn_mask + (1 - self.attn_gate(f_context)) * f_context

        pooled_tight = self.pool(f_tight)
        pooled_context = self.pool(f_context_att)

        fused = torch.cat([pooled_tight, pooled_context], dim=1)
        logits = self.classifier(self.flatten(fused))
        return logits

    def entropy_regularization(self, logits, targets, label_levels):
        """
        Adds regularization to encourage uniform logits for coarse labels.
        """
        if self.hierarchy_map is None:
            return torch.tensor(0.0, device=logits.device)
        probs = torch.softmax(logits, dim=1)
        reg_loss = 0.0
        count = 0

        for i, (label, level) in enumerate(zip(targets, label_levels)):
            if level == 'coarse':
                subclass_indices = self.hierarchy_map[label]
                subclass_probs = probs[i, subclass_indices]
                uniform = torch.full_like(subclass_probs, 1.0 / len(subclass_probs))
                reg_loss += F.mse_loss(subclass_probs, uniform)
                count += 1

        if count == 0:
            return torch.tensor(0.0, device=logits.device)
        return reg_loss / count
        
class SegmentationDataset(Dataset):
    def __init__(self, image_paths, dims,context_scale = 2, is_train=True):
        self.image_paths = image_paths
        self.dims = dims
        self.is_train = is_train
        self.context_scale = context_scale

        self.transforms = transforms.Compose([
            transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                   saturation=0.2, hue=0.1)
        ]) if is_train else transforms.Compose([])

        self.tensorify = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        self.data_info = []
        for ip in self.image_paths:
            stem = Path(ip).stem
            box_path    = f'/workspace/classification_pipeline/PumaDataset_yoloA/boxes/{stem}.npy' # Earlier: PumaDataset2
            nuclei_path = f'/workspace/classification_pipeline/PumaDataset_yoloA/nuclei/{stem}.npy'
            class_path  = f'/workspace/classification_pipeline/PumaDataset_yoloA/class/{stem}.npy'
            if os.path.exists(box_path) and os.path.exists(nuclei_path) and os.path.exists(class_path):
                nuclei = np.load(nuclei_path)
                for i in range(len(nuclei)):
                    self.data_info.append((ip, box_path, nuclei_path, class_path, i))

        #self.num_samples = 2048 * 4 if is_train else 512

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        ip, box_path, nuclei_path, class_path, idx_nuc = self.data_info[idx]

        boxes = np.load(box_path).astype(int)
        boxes[boxes < 8] = 8
        boxes[boxes > 992] = 992
        #nuclei = np.load(nuclei_path, allow_pickle=True)
        nuclei = np.load(nuclei_path)
        class_labels = np.load(class_path, allow_pickle=True)

        image = np.array(Image.open(ip).convert('RGB'))
        num_nuclei = nuclei.shape[0]
        #idx_nuc = random.randint(0, num_nuclei - 1)
        # Debug
        #print(f"Num of nuclei = {num_nuclei}")
        #print(f"Nucleus idx = {idx_nuc}")
        #print(f"Class labels len = {len(class_labels)}")
        #print(f"Class labels shape = {class_labels.shape}")
        #print(f"Is instance = {isinstance(class_labels[idx_nuc], (list, np.ndarray))}")

        #arr = nuclei[idx_nuc]
        arr = np.array(nuclei[idx_nuc], dtype=np.uint8)
        class_id = class_labels[idx_nuc][0] if isinstance(class_labels[idx_nuc], (list, np.ndarray)) else class_labels[idx_nuc]

        #Debug
        if not (0 <= class_id < 11):
            print(f"Invalid class id = {class_id}")
            class_id = 0  # or skip with a fallback dummy sample
        #class_id = class_labels[idx_nuc]
        #class_id_raw = class_labels[idx_nuc]
        # Debug 
        #print(f"Class id raw = {class_id_raw}")
        #class_id = int(class_id_raw[0]) if isinstance(class_id_raw, (list, np.ndarray)) else int(class_id_raw)
        #class_id = class_id_raw


        x1, y1, x2, y2 = boxes[idx_nuc][1:5]
        x1, y1, x2, y2 = x1 - 4, y1 - 4, x2 + 4, y2 + 4

        h, w = image.shape[:2]
        x1 = max(0, min(w-1, x1))
        x2 = max(0, min(w-1, x2))
        y1 = max(0, min(h-1, y1))
        y2 = max(0, min(h-1, y2))

        mask = arr[y1:y2, x1:x2]
        patch = image[y1:y2, x1:x2]

        #mask = cv2.resize(mask, (self.dims, self.dims), interpolation=cv2.INTER_NEAREST).astype(float)
        #patch = cv2.resize(patch, (self.dims, self.dims), interpolation=cv2.INTER_AREA)

        if mask.size == 0:
            mask = np.zeros((self.dims, self.dims), dtype=np.float32)
        else:
            mask = cv2.resize(mask, (self.dims, self.dims), interpolation=cv2.INTER_NEAREST).astype(float)

        if patch.size == 0:
            patch = np.zeros((self.dims, self.dims, 3), dtype=np.uint8)
        else:
            patch = cv2.resize(patch, (self.dims, self.dims), interpolation=cv2.INTER_AREA)

        #patch = Image.fromarray(patch)
        #patch = self.transforms(patch)
        #patch = self.tensorify(patch)
        #mask = torch.tensor(mask).unsqueeze(0)
        #label = torch.tensor(class_id, dtype=torch.long)

        ####### 2. CLASSIFICATION PATCH (original scale) #######
        # Center of the box
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        half_dim = self.dims // 2

        cls_x1 = cx - half_dim
        cls_x2 = cx + half_dim
        cls_y1 = cy - half_dim
        cls_y2 = cy + half_dim

        # Padding if out of bounds
        pad_left = max(0, -cls_x1)
        pad_top = max(0, -cls_y1)
        pad_right = max(0, cls_x2 - w)
        pad_bottom = max(0, cls_y2 - h)

        cls_x1 = max(0, cls_x1)
        cls_x2 = min(w, cls_x2)
        cls_y1 = max(0, cls_y1)
        cls_y2 = min(h, cls_y2)

        cls_patch = image[cls_y1:cls_y2, cls_x1:cls_x2]

        if pad_left or pad_top or pad_right or pad_bottom:
            cls_patch = cv2.copyMakeBorder(
                cls_patch,
                pad_top, pad_bottom, pad_left, pad_right,
                borderType=cv2.BORDER_CONSTANT, value=0
            )

        ####### 3. CONTEXT PATCH #######
        context_dim = self.dims * self.context_scale
        half_context = context_dim // 2

        cx1 = cx - half_context
        cx2 = cx + half_context
        cy1 = cy - half_context
        cy2 = cy + half_context

        pad_left = max(0, -cx1)
        pad_top = max(0, -cy1)
        pad_right = max(0, cx2 - w)
        pad_bottom = max(0, cy2 - h)

        cx1 = max(0, cx1)
        cx2 = min(w, cx2)
        cy1 = max(0, cy1)
        cy2 = min(h, cy2)

        context_patch = image[cy1:cy2, cx1:cx2]

        if pad_left or pad_top or pad_right or pad_bottom:
            context_patch = cv2.copyMakeBorder(
                context_patch,
                pad_top, pad_bottom, pad_left, pad_right,
                borderType=cv2.BORDER_CONSTANT, value=0
            )

        # Convert all to PIL
        patch = Image.fromarray(patch)
        cls_patch = Image.fromarray(cls_patch)
        context_patch = Image.fromarray(context_patch)

        # Apply transforms
        patch = self.transforms(patch)
        cls_patch = self.transforms(cls_patch)
        context_patch = self.transforms(context_patch)

        patch = self.tensorify(patch)
        cls_patch = self.tensorify(cls_patch)
        context_patch = self.tensorify(context_patch)
        mask = torch.tensor(mask).unsqueeze(0)
        label = torch.tensor(class_id, dtype=torch.long)
        
        return {'patch': patch, 'mask': mask, 'label': label, 'cls_patch': cls_patch, 'context_patch': context_patch}

def get_label_level_batch(label_tensor):
    """
    Args:
        label_tensor: torch.Tensor of shape (N,), values 0-10 (PUMA labels)
    Returns:
        label_levels: List[str] of 'fine' or 'coarse'
        remapped_labels: List[int] to match fine or coarse hierarchy
    """
    label_levels = []
    remapped_labels = []

    for label in label_tensor.tolist():
        name = PUMA_IDX_TO_NAME[label]

        # Remove "nuclei_" prefix
        stripped = name.replace('nuclei_', '').replace('_cell', '')

        # Try fine match
        fine_match = difflib.get_close_matches(stripped, FINE_CLASS_MAP.keys(), n=1, cutoff=0.6)
        if fine_match:
            label_levels.append('fine')
            remapped_labels.append(FINE_CLASS_MAP[fine_match[0]])
        else:
            # If no fine match, fall back to coarse
            for coarse_class, members in COARSE_CLASS_MEMBERS.items():
                #if any(term in stripped for term in members):
                if stripped in members:
                    label_levels.append('coarse')
                    remapped_labels.append(COARSE_CLASS_MAP[coarse_class])
                    break
            else: # for-else loop
                # fallback to "other"
                label_levels.append('coarse')
                remapped_labels.append(COARSE_CLASS_MAP['other'])

    return label_levels, remapped_labels

def main():
    global logger
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    dims = args.dims
    enc_depth = args.depth
    NUM_TYPES = 16  # e.g., nuclei_tumor → nuclei_apoptosis


    # Set up logs
    logs_dir = Path("logs_attention_context_2")
    logs_dir.mkdir(exist_ok=True)
    log_file = logs_dir / f"train_dims{dims}_depth{enc_depth}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(filename=log_file, level=logging.INFO, format="%(asctime)s %(message)s", filemode='w')
    logger = logging.getLogger()

    log(f"Training with dims={dims}, encoder_depth={enc_depth}, on GPU={args.gpu}")
    # print(f"Training with dims={dims}, encoder_depth={enc_depth}, on GPU={args.gpu}")

    metric_file_seg = logs_dir / f"seg_metrics_dims{dims}_depth{enc_depth}.csv"
    metric_file_cls = logs_dir / f"cls_metrics_dims{dims}_depth{enc_depth}.csv"

    # dq_list, sq_list, pq_list, aji_list = []

    ckpt_dir = Path(f"checkpoints_attention_context_2/dims{dims}_depth{enc_depth}")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    image_input_dir = '/workspace/HoVerNet_puma/PUMA/Images/01_training_dataset_tif_ROIs'
    image_paths = sorted([str(p) for p in Path(image_input_dir).rglob("*.tif")])
    val_paths = image_paths[:12]
    train_paths = image_paths[12:]

    log(f"Val set size: {len(val_paths)}, Train set size: {len(train_paths)}")
    # print(f"Val set size: {len(val_paths)}, Train set size: {len(train_paths)}")

    train_dataset = SegmentationDataset(train_paths, dims, context_scale=4, is_train=True)
    val_dataset   = SegmentationDataset(val_paths,   dims, context_scale=4, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=640, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_dataset,   batch_size=640, shuffle=False, num_workers=4)

    #log("Training and validation dataloaders prepared")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # model = UNetPlusPlusWithClassifier(encoder_depth=enc_depth).to(device)

    log("-------------------------------------------Training UNET++----------------------------------------------")
    model_seg = smp.UnetPlusPlus(
        encoder_name="resnet18",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation='sigmoid',
        encoder_depth=enc_depth,
        decoder_channels=[128,64,32,16][::-1][:enc_depth][::-1]
    ).to(device)
     

    optimizer_seg = torch.optim.AdamW(model_seg.parameters(), lr=5e-4, weight_decay=1e-9)
    criterion_seg = torch.nn.BCELoss().to(device)
    #criterion_cls = torch.nn.CrossEntropyLoss().to(device)
    # criterion_cls = HierarchicalCrossEntropyLoss(hierarchy_map)

    max_epochs_seg = 600
    patience_seg = 55
    best_val_loss_seg = float('inf')
    epochs_no_improve_seg = 0
    start_epoch_seg = 1

    train_losses_seg = []
    val_losses_seg = []

    if args.checkpoint_seg and args.continue_seg:
        checkpoint = torch.load(args.checkpoint_seg)
        model_seg.load_state_dict(checkpoint["model_state"])
        optimizer_seg.load_state_dict(checkpoint["optimizer_state"])
        best_val_loss_seg = checkpoint.get("best_val_loss", float('inf'))
        start_epoch_seg = checkpoint.get("epoch", 1) + 1
        train_losses_seg = checkpoint.get("train_losses", [])
        val_losses_seg = checkpoint.get("val_losses", [])
        epochs_no_improve_seg = checkpoint.get("epochs_no_improve", 0)
        log(f"Resumed training from checkpoint: {args.checkpoint_seg}, starting at epoch {start_epoch_seg}")
    elif args.checkpoint_seg and not args.continue_seg:
        log("Skipping UNET++ training")
        start_epoch_seg = max_epochs_seg + 1

    for epoch in range(start_epoch_seg, max_epochs_seg + 1):
        log(f"[dims={dims}, depth={enc_depth}] Epoch {epoch}/{max_epochs_seg}")
        # print(f"[dims={dims}, depth={enc_depth}] Epoch {epoch}/{max_epochs}")

        # --- train UNET---
        model_seg.train()
        train_loss_sum_seg = 0.0
        for batch in tqdm(train_loader, desc="Train", leave=False):
            patch = batch['patch'].to(device)
            mask  = batch['mask'].to(device)
            label = batch['label'].to(device)
            label_levels, remapped_labels = get_label_level_batch(label)

            optimizer_seg.zero_grad()
            # pred_mask, pred_class = model(patch)
            pred_mask = model_seg(patch)
            # Debug
            #print(f"GT mask: {mask}")
            #print(f"Pred mask: {pred_mask}")
            loss_seg = criterion_seg(pred_mask, mask.float())
            #loss_cls = criterion_cls(pred_class, label)
            # loss_cls = criterion_cls(pred_class, remapped_labels, label_levels)
            # loss = loss_seg + loss_cls
            loss_seg.backward()
            optimizer_seg.step()

            train_loss_sum_seg += loss_seg.item()

        train_loss_avg_seg = train_loss_sum_seg / len(train_loader)
        train_losses_seg.append(train_loss_avg_seg)
        log(f"  train_loss_seg = {train_loss_avg_seg:.5f}")
        # print(f"  train_loss = {train_loss_avg:.5f}")

        # --- val ---
        model_seg.eval()
        val_loss_sum_seg = 0.0
        dq_list, sq_list, pq_list, aji_list = [], [], [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Val", leave=False):
                patch = batch['patch'].to(device)
                mask  = batch['mask'].to(device)
                label = batch['label'].to(device)
                label_levels, remapped_labels = get_label_level_batch(label)

                #pred_mask, pred_class = model(patch)
                pred_mask = model_seg(patch)

                loss_seg = criterion_seg(pred_mask, mask.float())
                # Debug
                # print("Labels in batch:", label.cpu().unique())
                #loss_cls = criterion_cls(pred_class, label)
                #loss_cls = criterion_cls(pred_class, remapped_labels, label_levels)
                #loss = loss_seg + loss_cls

                val_loss_sum_seg += loss_seg.item()

                #mask = mask.cpu().numpy()
                #pred_mask = pred_mask.cpu().numpy()
                for i in range(mask.shape[0]):
                    true_arr = (mask[i, 0] > 0.5).cpu().numpy().astype(np.uint8)
                    pred_arr = (pred_mask[i, 0] > 0.5).cpu().numpy().astype(np.uint8)
                    #true_arr,_ = connected_components(mask[i,0]>0.5)
                    #pred_arr,_ = connected_components(pred_mask[i,0]>0.5)
                    # Debug 
                    #print(f"True arr: {true_arr}")
                    #print(f"Pred arr: {pred_arr}")
                    #print(f"True arr unique: {np.unique(true_arr)}")
                    #print(f"Pred arr unique: {np.unique(pred_arr)}")
                    result = process_one_mask_from_arrays(true_arr.astype(np.uint8), pred_arr.astype(np.uint8), stem=f"val_{i}")
                    if result:
                        dq, sq, pq, aji = result
                        dq_list.append(dq)
                        sq_list.append(sq)
                        pq_list.append(pq)
                        aji_list.append(aji)

        val_loss_avg_seg = val_loss_sum_seg / len(val_loader)
        val_losses_seg.append(val_loss_avg_seg)
        log(f"  val_loss_seg   = {val_loss_avg_seg:.5f}")
        # print(f"  val_loss   = {val_loss_avg:.5f}")
        # mean_mPQ = np.mean(type_f1s) if type_f1s else 0.0
        if dq_list:
            mean_dq = np.mean(dq_list)
            mean_sq = np.mean(sq_list)
            mean_bpq = np.mean(pq_list)
            mean_aji = np.mean(aji_list)
        log(f"Mean Segmentation Metrics - DQ: {mean_dq:.4f}, SQ: {mean_sq:.4f}, bPQ: {mean_bpq:.4f}, AJI: {mean_aji:.4f}") #, mPQ: {mean_mPQ:.4f}")
        write_header = not metric_file_seg.exists()
        with open(metric_file_seg, mode='a', newline='') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["epoch", "mean_dq", "mean_sq", "mean_bpq", "mean_aji"])#,"mean_mpq"])
            writer.writerow([epoch, mean_dq, mean_sq, mean_bpq, mean_aji])#, mean_mPQ])

        if val_loss_avg_seg < best_val_loss_seg:
            best_val_loss_seg = val_loss_avg_seg
            epochs_no_improve_seg = 0
            best_ckpt_seg = ckpt_dir / "best_model_seg.pt"
            # torch.save(model.state_dict(), best_ckpt)
            torch.save({
                "epoch": epoch,
                "model_state": model_seg.state_dict(),
                "optimizer_state": optimizer_seg.state_dict(),
                "best_val_loss": best_val_loss_seg,
                "train_losses": train_losses_seg,
                "val_losses": val_losses_seg,
                "epochs_no_improve": epochs_no_improve_seg
            }, best_ckpt_seg)
            log(f"  (New best val_loss: {val_loss_avg_seg:.5f}, checkpoint saved to {best_ckpt_seg})")
            # print(f"  (New best val_loss: {val_loss_avg:.5f}, checkpoint saved to {best_ckpt})")
        else:
            epochs_no_improve_seg += 1
            if epochs_no_improve_seg >= patience_seg:
                log(f"No improvement for {patience_seg} epochs, stopping early.")
                # print(f"No improvement for {patience} epochs, stopping early.")
                break
    if args.plots and args.continue_seg:
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses_seg, label='Train Loss (Segmentation)')
        plt.plot(val_losses_seg, label='Val Loss (Segmentation)')
        plt.xlabel('Epoch')
        plt.ylabel('Segmentation Loss')
        plt.title(f'Training vs Validation Loss (dims={dims}, depth={enc_depth}) - Segmentation')
        plt.legend()
        plt.grid(True)
        plot_path = logs_dir / f"seg_loss_curve_dims{dims}_depth{enc_depth}.png"
        plt.savefig(plot_path)
        log(f"Segmentation Loss curve saved to {plot_path}")

    log("-------------------------------------------Training Classifier-------------------------------------------")
    NUM_CLASSES = 16
    #best_ckpt_seg = ckpt_dir / "best_model_seg.pt"
    #checkpoint = torch.load(best_ckpt_seg)
    #model_seg.load_state_dict(checkpoint["model_state"])
    #enc = model_seg.encoder
    #bottleneck_channels = enc.out_channels[-1]
    
    # model_cls = torch.nn.Sequential(
    #    torch.nn.AdaptiveAvgPool2d((1, 1)),
    #    torch.nn.Flatten(),
    #    torch.nn.Linear(bottleneck_channels, NUM_CLASSES)  # 15 nucleus classes + 1 background class
    #).to(device)
    
    #model_cls = nn.Sequential(
    #    nn.AdaptiveAvgPool2d((1, 1)),
    #    nn.Flatten(),
    #    nn.Linear(bottleneck_channels, bottleneck_channels // 2),
    #    nn.BatchNorm1d(bottleneck_channels // 2),
    #    nn.ReLU(inplace=True),
    #    nn.Dropout(0.5),
    #    nn.Linear(bottleneck_channels // 2, NUM_CLASSES)
    #).to(device)
    
    model_cls = FusionClassifier(
        num_classes=NUM_CLASSES,
        hierarchy_map=hierarchy_map,
        pretrained=False
    ).to(device)
     

    optimizer_cls = torch.optim.AdamW(model_cls.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler_cls = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_cls, 'min', patience=5, factor=0.5)
    #criterion_cls = torch.nn.BCELoss().to(device)
    #criterion_cls = torch.nn.CrossEntropyLoss().to(device)
    criterion_cls = HierarchicalCrossEntropyLoss(hierarchy_map)

    max_epochs_cls = 600
    patience_cls = 55
    best_val_loss_cls = float('inf')
    epochs_no_improve_cls = 0
    start_epoch_cls = 1
    reg_weight = 0.1 # entropy regularization weight

    train_losses_cls = []
    val_losses_cls = []

    if args.checkpoint_cls and args.continue_cls:
        checkpoint = torch.load(args.checkpoint_cls)
        model_cls.load_state_dict(checkpoint["model_state"])
        optimizer_cls.load_state_dict(checkpoint["optimizer_state"])
        best_val_loss_cls = checkpoint.get("best_val_loss", float('inf'))
        start_epoch_cls = checkpoint.get("epoch", 1) + 1
        train_losses_cls = checkpoint.get("train_losses", [])
        val_losses_cls = checkpoint.get("val_losses", [])
        epochs_no_improve_cls = checkpoint.get("epochs_no_improve", 0)
        log(f"Resumed training from checkpoint: {args.checkpoint_cls}, starting at epoch {start_epoch_cls}")
    elif args.checkpoint_cls and not args.continue_cls:
        log("Skipping Classifier training")
        start_epoch_cls = max_epochs_cls + 1

    for epoch in range(start_epoch_cls, max_epochs_cls + 1):
        log(f"[dims={dims}, depth={enc_depth}] Epoch {epoch}/{max_epochs_cls}")
        # print(f"[dims={dims}, depth={enc_depth}] Epoch {epoch}/{max_epochs}")

        # --- train Classifier---
        model_cls.train()
        train_loss_sum_cls = 0.0
        for batch in tqdm(train_loader, desc="Train", leave=False):
            cls_patch = batch['cls_patch'].to(device)
            context_patch = batch['context_patch'].to(device)  # zoomed-out patch
            mask  = batch['mask'].to(device)
            label = batch['label'].to(device)
            label_levels, remapped_labels = get_label_level_batch(label)

            optimizer_cls.zero_grad()
            # pred_mask, pred_class = model(patch)
            # pred_mask = model_seg(patch)
            pred_class = model_cls(cls_patch,context_patch)
            # Debug
            #print(f"GT mask: {mask}")
            #print(f"Pred mask: {pred_mask}")
            #loss_seg = criterion_seg(pred_mask, mask.float())
            #loss_cls = criterion_cls(pred_class, label)
            loss_cls = criterion_cls(pred_class, remapped_labels, label_levels)
            # Add entropy regularization
            reg = model_cls.entropy_regularization(pred_class, remapped_labels, label_levels)
            loss_cls += reg_weight*reg
            loss_cls.backward()
            optimizer_cls.step()

            train_loss_sum_cls += loss_cls.item()

        train_loss_avg_cls = train_loss_sum_cls / len(train_loader)
        train_losses_cls.append(train_loss_avg_cls)
        log(f"  train_loss_cls = {train_loss_avg_cls:.5f}")
        # print(f"  train_loss = {train_loss_avg:.5f}")

        
        
        # --- val ---
        model_cls.eval()
        val_loss_sum_cls = 0.0
        # type_f1s = []
        # accs = []
        f1_list, precision_list, recall_list, accuracy_list, mPQ_list = [], [], [], [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Val", leave=False):
                cls_patch = batch['cls_patch'].to(device)
                context_patch = batch['context_patch'].to(device) # zoomed-out patch
                mask  = batch['mask'].to(device)
                label = batch['label'].to(device)
                label_levels, remapped_labels = get_label_level_batch(label)

                #pred_mask, pred_class = model(patch)
                pred_class = model_cls(cls_patch,context_patch)
                """
                paired_true = np.array(remapped_labels)
                paired_pred = pred_class.argmax(dim=1).detach().cpu().numpy()  # convert tensor to numpy
                unpaired_true = np.array([])
                unpaired_pred = np.array([])
                for type_id in range(NUM_TYPES):
                    f1, _, _ = cell_type_detection_scores(
                        paired_true, paired_pred, unpaired_true, unpaired_pred, type_id, #pred_class, remapped_labels, NUM_CLASSES
                    )
                    type_f1s.append(f1)
                    # accs.append(acc)
                """
                # Call the function
                metrics = compute_metrics_with_hierarchy3(
                    pred_class, remapped_labels, label_levels, hierarchy_map, NUM_CLASSES
                )
                # Debug 
                # print(f"GT: {remapped_labels}")
                # print(f"Predictions: {pred_class.argmax(dim=1).detach().cpu().numpy()}")
                if metrics:
                    f1, precision, recall, accuracy, mPQ = metrics
                    f1_list.append(f1)
                    precision_list.append(precision)
                    recall_list.append(recall)
                    accuracy_list.append(accuracy)
                    mPQ_list.append(mPQ)
                #loss_seg = criterion_seg(pred_mask, mask.float())
                # Debug
                # print("Labels in batch:", label.cpu().unique())
                #loss_cls = criterion_cls(pred_class, label)
                loss_cls = criterion_cls(pred_class, remapped_labels, label_levels)
                #loss = loss_seg + loss_cls

                val_loss_sum_cls += loss_cls.item()

        val_loss_avg_cls = val_loss_sum_cls / len(val_loader)
        val_losses_cls.append(val_loss_avg_cls)
        log(f"  val_loss_cls   = {val_loss_avg_cls:.5f}")
        # print(f"  val_loss   = {val_loss_avg:.5f}")
        
        # Step the scheduler
        scheduler_cls.step(val_loss_avg_cls)
        
        # mean_typef1s = np.mean(type_f1s) if type_f1s else 0.0
        mean_f1 = np.mean(f1_list) if f1_list else 0.0
        mean_precision = np.mean(precision_list) if precision_list else 0.0
        mean_recall = np.mean(recall_list) if recall_list else 0.0
        mean_accuracy = np.mean(accuracy_list) if accuracy_list else 0.0
        mean_mPQ = np.mean(mPQ_list) if mPQ_list else 0.0
        # mean_accs = np.mean(accs) if accs else 0.0
        # log(f"Mean Classification Metrics - type_F1: {mean_typef1s:.4f}")
        log(f"Mean Classification Metrics - F1: {mean_f1:.4f}, Precision: {mean_precision:.4f}, Recall: {mean_recall:.4f}, Accuracy: {mean_accuracy:.4f}, mPQ: {mean_mPQ:.4f}")
        write_header = not metric_file_cls.exists()
        with open(metric_file_cls, mode='a', newline='') as f:
            writer = csv.writer(f)
            if write_header:
                # writer.writerow(["epoch", "mean_type_F1"])
                writer.writerow(["epoch", "mean_f1", "mean_precision", "mean_recall", "mean_accuracy", "mean_mPQ"])
            # writer.writerow([epoch, mean_typef1s])
            writer.writerow([epoch, mean_f1, mean_precision, mean_recall, mean_accuracy, mean_mPQ])

        if val_loss_avg_cls < best_val_loss_cls:
            best_val_loss_cls = val_loss_avg_cls
            epochs_no_improve_cls = 0
            best_ckpt_cls = ckpt_dir / "best_model_cls.pt"
            # torch.save(model.state_dict(), best_ckpt)
            torch.save({
                "epoch": epoch,
                "model_state": model_cls.state_dict(),
                "optimizer_state": optimizer_cls.state_dict(),
                "best_val_loss": best_val_loss_cls,
                "train_losses": train_losses_cls,
                "val_losses": val_losses_cls,
                "epochs_no_improve": epochs_no_improve_cls
            }, best_ckpt_cls)
            log(f"  (New best val_loss_cls: {val_loss_avg_cls:.5f}, checkpoint saved to {best_ckpt_cls})")
            # print(f"  (New best val_loss: {val_loss_avg:.5f}, checkpoint saved to {best_ckpt})")
        else:
            epochs_no_improve_cls += 1
            if epochs_no_improve_cls >= patience_cls:
                log(f"No improvement for {patience_cls} epochs, stopping early.")
                # print(f"No improvement for {patience} epochs, stopping early.")
                break
    if args.plots and args.continue_cls:
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses_cls, label='Train Loss (Classification)')
        plt.plot(val_losses_cls, label='Val Loss (Classification)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Training vs Validation Loss (dims={dims}, depth={enc_depth}) - Classification')
        plt.legend()
        plt.grid(True)
        plot_path = logs_dir / f"cls_loss_curve_dims{dims}_depth{enc_depth}.png"
        plt.savefig(plot_path)
        log(f"Classification Loss curve saved to {plot_path}")


if __name__ == "__main__":
    main()
