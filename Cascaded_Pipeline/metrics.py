import warnings

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.optimize import linear_sum_assignment
import os
import glob
from pathlib import Path
from multiprocessing import Pool, cpu_count
import pandas as pd
from typing import List
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score


# --------------------------Optimised for Speed
def get_fast_aji(true, pred):
    """AJI version distributed by MoNuSeg, has no permutation problem but suffered from
    over-penalisation similar to DICE2.

    Fast computation requires instance IDs are in contiguous ordering i.e [1, 2, 3, 4]
    not [2, 3, 6, 10]. Please call `remap_label` beforehand and `by_size` flag has no
    effect on the result.
    """
    pred = remap_label(pred)
    true = remap_label(true)
    true = np.copy(true)
    pred = np.copy(pred)
    true_id_list = list(np.unique(true))
    pred_id_list = list(np.unique(pred))

    

    # Prepare instance masks
    true_masks = [None]
    for t in true_id_list[1:]:
        t_mask = (true == t).astype(np.uint8)
        true_masks.append(t_mask)

    pred_masks = [None]
    for p in pred_id_list[1:]:
        p_mask = (pred == p).astype(np.uint8)
        pred_masks.append(p_mask)

    pairwise_inter = np.zeros([len(true_id_list) - 1, len(pred_id_list) - 1], dtype=np.float64)
    pairwise_union = np.zeros([len(true_id_list) - 1, len(pred_id_list) - 1], dtype=np.float64)

    # Fill pairwise intersections/unions
    for t_id in true_id_list[1:]:  # 0-th is background
        t_mask = true_masks[t_id]
        pred_true_overlap = pred[t_mask > 0]
        pred_true_overlap_id = np.unique(pred_true_overlap)
        pred_true_overlap_id = list(pred_true_overlap_id)
        for p_id in pred_true_overlap_id:
            if p_id == 0:
                continue
            p_mask = pred_masks[p_id]
            total = (t_mask + p_mask).sum()
            inter = (t_mask * p_mask).sum()
            pairwise_inter[t_id - 1, p_id - 1] = inter
            pairwise_union[t_id - 1, p_id - 1] = total - inter

    pairwise_iou = pairwise_inter / (pairwise_union + 1.0e-6)
    # For each GT, pick the pred that gives highest IoU
    paired_pred = np.argmax(pairwise_iou, axis=1)
    max_iou = np.max(pairwise_iou, axis=1)
    # Filter out GT with no intersection
    paired_true = np.nonzero(max_iou > 0.0)[0]  # indices
    paired_pred = paired_pred[paired_true]

    overall_inter = (pairwise_inter[paired_true, paired_pred]).sum()
    overall_union = (pairwise_union[paired_true, paired_pred]).sum()

    # Convert from index to actual IDs
    paired_true = list(paired_true + 1)
    paired_pred = list(paired_pred + 1)

    # Add unpaired GT + Pred to union
    unpaired_true = [x for x in true_id_list[1:] if x not in paired_true]
    unpaired_pred = [x for x in pred_id_list[1:] if x not in paired_pred]
    for t_id in unpaired_true:
        overall_union += true_masks[t_id].sum()
    for p_id in unpaired_pred:
        overall_union += pred_masks[p_id].sum()

    aji_score = overall_inter / overall_union if overall_union else 0.0
    return aji_score


from scipy.optimize import linear_sum_assignment
import numpy as np

def get_fast_aji_plus(true, pred):
    """
    AJI+ (Aggregated Jaccard Index Plus) with maximal unique pairing.
    1. Remap labels to ensure contiguous IDs: 0,1,2,...
    2. Build dictionaries of {inst_id: mask}.
    3. Compute pairwise intersection & union for GT and Pred.
    4. Use Hungarian (linear_sum_assignment) to find the best 1-to-1 matching.
    5. Sum intersection over matched pairs, and union over matched + unpaired IDs.
    6. Return the AJI+ = total_inter / total_union.

    Note: 'true' and 'pred' must be 2D integer arrays with instance IDs.
    ID 0 is background (ignored).
    """
    # ---------------------
    # 1) Remap label => 0..N contiguous
    # ---------------------
    true = remap_label(true)
    pred = remap_label(pred)
    # Make copies to avoid side effects
    true = np.copy(true)
    pred = np.copy(pred)

    # Collect unique IDs
    true_id_list = sorted(list(np.unique(true)))
    pred_id_list = sorted(list(np.unique(pred)))

    # Remove background from ID lists
    if 0 in true_id_list:
        true_id_list.remove(0)
    if 0 in pred_id_list:
        pred_id_list.remove(0)

    # Edge case: if no nuclei in GT or Pred, AJI+ is 0
    if len(true_id_list) == 0 or len(pred_id_list) == 0:
        return 0.0

    # ---------------------
    # 2) Build dictionaries: {id: mask}
    # ---------------------
    # Each dict key = instance ID, value = binary mask for that instance
    true_masks = {}
    for t_id in true_id_list:
        mask = (true == t_id).astype(np.uint8)
        true_masks[t_id] = mask

    pred_masks = {}
    for p_id in pred_id_list:
        mask = (pred == p_id).astype(np.uint8)
        pred_masks[p_id] = mask

    # ---------------------
    # 3) Compute pairwise intersection & union
    # ---------------------
    # We'll store them in a 2D array, rows=GT IDs, cols=Pred IDs
    n_true = len(true_id_list)
    n_pred = len(pred_id_list)

    pairwise_inter = np.zeros((n_true, n_pred), dtype=np.float64)
    pairwise_union = np.zeros((n_true, n_pred), dtype=np.float64)

    # Create quick ID -> row/col mapping
    #   E.g. if true_id_list = [1,4,9], then row_idx_of[1] = 0, row_idx_of[4] = 1, row_idx_of[9]=2
    row_idx_of = {tid: r for r, tid in enumerate(true_id_list)}
    col_idx_of = {pid: c for c, pid in enumerate(pred_id_list)}

    # For each ground-truth instance t_id
    for t_id in true_id_list:
        t_mask = true_masks[t_id]
        # Find which predicted IDs actually overlap t_mask
        overlapped_pred_ids = np.unique(pred[t_mask > 0])
        # We only care about predicted IDs > 0
        overlapped_pred_ids = overlapped_pred_ids[overlapped_pred_ids > 0]

        for p_id in overlapped_pred_ids:
            p_mask = pred_masks[p_id]
            union_ = (t_mask + p_mask).sum()
            inter_ = (t_mask * p_mask).sum()
            r = row_idx_of[t_id]
            c = col_idx_of[p_id]
            pairwise_inter[r, c] = inter_
            pairwise_union[r, c] = union_ - inter_

    # ---------------------
    # 4) Hungarian matching for best 1-to-1 pairing
    # ---------------------
    pairwise_iou = pairwise_inter / (pairwise_union + 1e-6)
    row_idx, col_idx = linear_sum_assignment(-pairwise_iou)  # maximize IoU
    matched_iou = pairwise_iou[row_idx, col_idx]

    # Filter out zero-overlap matches
    valid_matches = matched_iou > 0
    row_idx = row_idx[valid_matches]
    col_idx = col_idx[valid_matches]

    # Summation of matched intersection and union
    matched_inter = pairwise_inter[row_idx, col_idx]
    matched_union = pairwise_union[row_idx, col_idx]
    overall_inter = matched_inter.sum()
    overall_union = matched_union.sum()

    # ---------------------
    # 5) Add union from unpaired GT and Pred
    # ---------------------
    # Convert row/col indices to actual IDs
    paired_true_ids = [true_id_list[r] for r in row_idx]
    paired_pred_ids = [pred_id_list[c] for c in col_idx]

    unpaired_true = [tid for tid in true_id_list if tid not in paired_true_ids]
    unpaired_pred = [pid for pid in pred_id_list if pid not in paired_pred_ids]

    for t_id in unpaired_true:
        overall_union += true_masks[t_id].sum()
    for p_id in unpaired_pred:
        overall_union += pred_masks[p_id].sum()

    # ---------------------
    # 6) Compute AJI+ score
    # ---------------------
    aji_plus = overall_inter / overall_union if overall_union > 0 else 0.0
    return aji_plus

def remap_label(mask, by_size=False):
    """
    Rename all instance IDs in 'mask' so that they are contiguous [0..N].
    If 'by_size' is True, larger instances get smaller IDs (sorted by descending size).
    """
    mask = np.copy(mask)
    inst_ids = sorted(np.unique(mask))
    if 0 in inst_ids:
        inst_ids.remove(0)
    if len(inst_ids) == 0:
        return mask

    if by_size:
        sizes = [(id_, (mask == id_).sum()) for id_ in inst_ids]
        sizes.sort(key=lambda x: x[1], reverse=True)
        inst_ids = [x[0] for x in sizes]

    new_mask = np.zeros_like(mask, dtype=np.int32)
    for new_id, old_id in enumerate(inst_ids, start=1):
        new_mask[mask == old_id] = new_id
    return new_mask




def get_fast_pq(true, pred, match_iou=0.5):
    """
    `match_iou` is the IoU threshold level to determine the pairing between
    GT instances and pred instances. 1:1 mapping. If match_iou < 0.5, we do
    Hungarian matching to maximize total IoU. If >=0.5, the assumption is that
    any IoU>0.5 pair is unique already.
    """
    assert match_iou >= 0.0, "Cant' be negative"
    pred = remap_label(pred)
    true = remap_label(true)
    true = np.copy(true)
    pred = np.copy(pred)

    true_id_list = list(np.unique(true))
    pred_id_list = list(np.unique(pred))

    true_masks = [None]
    for t_id in true_id_list[1:]:
        t_mask = (true == t_id).astype(np.uint8)
        true_masks.append(t_mask)

    pred_masks = [None]
    for p_id in pred_id_list[1:]:
        p_mask = (pred == p_id).astype(np.uint8)
        pred_masks.append(p_mask)

    pairwise_iou = np.zeros([len(true_id_list) - 1, len(pred_id_list) - 1], dtype=np.float64)

    # Calculate IoU for each pair
    for t_id in true_id_list[1:]:
        t_mask = true_masks[t_id]
        pred_true_overlap = list(np.unique(pred[t_mask > 0]))
        for p_id in pred_true_overlap:
            if p_id == 0:
                continue
            p_mask = pred_masks[p_id]
            total = (t_mask + p_mask).sum()
            inter = (t_mask * p_mask).sum()
            iou = inter / (total - inter + 1e-6)
            pairwise_iou[t_id - 1, p_id - 1] = iou

    if match_iou >= 0.5:
        # direct matching
        pairwise_iou[pairwise_iou <= match_iou] = 0.0
        r_idx, c_idx = np.nonzero(pairwise_iou)
        matched_iou = pairwise_iou[r_idx, c_idx]
        # shift instance IDs
        paired_true = list(r_idx + 1)
        paired_pred = list(c_idx + 1)
    else:
        # Hungarian
        r_idx, c_idx = linear_sum_assignment(-pairwise_iou)
        matched_iou = pairwise_iou[r_idx, c_idx]
        valid = matched_iou > match_iou
        r_idx = r_idx[valid]
        c_idx = c_idx[valid]
        matched_iou = matched_iou[valid]
        paired_true = list(r_idx + 1)
        paired_pred = list(c_idx + 1)

    tp = len(paired_true)
    unpaired_true = [t for t in true_id_list[1:] if t not in paired_true]
    unpaired_pred = [p for p in pred_id_list[1:] if p not in paired_pred]
    fp = len(unpaired_pred)
    fn = len(unpaired_true)

    # DQ
    dq = tp / (tp + 0.5 * fp + 0.5 * fn + 1e-6)
    # SQ
    sq = matched_iou.sum() / (tp + 1e-6)
    return [dq, sq, dq * sq], [paired_true, paired_pred, unpaired_true, unpaired_pred]

def get_fast_pq2(true, pred, match_iou=0.3):
    """
    Compute Panoptic Quality (PQ) in a 'fast' way by enforcing one-to-one
    matches using the Hungarian (Munkres) method for all IoU thresholds.
    This avoids the incorrect shortcut that assumes IoU >= 0.5 implies uniqueness.

    Args:
        true (np.ndarray): 2D integer array with ground-truth instance labels.
        pred (np.ndarray): 2D integer array with predicted instance labels.
        match_iou (float): IoU threshold for deciding matches. Defaults to 0.5.

    Returns:
        metrics (list): [DQ, SQ, PQ]
        pairing_info (list): [paired_true_ids, paired_pred_ids, unpaired_true_ids, unpaired_pred_ids]
    """

    # Remap labels so that instance IDs are contiguous: 0, 1, 2, ...
    true = remap_label(true)
    pred = remap_label(pred)
    true = np.copy(true)
    pred = np.copy(pred)

    true_id_list = list(np.unique(true))
    pred_id_list = list(np.unique(pred))

    # Build masks for each instance
    # Index 0 is None / placeholder for background
    true_masks = [None]
    for t_id in true_id_list[1:]:
        true_masks.append((true == t_id).astype(np.uint8))

    pred_masks = [None]
    for p_id in pred_id_list[1:]:
        pred_masks.append((pred == p_id).astype(np.uint8))

    # pairwise_iou will be (num_true, num_pred)
    pairwise_iou = np.zeros((len(true_id_list) - 1, len(pred_id_list) - 1), dtype=np.float64)

    # Compute IoU for every pair (true_id, pred_id)
    for t_idx, t_id in enumerate(true_id_list[1:], start=0):
        t_mask = true_masks[t_idx + 1]  # t_idx=0 => t_id=1
        # Find which predictions actually overlap t_mask
        pred_ids_in_true = np.unique(pred[t_mask > 0])
        for p_id in pred_ids_in_true:
            if p_id == 0:
                continue
            p_idx = p_id - 1
            p_mask = pred_masks[p_idx + 1]
            union = (t_mask + p_mask).sum()
            inter = (t_mask * p_mask).sum()
            iou = inter / (union - inter + 1.0e-6)  # = inter / (|A| + |B| - inter)
            pairwise_iou[t_idx, p_idx] = iou

    # ---------------------------------------------------------
    # ALWAYS do a bipartite matching using Hungarian method
    # ---------------------------------------------------------
    # Negative because linear_sum_assignment() finds minimal cost, we want maximal IoU
    row_ids, col_ids = linear_sum_assignment(-pairwise_iou)
    matched_ious = pairwise_iou[row_ids, col_ids]

    # Filter out matches with IoU below threshold
    valid = matched_ious > match_iou
    row_ids = row_ids[valid]
    col_ids = col_ids[valid]
    matched_ious = matched_ious[valid]

    # Convert row/col indices to actual instance IDs
    # (Because row=0 => true_id=1, col=0 => pred_id=1)
    paired_true_ids = row_ids + 1
    paired_pred_ids = col_ids + 1

    # ---------------------------------------------------------
    # Count TPs, FPs, FNs
    # ---------------------------------------------------------
    tp = len(paired_true_ids)
    unpaired_true_ids = [t for t in true_id_list[1:] if t not in paired_true_ids]
    unpaired_pred_ids = [p for p in pred_id_list[1:] if p not in paired_pred_ids]
    fp = len(unpaired_pred_ids)
    fn = len(unpaired_true_ids)

    # ---------------------------------------------------------
    # Calculate DQ, SQ, PQ
    # ---------------------------------------------------------
    # Detection Quality (DQ)
    #   = TP / (TP + 0.5 * FP + 0.5 * FN)
    dq = tp / (tp + 0.5 * fp + 0.5 * fn + 1.0e-6)

    # Segmentation Quality (SQ)
    #   = average IoU of matched pairs
    sq = matched_ious.mean() if tp > 0 else 0.0

    pq = dq * sq

    return [dq, sq, pq], [paired_true_ids, paired_pred_ids, unpaired_true_ids, unpaired_pred_ids]

def get_fast_dice_2(true, pred):
    """Ensemble dice with instance-level summation."""
    true = np.copy(true)
    pred = np.copy(pred)
    
    true_id = list(np.unique(true))
    pred_id = list(np.unique(pred))

    # Build separate arrays for each true instance
    true_masks = [np.zeros_like(true, dtype=np.uint8)]
    for t in true_id[1:]:
        true_masks.append((true == t).astype(np.uint8))

    # Build separate arrays for each predicted instance
    pred_masks = [np.zeros_like(pred, dtype=np.uint8)]
    for p in pred_id[1:]:
        pred_masks.append((pred == p).astype(np.uint8))

    overall_total = 0
    overall_inter = 0

    # For each true instance
    for t_idx in range(1, len(true_id)):
        t_mask = true_masks[t_idx]
        pred_true_overlap = list(np.unique(pred[t_mask > 0]))
        
        if 0 in pred_true_overlap:
            pred_true_overlap.remove(0)
        
        for p_idx in pred_true_overlap:
            p_mask = pred_masks[p_idx]
            total = (t_mask + p_mask).sum()
            inter = (t_mask * p_mask).sum()
            overall_total += total
            overall_inter += inter

    return 2 * overall_inter / (overall_total + 1e-6)


def get_dice_1(true, pred):
    """Traditional dice over entire mask (binary)."""
    true = np.copy(true)
    pred = np.copy(pred)
    true[true > 0] = 1
    pred[pred > 0] = 1
    inter = (true * pred).sum()
    union = (true + pred).sum()
    return 2.0 * inter / (union + 1e-6)


def get_dice_2(true, pred):
    """Ensemble Dice as used in certain challenges (with instance-level summation)."""
    true = np.copy(true)
    pred = np.copy(pred)
    true_id = list(np.unique(true))
    pred_id = list(np.unique(pred))

    # Safely remove background ID 0 if present
    if 0 in true_id:
        true_id.remove(0)
    if 0 in pred_id:
        pred_id.remove(0)

    total_markup = 0
    total_intersect = 0
    for t in true_id:
        t_mask = (true == t).astype(np.uint8)
        for p in pred_id:
            p_mask = (pred == p).astype(np.uint8)
            intersect = (p_mask * t_mask).sum()
            if intersect > 0:
                total_intersect += intersect
                total_markup += t_mask.sum() + p_mask.sum()
    return 2 * total_intersect / (total_markup + 1e-6)


def remap_label(pred, by_size=False):
    """
    Rename all instance IDs so they are contiguous [0,1,2,3,...].
    The ordering is preserved unless by_size=True, then bigger nuclei get smaller IDs.

    If 0 isn't in the unique labels, we skip removing it.
    """
    pred_id = list(np.unique(pred))
    if 0 in pred_id:
        pred_id.remove(0)

    if len(pred_id) == 0:
        return pred  # no label

    if by_size:
        # Sort by nucleus size (descending)
        pred_size = [(inst_id, (pred == inst_id).sum()) for inst_id in pred_id]
        pred_size.sort(key=lambda x: x[1], reverse=True)
        pred_id = [x[0] for x in pred_size]

    new_pred = np.zeros_like(pred, dtype=np.int32)
    for idx, inst_id in enumerate(pred_id):
        new_pred[pred == inst_id] = idx + 1
    return new_pred


def pair_coordinates(setA, setB, radius):
    """
    Use the Hungarian algorithm to find an optimal pairing of points in setB vs setA
    using Euclidean distance as cost. Only keep pairs with distance <= radius.
    Returns (pairing, unpairedA, unpairedB).
    """
    pair_distance = scipy.spatial.distance.cdist(setA, setB, metric="euclidean")
    indicesA, paired_indicesB = linear_sum_assignment(pair_distance)
    pair_cost = pair_distance[indicesA, paired_indicesB]

    valid = pair_cost <= radius
    pairedA = indicesA[valid]
    pairedB = paired_indicesB[valid]

    pairing = np.column_stack([pairedA, pairedB])
    unpairedA = np.delete(np.arange(setA.shape[0]), pairedA)
    unpairedB = np.delete(np.arange(setB.shape[0]), pairedB)
    return pairing, unpairedA, unpairedB


def combine_masks(mask_stack):
    """
    Combine a stack of binary masks (shape: [N, H, W])
    into a single label map (shape: [H, W]) where each nucleus gets a unique label.
    """
    label_map = np.zeros(mask_stack.shape[1:], dtype=np.int32)
    for i in range(mask_stack.shape[0]):
        label_map[mask_stack[i] > 0] = i + 1
    return label_map

def cell_type_detection_scores(
    paired_true,
    paired_pred,
    unpaired_true,
    unpaired_pred,
    type_id,
    #predictions,
    #gt,
    #num_classes,
    w: List = [2, 2, 1, 1],
    exhaustive: bool = True,
):
    paired_true = np.asarray(paired_true)
    paired_pred = np.asarray(paired_pred)
    unpaired_true = np.asarray(unpaired_true)
    unpaired_pred = np.asarray(unpaired_pred)

    type_samples = (paired_true == type_id) | (paired_pred == type_id)

    paired_true = paired_true[type_samples]
    paired_pred = paired_pred[type_samples]

    tp_dt = ((paired_true == type_id) & (paired_pred == type_id)).sum()
    tn_dt = ((paired_true != type_id) & (paired_pred != type_id)).sum()
    fp_dt = ((paired_true != type_id) & (paired_pred == type_id)).sum()
    fn_dt = ((paired_true == type_id) & (paired_pred != type_id)).sum()

    if not exhaustive:
        ignore = (paired_true == -1).sum()
        fp_dt -= ignore

    fp_d = (unpaired_pred == type_id).sum()
    fn_d = (unpaired_true == type_id).sum()

    prec_type = (tp_dt + tn_dt) / (tp_dt + tn_dt + w[0] * fp_dt + w[2] * fp_d + 1e-6)
    rec_type = (tp_dt + tn_dt) / (tp_dt + tn_dt + w[1] * fn_dt + w[3] * fn_d + 1e-6)

    f1_type = (2 * (tp_dt + tn_dt)) / (
        2 * (tp_dt + tn_dt) + w[0] * fp_dt + w[1] * fn_dt + w[2] * fp_d + w[3] * fn_d + 1e-6
    )

    # Calculate Accuracy
    # accuracy = accuracy_score(paired_true, paired_pred)
    
    return f1_type, prec_type, rec_type#, accuracy

# Helper function to map fine labels to coarse labels using the hierarchy map
def get_coarse_label(fine_label, hierarchy_map):
    for coarse_label, fine_labels in hierarchy_map.items():
        if fine_label in fine_labels:
            return coarse_label
    #return None  # In case fine label is not found in the hierarchy (should not happen)
    return ValueError(f"Fine label {fine_label} not found in hierarchy_map.")

def compute_metrics_with_hierarchy(
    logits,  # Raw logits from the model
    remapped_labels,  # Ground truth remapped labels
    label_levels,  # Fine or coarse labels
    hierarchy_map,  # Hierarchy map for fine/coarse labels
    num_classes,  # Number of classes
    w: List = [2, 2, 1, 1],  # Weights for FP and FN in paired and unpaired data
):
    # Convert logits to predicted class labels using argmax
    pred_class = np.argmax(logits.cpu().detach().numpy(), axis=1)  # Move tensor to CPU, detach it, and convert to numpy  # Shape: (batch_size, num_classes)
    
    # Initialize lists to store scores
    precision_scores = []
    recall_scores = []
    f1_scores = []
    accuracies = []
    pq_scores = []

    # Loop over each sample in the batch
    for i in range(logits.shape[0]):
        # Get the ground truth for the current sample based on the remapped labels
        true_labels = remapped_labels[i]
        level = label_levels[i]  # Fine or coarse

        # Ensure true_labels is a list (in case it is a single label)
        if isinstance(true_labels, int):  # If it's a single integer, wrap it in a list
            true_labels = [true_labels]

        # If level is 'fine', use the fine labels; if 'coarse', use the coarse labels
        if level == 'fine':
            # Map fine labels to coarse labels using the hierarchy map
            true_labels = [get_coarse_label(label, hierarchy_map) for label in true_labels]
        elif level == 'coarse':
            # Coarse labels are already in the correct format
            pass

        # Convert true_labels and pred_class[i] to arrays (in case they are single values)
        true_labels = np.array(true_labels)
        pred_class_i = np.array([pred_class[i]])  # Wrap pred_class[i] in an array

        # Calculate precision, recall, f1-score for the sample
        precision = precision_score(true_labels, pred_class_i, average='macro', zero_division=0)
        recall = recall_score(true_labels, pred_class_i, average='macro', zero_division=0)
        f1 = f1_score(true_labels, pred_class_i, average='macro', zero_division=0)

        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)
        accuracies.append(accuracy_score(true_labels, pred_class_i))

        # For mPQ, calculate PQ score for each class
        sample_pq_scores = []
        for j in range(num_classes):
            # Compute PQ for class j (using get_fast_pq or equivalent function)
            pred_class_j = (pred_class[i] == j).astype(int)
            true_class_j = (true_labels == j).astype(int)

            # Calculate PQ for the current class (use a function like get_fast_pq or similar)
            if np.sum(true_class_j) == 0:  # If ground truth for this class is empty
                pq_score = np.nan
            else:
                [dq, sq, pq_score], _ = get_fast_pq(pred_class_j, true_class_j, match_iou=0.5)
            
            sample_pq_scores.append(pq_score)

        # Calculate mPQ for the sample
        mPQ_sample = np.nanmean(sample_pq_scores)
        pq_scores.append(mPQ_sample)

    # Calculate average of metrics
    avg_precision = np.mean(precision_scores)
    avg_recall = np.mean(recall_scores)
    avg_f1 = np.mean(f1_scores)
    avg_accuracy = np.mean(accuracies)
    avg_mPQ = np.mean(pq_scores)

    return avg_f1, avg_precision, avg_recall, avg_accuracy, avg_mPQ

def compute_metrics_with_hierarchy2(
    logits,
    remapped_labels,
    label_levels,
    hierarchy_map,
    num_classes,
    get_fast_pq=get_fast_pq  # Pass a callable like: get_fast_pq(pred, gt, match_iou=0.5)
):
    # Incorrect as it computes metrics for each sample instead of averaging over a batch
    pred_class = np.argmax(logits.cpu().detach().numpy(), axis=1)

    # Reverse map: fine → coarse
    sub_to_super = {
        sub: super_cls for super_cls, sub_classes in hierarchy_map.items()
        for sub in sub_classes
    }

    precision_scores, recall_scores, f1_scores, accuracies, pq_scores = [], [], [], [], []

    for i in range(logits.shape[0]):
        pred_label = pred_class[i]
        true_label = remapped_labels[i]
        level = label_levels[i]

        if level == 'coarse':
            # Convert predicted fine label to coarse
            pred_label = sub_to_super.get(pred_label, -1)
            if pred_label == -1:
                raise ValueError(f"Prediction {pred_class[i]} not in hierarchy.")
            y_true = np.array([true_label])
            y_pred = np.array([pred_label])
        else:  # 'fine'
            y_true = np.array([true_label])
            y_pred = np.array([pred_class[i]])

        # Compute basic metrics
        precision_scores.append(precision_score(y_true, y_pred, average='macro', zero_division=0))
        recall_scores.append(recall_score(y_true, y_pred, average='macro', zero_division=0))
        f1_scores.append(f1_score(y_true, y_pred, average='macro', zero_division=0))
        accuracies.append(accuracy_score(y_true, y_pred))

        # Compute mPQ if callable is provided
        if get_fast_pq:
            sample_pq_scores = []
            for j in range(num_classes):
                pred_binary = np.array((pred_class[i] == j)).astype(int)
                true_binary = np.array((true_label == j)).astype(int)

                if np.sum(true_binary) == 0:
                    pq = np.nan
                else:
                    [_, _, pq], _ = get_fast_pq(pred_binary, true_binary, match_iou=0.5)

                sample_pq_scores.append(pq)

            pq_scores.append(np.nanmean(sample_pq_scores))

    avg_f1 = np.mean(f1_scores)
    avg_precision = np.mean(precision_scores)
    avg_recall = np.mean(recall_scores)
    avg_accuracy = np.mean(accuracies)
    avg_mPQ = np.mean(pq_scores) if pq_scores else None

    return avg_f1, avg_precision, avg_recall, avg_accuracy, avg_mPQ

def compute_metrics_with_hierarchy3(
    logits,
    remapped_labels,
    label_levels,
    hierarchy_map,
    num_classes,
    get_fast_pq=get_fast_pq  # Optional callable
):
    # Reverse map: fine → coarse
    sub_to_super = {
        sub: super_cls for super_cls, sub_classes in hierarchy_map.items()
        for sub in sub_classes
    }

    pred_class = np.argmax(logits.cpu().detach().numpy(), axis=1)

    all_preds = []
    all_targets = []

    for i in range(logits.shape[0]):
        pred_label = pred_class[i]
        true_label = remapped_labels[i]
        level = label_levels[i]

        if level == "coarse":
            # Map predicted fine class to its coarse class
            pred_label = sub_to_super.get(pred_label, -1)
            if pred_label == -1:
                raise ValueError(f"Prediction {pred_class[i]} not in hierarchy.")
            # true_label is already coarse
        else:
            # Keep both at fine level
            pass

        all_preds.append(pred_label)
        all_targets.append(true_label)

    # Compute batch-level metrics
    precision = precision_score(all_targets, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_targets, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    accuracy = accuracy_score(all_targets, all_preds)

    # mPQ section (optional)
    pq_scores = []
    if get_fast_pq:
        for j in range(num_classes):
            pred_binary = np.array([int(p == j) for p in all_preds])
            true_binary = np.array([int(t == j) for t in all_targets])

            if np.sum(true_binary) == 0:
                pq = np.nan
            else:
                [_, _, pq], _ = get_fast_pq(pred_binary, true_binary, match_iou=0.3)

            pq_scores.append(pq)

    avg_mPQ = np.nanmean(pq_scores) if pq_scores else None

    return f1, precision, recall, accuracy, avg_mPQ


def process_one_mask(true_path):
    """
    Reads the annotation mask (true_path) and its corresponding pred_path,
    computes metrics, and returns [stem, dq, sq, pq, aji].
    Returns None if something fails.
    
    """
    # print("ProcessONeMask")
    try:
        stem = Path(true_path).stem
        # Skip if not in patient_ids
        if stem not in patient_ids:
            return None

        # Load predicted mask
        # pred_path = f'/workspace/yolo-nuke/Segmentation/exp7/{stem}.npy'
        pred_path = f'{out_dir}/{stem}.npy'
        if not os.path.isfile(pred_path):
            print(f"No predicted mask found for: {stem}")
            return None

        pred_mask = np.load(pred_path)
        true_mask = np.load(true_path)

        # Example computations
        intersection = np.logical_and(pred_mask > 0, true_mask > 0)
        union = np.logical_or(pred_mask > 0, true_mask > 0)

        iou = intersection.sum() / union.sum() if union.sum() > 0 else 0.0
        dice = (2 * intersection.sum()) / ((pred_mask > 0).sum() + (true_mask > 0).sum() + 1e-6)

        # PQ, DQ, SQ
        [dq, sq, pq], _ = get_fast_pq(true_mask, pred_mask, match_iou=0.4)
        # print(pq)
        aji = get_fast_aji_plus(true_mask, pred_mask)
        # print("AJI Pass")
        print(f'{stem} -- DQ={dq:.4f} -- SQ={sq:.4f} -- PQ={pq:.4f} -- AJI={aji:.4f} PredSize={pred_mask.shape} GTSize={true_mask.shape}\n')

        return [stem, dq, sq, pq, aji]

    except Exception as e:
        print(f"Error with {true_path}: {e}")
        return None

def process_one_mask_from_arrays(true_mask, pred_mask, stem="unknown"):
    try:
        intersection = np.logical_and(pred_mask > 0, true_mask > 0)
        union = np.logical_or(pred_mask > 0, true_mask > 0)

        iou = intersection.sum() / union.sum() if union.sum() > 0 else 0.0
        dice = (2 * intersection.sum()) / ((pred_mask > 0).sum() + (true_mask > 0).sum() + 1e-6)

        [dq, sq, pq], _ = get_fast_pq(true_mask, pred_mask, match_iou=0.4)
        aji = get_fast_aji_plus(true_mask, pred_mask)

        #print(f'{stem} -- DQ={dq:.4f} -- SQ={sq:.4f} -- PQ={pq:.4f} -- AJI={aji:.4f}')
        return dq, sq, pq, aji

    except Exception as e:
        print(f"Error with {stem}: {e}")
        return None

