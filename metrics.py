"""Segmentation metrics — IoU, Dice, Recall, Precision, Specificity, Accuracy.

Ported verbatim from ``KMUNet/code/metrics_2.py`` so reported numbers are
directly comparable to the KMUNet paper.
"""

from typing import Dict, List

import numpy as np
import torch


def calculate_metrics(pred: torch.Tensor, target: torch.Tensor,
                      num_classes: int) -> Dict[str, List[float]]:
    """Per-class metric computation.

    Args:
        pred: ``(B, H, W)`` integer predictions.
        target: ``(B, H, W)`` integer labels.
        num_classes: number of classes.

    Returns:
        Dict with keys ``iou, accuracy, dice, recall, precision, specificity``,
        each a list with one entry per class.
    """
    iou_list: List[float] = []
    accuracy_list: List[float] = []
    dice_list: List[float] = []
    recall_list: List[float] = []
    precision_list: List[float] = []
    specificity_list: List[float] = []

    for class_id in range(num_classes):
        pred_class = (pred == class_id).cpu().numpy()
        target_class = (target == class_id).cpu().numpy()

        tp = np.sum(np.logical_and(pred_class == 1, target_class == 1))
        fp = np.sum(np.logical_and(pred_class == 1, target_class == 0))
        fn = np.sum(np.logical_and(pred_class == 0, target_class == 1))
        tn = np.sum(np.logical_and(pred_class == 0, target_class == 0))

        iou_list.append(tp / (tp + fp + fn + 1e-7))

        correct = np.sum(pred_class == target_class)
        total = np.prod(target_class.shape)
        accuracy_list.append(correct / total)

        dice_list.append(2 * tp / (2 * tp + fp + fn + 1e-7))
        recall_list.append(tp / (tp + fn + 1e-7))
        precision_list.append(tp / (tp + fp + 1e-7))
        specificity_list.append(tn / (tn + fp + 1e-7))

    return {
        "iou": iou_list,
        "accuracy": accuracy_list,
        "dice": dice_list,
        "recall": recall_list,
        "precision": precision_list,
        "specificity": specificity_list,
    }
