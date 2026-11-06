"""Reference supervision and metrics for the fixed three-view benchmark."""

import numpy as np
import torch
from torch.nn import functional as F


def class_weights_from_histogram(histogram, maximum=5.0):
    """Inverse square-root frequency; absent classes get zero; normalize twice."""
    # BEGIN YOUR CODE: class_weights_from_histogram
    return torch.ones(len(histogram), dtype=torch.float32)
    # END YOUR CODE


def completion_loss(
    logits,
    labels,
    completion_mask,
    output_mask,
    *,
    class_weights=None,
    occupancy_loss_weight=0.5,
    observed_consistency_weight=0.1,
):
    """Return (differentiable scalar, detached semantic/occupancy/observed values).

    Logits [B,10,Z,Y,X], labels/masks [B,Z,Y,X]; label -1 is ignored.
    Consistency is measured on output & ~completion; only completion is ranked.
    An empty completion mask contributes differentiable zero to its two terms.
    """
    # BEGIN YOUR CODE: completion_loss
    return logits.sum() * 0, dict(semantic=0.0, occupancy=0.0, observed=0.0)
    # END YOUR CODE


def metrics_from_confusion(confusion):
    """Compute target-present class mIoU (including free), accuracy and occupied IoU.

    Confusion is [target,prediction]. Pool samples WITHIN a house before calling;
    arithmetic-mean the resulting house scores, not sample scores, afterwards.
    """
    # BEGIN YOUR CODE: metrics_from_confusion
    return dict(
        completion_accuracy=0.0,
        completion_mean_iou=0.0,
        completion_occupied_iou=0.0,
        class_iou=[0.0] * len(confusion),
        class_support=[0] * len(confusion),
    )
    # END YOUR CODE
