"""Runnable blank algorithm interface. The all-free output is only a placeholder."""

import numpy as np


def load_model(model_dir, device):
    """Load once; model_dir is the method's weights/ directory, device cpu/cuda."""
    return dict(model_dir=str(model_dir), device=device)


def predict_sample(model, inputs, device):
    """Implement your own independent prediction using only this input dictionary.

    NumPy float32: rgb [3,3,H,W] in [0,1], depth_m [3,1,H,W] in metres
    (zero invalid) or None, intrinsics [3,3],
    t_world_camera [3,4,4], t_world_anchor [4,4]. Return integer [40,14,80]
    in z,y,x order, classes 0-9. No target, sample ID, or prior sample is input.
    """
    return np.zeros((40, 14, 80), dtype=np.int8)
