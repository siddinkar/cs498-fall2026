"""Stateless three-view baseline adapter; predictions never read annotations."""

from pathlib import Path

import numpy as np
import torch
from semocc.model import build_triview_model


def load_model(model_dir, device):
    """Load the packaged model specification and learned state once."""
    path = Path(model_dir) / "checkpoint_best.pt"
    if not path.is_file():
        raise FileNotFoundError("Train first: weights/checkpoint_best.pt is missing")
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    network = build_triview_model(ckpt["model_config"], 9)
    network.load_state_dict(ckpt["model_state"])
    return network.to(device).eval()


@torch.inference_mode()
def predict_sample(model, inputs, device):
    """Return int8 [40,14,80] labels for this sample only, without persistent state."""

    def tensor(key):
        """Convert one input array to a batched tensor on the selected device."""
        value = inputs[key]
        return (
            None
            if value is None
            else torch.from_numpy(np.ascontiguousarray(value)).to(device)[None]
        )

    depth = tensor("depth_m") if model.use_depth else None
    with torch.autocast(
        device_type=torch.device(device).type,
        dtype=torch.float16,
        enabled=torch.device(device).type == "cuda",
    ):
        logits = model(
            tensor("rgb"),
            depth,
            tensor("intrinsics"),
            tensor("t_world_camera"),
            tensor("t_world_anchor"),
        )
    return logits.argmax(1)[0].cpu().numpy().astype(np.int8)
