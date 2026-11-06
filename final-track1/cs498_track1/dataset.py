"""Provided loaders: inference inputs and supervision have separate APIs."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from .protocol import POLICY, SHAPE, VIEWS
from torch.utils.data import Dataset, Sampler


def sample_paths(root, split, houses=None):
    """Discover samples, optionally restricting exact house IDs."""
    paths = sorted(Path(root).glob(f"samples/{split}_house_*/sample_*/manifest.json"))
    if houses:
        paths = [p for p in paths if p.parent.parent.name in set(houses)]
    if not paths:
        raise FileNotFoundError(
            f"No {split} samples: run python run.py download --split {split} first"
        )
    return [p.parent for p in paths]


def load_inputs(sample):
    """Read only RGB, optional depth and calibration; never read targets.

    RGB: float32 [3,3,H,W] in [0,1]. Depth: z-depth metres [3,1,H,W],
    zero for invalid/out-of-range values. Poses map camera coordinates to world.
    """
    p = Path(sample)
    m = json.loads((p / "manifest.json").read_text())
    if m["target_policy"] != POLICY or tuple(m["view_names"]) != VIEWS:
        raise ValueError("Data contract mismatch")
    rgb = np.stack(
        [
            np.asarray(
                Image.open(p / "rgb" / f"{v}.png").convert("RGB"), dtype=np.float32
            ).transpose(2, 0, 1)
            / 255
            for v in VIEWS
        ]
    )
    depth = None
    if (p / "depth").is_dir():
        d = (
            np.stack(
                [
                    np.asarray(Image.open(p / "depth" / f"{v}.png"), dtype=np.float32)
                    for v in VIEWS
                ]
            )
            / m["depth_units_per_meter"]
        )
        depth = np.where(np.isfinite(d) & (d >= 0.1) & (d <= 10), d, 0)[:, None]
    return {
        "rgb": rgb,
        "depth_m": depth,
        "intrinsics": np.load(p / "K.npy", allow_pickle=False).astype(np.float32),
        "t_world_camera": np.load(p / "T_world_camera.npy", allow_pickle=False).astype(
            np.float32
        ),
        "t_world_anchor": np.load(p / "T_world_anchor.npy", allow_pickle=False).astype(
            np.float32
        ),
    }


def load_target(sample):
    """Read prepared 25 cm supervision; Test targets are never publicly released."""
    with np.load(Path(sample) / "target.npz", allow_pickle=False) as f:
        out = {k: f[k].copy() for k in ("labels", "completion_mask", "output_mask")}
    if any(a.shape != SHAPE for a in out.values()):
        raise ValueError("Target shape mismatch")
    mask = out["completion_mask"]
    if np.any(mask & ~out["output_mask"]):
        raise ValueError("Invalid completion mask")
    return out


class ThreeViewDataset(Dataset):
    """Supervised loader; inputs-only inference uses load_inputs instead."""

    def __init__(self, paths, use_depth=True):
        """Keep lightweight paths; no simulator or target conversion is needed."""
        self.paths = list(paths)
        self.use_depth = use_depth

    def __len__(self):
        """Return the number of independent samples."""
        return len(self.paths)

    def __getitem__(self, index):
        """Combine inputs and supervision only for training."""
        p = self.paths[index]
        values = load_inputs(p)
        if not self.use_depth:
            values.pop("depth_m")
        elif values["depth_m"] is None:
            raise ValueError("RGB-D baseline requires depth")
        values.update(load_target(p))
        result = {
            k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in values.items()
        }
        result.update(house_id=p.parent.name, sample_id=p.name)
        return result


class HouseBatchSampler(Sampler):
    """Deterministic shuffled house-local batches, matching the reference sampler."""

    def __init__(self, paths, batch_size, seed=498):
        """Group indices by house and retain final partial batches."""
        self.batch_size, self.seed, self.epoch = batch_size, seed, 0
        self.groups = {}
        for i, p in enumerate(paths):
            self.groups.setdefault(p.parent.name, []).append(i)

    def __len__(self):
        """Count batches including the final partial batch of each house."""
        return sum(
            (len(x) + self.batch_size - 1) // self.batch_size
            for x in self.groups.values()
        )

    def __iter__(self):
        """Shuffle within houses and across batches; never carry prediction state."""
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        batches = []
        for values in self.groups.values():
            values = values.copy()
            rng.shuffle(values)
            batches.extend(
                values[i : i + self.batch_size]
                for i in range(0, len(values), self.batch_size)
            )
        rng.shuffle(batches)
        yield from batches
