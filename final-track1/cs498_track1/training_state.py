"""Provided epoch-boundary recovery; load only your own training checkpoints."""

import random
import json
from pathlib import Path

import numpy as np
import torch

from .protocol import POLICY, sha256


def atomic_save(value, path):
    """Replace a complete checkpoint only after torch.save succeeds."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    torch.save(value, temporary)
    temporary.replace(path)


def random_state(cuda=False):
    """Use tensors/primitives so recovery supports weights_only=True loading."""
    numpy = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": [numpy[0], torch.tensor(numpy[1].astype(np.int64)), *numpy[2:]],
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if cuda else [],
    }


def restore_random_state(state):
    """Restore randomness after constructing/loading all training components."""
    random.setstate(state["python"])
    numpy = state["numpy"]
    np.random.set_state((numpy[0], numpy[1].numpy().astype(np.uint32), *numpy[2:]))
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        if (
            not torch.cuda.is_available()
            or len(state["cuda"]) != torch.cuda.device_count()
        ):
            raise ValueError("Resume needs the same CUDA device count")
        torch.cuda.set_rng_state_all(state["cuda"])


def signature(source, config, train_paths, val_paths, args, epochs, device):
    """Reject changes that would silently alter data, model or LR trajectory."""
    return {
        "target_policy": POLICY,
        "data_receipts": {
            split: json.loads(path.read_text()) if path.is_file() else None
            for split in ("train", "val")
            for path in [args.data_root / "download_receipts" / f"{split}.json"]
        },
        "code": {
            p.relative_to(source).as_posix(): sha256(p)
            for p in sorted(source.rglob("*.py"))
        },
        "model_config": config,
        "train_ids": [p.parent.name + "__" + p.name for p in train_paths],
        "val_ids": [p.parent.name + "__" + p.name for p in val_paths],
        "epochs": epochs,
        "batch_size": args.batch_size,
        "smoke": args.smoke,
        "device": device,
        "torch_version": str(torch.__version__),
    }
