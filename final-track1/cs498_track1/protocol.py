"""Provided, model-independent benchmark contract. Do not change for submission."""

from __future__ import annotations

import ast
import base64
import hashlib
import json
import textwrap
import zlib
from pathlib import Path

import numpy as np

TASK = "three_view_semantic_completion"
POLICY = "three_view_occlusion_room_v1"
SHAPE = (40, 14, 80)  # z, y, x; C-order serialization
VIEWS = ("front_left", "front", "front_right")
CLASSES = (
    "free",
    "floor",
    "wall_ceiling",
    "opening",
    "furniture",
    "appliance",
    "plumbing_fixture",
    "electronics",
    "small_object",
    "other",
)
COMPETITION = "cs498-fall-2026-multiview-occupancy"
MODEL_CONFIG = {
    "triview_model": {
        "name": "lift_fuse_3d_unet",
        "use_depth": True,
        "image_feature_channels": 24,
        "volume_feature_channels": 24,
        "base_channels": 24,
        "depth_truncation_m": 0.5,
        "volume": {
            "x_range_m": [-10.0, 10.0],
            "y_range_m": [-1.75, 1.75],
            "z_range_m": [0.0, 10.0],
            "voxel_size_m": 0.25,
        },
    }
}


def expected_sample_ids(split):
    """Return the immutable ordered coverage contract, including empty samples."""
    if split not in {"train", "val", "test"}:
        raise ValueError("Unknown split")
    houses, samples = (200, 150) if split == "train" else (10, 15)
    return [
        f"{split}_house_{h:03}__sample_{s:05}"
        for h in range(houses)
        for s in range(samples)
    ]


def sha256(path):
    """Hash a file without loading a large checkpoint into memory."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def implementation_fingerprint(source):
    """Hash a function portably across supported Python 3.11/3.12 AST schemas."""
    node = ast.parse(textwrap.dedent(source)).body[0]
    for child in ast.walk(node):
        # Python 3.12 adds this empty field even for ordinary non-generic defs.
        # Removing only empty values preserves detection of real code edits.
        if getattr(child, "type_params", None) == []:
            del child.type_params
    return hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()


def tree_hash(root):
    """Hash filenames and contents, excluding only bytecode caches."""
    h = hashlib.sha256()
    for p in sorted(Path(root).rglob("*")):
        if p.is_symlink():
            raise ValueError(f"Symlink not allowed: {p}")
        if not p.is_file() or "__pycache__" in p.parts or p.suffix == ".pyc":
            continue
        h.update(
            p.relative_to(root).as_posix().encode() + b"\0" + bytes.fromhex(sha256(p))
        )
    return h.hexdigest()


def write_json(path, value):
    """Write a complete UTF-8 JSON artifact, creating its parent directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def validate_prediction(value):
    """Require exactly the canonical integer grid; unknown predictions are illegal."""
    a = np.asarray(value)
    if a.shape != SHAPE or a.dtype.kind not in "iu" or a.min() < 0 or a.max() > 9:
        raise ValueError(
            f"Prediction must be integer {SHAPE} in [0,9]; got {a.shape}/{a.dtype}"
        )
    return a.astype(np.int8, copy=False)


def encode_labels(value):
    """Transport a grid as base64(zlib(int8 C-order bytes)); not encryption."""
    a = np.asarray(value)
    if a.shape != SHAPE or a.dtype.kind not in "iu" or a.min() < -1 or a.max() > 9:
        raise ValueError("Invalid grid")
    return base64.b64encode(
        zlib.compress(a.astype(np.int8).tobytes(order="C"), 9)
    ).decode("ascii")


def decode_labels(value, allow_masked=False):
    """Bound decoding and reject extra streams, incorrect shapes and labels."""
    n = int(np.prod(SHAPE))
    if not isinstance(value, str) or len(value) > 4 * n:
        raise ValueError("Invalid encoded length")
    try:
        raw = base64.b64decode(value, validate=True)
    except Exception as e:
        raise ValueError("Invalid Base64") from e
    dec = zlib.decompressobj()
    try:
        data = dec.decompress(raw, n + 1)
    except zlib.error as e:
        raise ValueError("Invalid zlib") from e
    if dec.unconsumed_tail or not dec.eof or dec.unused_data or len(data) != n:
        raise ValueError("Invalid decoded length/stream")
    a = np.frombuffer(data, dtype=np.int8).reshape(SHAPE)
    if a.min() < (-1 if allow_masked else 0) or a.max() > 9:
        raise ValueError("Invalid labels")
    return a


def confusion_metrics(confusion):
    """Trusted canonical metric, independent of student TODOs."""
    c = np.asarray(confusion, dtype=np.float64)
    target, predicted = c.sum(1), c.sum(0)
    intersection = np.diag(c)
    union = target + predicted - intersection
    iou = np.divide(intersection, union, out=np.zeros(10), where=union > 0)
    oi = c[1:, 1:].sum()
    ou = target[1:].sum() + predicted[1:].sum() - oi
    return {
        "completion_accuracy": float(intersection.sum() / max(c.sum(), 1)),
        "completion_mean_iou": float(iou[target > 0].mean())
        if (target > 0).any()
        else 0.0,
        "completion_occupied_iou": float(oi / max(ou, 1)),
        "class_iou": iou.tolist(),
        "class_support": target.astype(int).tolist(),
    }


def summarize(confusions):
    """Pool samples within houses, then give each house equal weight."""
    by_house = {h: confusion_metrics(c) for h, c in sorted(confusions.items())}
    names = ("completion_accuracy", "completion_mean_iou", "completion_occupied_iou")
    return {
        "default_aggregation": "macro_by_house",
        "by_house": by_house,
        "metrics": {
            k: float(np.mean([v[k] for v in by_house.values()])) for k in names
        },
    }
