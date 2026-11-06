"""Shared utilities for CS498 HW4.

This file contains provided loading, configuration, metric, and visualization
helpers. Student implementations live in the task files.
"""

from __future__ import annotations

import copy
import json
import random
from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


SMOKE_CONFIG: dict[str, object] = {
    "data_dir": "data/nerf_synthetic/lego",
    "output_dir": "outputs/task3/debug",
    "seed": 0,
    "device": "cpu",
    "white_background": True,
    "image_downscale": 2,
    "train_skip": 5,
    "val_skip": 5,
    "test_skip": 6,
    "num_freqs_xyz": 6,
    "num_freqs_dir": 3,
    "hidden_dim": 64,
    "num_layers": 4,
    "skip_layer": 2,
    "use_viewdirs": True,
    "near": 2.0,
    "far": 6.0,
    "num_samples": 32,
    "perturb": True,
    "batch_size": 512,
    "lr": 5.0e-4,
    "iters": 100,
    "log_every": 20,
    "val_every": 50,
    "checkpoint_every": 0,
    "num_val_views": 1,
    "metrics_plot_every": 100,
    "render_chunk": 4096,
}


TRAIN_CONFIG: dict[str, object] = {
    **SMOKE_CONFIG,
    "output_dir": "outputs/task3/lego_train",
    "device": "cuda",
    "image_downscale": 1,
    "train_skip": 1,
    "val_skip": 1,
    "test_skip": 1,
    "num_freqs_xyz": 8,
    "num_freqs_dir": 4,
    "hidden_dim": 128,
    "num_layers": 6,
    "skip_layer": 3,
    "num_samples": 64,
    "batch_size": 1024,
    "iters": 10000,
    "log_every": 50,
    "val_every": 2500,
    "checkpoint_every": 2500,
    "num_val_views": 2,
    "metrics_plot_every": 500,
    "render_chunk": 8192,
}


@dataclass
class BlenderScene:
    """Loaded RGB images, camera poses, intrinsics, and split membership."""

    images: torch.Tensor
    poses: torch.Tensor
    height: int
    width: int
    focal: float
    split_indices: dict[str, np.ndarray]


@dataclass
class Mesh:
    """Triangle mesh represented by world-space vertices and index triplets."""

    vertices: np.ndarray
    faces: np.ndarray


def default_config(kind: str = "smoke") -> dict[str, object]:
    """Return a mutable config dictionary for the requested run kind."""
    if kind == "smoke":
        return copy.deepcopy(SMOKE_CONFIG)
    if kind == "train":
        return copy.deepcopy(TRAIN_CONFIG)
    raise ValueError(f"unknown config kind {kind!r}")


def resolve_device(device_cfg: str) -> torch.device:
    """Resolve a requested device, falling back from CUDA when unavailable."""
    if device_cfg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_cfg == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_cfg)


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _load_rgba(path: Path, white_background: bool) -> np.ndarray:
    """Load an image as float RGB and composite any alpha channel."""
    image = imageio.imread(path).astype(np.float32) / 255.0
    if image.shape[-1] == 4:
        rgb = image[..., :3]
        alpha = image[..., 3:4]
        background = 1.0 if white_background else 0.0
        image = rgb * alpha + background * (1.0 - alpha)
    else:
        image = image[..., :3]
    return image.astype(np.float32)


def _resize_images(images: torch.Tensor, downscale: int) -> torch.Tensor:
    """Area-downsample a batch of channel-last RGB images."""
    if downscale == 1:
        return images
    images_nchw = images.permute(0, 3, 1, 2)
    new_hw = (images.shape[1] // downscale, images.shape[2] // downscale)
    resized = F.interpolate(images_nchw, size=new_hw, mode="area")
    return resized.permute(0, 2, 3, 1).contiguous()


def load_blender_scene(
    data_dir: str | Path,
    image_downscale: int = 4,
    white_background: bool = True,
    train_skip: int = 1,
    val_skip: int = 1,
    test_skip: int = 1,
) -> BlenderScene:
    """Load the Blender-format Lego scene used in the assignment."""
    data_dir = Path(data_dir)
    split_skips = {"train": train_skip, "val": val_skip, "test": test_skip}
    all_images: list[np.ndarray] = []
    all_poses: list[np.ndarray] = []
    split_indices: dict[str, np.ndarray] = {}
    cursor = 0
    camera_angle_x = None

    for split in ["train", "val", "test"]:
        meta_path = data_dir / f"transforms_{split}.json"
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        camera_angle_x = float(meta["camera_angle_x"])
        frames = meta["frames"][:: max(1, int(split_skips[split]))]
        for frame in frames:
            img_path = data_dir / f"{frame['file_path']}.png"
            all_images.append(_load_rgba(img_path, white_background))
            all_poses.append(np.asarray(frame["transform_matrix"], dtype=np.float32))
        split_indices[split] = np.arange(cursor, cursor + len(frames))
        cursor += len(frames)

    images = torch.from_numpy(np.stack(all_images, axis=0))
    poses = torch.from_numpy(np.stack(all_poses, axis=0))
    images = _resize_images(images, int(image_downscale))
    height, width = int(images.shape[1]), int(images.shape[2])
    assert camera_angle_x is not None
    focal = 0.5 * width / np.tan(0.5 * float(camera_angle_x))
    return BlenderScene(images, poses, height, width, float(focal), split_indices)


def mse_to_psnr(mse: torch.Tensor) -> torch.Tensor:
    """Convert mean squared error in the unit range to PSNR in decibels."""
    return -10.0 * torch.log10(torch.clamp(mse, min=1.0e-10))


def img_to_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return the mean squared error between predicted and target images."""
    return torch.mean((pred - target) ** 2)


def to8b(image: np.ndarray) -> np.ndarray:
    """Clamp a float image to the unit range and convert it to uint8."""
    return (255.0 * np.clip(image, 0.0, 1.0)).astype(np.uint8)


def save_image(path: str | Path, image: np.ndarray) -> None:
    """Write a float RGB image to disk as uint8."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(path, to8b(image))


def save_video(path: str | Path, frames: np.ndarray, fps: int = 24) -> None:
    """Encode float RGB frames as a video with stable frame dimensions."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(path, to8b(frames), fps=fps, quality=8, macro_block_size=8)


def save_contact_sheet(path: str | Path, images: np.ndarray, cols: int = 5) -> None:
    """Arrange RGB images in a compact grid and save the resulting figure."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(images)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.2 * cols, 2.2 * rows))
    axes = np.asarray(axes).reshape(rows, cols)
    for ax in axes.ravel():
        ax.axis("off")
    for i, image in enumerate(images):
        axes.ravel()[i].imshow(np.clip(image, 0.0, 1.0))
    fig.tight_layout(pad=0.1)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_metrics_csv(path: str | Path, rows: list[dict[str, float]]) -> None:
    """Write training step, loss, and PSNR records to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("step,loss,psnr\n")
        for row in rows:
            f.write(f"{int(row['step'])},{row['loss']:.8f},{row['psnr']:.6f}\n")


def save_psnr_curve(path: str | Path, rows: list[dict[str, float]]) -> None:
    """Plot training PSNR against optimization step."""
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4))
    plt.plot([row["step"] for row in rows], [row["psnr"] for row in rows], linewidth=1.5)
    plt.xlabel("Training iteration")
    plt.ylabel("Training PSNR (dB)")
    plt.title("Training PSNR")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_checkpoint(path: str | Path, model: torch.nn.Module, cfg: dict[str, object]) -> None:
    """Save model parameters together with the run configuration."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "cfg": cfg}, path)
