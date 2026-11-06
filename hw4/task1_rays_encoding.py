"""Task 1: camera rays and positional encoding."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import torch


def get_rays(height: int, width: int, focal: float, c2w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return one world-space ray per image pixel.

    Args:
        height: Image height ``H`` in pixels. This is the number of pixel rows
            and determines the first dimension of the returned ray tensors.
        width: Image width ``W`` in pixels. This is the number of pixel columns
            and determines the second dimension of the returned ray tensors.
        focal: Camera focal length, measured in pixels. Use this scalar to
            convert pixel offsets from the image center into camera-space ray
            directions.
        c2w: Camera-to-world transform with shape ``(4, 4)`` or ``(3, 4)``.
            The upper-left ``(3, 3)`` block is the camera-to-world rotation, and
            the last column ``c2w[:3, 3]`` is the camera center in world
            coordinates. The returned tensors should use this tensor's dtype and
            device.

    Returns:
        A tuple ``(rays_o, rays_d)``:
        ``rays_o`` has shape ``(H, W, 3)`` and stores the world-space origin of
        each pixel ray. Every entry is the same camera center.
        ``rays_d`` has shape ``(H, W, 3)`` and stores the world-space direction
        of each pixel ray. Directions should not be normalized by this function.
    """
    # ======= INSERT YOUR CODE BELOW =======
    # TODO(student): implement pinhole ray generation.
    #
    # 1. Build i, j pixel grids with torch.meshgrid(..., indexing="xy").
    # 2. Form camera-space directions:
    #      [(i - width * 0.5) / focal,
    #       -(j - height * 0.5) / focal,
    #       -1]
    # 3. Rotate directions by c2w[:3, :3].
    # 4. Broadcast c2w[:3, 3] to all pixels for ray origins.
    #
    # Do not normalize directions here. The renderer uses ||rays_d|| when
    # converting density to opacity.
    device = c2w.device
    dtype = c2w.dtype
    return torch.zeros((height, width, 3), device=device, dtype=dtype), torch.zeros(
        (height, width, 3), device=device, dtype=dtype
    )
    # ======= DO NOT MODIFY CODE OUTSIDE THIS BLOCK =======


def pose_spherical(theta_deg: float, phi_deg: float, radius: float, device: torch.device | None = None) -> torch.Tensor:
    """Return a homogeneous camera-to-world pose on a viewing sphere.

    Args:
        theta_deg: Azimuth angle in degrees. This rotates the camera around the
            world ``y`` axis after the elevation rotation is applied.
        phi_deg: Elevation angle in degrees. This rotates the camera around the
            camera/world ``x`` axis depending on the multiplication order shown
            in the handout.
        radius: Distance from the world origin to the camera center. The pose
            should first translate the camera by this amount along camera ``z``.
        device: Optional torch device for the returned tensor, such as
            ``torch.device("cpu")`` or ``torch.device("cuda")``.

    Returns:
        A float tensor with shape ``(4, 4)``. The upper-left ``(3, 3)`` block is
        the camera-to-world rotation, the last column ``[:3, 3]`` is the
        world-space camera center, and the final row is homogeneous
        coordinates ``[0, 0, 0, 1]``.
    """
    theta = math.radians(theta_deg)
    phi = math.radians(phi_deg)

    translate = torch.eye(4, device=device)
    translate[2, 3] = radius

    rot_phi = torch.eye(4, device=device)
    rot_phi[1, 1] = math.cos(phi)
    rot_phi[1, 2] = -math.sin(phi)
    rot_phi[2, 1] = math.sin(phi)
    rot_phi[2, 2] = math.cos(phi)

    rot_theta = torch.eye(4, device=device)
    rot_theta[0, 0] = math.cos(theta)
    rot_theta[0, 2] = -math.sin(theta)
    rot_theta[2, 0] = math.sin(theta)
    rot_theta[2, 2] = math.cos(theta)

    basis = torch.tensor(
        [
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        device=device,
    )
    return basis @ rot_theta @ rot_phi @ translate


def positional_encoding(
    x: torch.Tensor,
    num_freqs: int,
    include_input: bool = True,
    log_sampling: bool = True,
) -> torch.Tensor:
    """Apply NeRF sinusoidal positional encoding to the last dimension of x.

    Args:
        x: Input coordinates or directions with shape ``(..., D)``. ``D`` is
            usually ``3`` for xyz positions or view directions, and all leading
            dimensions ``...`` should be preserved. The output should stay on
            the same device and use the same dtype as ``x``.
        num_freqs: Number of sinusoidal frequency bands ``L`` to append.
        include_input: If ``True``, include the original ``x`` values as the
            first ``D`` output channels. If ``False``, return only sine/cosine
            features.
        log_sampling: If ``True``, use exponentially spaced frequencies
            ``1, 2, 4, ..., 2**(L-1)``. If ``False``, use linearly spaced
            frequencies from ``1`` to ``2**(L-1)``.

    Returns:
        Encoded tensor with shape ``(..., D * (2 * L + int(include_input)))``.
        For each frequency, concatenate ``sin(freq * x)`` and
        ``cos(freq * x)`` along the final dimension only.
    """
    # ======= INSERT YOUR CODE BELOW =======
    # TODO(student): concatenate [x, sin(freq*x), cos(freq*x), ...].
    #
    # For log_sampling=True, use frequencies 1, 2, 4, ..., 2**(num_freqs-1).
    # For log_sampling=False, use torch.linspace(1, 2**(num_freqs-1), ...).
    # Preserve every leading dimension and concatenate only on the final axis.
    output_dim = encoded_dim(x.shape[-1], num_freqs, include_input)
    return torch.zeros((*x.shape[:-1], output_dim), device=x.device, dtype=x.dtype)
    # ======= DO NOT MODIFY CODE OUTSIDE THIS BLOCK =======


def encoded_dim(input_dim: int, num_freqs: int, include_input: bool = True) -> int:
    """Return the feature width produced by positional_encoding.

    Args:
        input_dim: Size ``D`` of the final dimension passed into
            ``positional_encoding``.
        num_freqs: Number of sinusoidal frequency bands ``L`` used by
            ``positional_encoding``.
        include_input: Whether the original ``D`` input channels are included
            in the encoded output.

    Returns:
        Integer output width ``D * (2 * L + int(include_input))``.
    """
    return input_dim * (2 * num_freqs + int(include_input))


def save_ray_debug(path: str | Path, height: int = 64, width: int = 64, focal: float = 70.0) -> None:
    """Save a false-color visualization of normalized ray directions."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    c2w = pose_spherical(0.0, -30.0, 4.0)
    _, rays_d = get_rays(height, width, focal, c2w)
    norm = torch.linalg.norm(rays_d, dim=-1, keepdim=True).clamp_min(1.0e-8)
    directions = rays_d / norm
    image = 0.5 * (directions.detach().cpu().numpy() + 1.0)
    plt.imsave(path, image)
    print(f"saved {path}")


def _direction_rgb(rays_d: torch.Tensor) -> torch.Tensor:
    """Map normalized xyz ray directions into RGB values for visualization."""
    norm = torch.linalg.norm(rays_d, dim=-1, keepdim=True).clamp_min(1.0e-8)
    return (0.5 * (rays_d / norm + 1.0)).clamp(0.0, 1.0)


def save_task1_visual_checkpoint(path: str | Path, height: int = 64, width: int = 64, focal: float = 70.0) -> None:
    """Save the Task 1 visual checkpoint for rays and positional encoding."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    identity_pose = torch.eye(4)
    _, identity_dirs = get_rays(height, width, focal, identity_pose)
    turntable_pose = pose_spherical(45.0, -30.0, 4.0)
    _, turntable_dirs = get_rays(height, width, focal, turntable_pose)

    samples = torch.linspace(-1.0, 1.0, 96).reshape(-1, 1)
    encoded = positional_encoding(samples, num_freqs=4, include_input=True)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    axes[0].imshow(_direction_rgb(identity_dirs).detach().cpu().numpy())
    axes[0].set_title("Identity-pose ray directions")
    axes[0].axis("off")

    axes[1].imshow(_direction_rgb(turntable_dirs).detach().cpu().numpy())
    axes[1].set_title("Turntable-pose ray directions")
    axes[1].axis("off")

    axes[2].imshow(encoded.detach().cpu().T, aspect="auto", cmap="coolwarm", interpolation="nearest")
    axes[2].set_title("Positional-encoding channels")
    axes[2].set_xlabel("Input sample from -1 to 1")
    axes[2].set_ylabel("Encoded channel")

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"saved {path}")


def run_basic_checks() -> None:
    """Run lightweight interface checks that keep the starter executable."""
    c2w = torch.eye(4)
    rays_o, rays_d = get_rays(4, 5, 4.0, c2w)
    assert rays_o.shape == (4, 5, 3)
    assert rays_d.shape == (4, 5, 3)
    encoded = positional_encoding(torch.zeros(2, 3), num_freqs=4)
    assert encoded.shape == (2, encoded_dim(3, 4))
    pose = pose_spherical(0.0, -30.0, 4.0)
    assert pose.shape == (4, 4)
    print("Task 1 starter checks passed")


def main() -> None:
    """Run Task 1 checks and optionally write both visualization outputs."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/task1/ray_directions.png")
    parser.add_argument("--visual-output", default="outputs/task1/task1_visual_checkpoint.png")
    parser.add_argument("--skip-figure", action="store_true")
    args = parser.parse_args()
    run_basic_checks()
    if not args.skip_figure:
        save_ray_debug(args.output)
        save_task1_visual_checkpoint(args.visual_output)


if __name__ == "__main__":
    main()
