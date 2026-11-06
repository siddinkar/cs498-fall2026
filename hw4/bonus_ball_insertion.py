"""Bonus: insert an analytically shaded sphere into a trained NeRF scene."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from hw4_utils import default_config, resolve_device, save_contact_sheet, save_image, save_video
from task1_rays_encoding import get_rays, positional_encoding
from task2_nerf_rendering import raw_to_outputs, sample_along_rays
from task3_train_mesh import load_scene_from_config, load_trained_model


DEFAULT_SPHERE_CENTER = (0.00, -1.20, -0.18)
DEFAULT_SPHERE_RADIUS = 0.35
DEFAULT_SPHERE_COLOR = (0.08, 0.62, 0.24)
DEFAULT_LIGHT_DIRECTION = (-0.45, -0.65, 1.00)


def intersect_sphere(
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    center: torch.Tensor,
    radius: float,
    near: float,
    far: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the nearest valid ray parameter and hit mask for one sphere.

    Args:
        rays_o: World-space ray origins with shape ``(N, 3)``.
        rays_d: Unnormalized world-space ray directions with shape ``(N, 3)``.
        center: Sphere center with shape ``(3,)`` in world coordinates.
        radius: Positive sphere radius in world units.
        near: Inclusive lower bound for a valid ray parameter.
        far: Inclusive upper bound for a valid ray parameter.

    Returns:
        ``(hit_depth, hit_mask)``. ``hit_depth`` has shape ``(N,)`` and stores
        the nearest root in ``[near, far]``, or positive infinity for a miss.
        ``hit_mask`` is a boolean tensor with shape ``(N,)``.
    """
    if rays_o.shape != rays_d.shape or rays_o.ndim != 2 or rays_o.shape[-1] != 3:
        raise ValueError("rays_o and rays_d must both have shape (N, 3)")
    if radius <= 0.0:
        raise ValueError("radius must be positive")
    if near > far:
        raise ValueError("near must not exceed far")

    center = torch.as_tensor(center, device=rays_o.device, dtype=rays_o.dtype)
    if center.shape != (3,):
        raise ValueError("center must have shape (3,)")

    offset = rays_o - center
    a = torch.sum(rays_d * rays_d, dim=-1)
    b = 2.0 * torch.sum(offset * rays_d, dim=-1)
    c = torch.sum(offset * offset, dim=-1) - radius**2
    discriminant = b * b - 4.0 * a * c
    valid_direction = a > torch.finfo(rays_o.dtype).eps
    sqrt_discriminant = torch.sqrt(torch.clamp(discriminant, min=0.0))
    denominator = 2.0 * torch.clamp_min(a, torch.finfo(rays_o.dtype).eps)
    root_near = (-b - sqrt_discriminant) / denominator
    root_far = (-b + sqrt_discriminant) / denominator

    real = (discriminant >= 0.0) & valid_direction
    near_valid = real & (root_near >= near) & (root_near <= far)
    far_valid = real & (root_far >= near) & (root_far <= far)
    hit_mask = near_valid | far_valid
    nearest_valid = torch.where(near_valid, root_near, root_far)
    hit_depth = torch.where(hit_mask, nearest_valid, torch.full_like(nearest_valid, torch.inf))
    return hit_depth, hit_mask


def composite_opaque_sphere(
    nerf_outputs: dict[str, torch.Tensor],
    z_vals: torch.Tensor,
    hit_depth: torch.Tensor,
    hit_mask: torch.Tensor,
    ball_rgb: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Composite an opaque sphere using only NeRF contributions before its hit.

    Args:
        nerf_outputs: Result from ``raw_to_outputs`` for ``N`` rays and ``S``
            samples. The function uses ``rgb`` with shape ``(N, 3)``,
            ``weights`` with shape ``(N, S)``, and ``sample_rgb`` with shape
            ``(N, S, 3)``.
        z_vals: NeRF sample depths with shape ``(N, S)``.
        hit_depth: Nearest sphere hit parameter with shape ``(N,)``.
        hit_mask: Boolean sphere-hit mask with shape ``(N,)``.
        ball_rgb: Shaded sphere RGB with shape ``(N, 3)`` or a constant color
            with shape ``(3,)``. Values should lie in ``[0, 1]``.

    Returns:
        Dictionary with ``rgb`` of shape ``(N, 3)`` and ``visibility`` of shape
        ``(N,)``. Visibility is the NeRF transmittance reaching the sphere and
        is zero on rays that miss it.
    """
    # ======= INSERT YOUR CODE BELOW =======
    # TODO(bonus): composite the analytic opaque surface into the NeRF ray.
    #
    # 1. Keep only NeRF weights whose z_vals are strictly before hit_depth.
    # 2. Sum those weights times sample_rgb to obtain the front scene color.
    # 3. Compute sphere visibility as one minus the accumulated front weight.
    # 4. On hits, return front_rgb + visibility * ball_rgb. On misses, preserve
    #    nerf_outputs["rgb"]. Visibility should be zero for misses.
    num_rays = z_vals.shape[0]
    return {
        "rgb": nerf_outputs["rgb"].clone(),
        "visibility": torch.zeros(num_rays, device=z_vals.device, dtype=z_vals.dtype),
    }
    # ======= DO NOT MODIFY CODE OUTSIDE THIS BLOCK =======


def shade_sphere(
    points: torch.Tensor,
    center: torch.Tensor,
    albedo: torch.Tensor,
    light_direction: torch.Tensor,
    ambient: float = 0.22,
) -> torch.Tensor:
    """Apply provided ambient-plus-Lambertian shading at sphere hit points."""
    if not 0.0 <= ambient <= 1.0:
        raise ValueError("ambient must lie in [0, 1]")
    center = torch.as_tensor(center, device=points.device, dtype=points.dtype)
    albedo = torch.as_tensor(albedo, device=points.device, dtype=points.dtype)
    light_direction = torch.as_tensor(light_direction, device=points.device, dtype=points.dtype)
    normals = torch.nn.functional.normalize(points - center, dim=-1, eps=1.0e-8)
    light_direction = torch.nn.functional.normalize(light_direction, dim=-1, eps=1.0e-8)
    diffuse = torch.clamp(torch.sum(normals * light_direction, dim=-1, keepdim=True), min=0.0)
    return (albedo * (ambient + (1.0 - ambient) * diffuse)).clamp(0.0, 1.0)


def render_rays_with_sphere(
    model: torch.nn.Module,
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    cfg: dict[str, object],
    center: torch.Tensor,
    radius: float,
    albedo: torch.Tensor,
    light_direction: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Render scene-only and sphere-inserted RGB for a flattened ray batch."""
    pts, z_vals = sample_along_rays(
        rays_o,
        rays_d,
        float(cfg["near"]),
        float(cfg["far"]),
        int(cfg["num_samples"]),
        perturb=False,
    )
    encoded_pts = positional_encoding(pts.reshape(-1, 3), int(cfg["num_freqs_xyz"]))
    viewdirs = rays_d / torch.linalg.norm(rays_d, dim=-1, keepdim=True).clamp_min(1.0e-8)
    expanded_dirs = viewdirs[:, None, :].expand_as(pts).reshape(-1, 3)
    encoded_dirs = positional_encoding(expanded_dirs, int(cfg["num_freqs_dir"]))
    raw = model(encoded_pts, encoded_dirs).reshape(rays_o.shape[0], int(cfg["num_samples"]), 4)
    nerf_outputs = raw_to_outputs(raw, z_vals, rays_d, bool(cfg["white_background"]))

    hit_depth, hit_mask = intersect_sphere(
        rays_o,
        rays_d,
        center,
        radius,
        float(cfg["near"]),
        float(cfg["far"]),
    )
    safe_depth = torch.where(hit_mask, hit_depth, torch.zeros_like(hit_depth))
    hit_points = rays_o + safe_depth[:, None] * rays_d
    ball_rgb = shade_sphere(hit_points, center, albedo, light_direction)
    composite = composite_opaque_sphere(nerf_outputs, z_vals, hit_depth, hit_mask, ball_rgb)
    return {
        "scene_rgb": nerf_outputs["rgb"],
        "inserted_rgb": composite["rgb"],
        "visibility": composite["visibility"],
        "hit_mask": hit_mask,
        "hit_depth": hit_depth,
    }


@torch.no_grad()
def render_image_with_sphere(
    model: torch.nn.Module,
    height: int,
    width: int,
    focal: float,
    c2w: torch.Tensor,
    cfg: dict[str, object],
    chunk: int,
    center: torch.Tensor,
    radius: float,
    albedo: torch.Tensor,
    light_direction: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Render one full view and reshape all bonus outputs to image grids."""
    model.eval()
    rays_o, rays_d = get_rays(height, width, focal, c2w)
    rays_o = rays_o.reshape(-1, 3)
    rays_d = rays_d.reshape(-1, 3)
    chunks: dict[str, list[torch.Tensor]] = {
        "scene_rgb": [],
        "inserted_rgb": [],
        "visibility": [],
        "hit_mask": [],
    }
    for start in range(0, rays_o.shape[0], chunk):
        outputs = render_rays_with_sphere(
            model,
            rays_o[start : start + chunk],
            rays_d[start : start + chunk],
            cfg,
            center,
            radius,
            albedo,
            light_direction,
        )
        for name in chunks:
            chunks[name].append(outputs[name].detach().cpu())

    return {
        "scene_rgb": torch.cat(chunks["scene_rgb"]).reshape(height, width, 3),
        "inserted_rgb": torch.cat(chunks["inserted_rgb"]).reshape(height, width, 3),
        "visibility": torch.cat(chunks["visibility"]).reshape(height, width),
        "hit_mask": torch.cat(chunks["hit_mask"]).reshape(height, width),
    }


def render_ball_turntable(
    cfg: dict[str, object],
    checkpoint: str | Path,
    num_frames: int = 12,
    center: tuple[float, float, float] = DEFAULT_SPHERE_CENTER,
    radius: float = DEFAULT_SPHERE_RADIUS,
    albedo: tuple[float, float, float] = DEFAULT_SPHERE_COLOR,
    light_direction: tuple[float, float, float] = DEFAULT_LIGHT_DIRECTION,
) -> Path:
    """Render scene-only and ball-inserted views on held-out test poses."""
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    device = resolve_device(str(cfg.get("device", "auto")))
    test_meta_path = Path(str(cfg["data_dir"])) / "transforms_test.json"
    with test_meta_path.open("r", encoding="utf-8") as f:
        num_test_views = len(json.load(f)["frames"])
    if num_test_views == 0:
        raise ValueError("test split contains no views")

    render_cfg = dict(cfg)
    render_cfg["train_skip"] = 1_000_000_000
    render_cfg["val_skip"] = 1_000_000_000
    render_cfg["test_skip"] = max(1, num_test_views // num_frames)
    scene = load_scene_from_config(render_cfg)
    model = load_trained_model(cfg, checkpoint, device)
    dtype = next(model.parameters()).dtype
    center_tensor = torch.tensor(center, device=device, dtype=dtype)
    albedo_tensor = torch.tensor(albedo, device=device, dtype=dtype)
    light_tensor = torch.tensor(light_direction, device=device, dtype=dtype)

    test_indices = scene.split_indices["test"]
    selected_count = min(num_frames, len(test_indices))
    selected_positions = np.linspace(0, len(test_indices), selected_count, endpoint=False, dtype=np.int64)
    selected_indices = test_indices[selected_positions]
    frames = []
    visibility_images = []
    for test_idx in tqdm(selected_indices, desc="render[ball insertion]", leave=True):
        outputs = render_image_with_sphere(
            model,
            scene.height,
            scene.width,
            scene.focal,
            scene.poses[int(test_idx)].to(device),
            cfg,
            int(cfg["render_chunk"]),
            center_tensor,
            radius,
            albedo_tensor,
            light_tensor,
        )
        scene_rgb = outputs["scene_rgb"].numpy()
        inserted_rgb = outputs["inserted_rgb"].numpy()
        visibility = outputs["visibility"].numpy()
        frames.append(np.concatenate([scene_rgb, inserted_rgb], axis=1))
        visibility_images.append(np.repeat(visibility[..., None], 3, axis=-1))

    frames_array = np.stack(frames)
    output_dir = Path(str(cfg["output_dir"]))
    video_path = output_dir / "ball_insertion.mp4"
    contact_sheet_path = output_dir / "ball_insertion_contact_sheet.png"
    visibility_path = output_dir / "ball_visibility.png"
    save_video(video_path, frames_array, fps=24)
    sheet_count = min(8, len(frames_array))
    sheet_indices = np.linspace(0, len(frames_array), sheet_count, endpoint=False, dtype=np.int64)
    save_contact_sheet(contact_sheet_path, frames_array[sheet_indices], cols=min(4, sheet_count))
    save_image(visibility_path, visibility_images[0])
    print(f"saved {len(frames_array)} scene-vs-insertion frames to {video_path}")
    print(f"saved {contact_sheet_path}")
    print(f"saved {visibility_path}")
    return video_path


def run_api_smoke() -> None:
    """Run dataset-free shape checks for the two bonus implementation blocks."""
    rays_o = torch.zeros(3, 3)
    rays_d = torch.tensor([[0.0, 0.0, -1.0]]).expand(3, 3)
    hit_depth, hit_mask = intersect_sphere(rays_o, rays_d, torch.tensor([0.0, 0.0, -3.0]), 1.0, 0.0, 6.0)
    assert hit_depth.shape == (3,)
    assert hit_mask.shape == (3,)

    z_vals = torch.tensor([[1.0, 2.0, 3.0]]).expand(3, 3)
    outputs = {
        "rgb": torch.ones(3, 3),
        "weights": torch.zeros(3, 3),
        "sample_rgb": torch.zeros(3, 3, 3),
    }
    composite = composite_opaque_sphere(outputs, z_vals, hit_depth, hit_mask, torch.tensor([0.1, 0.7, 0.2]))
    assert composite["rgb"].shape == (3, 3)
    assert composite["visibility"].shape == (3,)
    print("Bonus ball-insertion API smoke check passed")


def main() -> None:
    """Run bonus API checks or render the held-out insertion turntable."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["check", "render"], default="check")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-frames", type=int, default=12)
    parser.add_argument("--center", type=float, nargs=3, default=DEFAULT_SPHERE_CENTER)
    parser.add_argument("--radius", type=float, default=DEFAULT_SPHERE_RADIUS)
    parser.add_argument("--color", type=float, nargs=3, default=DEFAULT_SPHERE_COLOR)
    parser.add_argument("--light-direction", type=float, nargs=3, default=DEFAULT_LIGHT_DIRECTION)
    args = parser.parse_args()

    if args.mode == "check":
        run_api_smoke()
        return

    cfg = default_config("train")
    if args.data_dir is not None:
        cfg["data_dir"] = args.data_dir
    if args.output_dir is not None:
        cfg["output_dir"] = args.output_dir
    if args.device is not None:
        cfg["device"] = args.device
    checkpoint = args.checkpoint or str(Path(str(cfg["output_dir"])) / "model.pt")
    render_ball_turntable(
        cfg,
        checkpoint,
        num_frames=args.num_frames,
        center=tuple(args.center),
        radius=args.radius,
        albedo=tuple(args.color),
        light_direction=tuple(args.light_direction),
    )


if __name__ == "__main__":
    main()
