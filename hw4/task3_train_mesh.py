"""Task 3: train NeRF, render novel views, and extract a textured mesh."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm, trange

from hw4_utils import (
    BlenderScene,
    Mesh,
    default_config,
    img_to_mse,
    load_blender_scene,
    mse_to_psnr,
    resolve_device,
    save_checkpoint,
    save_contact_sheet,
    save_image,
    save_psnr_curve,
    save_video,
    seed_everything,
    to8b,
    write_metrics_csv,
)
from task1_rays_encoding import encoded_dim, get_rays, positional_encoding
from task2_nerf_rendering import TinyNeRF, render_rays


def build_model(cfg: dict[str, object], device: torch.device) -> TinyNeRF:
    """Construct TinyNeRF from the configured encoding and MLP dimensions."""
    input_ch = encoded_dim(3, int(cfg["num_freqs_xyz"]), include_input=True)
    input_ch_dir = encoded_dim(3, int(cfg["num_freqs_dir"]), include_input=True)
    model = TinyNeRF(
        input_ch=input_ch,
        input_ch_dir=input_ch_dir,
        hidden_dim=int(cfg["hidden_dim"]),
        num_layers=int(cfg["num_layers"]),
        skip_layer=int(cfg["skip_layer"]),
        use_viewdirs=bool(cfg["use_viewdirs"]),
    )
    return model.to(device)


def load_scene_from_config(cfg: dict[str, object]) -> BlenderScene:
    """Load and downsample all configured Blender scene splits."""
    return load_blender_scene(
        cfg["data_dir"],
        image_downscale=int(cfg["image_downscale"]),
        white_background=bool(cfg["white_background"]),
        train_skip=int(cfg.get("train_skip", 1)),
        val_skip=int(cfg.get("val_skip", 1)),
        test_skip=int(cfg.get("test_skip", 1)),
    )


def render_image(
    model: torch.nn.Module,
    height: int,
    width: int,
    focal: float,
    c2w: torch.Tensor,
    cfg: dict[str, object],
    chunk: int,
) -> torch.Tensor:
    """Render one full image by chunking flattened rays."""
    model.eval()
    with torch.no_grad():
        rays_o, rays_d = get_rays(height, width, focal, c2w)
        rays_o = rays_o.reshape(-1, 3)
        rays_d = rays_d.reshape(-1, 3)
        rgbs = []
        old_perturb = cfg.get("perturb", False)
        cfg["perturb"] = False
        for start in range(0, rays_o.shape[0], chunk):
            out = render_rays(model, rays_o[start : start + chunk], rays_d[start : start + chunk], cfg)
            rgbs.append(out["rgb"].detach().cpu())
        cfg["perturb"] = old_perturb
    return torch.cat(rgbs, dim=0).reshape(height, width, 3)


def build_ray_pool(scene: BlenderScene, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Flatten every training pixel into aligned origin, direction, and RGB tensors."""
    ray_origins = []
    ray_dirs = []
    rgbs = []
    for idx in scene.split_indices["train"]:
        pose = scene.poses[int(idx)].to(device)
        rays_o, rays_d = get_rays(scene.height, scene.width, scene.focal, pose)
        ray_origins.append(rays_o.reshape(-1, 3))
        ray_dirs.append(rays_d.reshape(-1, 3))
        rgbs.append(scene.images[int(idx)].to(device).reshape(-1, 3))
    return torch.cat(ray_origins, 0), torch.cat(ray_dirs, 0), torch.cat(rgbs, 0)


def visualize_dataset(cfg: dict[str, object]) -> Path:
    """Save a training-image contact sheet and print basic dataset metadata."""
    scene = load_scene_from_config(cfg)
    out = Path(str(cfg["output_dir"])) / "dataset_contact_sheet.png"
    save_contact_sheet(out, scene.images[scene.split_indices["train"][:10]].numpy(), cols=5)
    split_sizes = {k: len(v) for k, v in scene.split_indices.items()}
    print(f"images={tuple(scene.images.shape)} height={scene.height} width={scene.width} focal={scene.focal:.3f}")
    print(f"split sizes={split_sizes}")
    print(f"saved {out}")
    return out


def train_tiny_nerf(cfg: dict[str, object]) -> Path:
    """Train the TinyNeRF model and write validation renders/checkpoints."""
    seed_everything(int(cfg.get("seed", 0)))
    device = resolve_device(str(cfg.get("device", "auto")))
    out_dir = Path(str(cfg["output_dir"]))
    out_dir.mkdir(parents=True, exist_ok=True)

    scene = load_scene_from_config(cfg)
    model = build_model(cfg, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"]))
    rays_o, rays_d, target_rgb = build_ray_pool(scene, device)

    rows: list[dict[str, float]] = []
    metrics_csv = out_dir / "train_metrics.csv"
    psnr_curve = out_dir / "train_psnr_curve.png"
    plot_every = int(cfg.get("metrics_plot_every", 0))

    pbar = trange(1, int(cfg["iters"]) + 1, desc="train[tiny_nerf]", leave=True)
    for step in pbar:
        model.train()
        select = torch.randint(0, rays_o.shape[0], (int(cfg["batch_size"]),), device=device)
        out = render_rays(model, rays_o[select], rays_d[select], cfg)
        loss = img_to_mse(out["rgb"], target_rgb[select])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % int(cfg["log_every"]) == 0 or step == 1:
            psnr = float(mse_to_psnr(loss.detach()).cpu())
            loss_value = float(loss.detach().cpu())
            rows.append({"step": float(step), "loss": loss_value, "psnr": psnr})
            write_metrics_csv(metrics_csv, rows)
            pbar.set_postfix(loss=f"{loss_value:.5f}", psnr=f"{psnr:.2f}")
            if plot_every > 0 and step % plot_every == 0:
                save_psnr_curve(psnr_curve, rows)

        if step % int(cfg["val_every"]) == 0 or step == int(cfg["iters"]):
            val_indices = scene.split_indices["val"][: max(1, int(cfg.get("num_val_views", 1)))]
            for view_idx, val_idx in enumerate(val_indices):
                pred = render_image(
                    model,
                    scene.height,
                    scene.width,
                    scene.focal,
                    scene.poses[int(val_idx)].to(device),
                    cfg,
                    int(cfg["render_chunk"]),
                )
                save_image(out_dir / f"val_{step:06d}_view_{view_idx:02d}.png", pred.numpy())

        checkpoint_every = int(cfg.get("checkpoint_every", 0))
        if checkpoint_every > 0 and step % checkpoint_every == 0:
            save_checkpoint(out_dir / f"ckpt_{step:06d}.pt", model, cfg)

    ckpt_path = out_dir / "model.pt"
    save_psnr_curve(psnr_curve, rows)
    save_checkpoint(ckpt_path, model, cfg)
    print(f"saved checkpoint {ckpt_path}")
    return ckpt_path


def load_trained_model(cfg: dict[str, object], checkpoint: str | Path, device: torch.device) -> TinyNeRF:
    """Restore a TinyNeRF checkpoint in evaluation mode."""
    model = build_model(cfg, device)
    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def render_turntable(
    cfg: dict[str, object],
    checkpoint: str | Path,
    num_frames: int = 12,
) -> Path:
    """Render test-set poses beside their matching ground-truth images."""
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")

    device = resolve_device(str(cfg.get("device", "auto")))
    test_meta_path = Path(str(cfg["data_dir"])) / "transforms_test.json"
    with test_meta_path.open("r", encoding="utf-8") as f:
        num_test_views = len(json.load(f)["frames"])
    if num_test_views == 0:
        raise ValueError("test split contains no views")

    # Load only the held-out frames needed for the video. The scene loader also
    # reads train/val, so skip those splits aggressively to keep memory bounded.
    turntable_cfg = dict(cfg)
    turntable_cfg["train_skip"] = 1_000_000_000
    turntable_cfg["val_skip"] = 1_000_000_000
    turntable_cfg["test_skip"] = max(1, num_test_views // num_frames)
    scene = load_scene_from_config(turntable_cfg)
    model = load_trained_model(cfg, checkpoint, device)

    test_indices = scene.split_indices["test"]
    selected_count = min(num_frames, len(test_indices))
    selected_positions = np.linspace(
        0, len(test_indices), selected_count, endpoint=False, dtype=np.int64
    )
    selected_indices = test_indices[selected_positions]
    frames = []
    for test_idx in tqdm(selected_indices, desc="render[test turntable]", leave=True):
        pose = scene.poses[int(test_idx)].to(device)
        target = scene.images[int(test_idx)].numpy()
        prediction = render_image(
            model,
            scene.height,
            scene.width,
            scene.focal,
            pose,
            cfg,
            int(cfg["render_chunk"]),
        ).numpy()
        frames.append(np.concatenate([target, prediction], axis=1))

    frames_array = np.stack(frames, axis=0)
    output_dir = Path(str(cfg["output_dir"]))
    out = output_dir / "turntable.mp4"
    contact_sheet = output_dir / "turntable_contact_sheet.png"
    save_video(out, frames_array, fps=24)
    sheet_count = min(8, len(frames))
    sheet_indices = np.linspace(
        0, len(frames), sheet_count, endpoint=False, dtype=np.int64
    )
    save_contact_sheet(contact_sheet, frames_array[sheet_indices], cols=min(4, sheet_count))
    print(f"saved {len(frames)} GT-vs-prediction frames to {out}")
    print(f"saved {contact_sheet}")
    return out


def _psnr_np(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute PSNR in decibels for two float RGB arrays in the unit range."""
    mse = float(np.mean((pred.astype(np.float32) - target.astype(np.float32)) ** 2))
    return float(-10.0 * np.log10(max(mse, 1.0e-10)))


def _compute_image_metrics(
    pred: np.ndarray,
    target: np.ndarray,
) -> dict[str, float]:
    """Compute PSNR and SSIM for one RGB image pair."""
    from skimage.metrics import structural_similarity

    pred = np.asarray(pred, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    if pred.shape != target.shape or pred.ndim != 3 or pred.shape[-1] != 3:
        raise ValueError(f"expected matching RGB images, got {pred.shape} and {target.shape}")

    pred = np.clip(pred, 0.0, 1.0)
    target = np.clip(target, 0.0, 1.0)
    return {
        "psnr": _psnr_np(pred, target),
        "ssim": float(structural_similarity(pred, target, data_range=1.0, channel_axis=-1)),
    }


def _write_validation_metrics(path: str | Path, rows: list[dict[str, float]]) -> dict[str, float]:
    """Write per-view validation metrics and a final aggregate row."""
    if not rows:
        raise ValueError("validation split contains no views")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metric_names = ["psnr", "ssim"]
    mean_metrics = {name: float(np.mean([row[name] for row in rows])) for name in metric_names}
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["view_index", *metric_names])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "view_index": int(row["view_index"]),
                    **{name: f"{row[name]:.6f}" for name in metric_names},
                }
            )
        writer.writerow(
            {
                "view_index": "mean",
                **{name: f"{mean_metrics[name]:.6f}" for name in metric_names},
            }
        )
    return mean_metrics


def evaluate_validation_set(cfg: dict[str, object], checkpoint: str | Path) -> Path:
    """Render the configured validation split and report PSNR and SSIM."""
    device = resolve_device(str(cfg.get("device", "auto")))
    scene = load_scene_from_config(cfg)
    model = load_trained_model(cfg, checkpoint, device)
    output_dir = Path(str(cfg["output_dir"]))
    render_dir = output_dir / "validation"
    render_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float]] = []
    val_indices = scene.split_indices["val"]
    for view_index, val_idx in enumerate(tqdm(val_indices, desc="eval[validation]", leave=True)):
        pose = scene.poses[int(val_idx)].to(device)
        target = scene.images[int(val_idx)].numpy()
        pred = render_image(
            model,
            scene.height,
            scene.width,
            scene.focal,
            pose,
            cfg,
            int(cfg["render_chunk"]),
        ).numpy()
        metrics = _compute_image_metrics(pred, target)
        rows.append({"view_index": float(view_index), **metrics})
        save_image(render_dir / f"view_{view_index:02d}_prediction.png", pred)
        save_image(render_dir / f"view_{view_index:02d}_comparison.png", np.concatenate([target, pred], axis=1))

    csv_path = output_dir / "validation_metrics.csv"
    mean_metrics = _write_validation_metrics(csv_path, rows)
    print(
        f"validation mean over {len(rows)} views: "
        f"PSNR={mean_metrics['psnr']:.4f} dB, "
        f"SSIM={mean_metrics['ssim']:.4f}"
    )
    print(f"saved validation renders under {render_dir}")
    print(f"saved {csv_path}")
    return csv_path


def task3_mesh_output_dir(cfg: dict[str, object]) -> Path:
    """Return the directory for Task 3 mesh, texture, and comparison outputs."""
    return Path(str(cfg["output_dir"])) / "mesh"


def build_density_grid(
    model: torch.nn.Module,
    cfg: dict[str, object],
    bounds: tuple[float, float],
    resolution: int,
    chunk_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Evaluate nonnegative density on a regular 3D grid.

    Args:
        model: Trained ``TinyNeRF`` or compatible module. It should accept
            encoded point features with shape ``(M, C_xyz)`` and encoded view
            directions with shape ``(M, C_dir)``, then return raw predictions
            with shape ``(M, 4)``. The fourth channel is raw density.
        cfg: Assignment configuration dictionary. This function needs
            ``cfg["num_freqs_xyz"]`` to encode grid positions and
            ``cfg["num_freqs_dir"]`` to encode the fixed viewing direction used
            when querying the RGB-conditioned model.
        bounds: Tuple ``(lo, hi)`` giving the inclusive world-coordinate extent
            of the cubic query volume. Grid coordinates should cover
            ``[lo, hi]`` along each of the x, y, and z axes.
        resolution: Grid resolution ``R`` along each axis. The full grid
            contains ``R * R * R`` query points.
        chunk_size: Maximum number ``M`` of flattened grid points to evaluate in
            one model call. Chunking keeps memory use manageable.
        device: Torch device on which to create the grid, encoded inputs, and
            returned density tensor.

    Returns:
        Tensor ``sigma_grid`` with shape ``(R, R, R)`` on ``device``. The axes
        are ordered as ``x, y, z`` because the grid should be built with
        ``torch.meshgrid(..., indexing="ij")``. Values are nonnegative densities
        after applying the same density activation used by the renderer.
    """
    # ======= INSERT YOUR CODE BELOW =======
    # TODO(student): build a regular xyz grid, query the trained model density,
    # and return nonnegative sigma values reshaped as (R, R, R).
    #
    # Suggested steps:
    #   1. Use torch.linspace(bounds[0], bounds[1], resolution, device=device).
    #   2. Use torch.meshgrid(..., indexing="ij") so grid axes are x, y, z.
    #   3. Flatten points to (R**3, 3), query in chunks, and concatenate.
    #   4. Encode positions with positional_encoding(..., cfg["num_freqs_xyz"]).
    #   5. Use any fixed normalized viewing direction for the RGB branch.
    #   6. Read raw[..., 3], apply the same nonnegative density activation as
    #      the renderer, and reshape to (R, R, R).
    return torch.zeros((resolution, resolution, resolution), device=device)
    # ======= DO NOT MODIFY CODE OUTSIDE THIS BLOCK =======


def extract_mesh_from_density(
    sigma_grid: torch.Tensor | np.ndarray,
    bounds: tuple[float, float],
    threshold: float,
) -> Mesh:
    """Extract a world-coordinate triangle mesh from a density grid.

    Args:
        sigma_grid: Density grid with shape ``(R, R, R)``. It may be a
            ``torch.Tensor`` or ``np.ndarray``. Axis ``0`` corresponds to x,
            axis ``1`` to y, and axis ``2`` to z. Values are scalar density
            samples on the regular grid produced by ``build_density_grid``.
        bounds: Tuple ``(lo, hi)`` matching the world-coordinate extent used to
            build ``sigma_grid``. Marching cubes returns vertices in voxel-index
            coordinates, so convert each coordinate from ``[0, R - 1]`` into
            ``[lo, hi]``.
        threshold: Density level set to extract. The returned surface should
            approximate locations where ``sigma_grid == threshold``.

    Returns:
        ``Mesh(vertices, faces)`` where ``vertices`` is a ``float32`` numpy array
        with shape ``(V, 3)`` containing world-space xyz vertex positions, and
        ``faces`` is an ``int64`` numpy array with shape ``(F, 3)`` containing
        triangle vertex indices. Normals and scalar values returned by
        ``skimage.measure.marching_cubes`` do not need to be stored.
    """
    from skimage import measure

    if isinstance(sigma_grid, torch.Tensor):
        volume = sigma_grid.detach().cpu().numpy()
    else:
        volume = np.asarray(sigma_grid)
    volume = volume.astype(np.float32, copy=False)
    if volume.ndim != 3:
        raise ValueError(f"sigma_grid must have shape (R, R, R), got {volume.shape}")
    if min(volume.shape) < 2:
        return Mesh(vertices=np.zeros((0, 3), dtype=np.float32), faces=np.zeros((0, 3), dtype=np.int64))
    if threshold < float(volume.min()) or threshold > float(volume.max()):
        return Mesh(vertices=np.zeros((0, 3), dtype=np.float32), faces=np.zeros((0, 3), dtype=np.int64))

    verts, faces, _normals, _values = measure.marching_cubes(volume, level=threshold)
    lo, hi = bounds
    scale = (hi - lo) / (volume.shape[0] - 1)
    verts_world = verts * scale + lo
    return Mesh(vertices=verts_world.astype(np.float32), faces=faces.astype(np.int64))


def carve_density_with_visual_hull(
    sigma_grid: torch.Tensor | np.ndarray,
    cfg: dict[str, object],
    bounds: tuple[float, float],
    min_view_fraction: float = 1.0,
    mask_dilation: int = 3,
) -> torch.Tensor | np.ndarray:
    """Remove density that is inconsistent with the training silhouettes.

    The Blender data contains an alpha channel for every training image. A
    point is retained when it projects into the (slightly dilated) foreground
    mask in at least ``min_view_fraction`` of the training cameras. Requiring
    every view by default is safe for this synthetic dataset because the poses
    and alpha masks are exact; dilation protects boundary geometry. This
    conservative visual hull removes the floating density that marching cubes
    would otherwise turn into shells and specks around the object.
    """
    import imageio.v2 as imageio
    from scipy.ndimage import binary_dilation

    if not 0.0 < min_view_fraction <= 1.0:
        raise ValueError("min_view_fraction must be in (0, 1]")
    if mask_dilation < 0:
        raise ValueError("mask_dilation must be nonnegative")

    is_tensor = isinstance(sigma_grid, torch.Tensor)
    volume = sigma_grid.detach().cpu().numpy() if is_tensor else np.asarray(sigma_grid)
    if volume.ndim != 3:
        raise ValueError(f"sigma_grid must have shape (R, R, R), got {volume.shape}")

    data_dir = Path(str(cfg["data_dir"]))
    metadata_path = data_dir / "transforms_train.json"
    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    lo, hi = bounds
    axes = [np.linspace(lo, hi, size, dtype=np.float32) for size in volume.shape]
    grid = np.meshgrid(*axes, indexing="ij")
    points = np.stack(grid, axis=-1).reshape(-1, 3)
    points_h = np.concatenate(
        [points, np.ones((len(points), 1), dtype=np.float32)], axis=1
    )
    support = np.zeros(len(points), dtype=np.uint16)
    num_masks = 0

    camera_angle_x = float(metadata["camera_angle_x"])
    for frame in metadata["frames"]:
        image_path = data_dir / f"{frame['file_path']}.png"
        image = imageio.imread(image_path)
        if image.ndim != 3 or image.shape[-1] < 4:
            raise RuntimeError(
                f"visual-hull cleanup requires an alpha channel, but {image_path} "
                f"has shape {image.shape}"
            )
        height, width = image.shape[:2]
        foreground = image[..., 3] > 8
        if mask_dilation:
            foreground = binary_dilation(foreground, iterations=mask_dilation)

        c2w = np.asarray(frame["transform_matrix"], dtype=np.float32)
        camera_points = points_h @ np.linalg.inv(c2w).T
        depth = -camera_points[:, 2]
        focal = 0.5 * width / np.tan(0.5 * camera_angle_x)
        safe_depth = np.maximum(depth, 1.0e-8)
        pixel_x = focal * camera_points[:, 0] / safe_depth + 0.5 * width
        pixel_y = -focal * camera_points[:, 1] / safe_depth + 0.5 * height
        pixel_x = np.rint(pixel_x).astype(np.int64)
        pixel_y = np.rint(pixel_y).astype(np.int64)
        visible = (
            (depth > 0.0)
            & (pixel_x >= 0)
            & (pixel_x < width)
            & (pixel_y >= 0)
            & (pixel_y < height)
        )
        visible_indices = np.flatnonzero(visible)
        support[visible_indices] += foreground[
            pixel_y[visible_indices], pixel_x[visible_indices]
        ].astype(np.uint16)
        num_masks += 1

    if num_masks == 0:
        raise RuntimeError(f"no training masks found in {metadata_path}")
    required_views = max(1, int(np.ceil(min_view_fraction * num_masks)))
    hull = (support >= required_views).reshape(volume.shape)
    cleaned = np.where(hull, volume, 0.0)
    print(
        f"visual-hull cleanup retained {int(hull.sum()):,}/{hull.size:,} voxels "
        f"supported by at least {required_views}/{num_masks} masks"
    )

    if is_tensor:
        return torch.as_tensor(cleaned, dtype=sigma_grid.dtype, device=sigma_grid.device)
    return cleaned.astype(volume.dtype, copy=False)


def keep_largest_mesh_component(mesh: Mesh) -> Mesh:
    """Discard disconnected marching-cubes fragments and compact the mesh."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if len(vertices) == 0 or len(faces) == 0:
        return Mesh(vertices=vertices.copy(), faces=faces.copy())

    edges = np.concatenate(
        [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0
    )
    rows = np.concatenate([edges[:, 0], edges[:, 1]])
    cols = np.concatenate([edges[:, 1], edges[:, 0]])
    graph = coo_matrix(
        (np.ones(len(rows), dtype=np.uint8), (rows, cols)),
        shape=(len(vertices), len(vertices)),
    ).tocsr()
    _num_components, labels = connected_components(graph, directed=False)
    face_labels = labels[faces[:, 0]]
    component_face_counts = np.bincount(face_labels)
    largest_label = int(np.argmax(component_face_counts))
    kept_faces = faces[face_labels == largest_label]
    kept_vertices, compact_faces = np.unique(kept_faces, return_inverse=True)
    compact_faces = compact_faces.reshape(-1, 3).astype(np.int64, copy=False)
    return Mesh(vertices=vertices[kept_vertices].copy(), faces=compact_faces)


def estimate_vertex_normals(mesh: Mesh) -> np.ndarray:
    """Estimate consistently oriented area-weighted vertex normals."""
    normals = np.zeros_like(mesh.vertices, dtype=np.float32)
    if len(mesh.faces) == 0:
        return normals

    triangles = mesh.vertices[mesh.faces]
    face_normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    center = mesh.vertices.mean(axis=0)
    face_centers = triangles.mean(axis=1)
    orientation = np.einsum("ij,ij->i", face_normals, face_centers - center).sum()
    if orientation < 0.0:
        face_normals *= -1.0
    for corner in range(3):
        np.add.at(normals, mesh.faces[:, corner], face_normals)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    return normals / np.maximum(lengths, 1.0e-8)


def unwrap_mesh_with_xatlas(mesh: Mesh) -> tuple[Mesh, np.ndarray, np.ndarray]:
    """Generate a non-overlapping UV atlas and remap geometry at chart seams."""
    try:
        import xatlas
    except ImportError as exc:
        raise RuntimeError("Task 3 mesh texturing requires `pip install xatlas`") from exc

    original_normals = estimate_vertex_normals(mesh)
    vmapping, indices, uvs = xatlas.parametrize(
        mesh.vertices.astype(np.float32, copy=False),
        mesh.faces.astype(np.uint32, copy=False),
    )
    atlas_mesh = Mesh(
        vertices=mesh.vertices[vmapping].astype(np.float32, copy=False),
        faces=indices.astype(np.int64, copy=False),
    )
    atlas_normals = original_normals[vmapping].astype(np.float32, copy=False)
    return atlas_mesh, uvs.astype(np.float32, copy=False), atlas_normals


def _barycentric(px: np.ndarray, py: np.ndarray, tri: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute barycentric weights for pixel coordinates relative to one triangle."""
    x0, y0 = tri[0]
    x1, y1 = tri[1]
    x2, y2 = tri[2]
    denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
    if abs(float(denom)) < 1.0e-8:
        nan = np.full_like(px, np.nan, dtype=np.float32)
        return nan, nan, nan
    w0 = ((y1 - y2) * (px - x2) + (x2 - x1) * (py - y2)) / denom
    w1 = ((y2 - y0) * (px - x2) + (x0 - x2) * (py - y2)) / denom
    w2 = 1.0 - w0 - w1
    return w0, w1, w2


def _dilate_texture(texture: np.ndarray, mask: np.ndarray, iterations: int) -> tuple[np.ndarray, np.ndarray]:
    """Pad valid UV texels into neighboring empty texels to reduce seam artifacts."""
    texture = texture.copy()
    mask = mask.copy()
    for _ in range(iterations):
        new_texture = texture.copy()
        new_mask = mask.copy()
        copies = [
            ((slice(0, -1), slice(None)), (slice(1, None), slice(None))),
            ((slice(1, None), slice(None)), (slice(0, -1), slice(None))),
            ((slice(None), slice(0, -1)), (slice(None), slice(1, None))),
            ((slice(None), slice(1, None)), (slice(None), slice(0, -1))),
        ]
        for src, dst in copies:
            fill = mask[src] & ~new_mask[dst]
            dst_tex = new_texture[dst]
            dst_mask = new_mask[dst]
            dst_tex[fill] = texture[src][fill]
            dst_mask[fill] = True
        texture = new_texture
        mask = new_mask
    return texture, mask


def rasterize_uv_surface(
    mesh: Mesh,
    uvs: np.ndarray,
    vertex_normals: np.ndarray,
    atlas_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rasterize world positions and normals for every covered UV texel."""
    positions = np.zeros((atlas_size, atlas_size, 3), dtype=np.float32)
    normals = np.zeros_like(positions)
    mask = np.zeros((atlas_size, atlas_size), dtype=bool)
    if len(mesh.faces) == 0:
        return positions, normals, mask

    pixel_uv = np.column_stack(
        [uvs[:, 0] * (atlas_size - 1), (1.0 - uvs[:, 1]) * (atlas_size - 1)]
    )
    for face in mesh.faces:
        tri = pixel_uv[face]
        min_xy = np.floor(tri.min(axis=0)).astype(int)
        max_xy = np.ceil(tri.max(axis=0)).astype(int)
        x0 = max(0, min_xy[0])
        y0 = max(0, min_xy[1])
        x1 = min(atlas_size - 1, max_xy[0])
        y1 = min(atlas_size - 1, max_xy[1])
        if x1 < x0 or y1 < y0:
            continue
        xs, ys = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
        w0, w1, w2 = _barycentric(xs.astype(np.float32), ys.astype(np.float32), tri.astype(np.float32))
        inside = (w0 >= -1.0e-4) & (w1 >= -1.0e-4) & (w2 >= -1.0e-4)
        if not inside.any():
            continue
        surface_points = (
            w0[..., None] * mesh.vertices[face[0]]
            + w1[..., None] * mesh.vertices[face[1]]
            + w2[..., None] * mesh.vertices[face[2]]
        )
        surface_normals = (
            w0[..., None] * vertex_normals[face[0]]
            + w1[..., None] * vertex_normals[face[1]]
            + w2[..., None] * vertex_normals[face[2]]
        )
        yy = ys[inside]
        xx = xs[inside]
        positions[yy, xx] = surface_points[inside]
        normals[yy, xx] = surface_normals[inside]
        mask[yy, xx] = True

    normal_lengths = np.linalg.norm(normals[mask], axis=1, keepdims=True)
    normals[mask] /= np.maximum(normal_lengths, 1.0e-8)
    return positions, normals, mask


@torch.no_grad()
def query_nerf_surface_rgb(
    model: torch.nn.Module,
    cfg: dict[str, object],
    points: np.ndarray,
    normals: np.ndarray,
    camera_centers: np.ndarray,
    device: torch.device,
    chunk_size: int,
    num_views: int = 4,
) -> np.ndarray:
    """Infer surface RGB with NeRF directions from front-facing train cameras."""
    if len(points) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    if len(camera_centers) == 0:
        raise ValueError("at least one training camera is required to bake the texture")

    model.eval()
    dtype = next(model.parameters()).dtype
    view_count = min(max(1, num_views), len(camera_centers))
    colors = []
    for start in range(0, len(points), chunk_size):
        point_chunk = points[start : start + chunk_size]
        normal_chunk = normals[start : start + chunk_size]
        to_cameras = camera_centers[None, :, :] - point_chunk[:, None, :]
        distance_sq = np.einsum("nci,nci->nc", to_cameras, to_cameras)
        to_cameras /= np.sqrt(np.maximum(distance_sq[..., None], 1.0e-8))
        facing = np.maximum(np.einsum("ni,nci->nc", normal_chunk, to_cameras), 0.0)
        scores = facing**4 / np.maximum(distance_sq, 1.0e-8)

        selected = np.argpartition(scores, scores.shape[1] - view_count, axis=1)[:, -view_count:]
        selected_scores = np.take_along_axis(scores, selected, axis=1)
        no_front_view = selected_scores.sum(axis=1) <= 1.0e-12
        if no_front_view.any():
            nearest = np.argmin(distance_sq[no_front_view], axis=1)
            selected[no_front_view, 0] = nearest
            selected_scores[no_front_view, 0] = 1.0
        selected_scores /= np.maximum(selected_scores.sum(axis=1, keepdims=True), 1.0e-12)

        point_tensor = torch.as_tensor(point_chunk, device=device, dtype=dtype)
        encoded_points = positional_encoding(point_tensor, int(cfg["num_freqs_xyz"]))
        blended = torch.zeros((len(point_chunk), 3), device=device, dtype=dtype)
        for view_index in range(view_count):
            centers = camera_centers[selected[:, view_index]]
            ray_dirs = point_chunk - centers
            ray_dirs /= np.maximum(np.linalg.norm(ray_dirs, axis=1, keepdims=True), 1.0e-8)
            direction_tensor = torch.as_tensor(ray_dirs, device=device, dtype=dtype)
            encoded_dirs = positional_encoding(direction_tensor, int(cfg["num_freqs_dir"]))
            rgb = torch.sigmoid(model(encoded_points, encoded_dirs)[..., :3])
            weight = torch.as_tensor(selected_scores[:, view_index], device=device, dtype=dtype)
            blended += weight[:, None] * rgb
        colors.append(blended.cpu())
    return torch.cat(colors, dim=0).numpy()


def bake_nerf_texture(
    model: torch.nn.Module,
    cfg: dict[str, object],
    mesh: Mesh,
    uvs: np.ndarray,
    vertex_normals: np.ndarray,
    camera_centers: np.ndarray,
    device: torch.device,
    chunk_size: int,
    atlas_size: int = 256,
    padding: int = 8,
) -> np.ndarray:
    """Query NeRF at xatlas texels and pad chart boundaries."""
    positions, normals, mask = rasterize_uv_surface(mesh, uvs, vertex_normals, atlas_size)
    texture = np.ones((atlas_size, atlas_size, 3), dtype=np.float32)
    texture[mask] = query_nerf_surface_rgb(
        model,
        cfg,
        positions[mask],
        normals[mask],
        camera_centers,
        device,
        chunk_size,
    )
    texture, _ = _dilate_texture(texture, mask, padding)
    print(f"queried NeRF RGB for {int(mask.sum())} covered UV texels")
    return texture


def save_textured_obj(path: str | Path, mesh: Mesh, uvs: np.ndarray, texture_name: str) -> None:
    """Save the xatlas-remapped mesh with one UV per remapped vertex."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mtl_path = path.with_suffix(".mtl")
    with mtl_path.open("w", encoding="utf-8") as f:
        f.write("newmtl nerf_texture\n")
        f.write("Kd 1.000 1.000 1.000\n")
        f.write(f"map_Kd {texture_name}\n")
    with path.open("w", encoding="utf-8") as f:
        f.write(f"mtllib {mtl_path.name}\n")
        for vertex in mesh.vertices:
            f.write(f"v {vertex[0]:.7f} {vertex[1]:.7f} {vertex[2]:.7f}\n")
        for uv in uvs:
            f.write(f"vt {uv[0]:.7f} {uv[1]:.7f}\n")
        f.write("usemtl nerf_texture\n")
        for face in mesh.faces:
            a, b, c = (int(x) + 1 for x in face)
            f.write(f"f {a}/{a} {b}/{b} {c}/{c}\n")


def save_vertex_color_ply(path: str | Path, mesh: Mesh, vertex_colors: np.ndarray) -> None:
    """Save a PLY with baked vertex colors for quick inspection."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    colors = to8b(vertex_colors)
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(mesh.vertices)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write(f"element face {len(mesh.faces)}\n")
        f.write("property list uchar int vertex_indices\nend_header\n")
        for vertex, color in zip(mesh.vertices, colors):
            f.write(
                f"{vertex[0]:.7f} {vertex[1]:.7f} {vertex[2]:.7f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )
        for face in mesh.faces:
            f.write(f"3 {int(face[0])} {int(face[1])} {int(face[2])}\n")


def _sample_texture(texture: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Bilinearly sample an RGB texture at normalized UV coordinates."""
    h, w = texture.shape[:2]
    x = np.clip(uv[..., 0] * (w - 1), 0.0, w - 1.0)
    y = np.clip((1.0 - uv[..., 1]) * (h - 1), 0.0, h - 1.0)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    wx = (x - x0)[..., None]
    wy = (y - y0)[..., None]
    c00 = texture[y0, x0]
    c10 = texture[y0, x1]
    c01 = texture[y1, x0]
    c11 = texture[y1, x1]
    return (1.0 - wx) * (1.0 - wy) * c00 + wx * (1.0 - wy) * c10 + (1.0 - wx) * wy * c01 + wx * wy * c11


def render_textured_mesh_view(
    mesh: Mesh,
    uvs: np.ndarray,
    texture: np.ndarray,
    c2w: torch.Tensor,
    height: int,
    width: int,
    focal: float,
) -> np.ndarray:
    """Render a textured mesh from a NeRF/OpenGL camera-to-world pose."""
    image = np.ones((height, width, 3), dtype=np.float32)
    zbuf = np.full((height, width), np.inf, dtype=np.float32)
    if len(mesh.faces) == 0:
        return image

    c2w_np = c2w.detach().cpu().numpy()
    world_to_cam = np.linalg.inv(c2w_np)
    homo = np.concatenate([mesh.vertices.astype(np.float32), np.ones((len(mesh.vertices), 1), dtype=np.float32)], axis=1)
    cam = (world_to_cam @ homo.T).T[:, :3]
    depth = -cam[:, 2]
    screen = np.empty((len(mesh.vertices), 2), dtype=np.float32)
    screen[:, 0] = focal * cam[:, 0] / np.maximum(depth, 1.0e-8) + width * 0.5
    screen[:, 1] = -focal * cam[:, 1] / np.maximum(depth, 1.0e-8) + height * 0.5

    for face in mesh.faces:
        if np.any(depth[face] <= 1.0e-6):
            continue
        tri = screen[face]
        min_xy = np.floor(tri.min(axis=0)).astype(int)
        max_xy = np.ceil(tri.max(axis=0)).astype(int)
        x0 = max(0, min_xy[0])
        y0 = max(0, min_xy[1])
        x1 = min(width - 1, max_xy[0])
        y1 = min(height - 1, max_xy[1])
        if x1 < x0 or y1 < y0:
            continue
        xs, ys = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
        w0, w1, w2 = _barycentric(xs.astype(np.float32), ys.astype(np.float32), tri.astype(np.float32))
        inside = (w0 >= -1.0e-4) & (w1 >= -1.0e-4) & (w2 >= -1.0e-4)
        if not inside.any():
            continue
        tri_depth = w0 * depth[face[0]] + w1 * depth[face[1]] + w2 * depth[face[2]]
        z_patch = zbuf[y0 : y1 + 1, x0 : x1 + 1]
        update = inside & (tri_depth < z_patch)
        if not update.any():
            continue
        uv = w0[..., None] * uvs[face[0]] + w1[..., None] * uvs[face[1]] + w2[..., None] * uvs[face[2]]
        colors = _sample_texture(texture, uv)
        image_patch = image[y0 : y1 + 1, x0 : x1 + 1]
        image_patch[update] = colors[update]
        z_patch[update] = tri_depth[update]

    return image


def compare_nerf_mesh_views(
    model: torch.nn.Module,
    mesh: Mesh,
    uvs: np.ndarray,
    texture: np.ndarray,
    scene: BlenderScene,
    cfg: dict[str, object],
    device: torch.device,
    output_dir: str | Path,
    num_views: int = 3,
) -> list[dict[str, float]]:
    """Score matched NeRF and mesh renders against validation ground truth."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float]] = []
    val_indices = scene.split_indices["val"][:num_views]
    for view_index, val_idx in enumerate(val_indices):
        pose = scene.poses[int(val_idx)].to(device)
        gt = scene.images[int(val_idx)].numpy()
        nerf = render_image(model, scene.height, scene.width, scene.focal, pose, cfg, int(cfg["render_chunk"])).numpy()
        mesh_rgb = render_textured_mesh_view(mesh, uvs, texture, pose, scene.height, scene.width, scene.focal)
        comparison = np.concatenate([gt, nerf, mesh_rgb], axis=1)
        save_image(output_dir / f"view_{view_index:02d}_comparison.png", comparison)
        nerf_metrics = _compute_image_metrics(nerf, gt)
        mesh_metrics = _compute_image_metrics(mesh_rgb, gt)
        rows.append(
            {
                "view_index": float(view_index),
                **{f"nerf_{name}_to_gt": value for name, value in nerf_metrics.items()},
                **{f"mesh_{name}_to_gt": value for name, value in mesh_metrics.items()},
            }
        )

    csv_path = output_dir / "comparison.csv"
    metric_names = [
        "nerf_psnr_to_gt",
        "nerf_ssim_to_gt",
        "mesh_psnr_to_gt",
        "mesh_ssim_to_gt",
    ]
    mean_metrics = {name: float(np.mean([row[name] for row in rows])) for name in metric_names}
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["view_index", *metric_names])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "view_index": int(row["view_index"]),
                    **{name: f"{row[name]:.6f}" for name in metric_names},
                }
            )
        writer.writerow(
            {
                "view_index": "mean",
                **{name: f"{mean_metrics[name]:.6f}" for name in metric_names},
            }
        )
    print(f"saved {csv_path}")
    return rows


def run_mesh_pipeline(
    cfg: dict[str, object],
    checkpoint: str | Path,
    resolution: int,
    threshold: float,
    bounds: tuple[float, float],
    atlas_size: int,
) -> Path:
    """Run provided texturing/comparison after the student mesh extraction."""
    device = resolve_device(str(cfg.get("device", "auto")))
    scene = load_scene_from_config(cfg)
    model = load_trained_model(cfg, checkpoint, device)
    sigma_grid = build_density_grid(model, cfg, bounds, resolution, int(cfg["render_chunk"]), device)
    sigma_grid = carve_density_with_visual_hull(sigma_grid, cfg, bounds)
    mesh = extract_mesh_from_density(sigma_grid, bounds, threshold)
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise RuntimeError("mesh extraction returned no geometry; complete the Task 3 TODOs first")
    unclean_vertices, unclean_faces = len(mesh.vertices), len(mesh.faces)
    mesh = keep_largest_mesh_component(mesh)
    print(
        f"component cleanup retained {len(mesh.vertices):,}/{unclean_vertices:,} vertices "
        f"and {len(mesh.faces):,}/{unclean_faces:,} faces"
    )

    out_dir = task3_mesh_output_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    atlas_mesh, uvs, atlas_normals = unwrap_mesh_with_xatlas(mesh)
    train_indices = scene.split_indices["train"]
    camera_centers = scene.poses[train_indices, :3, 3].numpy()
    texture = bake_nerf_texture(
        model,
        cfg,
        atlas_mesh,
        uvs,
        atlas_normals,
        camera_centers,
        device,
        int(cfg["render_chunk"]),
        atlas_size=atlas_size,
        padding=8,
    )
    texture_path = out_dir / "lego_texture.png"
    save_image(texture_path, texture)
    vertex_colors = _sample_texture(texture, uvs)
    save_textured_obj(out_dir / "lego_textured.obj", atlas_mesh, uvs, texture_path.name)
    save_vertex_color_ply(out_dir / "lego_vertex_color_mesh.ply", atlas_mesh, vertex_colors)
    compare_nerf_mesh_views(model, atlas_mesh, uvs, texture, scene, cfg, device, out_dir, num_views=2)
    print(
        f"xatlas remapped {len(mesh.vertices)} mesh vertices to "
        f"{len(atlas_mesh.vertices)} UV-chart vertices"
    )
    print(f"saved mesh outputs under {out_dir}")
    return out_dir


def run_api_smoke() -> None:
    """Dataset-free check for task imports, model construction, and rendering."""
    cfg = default_config("smoke")
    cfg.update({"num_samples": 8, "perturb": False})
    device = torch.device("cpu")
    model = build_model(cfg, device)
    c2w = torch.eye(4)
    rays_o, rays_d = get_rays(4, 5, 4.0, c2w)
    out = render_rays(model, rays_o.reshape(-1, 3), rays_d.reshape(-1, 3), cfg)
    assert out["rgb"].shape == (20, 3)
    assert torch.isfinite(out["rgb"]).all()
    print("Task 3 API smoke check passed")


def _apply_overrides(cfg: dict[str, object], args: argparse.Namespace) -> dict[str, object]:
    """Apply command-line data, output, device, and iteration overrides."""
    if args.output_dir is not None:
        cfg["output_dir"] = args.output_dir
    if args.device is not None:
        cfg["device"] = args.device
    if args.iters is not None:
        cfg["iters"] = args.iters
    if args.data_dir is not None:
        cfg["data_dir"] = args.data_dir
    return cfg


def main() -> None:
    """Dispatch the requested Task 3 visualization, training, evaluation, or mesh mode."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["check", "visualize", "smoke", "train", "eval", "turntable", "mesh", "all"],
        default="check",
    )
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output-dir", default=None, help="Task 3 run directory, e.g. outputs/task3/lego_train.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--iters", type=int, default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--num-frames", type=int, default=12)
    parser.add_argument("--mesh-resolution", type=int, default=96)
    parser.add_argument("--mesh-threshold", type=float, default=20.0)
    parser.add_argument("--mesh-bound-min", type=float, default=-1.5)
    parser.add_argument("--mesh-bound-max", type=float, default=1.5)
    parser.add_argument("--atlas-size", type=int, default=256)
    args = parser.parse_args()

    if args.mode == "check":
        run_api_smoke()
        return

    cfg_kind = "smoke" if args.mode in {"visualize", "smoke"} else "train"
    cfg = _apply_overrides(default_config(cfg_kind), args)
    checkpoint = args.checkpoint or str(Path(str(cfg["output_dir"])) / "model.pt")

    if args.mode == "visualize":
        visualize_dataset(cfg)
    elif args.mode == "smoke":
        train_tiny_nerf(cfg)
    elif args.mode == "train":
        train_tiny_nerf(cfg)
    elif args.mode == "eval":
        evaluate_validation_set(cfg, checkpoint)
    elif args.mode == "turntable":
        render_turntable(cfg, checkpoint, num_frames=args.num_frames)
    elif args.mode == "mesh":
        run_mesh_pipeline(
            cfg,
            checkpoint,
            resolution=args.mesh_resolution,
            threshold=args.mesh_threshold,
            bounds=(args.mesh_bound_min, args.mesh_bound_max),
            atlas_size=args.atlas_size,
        )
    elif args.mode == "all":
        visualize_dataset(default_config("smoke"))
        train_tiny_nerf(cfg)
        evaluate_validation_set(cfg, checkpoint)
        render_turntable(cfg, checkpoint, num_frames=args.num_frames)
        run_mesh_pipeline(
            cfg,
            checkpoint,
            resolution=args.mesh_resolution,
            threshold=args.mesh_threshold,
            bounds=(args.mesh_bound_min, args.mesh_bound_max),
            atlas_size=args.atlas_size,
        )


if __name__ == "__main__":
    main()
