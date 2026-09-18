"""Provided geometry, visualization, and output helpers for HW1."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
    os.environ.setdefault("EGL_PLATFORM", "surfaceless")

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
try:
    import open3d as o3d
except ImportError:
    o3d = None  # Numerical work is available with --skip-rendering.


def project_rigid_transform(transform: np.ndarray) -> np.ndarray:
    """Remove rounding-induced scale/shear from a supplied demo transform."""
    result = np.array(transform, dtype=np.float64, copy=True)
    left, _, right_t = np.linalg.svd(result[:3, :3])
    correction = np.eye(3)
    correction[2, 2] = np.linalg.det(left @ right_t)
    result[:3, :3] = left @ correction @ right_t
    result[3] = [0, 0, 0, 1]
    return result


SEMANTIC_COLORS = np.array(
    [
        [0.20, 0.45, 0.95], [0.95, 0.30, 0.75], [0.90, 0.18, 0.45],
        [0.15, 0.75, 0.95], [0.35, 0.65, 0.85], [0.95, 0.18, 0.18],
        [0.85, 0.35, 0.15], [0.95, 0.55, 0.20], [0.28, 0.30, 0.34],
        [0.52, 0.42, 0.65], [0.94, 0.78, 0.18], [0.60, 0.52, 0.34],
        [0.78, 0.45, 0.22], [0.64, 0.50, 0.38], [0.20, 0.72, 0.28],
        [0.46, 0.30, 0.18], [0.46, 0.72, 0.32], [0.75, 0.75, 0.78],
        [0.98, 0.63, 0.10],
    ]
)


def to_homogeneous(points: np.ndarray) -> np.ndarray:
    """Convert an [N, 3] library array to a [4, N] homogeneous point matrix."""
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Expected an [N, 3] point array")
    return np.vstack([points.T, np.ones((1, len(points)), dtype=points.dtype)])


def to_xyz(points_h: np.ndarray) -> np.ndarray:
    """Convert a [4, N] homogeneous point matrix to an [N, 3] library array."""
    points_h = np.asarray(points_h)
    return (points_h[:3] / points_h[3:4]).T


def transform_points(transform: np.ndarray, points_h: np.ndarray) -> np.ndarray:
    """Apply T_A_to_B to homogeneous columns: P_B = T_A_to_B @ P_A."""
    return np.asarray(transform) @ np.asarray(points_h)


def rotation_error_degrees(estimate: np.ndarray, reference: np.ndarray) -> float:
    """Return the geodesic SO(3) error in degrees."""
    delta = estimate[:3, :3] @ reference[:3, :3].T
    cosine = np.clip((np.trace(delta) - 1.0) / 2.0, -1.0, 1.0)
    if cosine > 1.0 - 1e-12:
        return 0.0
    return float(np.degrees(np.arccos(cosine)))


def write_json(path: Path, values: dict) -> None:
    """Write a result dictionary and convert NumPy scalar values."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(values, indent=2, default=lambda value: value.item()) + "\n",
        encoding="utf-8",
    )


def gt_voxel_majority(
    indices: np.ndarray,
    labels: np.ndarray,
    grid_shape: tuple[int, int, int],
    class_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return sparse coordinates and majority GT labels for staff evaluation."""
    valid = ((indices >= 0) & (indices < np.asarray(grid_shape))).all(axis=1)
    coordinates, inverse = np.unique(indices[valid], axis=0, return_inverse=True)
    histogram = np.zeros((len(coordinates), class_count), dtype=np.int64)
    labels = labels[valid]
    labeled = labels < class_count
    np.add.at(histogram, (inverse[labeled], labels[labeled]), 1)
    return coordinates, histogram.argmax(axis=1), histogram


def semantic_metrics(
    prediction: np.ndarray, target: np.ndarray, class_count: int
) -> dict[str, float | list[float]]:
    """Return voxel accuracy, per-class IoU, and mean IoU."""
    confusion = np.bincount(
        target * class_count + prediction, minlength=class_count**2
    ).reshape(class_count, class_count)
    intersection = np.diag(confusion)
    union = confusion.sum(0) + confusion.sum(1) - intersection
    iou = np.divide(
        intersection, union, out=np.full(class_count, np.nan), where=union > 0
    )
    return {
        "voxel_accuracy": float(np.mean(prediction == target)),
        "per_class_iou": iou.tolist(),
        "mean_iou": float(np.nanmean(iou)),
    }


def estimate_normals(points: np.ndarray, radius: float = 0.12) -> np.ndarray:
    """Estimate consistently oriented local normals with Open3D."""
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    cloud.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30)
    )
    cloud.orient_normals_consistent_tangent_plane(20)
    return np.asarray(cloud.normals).copy()


def make_surfel_mesh(
    points: np.ndarray,
    normals: np.ndarray,
    color: tuple[float, float, float],
    radius: float = 0.018,
) -> o3d.geometry.TriangleMesh:
    """Represent each point as an equal-metric square surfel."""
    reference = np.tile([0.0, 0.0, 1.0], (len(points), 1))
    reference[np.abs(normals[:, 2]) > 0.9] = [1.0, 0.0, 0.0]
    tangent = np.cross(normals, reference)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-12)
    bitangent = np.cross(normals, tangent)
    vertices = np.stack(
        [
            points - radius * tangent - radius * bitangent,
            points + radius * tangent - radius * bitangent,
            points + radius * tangent + radius * bitangent,
            points - radius * tangent + radius * bitangent,
        ],
        axis=1,
    ).reshape(-1, 3)
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    triangles = (faces[None] + 4 * np.arange(len(points))[:, None, None]).reshape(
        -1, 3
    )
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices),
        o3d.utility.Vector3iVector(triangles),
    )
    mesh.vertex_normals = o3d.utility.Vector3dVector(np.repeat(normals, 4, axis=0))
    mesh.vertex_colors = o3d.utility.Vector3dVector(
        np.tile(np.asarray(color), (4 * len(points), 1))
    )
    return mesh


def save_alignment_visuals(
    png_path: Path,
    source: np.ndarray,
    target: np.ndarray,
    transforms: np.ndarray,
    video_path: Path | None = None,
) -> None:
    """Render a fixed-view before/after image and optional iteration video."""
    source_xyz = to_xyz(source)
    target_xyz = to_xyz(target)
    source_normals = estimate_normals(source_xyz)
    target_normals = estimate_normals(target_xyz)
    camera_points = np.concatenate(
        [
            target_xyz,
            to_xyz(transform_points(transforms[0], source)),
            to_xyz(transform_points(transforms[-1], source)),
        ]
    )
    lower, upper = camera_points.min(axis=0), camera_points.max(axis=0)
    look_at = 0.5 * (lower + upper)
    half_width = 0.5 * (upper[0] - lower[0])
    half_height = 0.5 * (upper[1] - lower[1])
    distance = 1.18 * max(half_height, half_width / (900.0 / 700.0)) / np.tan(
        np.deg2rad(24.0)
    )
    eye = look_at + np.array([0.0, -0.06 * (upper[1] - lower[1]), -distance])

    renderer = o3d.visualization.rendering.OffscreenRenderer(900, 700)
    renderer.scene.set_background(np.array([0.94, 0.95, 0.97, 1.0]))
    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultLit"
    renderer.scene.add_geometry(
        "target",
        make_surfel_mesh(target_xyz, target_normals, (0.10, 0.42, 0.88)),
        material,
    )
    renderer.setup_camera(
        48.0,
        look_at.astype(np.float32),
        eye.astype(np.float32),
        np.array([0.0, -1.0, 0.0], dtype=np.float32),
    )

    def render(transform: np.ndarray) -> np.ndarray:
        if renderer.scene.has_geometry("source"):
            renderer.scene.remove_geometry("source")
        aligned = to_xyz(transform_points(transform, source))
        aligned_normals = source_normals @ transform[:3, :3].T
        renderer.scene.add_geometry(
            "source",
            make_surfel_mesh(aligned, aligned_normals, (0.95, 0.42, 0.08)),
            material,
        )
        return np.asarray(renderer.render_to_image())

    before, after = render(transforms[0]), render(transforms[-1])
    divider = np.full((before.shape[0], 12, 3), 30, dtype=np.uint8)
    imageio.imwrite(png_path, np.concatenate([before, divider, after], axis=1))
    if video_path is None:
        return
    frame_ids = [0] * 12
    frame_ids += [index for index in range(len(transforms)) for _ in range(2)]
    frame_ids += [len(transforms) - 1] * 12
    with imageio.get_writer(
        video_path, fps=24, codec="libx264", quality=8, macro_block_size=None
    ) as writer:
        for index in frame_ids:
            writer.append_data(render(transforms[index]))


def save_ransac_plot(
    path: Path,
    aligned: np.ndarray,
    inliers: np.ndarray,
    residuals: np.ndarray,
    threshold: float,
) -> None:
    """Save the Task 2 residual histogram and inlier visualization."""
    aligned = to_xyz(aligned)
    figure, axes = plt.subplots(1, 2, figsize=(9.0, 4.0))
    axes[0].hist(residuals, bins=35, color="tab:blue")
    axes[0].axvline(threshold, color="black", linestyle="--", label="threshold")
    axes[0].set(xlabel="pair residual (m)", ylabel="count", title="RANSAC residuals")
    axes[0].legend()
    axes[1].scatter(
        aligned[:, 0], aligned[:, 1], c=inliers, s=10, cmap="coolwarm"
    )
    axes[1].set(
        aspect="equal", title="Blue: rejected; red: inlier", xlabel="x", ylabel="y"
    )
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def save_convergence_plot(path: Path, histories: list[np.ndarray]) -> None:
    """Plot the median and 10--90% range of pairwise ICP histories."""
    histories = [history for history in histories if len(history)]
    figure, axis = plt.subplots(figsize=(6.2, 3.8))
    if histories:
        curves = np.full((len(histories), max(map(len, histories))), np.nan)
        for row, history in enumerate(histories):
            curves[row, : len(history)] = history
        iteration = np.arange(1, curves.shape[1] + 1)
        axis.fill_between(
            iteration,
            np.nanpercentile(curves, 10, axis=0),
            np.nanpercentile(curves, 90, axis=0),
            color="#4c78a8",
            alpha=0.22,
            label="10--90% pairs",
        )
        axis.plot(
            iteration,
            np.nanmedian(curves, axis=0),
            color="#1f4e79",
            linewidth=2.2,
            label="median pair",
        )
        axis.legend()
    else:
        axis.text(
            0.5,
            0.5,
            "Implement ICP to populate convergence",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
    axis.set(
        xlabel="ICP iteration",
        ylabel="trimmed RMSE (m)",
        title="Sequential LiDAR ICP convergence",
    )
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def save_trajectory_plot(
    path: Path, ground_truth: np.ndarray, odometry: np.ndarray
) -> None:
    """Save an equal-scale 2D GT/odometry trajectory comparison."""
    gt_xy = ground_truth[:, :2, 3]
    odometry_xy = odometry[:, :2, 3]
    figure, axis = plt.subplots(figsize=(7.2, 6.2))
    axis.plot(gt_xy[:, 0], gt_xy[:, 1], "k-", linewidth=2.5, label="GT")
    axis.plot(
        odometry_xy[:, 0],
        odometry_xy[:, 1],
        "--",
        linewidth=2.0,
        label="ICP odometry",
    )
    axis.scatter(
        *gt_xy[0], c="limegreen", edgecolors="black", s=45, zorder=4, label="start"
    )
    axis.set(
        aspect="equal",
        xlabel="world x (m)",
        ylabel="world y (m)",
        title="Curved sequence: ICP odometry versus GT",
    )
    axis.legend(loc="best")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _accumulate_lidar(scans: list[np.ndarray], poses: np.ndarray) -> np.ndarray:
    points = []
    for scan, pose in zip(scans, poses):
        distance = np.linalg.norm(scan[:2], axis=0)
        keep = (distance < 40.0) & (scan[2] > -2.8) & (scan[2] < 3.0)
        points.append(to_xyz(transform_points(pose, scan[:, keep])))
    return np.concatenate(points)


def save_bev_comparison(
    path: Path,
    scans: list[np.ndarray],
    ground_truth: np.ndarray,
    odometry: np.ndarray,
) -> None:
    """Save equal-scale accumulated LiDAR BEVs for GT and odometry."""
    methods = [("Ground truth", ground_truth), ("ICP odometry", odometry)]
    maps = [_accumulate_lidar(scans, poses) for _, poses in methods]
    lower, upper = np.percentile(np.concatenate(maps)[:, :2], [0.2, 99.8], axis=0)
    lower -= 4.0
    upper += 4.0
    resolution = max(0.18, float(np.max(upper - lower) / 1000.0))
    bins = np.maximum(2, np.ceil((upper - lower) / resolution).astype(int))
    gt_xy = ground_truth[:, :2, 3]
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 5.6), sharex=True, sharey=True)
    for axis, (title, poses), points in zip(axes, methods, maps):
        density, _, _ = np.histogram2d(
            points[:, 1],
            points[:, 0],
            bins=(bins[1], bins[0]),
            range=((lower[1], upper[1]), (lower[0], upper[0])),
        )
        density = np.log1p(density)
        positive = density[density > 0]
        maximum = np.percentile(positive, 99.5) if len(positive) else 1.0
        axis.imshow(
            density,
            cmap="gray",
            origin="lower",
            extent=(lower[0], upper[0], lower[1], upper[1]),
            vmin=0.0,
            vmax=maximum,
        )
        trajectory = poses[:, :2, 3]
        if title != "Ground truth":
            axis.plot(
                gt_xy[:, 0], gt_xy[:, 1], "--", color="cyan", linewidth=1.4, label="GT"
            )
        axis.plot(
            trajectory[:, 0],
            trajectory[:, 1],
            color="red",
            linewidth=1.8,
            label="trajectory",
        )
        axis.scatter(
            *trajectory[0], c="lime", s=28, edgecolors="black", linewidths=0.4
        )
        axis.set(
            title=title, xlabel="world x (m)", ylabel="world y (m)", aspect="equal"
        )
        axis.legend(loc="upper right")
    figure.suptitle("Accumulated LiDAR BEV: sharp structure indicates consistent poses")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def save_icp_odometry_video(
    video_path: Path,
    points: np.ndarray,
    offsets: np.ndarray,
    poses: np.ndarray,
    local_range: float = 50.0,
    panel_size: int = 636,
) -> None:
    """Render online ICP odometry as current-sweep and accumulated-intensity BEVs.

    Both panels stay centered on the current LiDAR with forward pointing up and
    cover ``[-local_range, local_range]`` meters on each horizontal axis. The
    left panel uses every point in the stored current sweep. The right panel
    contains only measurements accumulated through the current frame.
    """
    frame_count = len(offsets) - 1
    if len(poses) != frame_count:
        raise ValueError("Expected one LiDAR-to-world pose per frame")

    # A world-aligned raster stores the online map. Its resolution matches the
    # displayed local window, so the renderer never changes physical scale.
    resolution = 2.0 * local_range / (panel_size - 1)
    margin = local_range + 2.0
    lower = poses[:, :2, 3].min(axis=0) - margin
    upper = poses[:, :2, 3].max(axis=0) + margin
    grid_shape = np.ceil((upper - lower) / resolution).astype(np.int64) + 1
    map_intensity = np.zeros((grid_shape[1], grid_shape[0]), dtype=np.float32)
    map_occupied = np.zeros_like(map_intensity, dtype=bool)

    coordinate = np.linspace(local_range, -local_range, panel_size)
    local_x, local_y = np.meshgrid(coordinate, coordinate, indexing="ij")
    inferno = plt.get_cmap("inferno")
    background = np.array([6, 8, 14], dtype=np.uint8)

    def colorize(values: np.ndarray, occupied: np.ndarray) -> np.ndarray:
        color_coordinate = 0.12 + 0.86 * np.clip(values / 0.65, 0.0, 1.0)
        image = (255.0 * inferno(color_coordinate)[..., :3]).astype(np.uint8)
        image[~occupied] = background
        return image

    def current_sweep_bev(scan: np.ndarray) -> np.ndarray:
        keep = (np.abs(scan[:, 0]) <= local_range) & (
            np.abs(scan[:, 1]) <= local_range
        )
        scan = scan[keep]
        rows = np.rint(
            (local_range - scan[:, 0]) * (panel_size - 1) / (2.0 * local_range)
        ).astype(np.int64)
        columns = np.rint(
            (local_range - scan[:, 1]) * (panel_size - 1) / (2.0 * local_range)
        ).astype(np.int64)
        values = np.zeros((panel_size, panel_size), dtype=np.float32)
        occupied = np.zeros_like(values, dtype=bool)
        # Expand each projected sample to 2x2 pixels for legibility while still
        # rasterizing every point in the current sweep.
        for row_shift, column_shift in ((0, 0), (1, 0), (0, 1), (1, 1)):
            rr = np.clip(rows + row_shift, 0, panel_size - 1)
            cc = np.clip(columns + column_shift, 0, panel_size - 1)
            np.maximum.at(values, (rr, cc), scan[:, 3])
            occupied[rr, cc] = True
        return colorize(values, occupied)

    def draw_trajectory(image: np.ndarray, frame: int, pose: np.ndarray) -> None:
        """Overlay inferred poses through ``frame`` in the current LiDAR view."""
        world_history = poses[: frame + 1, :2, 3]
        local_history = (world_history - pose[:2, 3]) @ pose[:2, :2]
        rows = (local_range - local_history[:, 0]) * (panel_size - 1) / (
            2.0 * local_range
        )
        columns = (local_range - local_history[:, 1]) * (panel_size - 1) / (
            2.0 * local_range
        )
        line_rows, line_columns = [], []
        for start in range(len(rows) - 1):
            steps = max(
                2,
                int(
                    np.ceil(
                        max(
                            abs(rows[start + 1] - rows[start]),
                            abs(columns[start + 1] - columns[start]),
                        )
                    )
                ),
            )
            line_rows.append(np.linspace(rows[start], rows[start + 1], steps))
            line_columns.append(
                np.linspace(columns[start], columns[start + 1], steps)
            )
        if not line_rows:
            return
        rr = np.rint(np.concatenate(line_rows)).astype(np.int64)
        cc = np.rint(np.concatenate(line_columns)).astype(np.int64)
        keep = (rr >= 0) & (cc >= 0) & (rr < panel_size) & (cc < panel_size)
        rr, cc = rr[keep], cc[keep]
        for row_shift in (-1, 0, 1):
            for column_shift in (-1, 0, 1):
                image[
                    np.clip(rr + row_shift, 0, panel_size - 1),
                    np.clip(cc + column_shift, 0, panel_size - 1),
                ] = [35, 225, 255]

    video_path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(
        video_path, fps=24, codec="libx264", quality=6, macro_block_size=None
    ) as writer:
        for frame, pose in enumerate(poses):
            scan = points[offsets[frame] : offsets[frame + 1]]
            local_keep = (np.abs(scan[:, 0]) <= local_range) & (
                np.abs(scan[:, 1]) <= local_range
            )
            local_scan = scan[local_keep]
            world_xy = np.column_stack(
                [
                    pose[0, 0] * local_scan[:, 0]
                    + pose[0, 1] * local_scan[:, 1]
                    + pose[0, 3],
                    pose[1, 0] * local_scan[:, 0]
                    + pose[1, 1] * local_scan[:, 1]
                    + pose[1, 3],
                ]
            )
            grid_xy = np.rint((world_xy - lower) / resolution).astype(np.int64)
            valid = ((grid_xy >= 0) & (grid_xy < grid_shape)).all(axis=1)
            gx, gy = grid_xy[valid, 0], grid_xy[valid, 1]
            np.maximum.at(map_intensity, (gy, gx), local_scan[valid, 3])
            map_occupied[gy, gx] = True

            # Sample the online world map through the current LiDAR pose so the
            # right panel follows both its position and heading.
            world_x = (
                pose[0, 0] * local_x
                + pose[0, 1] * local_y
                + pose[0, 3]
            )
            world_y = (
                pose[1, 0] * local_x
                + pose[1, 1] * local_y
                + pose[1, 3]
            )
            sample_x = np.rint((world_x - lower[0]) / resolution).astype(np.int64)
            sample_y = np.rint((world_y - lower[1]) / resolution).astype(np.int64)
            valid = (
                (sample_x >= 0)
                & (sample_y >= 0)
                & (sample_x < grid_shape[0])
                & (sample_y < grid_shape[1])
            )
            accumulated = np.zeros((panel_size, panel_size), dtype=np.float32)
            accumulated_occupied = np.zeros_like(accumulated, dtype=bool)
            accumulated[valid] = map_intensity[sample_y[valid], sample_x[valid]]
            accumulated_occupied[valid] = map_occupied[
                sample_y[valid], sample_x[valid]
            ]

            left = current_sweep_bev(scan)
            right = colorize(accumulated, accumulated_occupied)
            draw_trajectory(right, frame, pose)
            center = panel_size // 2
            for image in (left, right):
                image[center - 28 : center + 2, center - 1 : center + 2] = 255
                image[center - 30 : center - 24, center - 6 : center + 7] = 255
                image[center - 3 : center + 4, center - 3 : center + 4] = [40, 220, 255]
            divider = np.full((panel_size, 8, 3), 235, dtype=np.uint8)
            writer.append_data(np.concatenate([left, divider, right], axis=1))


def write_semantic_ply(
    path: Path, xyz: np.ndarray, labels: np.ndarray, colors: np.ndarray
) -> None:
    """Write colored voxel centers as a compact Open3D PLY point cloud."""
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
    cloud.colors = o3d.utility.Vector3dVector(colors[labels])
    path.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_point_cloud(str(path), cloud, compressed=True):
        raise OSError(f"Could not write {path}")


def make_voxel_cube_mesh(
    centers: np.ndarray,
    labels: np.ndarray,
    voxel_size: float,
    colors: np.ndarray,
) -> o3d.geometry.TriangleMesh:
    """Build one Open3D triangle mesh containing one cube per voxel."""
    corners = 0.46 * voxel_size * np.array(
        [
            [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
            [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
        ],
        dtype=np.float64,
    )
    faces = np.array(
        [
            [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
            [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7],
        ],
        dtype=np.int32,
    )
    vertices = (centers[:, None] + corners).reshape(-1, 3)
    triangles = (faces[None] + 8 * np.arange(len(centers))[:, None, None]).reshape(
        -1, 3
    )
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(triangles)
    )
    mesh.vertex_colors = o3d.utility.Vector3dVector(
        np.repeat(colors[labels], 8, axis=0)
    )
    mesh.compute_vertex_normals()
    return mesh


def save_voxel_turntable(
    mesh: o3d.geometry.TriangleMesh,
    centers: np.ndarray,
    png_path: Path,
    video_path: Path,
) -> None:
    """Render equal-scale cubes from a shoulder-view orbit."""
    renderer = o3d.visualization.rendering.OffscreenRenderer(1280, 720)
    renderer.scene.set_background(np.array([0.035, 0.045, 0.065, 1.0]))
    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultLit"
    renderer.scene.add_geometry("semantic_voxels", mesh, material)
    center = 0.5 * (centers.min(axis=0) + centers.max(axis=0))
    distance = 0.95 * float(np.max(np.ptp(centers, axis=0)))
    up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    with imageio.get_writer(
        video_path, fps=24, codec="libx264", quality=8, macro_block_size=None
    ) as writer:
        for frame in range(96):
            azimuth = np.deg2rad(35.0 + 360.0 * frame / 96)
            eye = center + distance * np.array(
                [np.cos(azimuth), np.sin(azimuth), 0.52]
            )
            renderer.setup_camera(
                52.0, center.astype(np.float32), eye.astype(np.float32), up
            )
            image = np.asarray(renderer.render_to_image())
            if frame == 0:
                imageio.imwrite(png_path, image)
            writer.append_data(image)


def save_lidar_follow_video(
    mesh: o3d.geometry.TriangleMesh,
    points: np.ndarray,
    offsets: np.ndarray,
    poses: np.ndarray,
    video_path: Path,
) -> None:
    """Render each full stored sweep and the map from one moving camera."""
    renderer = o3d.visualization.rendering.OffscreenRenderer(640, 720)
    renderer.scene.set_background(np.array([0.025, 0.032, 0.048, 1.0]))
    map_material = o3d.visualization.rendering.MaterialRecord()
    map_material.shader = "defaultLit"
    renderer.scene.add_geometry("semantic_voxels", mesh, map_material)
    scan_material = o3d.visualization.rendering.MaterialRecord()
    scan_material.shader = "defaultUnlit"
    scan_material.point_size = 3.0
    inferno = plt.get_cmap("inferno")
    with imageio.get_writer(
        video_path, fps=24, codec="libx264", quality=6, macro_block_size=None
    ) as writer:
        for frame, pose in enumerate(poses):
            scan = points[offsets[frame] : offsets[frame + 1], :3]
            distance = np.linalg.norm(scan, axis=1)
            scan_h = to_homogeneous(scan)
            cloud = o3d.geometry.PointCloud(
                o3d.utility.Vector3dVector(to_xyz(transform_points(pose, scan_h)))
            )
            # Skip the near-black end of inferno so close points stay visible.
            color_coordinate = 0.12 + 0.86 * np.clip(distance / 60.0, 0.0, 1.0)
            cloud.colors = o3d.utility.Vector3dVector(
                inferno(color_coordinate)[:, :3]
            )
            if renderer.scene.has_geometry("current_scan"):
                renderer.scene.remove_geometry("current_scan")
            renderer.scene.add_geometry("current_scan", cloud, scan_material)
            position, forward, up = pose[:3, 3], pose[:3, 0], pose[:3, 2]
            renderer.setup_camera(
                70.0,
                (position + 12.0 * forward).astype(np.float32),
                (position + 3.0 * up).astype(np.float32),
                up.astype(np.float32),
            )
            renderer.scene.show_geometry("semantic_voxels", False)
            scan_image = np.asarray(renderer.render_to_image())
            renderer.scene.show_geometry("semantic_voxels", True)
            renderer.scene.show_geometry("current_scan", False)
            map_image = np.asarray(renderer.render_to_image())
            renderer.scene.show_geometry("current_scan", True)
            writer.append_data(np.concatenate([scan_image, map_image], axis=1))
