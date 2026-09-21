"""HW0 Task 2: full camera projection and 3D bunny insertion.

Work from Checkpoint 2A through 2C, then run:

    python task2_projection.py
"""

from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

from hw0_utils import (
    LANDMARK_UV,
    LANDMARK_XYZ,
    decompose_projection,
    float_image,
    load_bunny,
    project_points,
    save_rgb,
    write_json,
)


# ------------------------ Task 2: Modify your code below ------------------------

def estimate_projection_matrix(xyz: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Estimate a 3 x 4 camera matrix from N >= 6 3D-to-2D pairs.

    Replace the runnable placeholder with projection-matrix DLT. Convert each
    XYZ point to homogeneous coordinates, contribute two rows to a 2N x 12
    design matrix using its matching UV pixel, solve the right null space with
    SVD, reshape to 3 x 4, and choose a stable scale.
    """
    cent_xyz = xyz.mean(axis=0)
    cent_uv = uv.mean(axis=0)
    N = len(xyz)
    
    s_xyz = np.sqrt(3) / np.linalg.norm(xyz - cent_xyz, axis=1).mean()
    s_uv = np.sqrt(2) / np.linalg.norm(uv - cent_uv, axis=1).mean()
    T_xyz = np.array([[s_xyz, 0, 0, -s_xyz * cent_xyz[0]], [0, s_xyz, 0, -s_xyz * cent_xyz[1]], [0, 0, s_xyz, -s_xyz * cent_xyz[2]], [0, 0, 0, 1]])
    T_uv = np.array([[s_uv, 0, -s_uv * cent_uv[0]], [0, s_uv, -s_uv * cent_uv[1]], [0, 0, 1]])

    hom_xyz = np.hstack([xyz, np.ones((N, 1))])
    hom_uv = np.hstack([uv, np.ones((N, 1))])

    normalized_xyz = (T_xyz @ hom_xyz.T).T
    normalized_uv = (T_uv @ hom_uv.T).T

    A = np.zeros((2 * N, 12))
    for i in range(N):
        x, y, z = normalized_xyz[i, 0], normalized_xyz[i, 1], normalized_xyz[i, 2]
        u, v = normalized_uv[i, 0], normalized_uv[i, 1]
        A[2 * i] = [-x, -y, -z, -1,  0,  0,  0, 0, u*x, u*y, u*z, u]
        A[2 * i + 1] = [0,  0,  0, 0, -x, -y, -z, -1,  v*x, v*y, v*z, v]

    U, S, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3, 4)

    H_final = np.linalg.inv(T_uv) @ H @ T_xyz
    return H_final / np.linalg.norm(H_final)


# ------------------- DO NOT MODIFY CODE OUTSIDE THE BLOCK --------------------


def main() -> dict[str, float | int]:
    """Run the three Task 2 checkpoints in the same order as the handout."""
    root = Path(__file__).resolve().parent
    output_dir = root / "outputs/task2"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Checkpoint 2A: estimate P and inspect landmark reprojection.
    projection = estimate_projection_matrix(LANDMARK_XYZ, LANDMARK_UV)
    reprojected_uv = project_points(LANDMARK_XYZ, projection)
    reprojection_error = np.linalg.norm(reprojected_uv - LANDMARK_UV, axis=1)
    court = float_image(root / "data/court/court.png")[..., :3]
    figure, axis = plt.subplots(figsize=(11, 6.5))
    axis.imshow(court)
    axis.scatter(LANDMARK_UV[:, 0], LANDMARK_UV[:, 1], c="#e84a27", marker="x", s=48, label="given")
    axis.scatter(
        reprojected_uv[:, 0],
        reprojected_uv[:, 1],
        facecolors="none",
        edgecolors="#00b7ff",
        s=70,
        label="reprojected",
    )
    axis.legend(loc="lower left")
    axis.set_title("Checkpoint 2A: projection-matrix reprojection")
    axis.axis("off")
    figure.tight_layout()
    figure.savefig(output_dir / "task2a_reprojection.png", dpi=150)
    plt.close(figure)
    print(f"Checkpoint 2A saved; mean error = {np.mean(reprojection_error):.3f} px")

    # Checkpoint 2B: place, project, and render the Stanford bunny.
    task1_background = root / "outputs/task1/task1b_basketball_insertion.png"
    base_image = float_image(task1_background)[..., :3] if task1_background.exists() else court
    bunny_xyz = load_bunny(root / "data/court/bunny.ply")
    bunny_xyz[:, 0] += 28.65 / 2.0
    bunny_xyz[:, 1] += 15.24 / 2.0
    bunny_xyz[:, 2] -= bunny_xyz[:, 2].min()
    bunny_uv = project_points(bunny_xyz, projection)
    visible = (
        (bunny_uv[:, 0] >= 0)
        & (bunny_uv[:, 0] < base_image.shape[1])
        & (bunny_uv[:, 1] >= 0)
        & (bunny_uv[:, 1] < base_image.shape[0])
    )
    rendered = base_image.copy()
    height_fraction = bunny_xyz[:, 2] / max(float(np.ptp(bunny_xyz[:, 2])), 1e-12)
    colors = plt.colormaps["cool"](height_fraction)[:, :3]
    for (u, v), color in zip(bunny_uv[visible][::2], colors[visible][::2], strict=True):
        cv2.circle(rendered, (int(round(u)), int(round(v))), 1, tuple(color), -1)
    save_rgb(output_dir / "task2b_bunny_insertion.png", rendered)
    print("Checkpoint 2B saved: task2b_bunny_insertion.png")

    # Checkpoint 2C: decompose P and visualize the recovered camera pose.
    pose = decompose_projection(projection)
    center = pose["C"]
    optical_axis = pose["R"].T @ np.array([0.0, 0.0, 1.0])
    figure, axis = plt.subplots(figsize=(9.5, 6))
    axis.add_patch(plt.Rectangle((0, 0), 28.65, 15.24, facecolor="#f6b26b", edgecolor="#13294b"))
    axis.scatter(center[0], center[1], marker="*", s=180, color="#e84a27", label="camera center")
    axis.arrow(
        center[0],
        center[1],
        12 * optical_axis[0],
        12 * optical_axis[1],
        color="#00a6d2",
        width=0.12,
        head_width=0.8,
    )
    axis.text(center[0] - 14, center[1] - 1, f"C = ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}) m")
    axis.set_aspect("equal")
    axis.set_xlim(min(-3, center[0] - 3), max(31.5, center[0] + 3))
    axis.set_ylim(min(-3, center[1] - 3), max(18.5, center[1] + 3))
    axis.set_xlabel("world x (m)")
    axis.set_ylabel("world y (m)")
    axis.legend(loc="upper right")
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_dir / "task2c_camera_pose.png", dpi=150)
    plt.close(figure)

    metrics: dict[str, float | int] = {
        "mean_reprojection_error_px": float(np.mean(reprojection_error)),
        "max_reprojection_error_px": float(np.max(reprojection_error)),
        "camera_height_m": float(center[2]),
        "visible_bunny_vertices": int(np.count_nonzero(visible)),
    }
    write_json(output_dir / "task2c_metrics.json", metrics)
    np.savez(output_dir / "task2_results.npz", projection=projection, **pose)
    print("Checkpoint 2C saved: camera pose and task2c_metrics.json")
    return metrics


if __name__ == "__main__":
    print("Task 2 complete:", main())
