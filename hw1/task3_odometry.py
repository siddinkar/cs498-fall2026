"""HW1 Task 3: ICP followed by LiDAR odometry.

Run directly:

    python task3_odometry.py
"""

from __future__ import annotations

from pathlib import Path
import argparse

import numpy as np

from hw1_utils import (
    project_rigid_transform,
    rotation_error_degrees,
    save_alignment_visuals,
    save_bev_comparison,
    save_convergence_plot,
    save_icp_odometry_video,
    save_trajectory_plot,
    to_homogeneous,
    transform_points,
    write_json,
)
from task1_alignment import fit_rigid_svd, nearest_correspondences


# ------------------------ Task 3: Modify your code below ------------------------

def icp(
    source: np.ndarray,
    target: np.ndarray,
    max_iterations: int = 60,
    max_correspondence_distance: float = 1.5,
    trim_fraction: float = 0.8,
    tolerance: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return T_source_to_target, RMSE history, and transform history.

    ``source`` and ``target`` are homogeneous [4, N] matrices. Repeatedly
    transform the original source, find nearest neighbors with Task
    1B, retain the closest ``trim_fraction`` pairs, fit an SVD update with Task
    1A, and left-compose it. If the current estimate is T_source_to_target and
    the fitted incremental correction is delta_T, the new estimate is
    delta_T @ T_source_to_target. Stop when the RMSE change is below tolerance.
    Keep max(3, ceil(trim_fraction * accepted_count)) matches. If fewer than
    three are accepted, stop. Record post-update RMSE on the retained pairs,
    without re-matching, then compare absolute change with the previous value.
    The transform trace begins with identity and includes every update.
    """
    del target, max_iterations, max_correspondence_distance, trim_fraction, tolerance
    return np.eye(4), np.empty(0), np.eye(4)[None]


def chain_relative_poses(relative_transforms: np.ndarray) -> np.ndarray:
    """Chain T_(L_i+1 -> L_i) transforms into T_(L_i -> W) poses.

    Use T_(L_i+1 -> W) = T_(L_i -> W) @ T_(L_i+1 -> L_i).
    Do not add pose components.
    """
    return np.repeat(np.eye(4)[None], len(relative_transforms) + 1, axis=0)


def relative_trajectory_error(
    estimated_poses: np.ndarray,
    ground_truth_poses: np.ndarray,
    delta: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Return translation error in meters and rotation error in degrees.

    Each pose is T_(L_i -> W). Compare inv(T_(L_i -> W)) @
    T_(L_i+delta -> W) for the estimate and reference, then measure the error
    transform using the equations in the handout.
    """
    del ground_truth_poses
    count = max(len(estimated_poses) - delta, 0)
    return np.zeros(count), np.zeros(count)


# ------------------- DO NOT MODIFY CODE OUTSIDE THE BLOCK --------------------


def main(render: bool = True) -> dict[str, float | int | str | None]:
    """Debug ICP on real fragments, then estimate and evaluate the LiDAR path."""
    root = Path(__file__).resolve().parent
    output_dir = root / "outputs/task3"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Checkpoint 3A: run handwritten ICP on the real Open3D pair.
    with np.load(root / "data/registration_demo.npz") as data:
        demo_source = to_homogeneous(data["source"].astype(np.float64))
        demo_target = to_homogeneous(data["target"].astype(np.float64))
        initial_transform = project_rigid_transform(data["initial_transform"])
    initialized = transform_points(initial_transform, demo_source)
    _, demo_history, correction_trace = icp(
        initialized,
        demo_target,
        max_iterations=80,
        max_correspondence_distance=0.12,
        trim_fraction=0.9,
        tolerance=1e-6,
    )
    total_trace = correction_trace @ initial_transform
    print("Checkpoint 3A computed: ICP demo; rendering deferred until metrics are saved")

    # Checkpoint 3B: align all adjacent pairs in the curved LiDAR clip.
    with np.load(root / "data/sequence00.npz") as data:
        offsets = data["frame_offsets"].astype(np.int64)
        points = data["points"].astype(np.float32)
        ground_truth = data["poses_lidar_to_world"].astype(np.float64)
    scans = [
        to_homogeneous(points[offsets[i] : offsets[i + 1], :3])
        for i in range(len(offsets) - 1)
    ]
    ground_truth = np.stack(
        [np.linalg.inv(ground_truth[0]) @ pose for pose in ground_truth]
    )
    relative, histories = [], []
    for frame in range(len(scans) - 1):
        # ``initial`` and the saved result both map L_(frame+1) -> L_frame.
        initial = relative[-1] if relative else np.eye(4)
        initialized = transform_points(initial, scans[frame + 1])
        correction, history, _ = icp(initialized, scans[frame])
        relative.append(correction @ initial)
        histories.append(history)
        if (frame + 1) % 50 == 0:
            print(f"Odometry: {frame + 1}/{len(scans) - 1} pairs", flush=True)
    relative = np.asarray(relative)
    np.save(output_dir / "relative_transforms.npy", relative)
    save_convergence_plot(output_dir / "icp_convergence.png", histories)
    print("Checkpoint 3B saved: relative_transforms.npy and icp_convergence.png")

    # Checkpoint 3C: chain poses, evaluate them, and write trajectory/BEV plots.
    odometry = chain_relative_poses(relative)
    np.save(output_dir / "icp_poses.npy", odometry)
    translation_rte, rotation_rte = relative_trajectory_error(odometry, ground_truth)
    save_trajectory_plot(
        output_dir / "trajectory_comparison.png", ground_truth, odometry
    )
    save_bev_comparison(
        output_dir / "bev_comparison.png", scans, ground_truth, odometry
    )
    completed = bool(histories) and all(len(history) > 0 for history in histories)
    position_error = odometry[:, :2, 3] - ground_truth[:, :2, 3]
    metrics = {
        "status": "complete" if completed else "starter_placeholder",
        "ate_definition": "XY translation RMSE; first pose identity; no additional alignment",
        "demo_final_rmse_m": float(demo_history[-1]) if len(demo_history) else None,
        "mean_relative_translation_error_m": (
            float(translation_rte.mean()) if completed else None
        ),
        "mean_relative_rotation_error_deg": (
            float(rotation_rte.mean()) if completed else None
        ),
        "odometry_ate_rmse_m": (
            float(np.sqrt(np.mean(np.sum(position_error**2, axis=1))))
            if completed
            else None
        ),
        "frame_count": len(scans),
        "online_video_frames": 0,
        "online_video_fps": None,
    }
    write_json(output_dir / "metrics.json", metrics)
    if render:
        save_alignment_visuals(
            output_dir / "icp_demo_alignment.png", demo_source, demo_target,
            total_trace, output_dir / "icp_iterations.mp4",
        )
        if completed:
            save_icp_odometry_video(
                output_dir / "icp_odometry_online.mp4",
                points, offsets, odometry, local_range=50.0,
            )
            metrics.update(online_video_frames=len(scans), online_video_fps=24)
            write_json(output_dir / "metrics.json", metrics)
    print("Checkpoint 3C saved: poses, trajectory, BEV, and metrics")
    print("Rendering enabled" if render else "3D and video rendering skipped")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-rendering", action="store_true")
    print("Task 3 complete:", main(render=not parser.parse_args().skip_rendering))
