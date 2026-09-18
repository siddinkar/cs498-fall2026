"""HW1 Task 4: semantic evidence fusion in a sparse voxel map.

Run directly:

    python task4_mapping.py
"""

from __future__ import annotations

from pathlib import Path
import argparse

import numpy as np
import torch

from hw1_utils import (
    SEMANTIC_COLORS,
    gt_voxel_majority,
    make_voxel_cube_mesh,
    save_lidar_follow_video,
    save_voxel_turntable,
    semantic_metrics,
    to_homogeneous,
    transform_points,
    write_json,
    write_semantic_ply,
)


# ------------------------ Task 4: Modify your code below ------------------------

def points_to_voxels(
    points: torch.Tensor, origin: torch.Tensor, voxel_size: float
) -> torch.Tensor:
    """Return floor((points - origin) / voxel_size) as torch.long indices.

    ``points`` is a homogeneous [4, N] tensor of world-frame point columns;
    ``origin`` is [3]. Divide xyz by w, transpose to [N, 3], use
    ``torch.floor``, cast to ``torch.long``, and do not clip.
    """
    return torch.zeros(
        (points.shape[1], 3), dtype=torch.long, device=points.device
    )


def fuse_voxel_evidence(
    voxel_indices: torch.Tensor,
    raw_logits: torch.Tensor,
    grid_shape: tuple[int, int, int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return sparse coordinates, occupancy, summed log evidence, and counts.

    Inputs are ``voxel_indices`` [N, 3] and ``raw_logits`` [N, C].
    The archive supplies int8 logits. Convert to float64 BEFORE log_softmax.
    Reject out-of-bounds indices. Use ``torch.log_softmax`` for stable evidence,
    merge repeated coordinates, and accumulate with ``index_add_`` or an
    equivalent PyTorch operation. Return lexicographically sorted coordinates.
    Accumulate evidence in float64 so exact class-count ties remain stable. Do
    not let a later observation overwrite earlier evidence.
    """
    shape = torch.tensor(grid_shape, dtype=torch.long, device=voxel_indices.device)
    valid = ((voxel_indices >= 0) & (voxel_indices < shape)).all(dim=1)
    coordinates, inverse = torch.unique(
        voxel_indices[valid], dim=0, sorted=True, return_inverse=True
    )
    counts = torch.bincount(inverse, minlength=len(coordinates))
    log_evidence = torch.zeros(
        (len(coordinates), raw_logits.shape[1]),
        dtype=torch.float64,
        device=raw_logits.device,
    )
    return coordinates, counts > 0, log_evidence, counts


# ------------------- DO NOT MODIFY CODE OUTSIDE THE BLOCK --------------------


def main(render: bool = True) -> dict[str, float | int | str | list[float] | None]:
    """Transform all scans, fuse voxels, evaluate semantics, and render the map."""
    root = Path(__file__).resolve().parent
    output_dir = root / "outputs/task4"
    output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(root / "data/sequence00.npz") as data:
        points = data["points"].astype(np.float32)
        offsets = data["frame_offsets"].astype(np.int64)
        point_labels = data["semantic_gt"].astype(np.int64)
        point_logits = data["semantic_logits_gt"]
        poses = data["poses_lidar_to_world"].astype(np.float64)
    frame_count = len(offsets) - 1
    class_count = point_logits.shape[1]

    # Each pose is T_Li_to_W. Transform a fixed-size sample from every scan.
    world_points, labels, raw_logits = [], [], []
    for frame, pose in enumerate(poses):
        frame_slice = slice(offsets[frame], offsets[frame + 1])
        scan = points[frame_slice, :3]
        scan_labels = point_labels[frame_slice]
        distance = np.linalg.norm(scan[:, :2], axis=1)
        keep = (
            (distance < 40.0)
            & (scan[:, 2] > -2.8)
            & (scan[:, 2] < 3.5)
            & (scan_labels < class_count)
        )
        keep = np.flatnonzero(keep)
        if len(keep) > 2500:
            keep = keep[np.linspace(0, len(keep) - 1, 2500, dtype=np.int64)]
        world_points.append(
            transform_points(pose, to_homogeneous(scan[keep]))
        )
        labels.append(scan_labels[keep])
        raw_logits.append(point_logits[frame_slice][keep])
    world_points = np.concatenate(world_points, axis=1)
    labels = np.concatenate(labels)
    raw_logits = np.concatenate(raw_logits)

    voxel_size = 0.25
    world_xyz = world_points[:3] / world_points[3:4]
    origin = np.floor(world_xyz.min(axis=1) / voxel_size) * voxel_size - voxel_size
    grid_shape = tuple(
        np.ceil((world_xyz.max(axis=1) - origin) / voxel_size).astype(int) + 2
    )
    world_points_tensor = torch.from_numpy(world_points)
    origin_tensor = torch.from_numpy(origin).to(world_points_tensor.dtype)
    voxel_indices_tensor = points_to_voxels(
        world_points_tensor, origin_tensor, voxel_size
    )
    voxel_indices = voxel_indices_tensor.cpu().numpy()
    expected_indices = np.floor((world_xyz.T - origin) / voxel_size).astype(np.int64)
    coordinates_correct = np.array_equal(voxel_indices, expected_indices)

    # Checkpoint 4A: sparse coordinates and binary occupancy over all frames.
    gt_coordinates, gt_labels, gt_histogram = gt_voxel_majority(
        expected_indices, labels, grid_shape, class_count
    )
    coordinates_tensor, occupancy_tensor, evidence_tensor, count_tensor = (
        fuse_voxel_evidence(
            voxel_indices_tensor, torch.from_numpy(raw_logits), grid_shape
        )
    )
    coordinates = coordinates_tensor.cpu().numpy()
    occupancy = occupancy_tensor.cpu().numpy()
    log_evidence = evidence_tensor.cpu().numpy()
    observation_count = count_tensor.cpu().numpy()
    if coordinates_correct and not np.array_equal(coordinates, gt_coordinates):
        raise ValueError("GT and fused coordinates must match")
    print(f"Checkpoint 4A: {len(coordinates):,} occupied voxels from {frame_count} frames")

    # Checkpoint 4B: decode fused semantics and report quantitative metrics.
    average = log_evidence / np.maximum(observation_count[:, None], 1)
    probability = np.exp(average - average.max(axis=1, keepdims=True))
    probability /= probability.sum(axis=1, keepdims=True)
    # Make exact evidence ties deterministic across float32/float64 solutions.
    prediction = np.round(average, 6).argmax(axis=1)
    completed = coordinates_correct and bool(np.any(log_evidence))
    metrics = (
        semantic_metrics(prediction, gt_labels, class_count)
        if completed
        else {"voxel_accuracy": None, "per_class_iou": [], "mean_iou": None}
    )
    entropy = -np.sum(
        probability * np.log(np.maximum(probability, 1e-12)), axis=1
    ) / np.log(class_count)
    metrics.update(
        {
            "status": "complete" if completed else "starter_placeholder",
            "voxel_coordinates_correct": coordinates_correct,
            "entropy_definition": "softmax of mean log evidence; not posterior confidence",
            "frames_aggregated": frame_count,
            "occupied_voxels": int(occupancy.sum()),
            "mean_observations_per_voxel": float(observation_count.mean()),
            "maximum_observations_per_voxel": int(observation_count.max()),
            "mean_normalized_entropy": float(entropy.mean()),
        }
    )
    centers = origin + (coordinates + 0.5) * voxel_size
    write_json(output_dir / "semantic_metrics.json", metrics)
    write_json(
        output_dir / "voxel_statistics.json",
        {
            "frames_aggregated": frame_count,
            "selected_mapping_points": world_points.shape[1],
            "occupied_voxels": len(coordinates),
            "voxel_size_m": voxel_size,
            "origin_m": origin.tolist(),
            "grid_shape": list(grid_shape),
            "class_histogram": gt_histogram.sum(axis=0).tolist(),
        },
    )

    # Checkpoint 4C: render equal-metric cubes when fusion is implemented.
    if render:
        write_semantic_ply(
            output_dir / "semantic_occupancy.ply", centers, prediction, SEMANTIC_COLORS
        )
    if completed and render:
        mesh = make_voxel_cube_mesh(
            centers, prediction, voxel_size, SEMANTIC_COLORS
        )
        save_voxel_turntable(
            mesh,
            centers,
            output_dir / "semantic_occupancy.png",
            output_dir / "semantic_occupancy.mp4",
        )
        save_lidar_follow_video(
            mesh,
            points,
            offsets,
            poses,
            output_dir / "semantic_lidar_follow.mp4",
        )
        print("Checkpoint 4C saved: shoulder PNG, turntable MP4, and follow MP4")
    else:
        print("Checkpoint 4C pending: verify voxel coordinates/fusion and enable rendering")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-rendering", action="store_true")
    print("Task 4 complete:", main(render=not parser.parse_args().skip_rendering))
