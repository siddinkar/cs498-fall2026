"""HW1 Task 1: rigid fitting and nearest correspondences.

Run directly:

    python task1_alignment.py
"""

from __future__ import annotations

from pathlib import Path
import argparse

import numpy as np
from scipy.spatial import cKDTree

from hw1_utils import (
    project_rigid_transform,
    rotation_error_degrees,
    save_alignment_visuals,
    to_homogeneous,
    transform_points,
    write_json,
)


# ------------------------ Task 1: Modify your code below ------------------------

def fit_rigid_svd(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Fit T_source_to_target for paired source and target points.

    Center both point sets, form their 3x3 cross-covariance, solve for rotation
    with SVD, correct a reflection if necessary, and recover translation.

    ``source`` and ``target`` are homogeneous [4, N] point matrices. Lecture
    notation is used directly: P_target = T_source_to_target @ P_source.
    """
    del target
    return np.eye(4)


def nearest_correspondences(
    source: np.ndarray,
    target: np.ndarray,
    max_distance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return source indices, nearest target indices, and accepted distances.

    Transpose xyz to [N, 3] only for the SciPy cKDTree boundary, query one
    neighbor per source column, and retain distances <= ``max_distance``.
    """
    count = min(source.shape[1], target.shape[1])
    source_ids = np.arange(count)
    target_ids = np.arange(count)
    distances = np.linalg.norm(source[:3, :count] - target[:3, :count], axis=0)
    keep = distances <= max_distance
    return source_ids[keep], target_ids[keep], distances[keep]


# ------------------- DO NOT MODIFY CODE OUTSIDE THE BLOCK --------------------


def main(render: bool = True) -> dict[str, float | int | str | None]:
    """Fit real pairs, evaluate the full fragments, and save Task 1 outputs."""
    root = Path(__file__).resolve().parent
    output_dir = root / "outputs/task1"
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = root / "data/registration_demo.npz"
    if not data_path.exists():
        raise FileNotFoundError("Missing data/registration_demo.npz")
    with np.load(data_path, allow_pickle=False) as data:
        source = to_homogeneous(data["source"].astype(np.float64))
        target = to_homogeneous(data["target"].astype(np.float64))
        paired_source = to_homogeneous(data["task1_source"].astype(np.float64))
        paired_target = to_homogeneous(data["task1_target"].astype(np.float64))
        reference = project_rigid_transform(data["reference_transform"])

    # Checkpoint 1A: recover a rigid transform from known point pairs.
    estimate = fit_rigid_svd(paired_source, paired_target)
    aligned = transform_points(estimate, source)
    translation_error = float(np.linalg.norm(estimate[:3, 3] - reference[:3, 3]))
    rotation_error = rotation_error_degrees(estimate, reference)
    np.savetxt(output_dir / "estimated_transform.txt", estimate, fmt="%.8f")
    print(
        f"Checkpoint 1A: translation error={translation_error:.4f} m, "
        f"rotation error={rotation_error:.3f} deg"
    )

    # Checkpoint 1B: evaluate nearest-neighbor matches and render before/after.
    source_ids, _, distances = nearest_correspondences(aligned, target, 0.05)
    rmse = float(np.sqrt(np.mean(distances**2))) if len(distances) else None
    metrics = {
        "translation_error_m": translation_error,
        "rotation_error_deg": rotation_error,
        "known_pair_count": paired_source.shape[1],
        "nearest_correspondence_count": len(source_ids),
        "inlier_ratio": len(source_ids) / source.shape[1],
        "inlier_rmse_m": rmse,
        "status": "complete" if len(distances) else "starter_placeholder",
    }
    write_json(output_dir / "svd_metrics.json", metrics)
    if render:
        save_alignment_visuals(
            output_dir / "svd_alignment.png", source, target,
            np.stack([np.eye(4), estimate]),
        )
    print("Checkpoint 1B saved: svd_metrics.json")
    print("Alignment image saved" if render else "Alignment rendering skipped")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-rendering", action="store_true")
    print("Task 1 complete:", main(render=not parser.parse_args().skip_rendering))
