"""HW0 Task 3: two-view geometry and epipolar lines.

Work from Checkpoint 3A through 3C, then run:

    python task3_epipolar.py
"""

from pathlib import Path

import imageio.v3 as iio
import numpy as np

from hw0_utils import plot_two_views, symmetric_epipolar_distance, write_json


# ------------------------ Task 3: Modify your code below ------------------------

def estimate_fundamental_matrix(matches: np.ndarray) -> np.ndarray:
    """Estimate a rank-two F from rows (u1, v1, u2, v2), N >= 8.

    Replace the runnable placeholder with the normalized eight-point method:
      1. normalize the points in each view independently;
      2. build the N x 9 design matrix from (u1, v1, u2, v2);
      3. solve its right null space with SVD;
      4. enforce rank two by zeroing the smallest singular value; and
      5. denormalize and choose a stable scale.
    """
    centroid_x1 = matches[:, 0:2].mean(axis=0)
    centroid_x2 = matches[:, 2:4].mean(axis=0)
    N = len(matches)

    s_x1 = np.sqrt(2) / np.linalg.norm(matches[:, 0:2] - centroid_x1, axis=1).mean()
    s_x2 = np.sqrt(2) / np.linalg.norm(matches[:, 2:4] - centroid_x2, axis=1).mean()
    T_x1 = np.array([[s_x1, 0, -s_x1 * centroid_x1[0]], [0, s_x1, -s_x1 * centroid_x1[1]], [0, 0, 1]])
    T_x2 = np.array([[s_x2, 0, -s_x2 * centroid_x2[0]], [0, s_x2, -s_x2 * centroid_x2[1]], [0, 0, 1]])

    hom_x1 = np.hstack([matches[:, 0:2], np.ones((N, 1))])
    hom_x2 = np.hstack([matches[:, 2:4], np.ones((N, 1))])

    normalized_x1 = (T_x1 @ hom_x1.T).T
    normalized_x2 = (T_x2 @ hom_x2.T).T

    A = np.zeros((N, 9))
    for i in range(N):
        u1, v1 = normalized_x1[i, 0], normalized_x1[i, 1]
        u2, v2 = normalized_x2[i, 0], normalized_x2[i, 1]
        A[i] = [u2*u1, u2*v1, u2, v2*u1, v2*v1, v2, u1, v1, 1]

    U, S, Vt = np.linalg.svd(A)
    F = Vt[-1].reshape(3, 3)
    
    U_f, S_f, Vt_f = np.linalg.svd(F)
    S_f[-1] = 0

    F = U_f @ np.diag(S_f) @ Vt_f

    F_final = T_x2.T @ F @ T_x1
    return F_final / np.linalg.norm(F_final)


# ------------------- DO NOT MODIFY CODE OUTSIDE THE BLOCK --------------------


def main() -> dict[str, float | int]:
    """Run the three Task 3 checkpoints in the same order as the handout."""
    root = Path(__file__).resolve().parent
    output_dir = root / "outputs/task3"
    output_dir.mkdir(parents=True, exist_ok=True)
    all_matches = np.load(root / "data/epipolar/all_good_matches.npy")
    eight_matches = np.load(root / "data/epipolar/eight_good_matches.npy")
    images = [
        iio.imread(root / "data/epipolar/images/0000.png"),
        iio.imread(root / "data/epipolar/images/0005.png"),
    ]

    # Checkpoint 3A: estimate F from the supplied eight correspondences.
    fundamental = estimate_fundamental_matrix(eight_matches)
    np.savetxt(
        output_dir / "task3a_fundamental_matrix.txt",
        fundamental,
        fmt="%.10e",
        header="Estimated fundamental matrix F",
    )
    print(f"Checkpoint 3A saved; rank(F) = {np.linalg.matrix_rank(fundamental, tol=1e-8)}")

    # Checkpoint 3B: visualize matches first, then the epipolar lines from F.
    plot_two_views(images, all_matches, output_dir / "task3b_correspondences.png")
    plot_two_views(images, all_matches, output_dir / "task3b_epipolar_lines.png", fundamental)
    print("Checkpoint 3B saved: correspondences and epipolar lines")

    # Checkpoint 3C: evaluate F on all 200 correspondences.
    errors = symmetric_epipolar_distance(all_matches, fundamental)
    metrics: dict[str, float | int] = {
        "fundamental_rank": int(np.linalg.matrix_rank(fundamental, tol=1e-8)),
        "mean_symmetric_epipolar_error_px": float(np.mean(errors)),
        "median_symmetric_epipolar_error_px": float(np.median(errors)),
    }
    write_json(output_dir / "task3c_metrics.json", metrics)
    np.savez(output_dir / "task3_results.npz", fundamental=fundamental)
    print("Checkpoint 3C saved: task3c_metrics.json")
    return metrics


if __name__ == "__main__":
    print("Task 3 complete:", main())
