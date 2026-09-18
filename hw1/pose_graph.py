"""Provided planar optimizer for the optional loop-closure bonus."""
from __future__ import annotations
import numpy as np

def pose_from_xytheta(x: float, y: float, theta: float) -> np.ndarray:
    """Create a planar local-to-world pose as a 4x4 matrix."""
    cosine, sine = np.cos(theta), np.sin(theta)
    pose = np.eye(4)
    pose[:2, :2] = [[cosine, -sine], [sine, cosine]]
    pose[:2, 3] = [x, y]
    return pose


def xytheta_from_pose(pose: np.ndarray) -> np.ndarray:
    return np.array([pose[0, 3], pose[1, 3], np.arctan2(pose[1, 0], pose[0, 0])])


def wrap_angle(angle: np.ndarray | float):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def between_xytheta(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    cosine, sine = np.cos(first[2]), np.sin(first[2])
    delta = second[:2] - first[:2]
    return np.array(
        [
            cosine * delta[0] + sine * delta[1],
            -sine * delta[0] + cosine * delta[1],
            wrap_angle(second[2] - first[2]),
        ]
    )


def optimize_pose_graph(
    initial_poses: np.ndarray, edges: np.ndarray, iterations: int = 8
) -> np.ndarray:
    """Optimize a planar pose graph, fixing the first pose.

    initial_poses: float [K,3], rows (world x meters, world y meters, yaw radians).
    edges: float [E,6], rows (i,j,dx,dy,dtheta,weight). The measured transform
    maps frame j INTO frame i: inv(T_i_to_W) @ T_j_to_W.
    dx/dy are in frame i, dtheta is radians, weight must be positive.
    All nodes must be connected to node 0. Use a compact keyframe graph
    (e.g. every fourth frame); this educational dense solver is expensive.
    Return optimized [K,3] poses. Retrieval and verification are your work.
    """
    first_pose = initial_poses[0].copy()
    state = initial_poses[1:].reshape(-1).copy()

    def residual(values: np.ndarray) -> np.ndarray:
        poses = np.vstack([first_pose, values.reshape(-1, 3)])
        errors = []
        for first, second, dx, dy, dtheta, weight in edges:
            error = between_xytheta(poses[int(first)], poses[int(second)])
            error -= [dx, dy, dtheta]
            error[2] = wrap_angle(error[2])
            errors.append(np.sqrt(weight) * error)
        return np.concatenate(errors)

    for _ in range(iterations):
        error = residual(state)
        jacobian = np.empty((len(error), len(state)))
        for column in range(len(state)):
            perturbed = state.copy()
            perturbed[column] += 1e-6
            jacobian[:, column] = (residual(perturbed) - error) / 1e-6
        augmented = np.vstack([jacobian, np.sqrt(1e-5) * np.eye(len(state))])
        update = np.linalg.lstsq(
            augmented,
            np.concatenate([-error, np.zeros(len(state))]),
            rcond=None,
        )[0]
        state += update
        state[2::3] = wrap_angle(state[2::3])
        if np.linalg.norm(update) < 1e-6:
            break
    return np.vstack([first_pose, state.reshape(-1, 3)])


if __name__ == "__main__":
    initial = np.array([[0., 0., 0.], [1.1, 0., 0.], [2.2, 0., 0.]])
    edges = np.array([[0, 1, 1, 0, 0, 1], [1, 2, 1, 0, 0, 1],
                      [0, 2, 2, 0, 0, 2]], dtype=float)
    optimized = optimize_pose_graph(initial, edges)
    np.testing.assert_allclose(optimized[:, 0], [0, 1, 2], atol=1e-5)
    print("Three-node example passed:", optimized)
