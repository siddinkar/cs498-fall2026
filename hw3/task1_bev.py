"""Task 1: rasterize one LiDAR sweep into a four-channel BEV tensor.

Run directly with: python task1_bev.py
"""

import numpy as np

from hw3_utils import (
    DATA_ROOT, GRID_H, GRID_W, OUTPUT_ROOT, RESOLUTION, Z_MAX, Z_MIN,
    load_debug_frames, metric_to_grid, save_bev_boxes, save_bev_channels, write_json,
)


def rasterize_points(points: np.ndarray) -> np.ndarray:
    """Return `[4,400,352]` float32 BEV values in `[0,1]`.

    Input `points` has shape `[N,4]` with columns `(x,y,z,intensity)` in the
    LiDAR frame. The four output channels are max height, mean height, mean
    intensity, and log-density. Empty cells must be zero.

    Use `metric_to_grid`, a flat index `row * GRID_W + column`,
    `np.bincount` for counts/sums, and `np.maximum.at` for maximum height.
    Do not loop over BEV cells.
    """
    # ------------------------ Task 1 Modify your code below ---------------------
    x, y, z, intensity = points[:, 0], points[:, 1], points[:, 2], points[:, 3]
    rows, columns, valid = metric_to_grid(x, y)
    valid &= (z >= Z_MIN) & (z <= Z_MAX)
    z, intensity = z[valid], intensity[valid]
    flat = rows[valid] * GRID_W + columns[valid]
    cells = GRID_H * GRID_W

    count = np.bincount(flat, minlength=cells).astype(np.float64)
    z_sum = np.bincount(flat, weights=z, minlength=cells)
    i_sum = np.bincount(flat, weights=intensity, minlength=cells)
    z_max = np.full(cells, Z_MIN)
    np.maximum.at(z_max, flat, z)

    occupied = count > 0
    safe = np.maximum(count, 1.0)
    z_range = Z_MAX - Z_MIN
    max_height = np.where(occupied, (z_max - Z_MIN) / z_range, 0.0)
    mean_height = np.where(occupied, (z_sum / safe - Z_MIN) / z_range, 0.0)
    mean_intensity = np.where(occupied, np.clip(i_sum / safe, 0.0, 1.0), 0.0)
    density = np.clip(np.log1p(count) / np.log1p(64.0), 0.0, 1.0)

    bev = np.stack([max_height, mean_height, mean_intensity, density])
    return bev.reshape(4, GRID_H, GRID_W).astype(np.float32)
    # ---------------------- DO NOT MODIFY CODE OUTSIDE THE BLOCK ----------------


def main():
    points, boxes, frame_ids = load_debug_frames(DATA_ROOT / "debug_frames.npz")
    output = OUTPUT_ROOT / "task1"
    bev = rasterize_points(points[0])

    save_bev_channels(bev, output / "bev_channels.png")
    save_bev_boxes(bev, boxes[0], output / "bev_boxes.png", title=f"KITTI frame {frame_ids[0]}")

    occupied = np.max(bev, axis=0) > 0
    metrics = {
        "shape": list(bev.shape),
        "resolution_m": RESOLUTION,
        "occupied_cells": int(occupied.sum()),
        "occupancy_fraction": float(occupied.mean()),
        "channel_min": bev.reshape(4, -1).min(axis=1).tolist(),
        "channel_max": bev.reshape(4, -1).max(axis=1).tolist(),
    }
    write_json(output / "metrics.json", metrics)
    print("Checkpoint 1: bev_channels.png, bev_boxes.png, and metrics.json")
    return metrics


if __name__ == "__main__":
    print("Task 1 complete:", main())
