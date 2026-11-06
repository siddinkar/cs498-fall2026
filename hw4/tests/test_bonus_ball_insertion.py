"""Synthetic correctness checks for the optional ball-insertion bonus."""

import sys
import unittest
from pathlib import Path

import torch


HANDOUT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HANDOUT_DIR))

from bonus_ball_insertion import composite_opaque_sphere, intersect_sphere  # noqa: E402


class BonusBallInsertionTests(unittest.TestCase):
    """Check analytic intersections and NeRF-aware opaque compositing."""

    def test_intersection_selects_nearest_valid_root_and_marks_misses(self):
        rays_o = torch.zeros(4, 3, dtype=torch.float64)
        rays_d = torch.tensor(
            [
                [0.0, 0.0, -1.0],
                [0.0, 0.0, -2.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=torch.float64,
        )
        center = torch.tensor([0.0, 0.0, -3.0], dtype=torch.float64)

        hit_depth, hit_mask = intersect_sphere(rays_o, rays_d, center, radius=1.0, near=0.0, far=6.0)

        torch.testing.assert_close(hit_depth[:2], torch.tensor([2.0, 1.0], dtype=torch.float64))
        self.assertEqual(hit_mask.tolist(), [True, True, False, False])
        self.assertTrue(torch.isinf(hit_depth[2:]).all())
        self.assertEqual(hit_depth.dtype, torch.float64)

    def test_intersection_uses_exit_root_when_ray_starts_inside(self):
        rays_o = torch.zeros(1, 3)
        rays_d = torch.tensor([[0.0, 0.0, 1.0]])

        hit_depth, hit_mask = intersect_sphere(
            rays_o,
            rays_d,
            center=torch.zeros(3),
            radius=1.0,
            near=0.0,
            far=2.0,
        )

        torch.testing.assert_close(hit_depth, torch.tensor([1.0]))
        self.assertTrue(bool(hit_mask[0]))

    def test_compositing_keeps_front_scene_and_removes_scene_behind_ball(self):
        z_vals = torch.tensor(
            [
                [1.0, 2.0, 3.0],
                [1.0, 2.0, 3.0],
                [1.0, 2.0, 3.0],
            ]
        )
        sample_rgb = torch.tensor(
            [
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
            ]
        )
        weights = torch.tensor(
            [
                [0.25, 0.50, 0.25],
                [0.25, 0.50, 0.25],
                [0.25, 0.50, 0.25],
            ]
        )
        scene_rgb = torch.sum(weights[..., None] * sample_rgb, dim=1)
        outputs = {"rgb": scene_rgb, "weights": weights, "sample_rgb": sample_rgb}
        hit_depth = torch.tensor([1.5, 2.5, torch.inf])
        hit_mask = torch.tensor([True, True, False])
        ball_rgb = torch.tensor([0.0, 0.0, 1.0])

        composite = composite_opaque_sphere(outputs, z_vals, hit_depth, hit_mask, ball_rgb)

        expected_rgb = torch.stack(
            [
                torch.tensor([0.25, 0.00, 0.75]),
                torch.tensor([0.25, 0.50, 0.25]),
                scene_rgb[2],
            ]
        )
        torch.testing.assert_close(composite["rgb"], expected_rgb)
        torch.testing.assert_close(composite["visibility"], torch.tensor([0.75, 0.25, 0.00]))

    def test_opaque_foreground_hides_ball_completely(self):
        outputs = {
            "rgb": torch.tensor([[0.8, 0.1, 0.1]]),
            "weights": torch.tensor([[1.0, 0.0]]),
            "sample_rgb": torch.tensor([[[0.8, 0.1, 0.1], [0.0, 0.0, 0.0]]]),
        }
        composite = composite_opaque_sphere(
            outputs,
            z_vals=torch.tensor([[1.0, 3.0]]),
            hit_depth=torch.tensor([2.0]),
            hit_mask=torch.tensor([True]),
            ball_rgb=torch.tensor([0.0, 0.0, 1.0]),
        )

        torch.testing.assert_close(composite["rgb"], outputs["rgb"])
        torch.testing.assert_close(composite["visibility"], torch.zeros(1))


if __name__ == "__main__":
    unittest.main()
