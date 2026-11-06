"""Small correctness checks for Task 1."""

import sys
import unittest
from pathlib import Path

import torch


HANDOUT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HANDOUT_DIR))

from task1_rays_encoding import encoded_dim, get_rays, pose_spherical, positional_encoding  # noqa: E402


class Task1RaysEncodingTests(unittest.TestCase):
    def test_get_rays_identity_pose(self):
        c2w = torch.eye(4)
        rays_o, rays_d = get_rays(2, 2, 1.0, c2w)
        expected_dirs = torch.tensor(
            [[[-1.0, 1.0, -1.0], [0.0, 1.0, -1.0]], [[-1.0, 0.0, -1.0], [0.0, 0.0, -1.0]]]
        )
        self.assertEqual(rays_o.shape, (2, 2, 3))
        self.assertEqual(rays_d.shape, (2, 2, 3))
        torch.testing.assert_close(rays_o, torch.zeros(2, 2, 3))
        torch.testing.assert_close(rays_d, expected_dirs)

    def test_pose_spherical_has_homogeneous_bottom_row(self):
        pose = pose_spherical(0.0, -30.0, 4.0)
        self.assertEqual(pose.shape, (4, 4))
        torch.testing.assert_close(pose[3], torch.tensor([0.0, 0.0, 0.0, 1.0]))

    def test_positional_encoding_values_and_shape(self):
        x = torch.tensor([[0.0, 1.0]])
        enc = positional_encoding(x, num_freqs=2, include_input=True, log_sampling=True)
        expected = torch.cat([x, torch.sin(x), torch.cos(x), torch.sin(2.0 * x), torch.cos(2.0 * x)], dim=-1)
        self.assertEqual(enc.shape, (1, encoded_dim(2, 2)))
        torch.testing.assert_close(enc, expected)

    def test_positional_encoding_preserves_leading_dimensions(self):
        x = torch.zeros(2, 3, 4)
        enc = positional_encoding(x, num_freqs=3, include_input=False)
        self.assertEqual(enc.shape, (2, 3, encoded_dim(4, 3, include_input=False)))


if __name__ == "__main__":
    unittest.main()
