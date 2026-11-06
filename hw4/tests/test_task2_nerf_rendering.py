"""Small correctness checks for Task 2."""

import sys
import unittest
from pathlib import Path

import torch


HANDOUT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HANDOUT_DIR))

from task1_rays_encoding import encoded_dim  # noqa: E402
from task2_nerf_rendering import TinyNeRF, raw_to_outputs, sample_along_rays  # noqa: E402


class Task2NerfRenderingTests(unittest.TestCase):
    def test_tiny_nerf_shapes_and_gradients(self):
        torch.manual_seed(0)
        input_ch = encoded_dim(3, 4)
        input_ch_dir = encoded_dim(3, 2)
        model = TinyNeRF(input_ch=input_ch, input_ch_dir=input_ch_dir, hidden_dim=32, num_layers=4, skip_layer=2)
        x = torch.randn(2, 5, input_ch)
        viewdirs = torch.randn(2, 5, input_ch_dir)
        raw = model(x, viewdirs)
        self.assertEqual(raw.shape, (2, 5, 4))
        loss = raw.square().mean()
        loss.backward()
        self.assertTrue(any(param.grad is not None for param in model.parameters()))

    def test_viewdirs_do_not_change_density(self):
        torch.manual_seed(1)
        input_ch = encoded_dim(3, 2)
        input_ch_dir = encoded_dim(3, 1)
        model = TinyNeRF(input_ch=input_ch, input_ch_dir=input_ch_dir, hidden_dim=16, num_layers=3, skip_layer=2)
        x = torch.randn(4, input_ch)
        view_a = torch.zeros(4, input_ch_dir)
        view_b = torch.ones(4, input_ch_dir)
        sigma_a = model(x, view_a)[..., 3]
        sigma_b = model(x, view_b)[..., 3]
        torch.testing.assert_close(sigma_a, sigma_b)

    def test_sample_along_rays_deterministic_depths(self):
        rays_o = torch.zeros(3, 3)
        rays_d = torch.tensor([[0.0, 0.0, -1.0]]).expand(3, 3)
        pts, z_vals = sample_along_rays(rays_o, rays_d, 2.0, 6.0, 5, perturb=False)
        expected = torch.linspace(2.0, 6.0, 5).expand(3, 5)
        torch.testing.assert_close(z_vals, expected)
        torch.testing.assert_close(pts[..., 2], -expected)

    def test_raw_to_outputs_front_sample_dominates(self):
        rays_d = torch.tensor([[0.0, 0.0, -1.0]])
        z_vals = torch.linspace(1.0, 2.0, 4).reshape(1, 4)
        raw = torch.zeros(1, 4, 4)
        raw[0, 0, :3] = torch.tensor([4.0, -4.0, -4.0])
        raw[0, 0, 3] = 20.0
        out = raw_to_outputs(raw, z_vals, rays_d, white_background=False)
        self.assertGreater(out["weights"][0, 0].item(), 0.95)
        self.assertGreater(out["rgb"][0, 0].item(), out["rgb"][0, 1].item())
        self.assertTrue(torch.all((out["acc"] >= 0.0) & (out["acc"] <= 1.0001)).item())


if __name__ == "__main__":
    unittest.main()
