"""Small correctness checks for provided Task 3 utilities and mesh extraction."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch


HANDOUT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HANDOUT_DIR))

import task3_train_mesh as task3  # noqa: E402
from hw4_utils import BlenderScene, Mesh  # noqa: E402
from task3_train_mesh import (  # noqa: E402
    _compute_image_metrics,
    _write_validation_metrics,
    carve_density_with_visual_hull,
    extract_mesh_from_density,
    keep_largest_mesh_component,
    render_turntable,
)


class Task3MeshTests(unittest.TestCase):
    def test_provided_turntable_places_gt_left_and_prediction_right(self):
        images = torch.zeros(4, 8, 8, 3)
        images[2] = 0.25
        images[3] = 0.50
        scene = BlenderScene(
            images=images,
            poses=torch.eye(4).repeat(4, 1, 1),
            height=8,
            width=8,
            focal=8.0,
            split_indices={
                "train": np.array([0]),
                "val": np.array([1]),
                "test": np.array([2, 3]),
            },
        )
        captured: dict[str, np.ndarray] = {}

        def fake_load_scene(cfg):
            self.assertEqual(cfg["test_skip"], 2)
            return scene

        def fake_save_video(_path, frames, fps):
            self.assertEqual(fps, 24)
            captured["video"] = frames

        def fake_save_contact_sheet(_path, frames, cols):
            self.assertEqual(cols, 2)
            captured["sheet"] = frames

        with tempfile.TemporaryDirectory() as temporary_dir:
            data_dir = Path(temporary_dir) / "data"
            data_dir.mkdir()
            metadata = {"frames": [{"file_path": f"./test/r_{i}"} for i in range(4)]}
            (data_dir / "transforms_test.json").write_text(json.dumps(metadata), encoding="utf-8")
            cfg = {
                "data_dir": str(data_dir),
                "output_dir": temporary_dir,
                "device": "cpu",
                "render_chunk": 16,
            }
            prediction = torch.full((8, 8, 3), 0.75)
            with (
                patch.object(task3, "load_scene_from_config", side_effect=fake_load_scene),
                patch.object(task3, "load_trained_model", return_value=torch.nn.Identity()),
                patch.object(task3, "render_image", return_value=prediction),
                patch.object(task3, "save_video", side_effect=fake_save_video),
                patch.object(task3, "save_contact_sheet", side_effect=fake_save_contact_sheet),
            ):
                output = render_turntable(cfg, "model.pt", num_frames=2)

        self.assertEqual(output.name, "turntable.mp4")
        self.assertEqual(captured["video"].shape, (2, 8, 16, 3))
        np.testing.assert_allclose(captured["video"][0, :, :8], 0.25)
        np.testing.assert_allclose(captured["video"][1, :, :8], 0.50)
        np.testing.assert_allclose(captured["video"][:, :, 8:], 0.75)
        np.testing.assert_allclose(captured["sheet"], captured["video"])

    def test_provided_validation_metrics_and_csv_summary(self):
        image = np.linspace(0.0, 1.0, 16 * 16 * 3, dtype=np.float32).reshape(16, 16, 3)
        metrics = _compute_image_metrics(image, image)
        self.assertAlmostEqual(metrics["psnr"], 100.0, places=5)
        self.assertAlmostEqual(metrics["ssim"], 1.0, places=5)

        rows = [
            {"view_index": 0.0, "psnr": 20.0, "ssim": 0.8},
            {"view_index": 1.0, "psnr": 24.0, "ssim": 0.9},
        ]
        with tempfile.TemporaryDirectory() as temporary_dir:
            csv_path = Path(temporary_dir) / "validation_metrics.csv"
            summary = _write_validation_metrics(csv_path, rows)
            contents = csv_path.read_text(encoding="utf-8")
        self.assertAlmostEqual(summary["psnr"], 22.0)
        self.assertAlmostEqual(summary["ssim"], 0.85)
        self.assertIn("view_index,psnr,ssim", contents)
        self.assertIn("mean,22.000000,0.850000", contents)

    def test_extract_mesh_from_density_returns_world_vertices(self):
        resolution = 24
        bounds = (-1.0, 1.0)
        coords = np.linspace(bounds[0], bounds[1], resolution, dtype=np.float32)
        x, y, z = np.meshgrid(coords, coords, coords, indexing="ij")
        sigma = 0.55 - np.sqrt(x**2 + y**2 + z**2)

        mesh = extract_mesh_from_density(sigma, bounds, threshold=0.0)

        self.assertGreater(len(mesh.vertices), 0)
        self.assertGreater(len(mesh.faces), 0)
        self.assertEqual(mesh.vertices.shape[1], 3)
        self.assertEqual(mesh.faces.shape[1], 3)
        self.assertTrue(np.all(mesh.vertices >= bounds[0] - 1.0e-5))
        self.assertTrue(np.all(mesh.vertices <= bounds[1] + 1.0e-5))

    def test_visual_hull_removes_points_behind_camera(self):
        import imageio.v2 as imageio

        with tempfile.TemporaryDirectory() as temporary_dir:
            data_dir = Path(temporary_dir)
            image = np.zeros((11, 11, 4), dtype=np.uint8)
            image[..., 3] = 255
            imageio.imwrite(data_dir / "view.png", image)
            metadata = {
                "camera_angle_x": np.pi / 2.0,
                "frames": [
                    {
                        "file_path": "view",
                        "transform_matrix": np.eye(4).tolist(),
                    }
                ],
            }
            (data_dir / "transforms_train.json").write_text(
                json.dumps(metadata), encoding="utf-8"
            )
            density = np.ones((5, 5, 5), dtype=np.float32)
            cleaned = carve_density_with_visual_hull(
                density,
                {"data_dir": str(data_dir)},
                (-1.0, 1.0),
                min_view_fraction=1.0,
                mask_dilation=0,
            )

        self.assertGreater(np.count_nonzero(cleaned[:, :, :2]), 0)
        self.assertEqual(np.count_nonzero(cleaned[:, :, 2:]), 0)

    def test_largest_component_cleanup_compacts_vertices_and_faces(self):
        mesh = Mesh(
            vertices=np.array(
                [
                    [0, 0, 0],
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                    [4, 0, 0],
                    [5, 0, 0],
                    [4, 1, 0],
                ],
                dtype=np.float32,
            ),
            faces=np.array(
                [[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2], [4, 5, 6]],
                dtype=np.int64,
            ),
        )
        cleaned = keep_largest_mesh_component(mesh)

        self.assertEqual(cleaned.vertices.shape, (4, 3))
        self.assertEqual(cleaned.faces.shape, (4, 3))
        self.assertEqual(int(cleaned.faces.max()), 3)


if __name__ == "__main__":
    unittest.main()
