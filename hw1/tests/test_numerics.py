"""Small student checks; failures are expected until the edit blocks are filled."""
import sys
import unittest
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from task1_alignment import fit_rigid_svd, nearest_correspondences
from task2_ransac import ransac_rigid
from task3_odometry import icp, chain_relative_poses, relative_trajectory_error
from task4_mapping import points_to_voxels, fuse_voxel_evidence


class NumericalExamples(unittest.TestCase):
    def setUp(self):
        self.p = np.vstack([np.random.default_rng(42).normal(size=(3, 40)), np.ones(40)])
        self.t = np.eye(4)
        self.t[:2, :2] = [[0, -1], [1, 0]]
        self.t[:3, 3] = [.3, -.2, .1]

    def test_svd_rotation_and_translation(self):
        result = fit_rigid_svd(self.p, self.t @ self.p)
        np.testing.assert_allclose(result, self.t, atol=1e-10)
        reflected = self.p.copy()
        reflected[0] *= -1
        self.assertAlmostEqual(np.linalg.det(fit_rigid_svd(self.p, reflected)[:3, :3]), 1)

    def test_nearest_indices_threshold_and_empty(self):
        src, tgt, dist = nearest_correspondences(self.p[:, :3], self.p[:, ::-1], 0.)
        np.testing.assert_array_equal(src, [0, 1, 2])
        np.testing.assert_array_equal(tgt, [39, 38, 37])
        np.testing.assert_array_equal(dist, [0, 0, 0])
        self.assertTrue(all(len(x) == 0 for x in nearest_correspondences(self.p[:, :0], self.p, 1)))

    def test_ransac_outliers_and_final_mask(self):
        target = self.t @ self.p
        target[:3, :10] += 20
        result, mask = ransac_rigid(self.p, target, .001, 150, np.random.default_rng(10))
        np.testing.assert_allclose(result, self.t, atol=1e-10)
        np.testing.assert_array_equal(mask, np.arange(40) >= 10)

    def test_icp_and_trace(self):
        motion = np.eye(4)
        motion[:3, 3] = [.02, -.01, .01]
        result, history, trace = icp(self.p, motion @ self.p, max_correspondence_distance=.3)
        np.testing.assert_allclose(result, motion, atol=1e-8)
        self.assertGreater(len(history), 0)
        self.assertEqual(len(trace), len(history) + 1)
        np.testing.assert_allclose(trace[0], np.eye(4))
        np.testing.assert_allclose(trace[-1], result)

    def test_pose_order_and_nonzero_rte(self):
        shift = np.eye(4)
        shift[0, 3] = 1
        poses = chain_relative_poses(np.stack([self.t, shift]))
        np.testing.assert_allclose(poses, np.stack([np.eye(4), self.t, self.t @ shift]))
        reference = np.repeat(np.eye(4)[None], 3, axis=0)
        estimate = reference.copy()
        estimate[:, 0, 3] = [0, 1, 2]
        trans, rot = relative_trajectory_error(estimate, reference, delta=2)
        np.testing.assert_allclose(trans, [2])
        np.testing.assert_allclose(rot, [0])

    def test_voxel_floor_and_homogeneous_coordinates(self):
        points = torch.tensor([[-.02, .5], [0., 0.], [0., 0.], [2., 2.]])
        actual = points_to_voxels(points, torch.zeros(3), .25)
        self.assertEqual(actual.dtype, torch.long)
        torch.testing.assert_close(actual, torch.tensor([[-1, 0, 0], [1, 0, 0]]))

    def test_rte_nonzero_rotation_in_degrees(self):
        reference = np.repeat(np.eye(4)[None], 2, axis=0)
        estimated = np.stack([np.eye(4), self.t])
        trans, rot = relative_trajectory_error(estimated, reference)
        np.testing.assert_allclose(trans, [np.linalg.norm(self.t[:3, 3])])
        np.testing.assert_allclose(rot, [90], atol=1e-8)

    def test_int8_fusion_bounds_sort_and_counts(self):
        indices = torch.tensor([[1, 0, 0], [0, 1, 0], [1, 0, 0], [-1, 0, 0]])
        logits = torch.tensor([[4, -4], [-4, 4], [-4, 4], [4, -4]], dtype=torch.int8)
        coords, occupied, evidence, counts = fuse_voxel_evidence(indices, logits, (2, 2, 2))
        torch.testing.assert_close(coords, torch.tensor([[0, 1, 0], [1, 0, 0]]))
        torch.testing.assert_close(counts, torch.tensor([1, 2]))
        torch.testing.assert_close(occupied, torch.tensor([True, True]))
        expected = torch.log_softmax(logits[:3].double(), 1)
        torch.testing.assert_close(evidence, torch.stack([expected[1], expected[0] + expected[2]]))
        empty = fuse_voxel_evidence(indices[:0], logits[:0], (2, 2, 2))
        self.assertEqual(empty[0].shape, (0, 3))
        self.assertEqual(empty[2].shape, (0, 2))


if __name__ == "__main__":
    unittest.main()
