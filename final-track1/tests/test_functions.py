"""Eight focused correctness checks; untouched placeholders are explicitly XFAIL."""

import numpy as np
import torch
from conftest import xfail_if_untouched
from semocc.learning import (
    class_weights_from_histogram,
    completion_loss,
    metrics_from_confusion,
)
from semocc.model import (
    AnchorVolumeSpec,
    SpatialUNet3d,
    TriViewLiftFuseUNet,
    factorized_logits,
    project_voxel_centers,
)


def small_model():
    """Use a tiny grid for fast architecture and geometry tests."""
    v = AnchorVolumeSpec((-1.0, 1.0), (-1.0, 1.0), (0.0, 2.0), 0.5)
    return TriViewLiftFuseUNet(
        9, v, image_feature_channels=8, volume_feature_channels=8, base_channels=4
    )


def test_projection():
    xfail_if_untouched(project_voxel_centers)
    points = torch.tensor([[0.0, 0.5, 1.0], [100.0, 0.0, 1.0]])
    k = torch.tensor([[[16.0, 0, 15.5], [0, 16.0, 15.5], [0, 0, 1.0]]])
    cameras = torch.eye(4)[None, None].expand(1, 3, 4, 4)
    grid, valid, z = project_voxel_centers(
        points, k, cameras, torch.eye(4)[None], 32, 32
    )
    assert grid.shape == (3, 2, 1, 2) and valid.shape == (1, 3, 2)
    torch.testing.assert_close(grid[0, 0, 0], torch.tensor([0.0, -0.5]))
    assert valid[0, 0, 0] and not valid[0, 0, 1]
    torch.testing.assert_close(z, torch.ones_like(z))


def test_lifting():
    xfail_if_untouched(TriViewLiftFuseUNet._project_and_sample)
    m = small_model()
    features = torch.ones(1, 3, 8, 8, 8, requires_grad=True)
    k = torch.tensor([[[16.0, 0, 15.5], [0, 16.0, 15.5], [0, 0, 1.0]]])
    lifted = m._project_and_sample(
        features,
        torch.ones(1, 3, 1, 32, 32) * 1.5,
        k,
        torch.eye(4)[None, None].expand(1, 3, 4, 4),
        torch.eye(4)[None],
        image_height=32,
        image_width=32,
    )
    assert lifted.shape == (1, 36, 4, 4, 4) and torch.isfinite(lifted).all()
    assert lifted[:, :8].abs().sum() > 0
    lifted.sum().backward()
    assert features.grad is not None and features.grad.abs().sum() > 0


def test_view_fusion():
    xfail_if_untouched(TriViewLiftFuseUNet.fuse_views)
    m = small_model()
    x = torch.randn(1, 36, 4, 4, 4, requires_grad=True)
    actual = m.fuse_views(x)
    expected = m.view_fusion(x)
    torch.testing.assert_close(actual, expected)
    actual.square().mean().backward()
    assert x.grad.abs().sum() > 0


def test_unet():
    xfail_if_untouched(SpatialUNet3d.forward)
    m = SpatialUNet3d(8, 4, 10)
    x = torch.randn(1, 8, 7, 5, 9, requires_grad=True)
    output = m(x)
    assert output.shape == (1, 10, 7, 5, 9)
    output.square().mean().backward()
    assert x.grad.abs().sum() > 0
    assert all(p.grad is not None for p in m.parameters())


def test_factorized_logits():
    xfail_if_untouched(factorized_logits)
    raw = torch.randn(2, 10, 2, 2, 2, requires_grad=True)
    logits = factorized_logits(raw)
    probability = logits.softmax(1)
    torch.testing.assert_close(probability[:, 1:].sum(1), raw[:, 0].sigmoid())
    torch.testing.assert_close(
        probability[:, 1:] / probability[:, 1:].sum(1, keepdim=True),
        raw[:, 1:].softmax(1),
    )
    logits.sum().backward()
    assert raw.grad is not None


def test_loss():
    xfail_if_untouched(completion_loss)
    logits = torch.zeros(1, 3, 2, 1, 1, requires_grad=True)
    labels = torch.tensor([[[[2]], [[1]]]])
    completion = torch.tensor([[[[True]], [[False]]]])
    output = torch.ones_like(completion)
    loss, parts = completion_loss(logits, labels, completion, output)
    expected = np.log(3) + 0.5 * (-np.log(2 / 3)) + 0.1 * np.log(3)
    assert abs(float(loss.detach()) - expected) < 1e-6 and all(
        v > 0 for v in parts.values()
    )
    loss.backward()
    assert logits.grad is not None
    empty = torch.zeros_like(completion)
    empty_loss, empty_parts = completion_loss(logits, labels, empty, empty)
    assert float(empty_loss.detach()) == 0 and all(v == 0 for v in empty_parts.values())
    empty_loss.backward()


def test_class_weights():
    xfail_if_untouched(class_weights_from_histogram)
    weights = class_weights_from_histogram(np.array([1, 4, 16, 0]), maximum=10)
    torch.testing.assert_close(weights, torch.tensor([12 / 7, 6 / 7, 3 / 7, 0.0]))


def test_metrics():
    xfail_if_untouched(metrics_from_confusion)
    c = np.array([[3, 1, 0], [1, 3, 0], [0, 0, 0]])
    result = metrics_from_confusion(c)
    assert result["completion_accuracy"] == 0.75
    assert result["completion_mean_iou"] == 0.6
    assert result["completion_occupied_iou"] == 0.6
    assert result["class_support"] == [4, 4, 0]
