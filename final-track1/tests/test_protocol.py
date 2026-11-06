"""Always-active checks for provided transport, masking, and house aggregation."""

import base64
import zlib

import numpy as np
import pytest
from cs498_track1.protocol import SHAPE, decode_labels, encode_labels, summarize, validate_prediction


def test_roundtrip_and_shape():
    a = (np.arange(np.prod(SHAPE)) % 10).astype(np.int8).reshape(SHAPE)
    np.testing.assert_array_equal(a, decode_labels(encode_labels(a)))
    with pytest.raises(ValueError):
        validate_prediction(a.transpose(2, 1, 0))
    with pytest.raises(ValueError):
        validate_prediction(a.astype(float))


@pytest.mark.parametrize(
    "encoded",
    [
        "bad",
        base64.b64encode(zlib.compress(b"\x00")).decode(),
        base64.b64encode(zlib.compress(bytes(np.prod(SHAPE) + 1))).decode(),
        base64.b64encode(zlib.compress(bytes(np.prod(SHAPE))) + b"extra").decode(),
    ],
)
def test_invalid_encoding(encoded):
    with pytest.raises(ValueError):
        decode_labels(encoded)


def test_unknown_only_allowed_in_private_solution():
    a = np.full(SHAPE, -1, dtype=np.int8)
    encoded = encode_labels(a)
    with pytest.raises(ValueError):
        decode_labels(encoded)
    np.testing.assert_array_equal(a, decode_labels(encoded, allow_masked=True))


def test_houses_are_equal_weight_not_voxels():
    good = np.zeros((10, 10), int)
    good[1, 1] = 1000
    bad = np.zeros((10, 10), int)
    bad[1, 0] = 1
    assert (
        summarize({"large": good, "small": bad})["metrics"]["completion_mean_iou"]
        == 0.5
    )


def test_semantic_colors_match_source_and_do_not_depend_on_visibility():
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba
    from cs498_track1.visuals import COLORS, voxel_panel

    assert COLORS[1:].tolist() == [
        "#E8C77D",
        "#91A0A7",
        "#40C4E6",
        "#FF8A34",
        "#F05264",
        "#B887E8",
        "#5C91F2",
        "#F2D64B",
        "#62C985",
    ]
    # Separated cubes: each class has one observed and one hidden cell. All
    # twelve exposed faces for that class must keep the same semantic color.
    labels = np.zeros((1, 1, 11), dtype=np.int8)
    labels[0, 0, ::2] = [1, 1, 4, 4, 9, 9]
    completion = np.zeros_like(labels, dtype=bool)
    completion[0, 0, [2, 6, 10]] = True
    fig = plt.figure()
    try:
        ax = fig.add_subplot(projection="3d")
        voxel_panel(ax, labels, labels > 0, completion)
        colors = np.asarray(ax.collections[0].get_facecolor())
        assert colors.shape == (36, 4)
        for label in [1, 4, 9]:
            assert (
                np.all(np.isclose(colors, to_rgba(COLORS[label])), axis=1).sum() == 12
            )
    finally:
        plt.close(fig)


def test_comparison_crop_keeps_both_images_aligned():
    from cs498_track1.visuals import comparison_images

    target = np.zeros((4, 3, 12), dtype=np.int8)
    target[1, 1, 3] = 4
    prediction = target.copy()
    prediction[2, 1, 10] = 9  # a distant error must not get independently cropped
    images = comparison_images([target, prediction], np.ones_like(target, dtype=bool))
    assert images[0].size == images[1].size
    alpha_a = np.asarray(images[0])[:, :, 3] > 0
    alpha_b = np.asarray(images[1])[:, :, 3] > 0
    assert alpha_a.any() and alpha_b.sum() > alpha_a.sum()
    assert (alpha_a & alpha_b).sum() > 0.9 * alpha_a.sum()
