"""Provided scientific plotting helpers; display masks are never inference inputs."""

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image

# Stable class-ID palette shared by semantic plots and their legends.
# Visibility must never change a voxel's semantic class color.
COLORS = np.array(
    [
        "#eeeeee",
        "#E8C77D",  # 1 floor
        "#91A0A7",  # 2 wall / ceiling
        "#40C4E6",  # 3 opening
        "#FF8A34",  # 4 furniture
        "#F05264",  # 5 appliance
        "#B887E8",  # 6 plumbing fixture
        "#5C91F2",  # 7 electronics
        "#F2D64B",  # 8 small object
        "#62C985",  # 9 other
    ]
)


def style():
    """Use the approved regular-sans paper style with restrained line weights."""
    installed = {f.name for f in font_manager.fontManager.ttflist}
    family = next(
        name for name in ("Avenir", "Nimbus Sans", "DejaVu Sans") if name in installed
    )
    plt.rcParams.update(
        {
            "font.family": family,
            "font.size": 9,
            "text.color": "#171717",
            "axes.labelcolor": "#171717",
            "pdf.fonttype": 3,
            "savefig.facecolor": "white",
        }
    )


def voxel_panel(ax, labels, mask, completion=None, zoom=1.0):
    """Draw exposed faces with fixed class colors; outline hidden cells more strongly.

    A completion mask changes only edge emphasis, never the semantic color.
    """
    labels = np.asarray(labels)
    occupied = (labels > 0) & np.asarray(mask)
    # Geometry axes are x-right, forward, up. The lattice uses z,y,x order.
    cells = np.argwhere(occupied)
    if not len(cells):
        ax.set_axis_off()
        return
    faces = []
    colors = []
    widths = []
    delta = 0.25
    offsets = np.array(
        [[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1]]
    )
    corners = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
        ]
    )
    face_ids = [
        [0, 1, 2, 3],
        [4, 7, 6, 5],
        [0, 4, 5, 1],
        [3, 2, 6, 7],
        [0, 3, 7, 4],
        [1, 5, 6, 2],
    ]
    # neighbors in z,y,x correspond to forward/up/right face normals.
    mapping = [2, 3, 0, 1, 4, 5]
    for cell in cells:
        z, y, x = cell
        center = np.array([-10 + x * delta, z * delta, -1.75 + y * delta])
        vertices = center + corners * delta
        for k, offset in enumerate(offsets):
            neighbor = cell + offset
            if (
                np.all(neighbor >= 0)
                and np.all(neighbor < occupied.shape)
                and occupied[tuple(neighbor)]
            ):
                continue
            face = face_ids[mapping[k]]
            faces.append(vertices[face])
            colors.append(COLORS[int(labels[z, y, x])])
            widths.append(
                0.20 if completion is not None and completion[z, y, x] else 0.06
            )
    collection = Poly3DCollection(
        faces,
        facecolors=colors,
        edgecolors="#ffffff",
        linewidths=widths,
        rasterized=True,
    )
    ax.add_collection3d(collection)
    # Use the shared display domain, not prediction-dependent occupied extrema.
    # This keeps target/prediction framing identical and avoids hiding errors.
    domain = np.argwhere(mask)
    points = np.stack(
        [
            -10 + (domain[:, 2] + 0.5) * delta,
            (domain[:, 0] + 0.5) * delta,
            -1.75 + (domain[:, 1] + 0.5) * delta,
        ],
        axis=1,
    )
    lo = points.min(0) - 0.25
    hi = points.max(0) + 0.25
    ax.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]), zlim=(lo[2], hi[2]))
    ax.set_box_aspect(np.maximum(hi - lo, 0.5), zoom=zoom)
    ax.view_init(elev=24, azim=-65)
    ax.set_axis_off()
    ax.set_proj_type("ortho")


def save(fig, path):
    """Export high-resolution evidence and close the figure."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def comparison_images(labels, mask, completion=None):
    """Render an aligned pair with one shared crop of transparent canvas padding.

    All voxels in the shared display domain remain visible to the renderer.
    The union crop preserves both images' errors and identical metric scale.
    """
    images = []
    for values in labels:
        fig = plt.figure(figsize=(4.8, 3.0), dpi=240, facecolor="none")
        ax = fig.add_axes([0, 0, 1, 1], projection="3d")
        ax.patch.set_alpha(0)
        voxel_panel(ax, values, mask, completion)
        fig.canvas.draw()
        pixels = np.asarray(fig.canvas.buffer_rgba()).copy()
        plt.close(fig)
        if np.any(pixels[[0, -1], :, 3]) or np.any(pixels[:, [0, -1], 3]):
            raise ValueError(
                "Scene exceeds the render canvas; do not clip the evidence"
            )
        images.append(Image.fromarray(pixels))
    bounds = [im.getchannel("A").getbbox() for im in images]
    valid = [b for b in bounds if b is not None]
    if not valid:
        return images
    crop = (
        min(b[0] for b in valid),
        min(b[1] for b in valid),
        max(b[2] for b in valid),
        max(b[3] for b in valid),
    )
    return [im.crop(crop) for im in images]
