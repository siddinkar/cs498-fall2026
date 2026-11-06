"""Tiny format demonstration, NOT a reconstruction baseline or valid final method.

Replace predict with your algorithm. This intentionally simple example creates
one colored box to let you test input loading and GLB export before integrating
models. It does not locate objects or reconstruct a table, and earns no claim
of reconstruction quality.
"""
import numpy as np
import trimesh


def predict(scene):
    """Accept RGBScene and return {instance_id: world-space Trimesh}."""
    image = scene.rgb(0)
    camera = scene.calibration['camera_to_world_cv'][0]
    center = camera[:3, 3] + camera[:3, :3] @ np.array([0., 0., 1.])
    mesh = trimesh.creation.box(extents=[.1, .1, .1])
    mesh.apply_translation(center)
    mesh.visual.face_colors = np.r_[np.mean(image, axis=(0, 1)).astype(np.uint8), 255]
    return {1: mesh}
