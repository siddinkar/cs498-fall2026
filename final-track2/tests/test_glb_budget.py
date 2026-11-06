"""Strict scene export cap, texture trade-offs and packed-UV compatibility."""
from io import BytesIO

import numpy as np
from PIL import Image
import pytest
import trimesh

from baselines.glb_budget import fit_scene_glbs, for_trimesh, geometry_digest


def textured_blob():
    mesh = trimesh.creation.box()
    rng = np.random.default_rng(498)
    image = Image.fromarray(rng.integers(0, 256, (256, 256, 3), dtype=np.uint8))
    material = trimesh.visual.material.PBRMaterial(baseColorTexture=image,
                                                  metallicRoughnessTexture=image.copy())
    mesh.visual = trimesh.visual.TextureVisuals(uv=rng.random((len(mesh.vertices), 2)), material=material)
    return mesh.export(file_type='glb')


def test_budget_preserves_geometry_and_placement_and_bounds_uv_error():
    original = textured_blob()
    blobs, report = fit_scene_glbs([original], limit=150000)
    assert report['changed'] and sum(map(len, blobs)) < 150000
    assert geometry_digest(original) == geometry_digest(blobs[0])
    before = trimesh.load(BytesIO(original), file_type='glb', force='mesh', process=False)
    after = trimesh.load(BytesIO(for_trimesh(blobs[0])), file_type='glb', force='mesh', process=False)
    np.testing.assert_array_equal(before.vertices, after.vertices)
    np.testing.assert_array_equal(before.faces, after.faces)
    assert abs(before.visual.uv-after.visual.uv).max() <= 1/131070+1e-7
    assert after.visual.material.baseColorTexture.size == (128, 128)
    assert original != blobs[0]


def test_compliant_exports_unchanged_and_unfit_scene_fails():
    original = textured_blob()
    blobs, report = fit_scene_glbs([original], limit=len(original)+1)
    assert blobs == [original] and report['changed'] is False
    with pytest.raises(ValueError, match='without changing geometry'):
        fit_scene_glbs([original], limit=100)
