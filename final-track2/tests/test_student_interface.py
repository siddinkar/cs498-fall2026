"""Offline input/export, anti-leakage, packaging, and grading contract tests."""

import copy
import importlib.util
import json
import hashlib
from io import BytesIO
import os
from pathlib import Path
import subprocess
import sys
import zipfile

import numpy as np
from PIL import Image
import pytest
import trimesh

from track2_io import load_scene, export_scene, validate_scene, scene_ids, write_json
from course_metrics import summarize, performance_points, official_metric_identity
from evaluation.portable_metric import PROTOCOL_SHA256
from package_submission import package
from baselines.mesh_output import unpack

spec = importlib.util.spec_from_file_location(
    "download_data", Path(__file__).parents[1] / "data/download_data.py"
)
download = importlib.util.module_from_spec(spec)
spec.loader.exec_module(download)


def input_scene(root, identity="test_scene_000"):
    split, _, index = identity.partition("_scene_")
    path = root / split / f"scene_{index}"
    (path / "rgb").mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), (130, 90, 70)).save(path / "rgb/frame_0000.png")
    np.savez_compressed(
        path / "cameras.npz",
        K=np.eye(3),
        camera_to_world_cv=np.eye(4)[None],
        camera_to_world_blender=np.eye(4)[None],
        frame_indices=np.array([0]),
        image_size_wh=np.array([16, 16]),
    )
    return load_scene(root, identity)


def test_rgb_only_export_unknown_category_and_world_coordinates(tmp_path):
    scene = input_scene(tmp_path / "data")
    assert scene.rgb(0).shape == (16, 16, 3)
    mesh = trimesh.creation.box(extents=[0.2, 0.3, 0.4])
    mesh.apply_translation([0.1, 0.2, 0.8])
    output = tmp_path / scene.scene_uid / "evaluation"
    manifest = export_scene(scene, {7: mesh}, output)
    record, files = validate_scene(manifest, tmp_path / "data")
    assert record["meshes"][0]["category"] == "unknown"
    np.testing.assert_allclose(record["meshes"][0]["bbox_dataset_world_m"], mesh.bounds)
    assert len(files) == 4
    assert not (tmp_path / "data/test/scene_000/scene.blend").exists()
    assert not (output / "render").exists()
    with pytest.raises(FileExistsError):
        export_scene(scene, {7: mesh}, output)


@pytest.mark.parametrize("alpha_mode", ["OPAQUE", "MASK", "BLEND"])
def test_custom_pbr_material_survives_glb_export(tmp_path, alpha_mode):
    """Existing color/linear factors and alpha must reach the rendered GLB."""
    scene = input_scene(tmp_path / "data")
    mesh = trimesh.creation.box()
    base = Image.new("RGBA", (2, 2), (200, 150, 100, 80))
    metallic_roughness = Image.new("RGB", (2, 2), (255, 120, 180))
    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=base,
        baseColorFactor=[128, 64, 192, 160],
        metallicRoughnessTexture=metallic_roughness,
        metallicFactor=.2,
        roughnessFactor=.3,
        alphaMode=alpha_mode,
        alphaCutoff=.4,
        doubleSided=False,
        emissiveFactor=[.1, .2, .3],
        emissiveTexture=Image.new("RGB", (2, 2), (20, 40, 60)),
        normalTexture=Image.new("RGB", (2, 2), (128, 128, 255)),
        occlusionTexture=Image.new("RGB", (2, 2), (120, 120, 120)),
    )
    mesh.visual = trimesh.visual.TextureVisuals(
        uv=np.tile([.25, .75], (len(mesh.vertices), 1)), material=material
    )
    manifest = export_scene(scene, {7: mesh}, tmp_path / "evaluation")
    validate_scene(manifest, tmp_path / "data")
    glb, binary = unpack((manifest.parent / "meshes/obj_007.glb").read_bytes())
    exported = glb["materials"][0]
    pbr = exported["pbrMetallicRoughness"]
    np.testing.assert_allclose(
        pbr["baseColorFactor"], np.array([128, 64, 192, 160]) / 255
    )
    assert pbr["metallicFactor"] == .2
    assert pbr["roughnessFactor"] == .3
    assert exported["alphaMode"] == alpha_mode
    assert exported["alphaCutoff"] == .4
    assert exported.get("doubleSided", False) is False
    assert exported["emissiveFactor"] == [.1, .2, .3]

    def texture_pixels(reference):
        image = glb["images"][glb["textures"][reference["index"]]["source"]]
        view = glb["bufferViews"][image["bufferView"]]
        start = view.get("byteOffset", 0)
        with Image.open(BytesIO(binary[start : start + view["byteLength"]])) as decoded:
            return np.asarray(decoded).copy()

    np.testing.assert_array_equal(texture_pixels(pbr["baseColorTexture"]), base)
    np.testing.assert_array_equal(
        texture_pixels(pbr["metallicRoughnessTexture"]), metallic_roughness
    )
    for channel in ("emissiveTexture", "normalTexture", "occlusionTexture"):
        np.testing.assert_array_equal(
            texture_pixels(exported[channel]), getattr(material, channel)
        )
    # Export must not mutate the method's input material or its texture pixels.
    assert mesh.visual.material is material
    assert material.metallicFactor == .2 and material.roughnessFactor == .3
    np.testing.assert_array_equal(material.baseColorTexture, base)


@pytest.mark.parametrize("with_base_texture", [False, True])
def test_missing_pbr_textures_are_neutral_not_nonmetal_overrides(
    tmp_path, with_base_texture
):
    scene = input_scene(tmp_path / "data")
    mesh = trimesh.creation.box()
    material = trimesh.visual.material.PBRMaterial(
        baseColorFactor=[128, 64, 192, 255],
        metallicFactor=.6,
        roughnessFactor=.25,
    )
    if with_base_texture:
        material.baseColorTexture = Image.new("RGB", (2, 2), (100, 150, 200))
    mesh.visual = trimesh.visual.TextureVisuals(
        uv=np.zeros((len(mesh.vertices), 2)) if with_base_texture else None,
        material=material,
    )
    manifest = export_scene(scene, {1: mesh}, tmp_path / "evaluation")
    validate_scene(manifest, tmp_path / "data")
    glb, binary = unpack((manifest.parent / "meshes/obj_001.glb").read_bytes())
    pbr = glb["materials"][0]["pbrMetallicRoughness"]
    assert pbr["metallicFactor"] == .6
    assert pbr["roughnessFactor"] == .25
    np.testing.assert_allclose(
        pbr["baseColorFactor"], np.array([128, 64, 192, 255]) / 255
    )
    for channel in ("baseColorTexture", "metallicRoughnessTexture"):
        image = glb["images"][glb["textures"][pbr[channel]["index"]]["source"]]
        view = glb["bufferViews"][image["bufferView"]]
        start = view.get("byteOffset", 0)
        with Image.open(BytesIO(binary[start : start + view["byteLength"]])) as decoded:
            if channel == "metallicRoughnessTexture" or not with_base_texture:
                assert np.all(np.asarray(decoded) == 255)
    assert material.metallicRoughnessTexture is None
    if not with_base_texture:
        assert material.baseColorTexture is None
        assert mesh.visual.uv is None


@pytest.mark.parametrize(
    "settings",
    [{"metallicFactor": 1.1}, {"roughnessFactor": float("nan")}, {"alphaCutoff": -.1}],
)
def test_invalid_pbr_factors_do_not_produce_invalid_glbs(tmp_path, settings):
    scene = input_scene(tmp_path / "data")
    mesh = trimesh.creation.box()
    mesh.visual = trimesh.visual.TextureVisuals(
        material=trimesh.visual.material.PBRMaterial(**settings)
    )
    with pytest.raises(ValueError, match="Invalid glTF PBR"):
        export_scene(scene, {1: mesh}, tmp_path / "evaluation")


def test_custom_material_through_student_method_cli(tmp_path):
    scene = input_scene(tmp_path / "data")
    (tmp_path / "custom_material.py").write_text(
        "import numpy as np\n"
        "import trimesh\n"
        "from PIL import Image\n"
        "def predict(scene):\n"
        "    mesh = trimesh.creation.box(extents=[.2, .3, .4])\n"
        "    material = trimesh.visual.material.PBRMaterial(\n"
        "        baseColorTexture=Image.fromarray(scene.rgb(0)),\n"
        "        baseColorFactor=[128, 64, 192, 255],\n"
        "        metallicFactor=.2, roughnessFactor=.3, alphaMode='MASK',\n"
        "        alphaCutoff=.4)\n"
        "    mesh.visual = trimesh.visual.TextureVisuals(\n"
        "        uv=np.zeros((len(mesh.vertices), 2)), material=material)\n"
        "    return {7: mesh}\n"
    )
    root = Path(__file__).parents[1]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(tmp_path) + os.pathsep + environment.get(
        "PYTHONPATH", ""
    )
    result = subprocess.run(
        [
            sys.executable,
            str(root / "scripts/student/run_method.py"),
            "--method", "custom_material:predict",
            "--scene", scene.scene_uid,
            "--data-root", str(tmp_path / "data"),
            "--output-root", str(tmp_path / "outputs"),
        ],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    manifest = tmp_path / "outputs" / scene.scene_uid / "evaluation/mesh_manifest.json"
    validate_scene(manifest, tmp_path / "data")
    glb, _ = unpack((manifest.parent / "meshes/obj_007.glb").read_bytes())
    material = glb["materials"][0]
    np.testing.assert_allclose(
        material["pbrMetallicRoughness"]["baseColorFactor"],
        np.array([128, 64, 192, 255]) / 255,
    )
    assert material["pbrMetallicRoughness"]["metallicFactor"] == .2
    assert material["pbrMetallicRoughness"]["roughnessFactor"] == .3
    assert material["alphaMode"] == "MASK" and material["alphaCutoff"] == .4


def test_camera_tampering_and_mesh_hash_tampering_fail(tmp_path):
    scene = input_scene(tmp_path / "data")
    manifest = export_scene(scene, {1: trimesh.creation.box()}, tmp_path / "evaluation")
    camera = manifest.parent / "inputs/cameras.npz"
    modified = copy.deepcopy(scene.calibration)
    modified["K"][0, 0] = 2
    np.savez_compressed(camera, **modified)
    with pytest.raises(AssertionError):
        validate_scene(manifest, tmp_path / "data")
    np.savez_compressed(camera, **scene.calibration)
    (manifest.parent / "meshes/obj_001.glb").write_bytes(b"not a GLB")
    with pytest.raises(ValueError):
        validate_scene(manifest, tmp_path / "data")


def test_combined_scene_glb_limit_is_strict(tmp_path, monkeypatch):
    scene = input_scene(tmp_path / "data")
    manifest = export_scene(
        scene,
        {1: trimesh.creation.box(), 2: trimesh.creation.box()},
        tmp_path / "evaluation",
    )
    size = sum(p.stat().st_size for p in (manifest.parent / "meshes").glob("*.glb"))
    monkeypatch.setattr("track2_io.MAX_SCENE_GLB_BYTES", size)
    with pytest.raises(ValueError, match="strictly less than 20 MB"):
        validate_scene(manifest, tmp_path / "data")
    monkeypatch.setattr("track2_io.MAX_SCENE_GLB_BYTES", size + 1)
    validate_scene(manifest, tmp_path / "data")


def test_complete_twenty_case_minimal_deterministic_bundle(tmp_path):
    for identity in scene_ids("test"):
        scene = input_scene(tmp_path / "data", identity)
        export_scene(
            scene,
            {1: trimesh.creation.box()},
            tmp_path / "predictions" / identity / "evaluation",
        )
    preview = (
        tmp_path / "predictions/test_scene_000/evaluation/prediction_preview/index.html"
    )
    preview.parent.mkdir()
    preview.write_text("Qualitative report evidence, not part of the minimal Test ZIP")
    receipt = package(tmp_path / "predictions", tmp_path / "data", tmp_path / "a.zip")
    package(tmp_path / "predictions", tmp_path / "data", tmp_path / "b.zip")
    assert (tmp_path / "a.zip").read_bytes() == (tmp_path / "b.zip").read_bytes()
    assert len(receipt["scene_ids"]) == 20
    with zipfile.ZipFile(tmp_path / "a.zip") as archive:
        assert len(archive.namelist()) == 81
        assert not any("report" in p or "scene.blend" in p for p in archive.namelist())
        assert not any("prediction_preview" in p for p in archive.namelist())
    (tmp_path / "predictions/test_scene_019/evaluation/mesh_manifest.json").unlink()
    with pytest.raises(FileNotFoundError):
        package(
            tmp_path / "predictions", tmp_path / "data", tmp_path / "incomplete.zip"
        )
    assert not (tmp_path / "incomplete.zip").exists()


def test_archive_traversal_and_test_gt_rejected(tmp_path):
    for name in [
        "../bad",
        "test/scene_000/scene.blend",
        "test/scene_000/scene.json",
        "test/scene_020/cameras.npz",
    ]:
        archive = tmp_path / "unsafe.zip"
        with zipfile.ZipFile(archive, "w") as zipped:
            zipped.writestr(name, b"bad")
        with pytest.raises(ValueError):
            download.safe_extract(archive, tmp_path / "data")


def test_missing_cases_are_zeros_before_score_curve(tmp_path):
    groups = {
        key: 1
        for key in [
            "matching",
            "scene_geometry",
            "object_geometry",
            "scene_appearance",
            "object_appearance",
            "collision_free",
        ]
    }
    report = dict(
        schema="cs498.track2.case_metrics.v2",
        protocol_sha256=PROTOCOL_SHA256,
        scene_id="val_scene_000",
        passed=True,
        status="complete",
        groups=groups,
        final_score=100,
        fingerprint=official_metric_identity(),
    )
    report["signature"] = hashlib.sha256(
        json.dumps(report["fingerprint"], sort_keys=True).encode()
    ).hexdigest()
    write_json(
        tmp_path / "val_scene_000/evaluation/reports/evaluation/default/metrics.json",
        report,
    )
    summary = summarize(tmp_path, "val")
    assert summary["completed"] == 1 and len(summary["missing"]) == 9
    assert summary["means"]["matching"] == 0.1
    assert summary["means"]["benchmark"] == 10


def test_full_credit_uses_split_means_not_case_or_baseline_attainment():
    means = dict(
        matching=0.75,
        scene_geometry=0.75,
        object_geometry=0.75,
        scene_appearance=0.75,
        object_appearance=0.75,
        collision_free=0.75,
        benchmark=20,
    )
    val = dict(
        schema="cs498.track2.split_summary.v2",
        protocol_sha256=PROTOCOL_SHA256,
        means=means.copy(),
    )
    test = dict(
        schema="cs498.track2.split_summary.v2",
        protocol_sha256=PROTOCOL_SHA256,
        means=means.copy(),
    )
    reference = dict(protocol_sha256=PROTOCOL_SHA256)
    assert performance_points(val, test, reference) == pytest.approx(
        dict(perception=1.5, reconstruction=1.5, collision_free=1.0)
    )
    test["means"]["matching"] = 0
    assert performance_points(val, test, reference)["perception"] == pytest.approx(0.75)
    assert sum(performance_points(val, None, reference).values()) == pytest.approx(4)


def test_old_simulation_results_are_not_relabelled_as_collision_free(tmp_path):
    path = tmp_path / "val_scene_000/evaluation/reports/evaluation/default/metrics.json"
    write_json(path, dict(scene_id="val_scene_000", status="complete", passed=True))
    with pytest.raises(ValueError, match="obsolete"):
        summarize(tmp_path, "val")
    with pytest.raises(ValueError, match="current collision-free protocol"):
        performance_points(
            dict(schema="old", means={}), None, dict(protocol_sha256=PROTOCOL_SHA256)
        )
