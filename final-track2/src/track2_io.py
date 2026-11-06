"""Algorithm-independent RGB input, textured mesh export, and bundle validation.

Public methods see only RGB and calibration. Geometry passed to export_scene is
already in dataset world coordinates (right-handed, Z-up, meters).
"""

from dataclasses import dataclass
import json
from pathlib import Path
import re
import shutil
import struct
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from baselines.mesh_output import (
    DATASET_FROM_GLTF,
    GLTF_FROM_DATASET,
    SCHEMA,
    load_mesh_manifest,
    load_textured_meshes,
)
from baselines.pi3x_fusion.common import sha256, write_json

SPLIT_COUNTS = {"dev": 30, "val": 10, "test": 20}
MAX_SCENE_GLB_BYTES = (
    20_000_000  # Strictly below 20 decimal MB, all instances combined.
)


def scene_ids(split):
    """Return the complete, fixed scene list; missing cases must not disappear."""
    return [f"{split}_scene_{i:03d}" for i in range(SPLIT_COUNTS[split])]


def scene_path(data_root, scene_uid):
    """Resolve a canonical scene ID without accepting traversal or extra cases."""
    match = re.fullmatch(r"(dev|val|test)_scene_(\d{3})", scene_uid)
    if not match or scene_uid not in scene_ids(match[1]):
        raise ValueError(f"Invalid scene ID: {scene_uid}")
    return Path(data_root).resolve() / match[1] / f"scene_{match[2]}"


@dataclass(frozen=True)
class RGBScene:
    """Input-only view: no depth, masks, GT meshes, or scene-generation metadata."""

    scene_uid: str
    rgb_paths: tuple
    calibration: dict

    def rgb(self, position):
        """Load one H x W x 3 uint8 RGB frame in the supplied temporal order."""
        with Image.open(self.rgb_paths[position]) as image:
            return np.asarray(image.convert("RGB"))


def load_scene(data_root, scene_uid):
    """Load input paths and numeric calibration only; never open GT artifacts."""
    source = scene_path(data_root, scene_uid)
    with np.load(source / "cameras.npz", allow_pickle=False) as archive:
        calibration = {key: archive[key].copy() for key in archive.files}
    required = {
        "K",
        "camera_to_world_cv",
        "camera_to_world_blender",
        "frame_indices",
        "image_size_wh",
    }
    if not required <= set(calibration):
        raise ValueError(f"Missing calibration fields: {required - set(calibration)}")
    indices = calibration["frame_indices"]
    if indices.ndim != 1 or not len(indices) or np.any(np.diff(indices) <= 0):
        raise ValueError("Frame indices must be nonempty and strictly increasing")
    if calibration["K"].shape not in {(3, 3), (len(indices), 3, 3)}:
        raise ValueError("K must be a shared or per-frame 3 x 3 matrix")
    for key in ["camera_to_world_cv", "camera_to_world_blender"]:
        if calibration[key].shape != (len(indices), 4, 4):
            raise ValueError(f"Invalid shape for {key}")
    if any(not np.isfinite(value).all() for value in calibration.values()):
        raise ValueError("Calibration must be finite numeric arrays")
    paths = tuple(source / "rgb" / f"frame_{int(i):04d}.png" for i in indices)
    if not all(path.is_file() for path in paths):
        raise FileNotFoundError(f"Incomplete RGB frames for {scene_uid}")
    return RGBScene(scene_uid, paths, calibration)


def _embedded_pbr(mesh):
    """Preserve PBR materials; bake vertex/face colors to a simple face atlas.

    Constant metallic/roughness maps satisfy the file contract, not a material
    quality target. This adapter imposes no neural reconstruction requirement.
    """
    import trimesh

    mesh = mesh.copy()
    if (
        not len(mesh.vertices)
        or not len(mesh.faces)
        or not np.isfinite(mesh.vertices).all()
    ):
        raise ValueError("Meshes must contain finite, nonempty triangle geometry")
    material = getattr(mesh.visual, "material", None)
    if isinstance(material, trimesh.visual.material.PBRMaterial):
        # glTF texture RGB is sRGB, while material factors are linear. Keep
        # them separate rather than multiplying encoded image bytes. Copying
        # also retains alpha mode/cutoff, sidedness and the other PBR channels.
        pbr = material.copy()
        for field in ("metallicFactor", "roughnessFactor", "alphaCutoff"):
            value = getattr(pbr, field)
            if value is not None and (
                not np.isfinite(value)
                or value < 0
                or (field != "alphaCutoff" and value > 1)
            ):
                raise ValueError(f"Invalid glTF PBR {field}")
        uv = getattr(mesh.visual, "uv", None)
        if uv is None:
            if any(
                getattr(pbr, field) is not None
                for field in (
                    "baseColorTexture",
                    "metallicRoughnessTexture",
                    "normalTexture",
                    "occlusionTexture",
                    "emissiveTexture",
                )
            ):
                raise ValueError("PBR textures require per-vertex UV coordinates")
            uv = np.zeros((len(mesh.vertices), 2))
        else:
            uv = np.asarray(uv).copy()
            if uv.shape != (len(mesh.vertices), 2) or not np.isfinite(uv).all():
                raise ValueError("PBR UV coordinates must be finite vertex pairs")
        # An absent glTF base texture samples as RGBA=1, and an absent
        # metallic/roughness texture samples as G=B=1. Embed neutral textures
        # to satisfy the course format without changing those scalar factors.
        if pbr.baseColorTexture is None:
            pbr.baseColorTexture = Image.new("RGBA", (2, 2), (255, 255, 255, 255))
        if pbr.metallicRoughnessTexture is None:
            pbr.metallicRoughnessTexture = Image.new("RGB", (2, 2), (255, 255, 255))
        mesh.visual = trimesh.visual.TextureVisuals(uv=uv, material=pbr)
        return mesh
    base = getattr(material, "baseColorTexture", None)
    if base is None and material is not None:
        base = getattr(material, "image", None)
    if base is not None and getattr(mesh.visual, "uv", None) is not None:
        base = base.convert("RGB")
        uv = np.asarray(mesh.visual.uv).copy()
        roughness = getattr(material, "metallicRoughnessTexture", None)
    else:
        colors = np.asarray(mesh.visual.face_colors, dtype=np.uint8)[:, :3]
        side = int(np.ceil(np.sqrt(len(mesh.faces))))
        pixels = np.full((side * side, 3), 180, np.uint8)
        pixels[: len(colors)] = colors
        base = Image.fromarray(pixels.reshape(side, side, 3))
        indices = np.arange(len(mesh.faces))
        centers = np.column_stack(
            ((indices % side + 0.5) / side, 1 - (indices // side + 0.5) / side)
        )
        uv = np.repeat(centers, 3, axis=0)
        mesh = trimesh.Trimesh(
            vertices=mesh.vertices[mesh.faces].reshape(-1, 3),
            faces=np.arange(len(mesh.faces) * 3).reshape(-1, 3),
            process=False,
        )
        roughness = None
    if roughness is None:
        roughness = Image.new("RGB", (2, 2), (255, 255, 0))
    pbr = trimesh.visual.material.PBRMaterial(
        baseColorTexture=base,
        metallicRoughnessTexture=roughness,
        metallicFactor=1.0,
        roughnessFactor=1.0,
        baseColorFactor=[255, 255, 255, 255],
        doubleSided=True,
    )
    mesh.visual = trimesh.visual.TextureVisuals(uv=uv, material=pbr)
    return mesh


def export_scene(scene, meshes, output):
    """Export {positive_instance_id: world_space_Trimesh} to evaluation/.

    Include the reconstructed table as its own instance. IDs are arbitrary,
    unique within a scene, and need not match GT IDs. Category names are unused.
    """
    output = Path(output).resolve()
    if output.name != "evaluation":
        raise ValueError("Output directory must be named evaluation")
    if (
        not meshes
        or len(meshes) > 64
        or any(type(k) is not int or not 1 <= k <= 32767 for k in meshes)
    ):
        raise ValueError(
            "Provide up to 64 meshes with integer instance IDs in [1, 32767]"
        )
    if (output / "mesh_manifest.json").exists():
        raise FileExistsError(
            "Refusing to overwrite a completed result; use a new method/run directory"
        )
    (output / "meshes").mkdir(parents=True, exist_ok=True)
    (output / "inputs").mkdir(exist_ok=True)
    entries, blobs = [], []
    for identity, original in sorted(meshes.items()):
        mesh = _embedded_pbr(original)
        bounds = mesh.bounds.copy()
        mesh.apply_transform(GLTF_FROM_DATASET)
        relative = f"meshes/obj_{identity:03d}.glb"
        path = output / relative
        blobs.append(mesh.export(file_type="glb"))
        entries.append(
            dict(
                instance_id=identity,
                category="unknown",
                glb_path=relative,
                sha256="",
                bbox_dataset_world_m=bounds.tolist(),
                vertices=len(mesh.vertices),
                faces=len(mesh.faces),
            )
        )
    from baselines.glb_budget import fit_scene_glbs

    blobs, budget = fit_scene_glbs(blobs, MAX_SCENE_GLB_BYTES)
    for entry, blob in zip(entries, blobs):
        path = output / entry["glb_path"]
        path.write_bytes(blob)
        entry["sha256"] = sha256(path)
    write_json(output / "reports/export_budget.json", budget)
    np.savez_compressed(output / "inputs/cameras.npz", **scene.calibration)
    write_json(
        output / "inputs/source.json",
        dict(
            scene_id=scene.scene_uid,
            frame_indices=scene.calibration["frame_indices"].tolist(),
            input_contract="RGB + calibration only",
        ),
    )
    manifest = dict(
        schema=SCHEMA,
        scene_id=scene.scene_uid,
        units="meters",
        mesh_coordinate_frame="gltf_world_y_up_right_handed",
        dataset_coordinate_frame="blender_world_z_up_right_handed",
        mesh_to_dataset_world=DATASET_FROM_GLTF.tolist(),
        placement="World placement baked into GLB vertices",
        appearance="Embedded PBR textures; category names are not evaluated",
        camera_path="inputs/cameras.npz",
        source_path="inputs/source.json",
        meshes=entries,
        skipped_instances=[],
        frozen_profile=None,
        render=dict(
            rgb_directory="render/rgb",
            video_path="render/reconstruction.mp4",
            comparison_path="render/comparison.mp4",
            environment_path="render/environment.blend",
        ),
    )
    write_json(output / "mesh_manifest.json", manifest)
    load_textured_meshes(output / "mesh_manifest.json")
    return output / "mesh_manifest.json"


def _check_glb(path):
    """Reject external resources before handing a mesh to a rendering process."""
    with path.open("rb") as stream:
        header = stream.read(20)
        if len(header) != 20:
            raise ValueError("Truncated GLB")
        magic, version, size, chunk_size, chunk_type = struct.unpack("<4sIIII", header)
        if (
            magic != b"glTF"
            or version != 2
            or size != path.stat().st_size
            or chunk_type != 0x4E4F534A
        ):
            raise ValueError("Expected a self-contained glTF 2 GLB")
        if chunk_size > size - 20:
            raise ValueError("Invalid GLB JSON chunk")
        data = json.loads(stream.read(chunk_size))
    for key in ["buffers", "images"]:
        if any("uri" in row for row in data.get(key, [])):
            raise ValueError(
                "External/data URI resources are not permitted; embed them in GLB buffer views"
            )


def validate_scene(
    manifest_path, data_root, max_mesh_bytes=20_000_000, max_scene_bytes=22_000_000
):
    """Validate geometry, hashes, embedded resources, and canonical calibration."""
    raw_path = Path(manifest_path).absolute()
    if any(p.is_symlink() for p in [raw_path, *raw_path.parents]):
        raise ValueError("Submission paths must not use symbolic links")
    manifest = load_mesh_manifest(raw_path)
    if len(manifest["meshes"]) > 64 or any(
        row["instance_id"] > 32767 for row in manifest["meshes"]
    ):
        raise ValueError("At most 64 instances, with IDs in [1, 32767], are permitted")
    if raw_path.parent.name != "evaluation":
        raise ValueError("Manifest must be under evaluation/")
    scene = load_scene(data_root, manifest["scene_id"])
    files = [
        raw_path,
        raw_path.parent / "inputs/cameras.npz",
        raw_path.parent / "inputs/source.json",
    ]
    files += [raw_path.parent / row["glb_path"] for row in manifest["meshes"]]
    glb_bytes = sum(path.stat().st_size for path in files if path.suffix == ".glb")
    if glb_bytes >= MAX_SCENE_GLB_BYTES:
        raise ValueError(
            f"All GLBs in one scene must total strictly less than 20 MB (20,000,000 bytes); got {glb_bytes}"
        )
    for path in files:
        if path.is_symlink() or any(p.is_symlink() for p in path.parents):
            raise ValueError("Symbolic links are not permitted")
        if path.stat().st_size > max_mesh_bytes:
            raise ValueError(f"File exceeds {max_mesh_bytes} bytes: {path.name}")
        if path.suffix == ".glb":
            _check_glb(path)
    if sum(p.stat().st_size for p in files) > max_scene_bytes:
        raise ValueError("Scene exceeds the result size limit")
    with np.load(files[1], allow_pickle=False) as proposed:
        if set(proposed.files) != set(scene.calibration):
            raise ValueError("Camera fields differ from released inputs")
        for key, value in scene.calibration.items():
            np.testing.assert_array_equal(
                proposed[key], value, err_msg=f"Camera mismatch: {key}"
            )
    source = json.loads(files[2].read_text())
    if source.get("frame_indices") != scene.calibration["frame_indices"].tolist():
        raise ValueError("Source frame indices differ from released inputs")
    total_vertices = total_faces = 0
    for entry, mesh in load_textured_meshes(raw_path):
        total_vertices += len(mesh.vertices)
        total_faces += len(mesh.faces)
        if total_vertices > 2_000_000 or total_faces > 2_000_000:
            raise ValueError(
                "Scene exceeds the combined two-million vertex/triangle transport limit"
            )
        if not np.isfinite(mesh.vertices).all() or mesh.faces.max() >= len(
            mesh.vertices
        ):
            raise ValueError("Invalid mesh geometry")
        if len(mesh.vertices) != entry["vertices"] or len(mesh.faces) != entry["faces"]:
            raise ValueError("Mesh counts do not match manifest")
    return manifest, files
