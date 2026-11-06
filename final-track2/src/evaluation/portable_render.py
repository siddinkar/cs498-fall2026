"""Canonical local-only Blender rendering; Kaggle never invokes this module.

Scene cameras depend only on public input calibration. Object cameras use each
side's own AABB, with common world-axis directions and uniform framing. GT mode
is for authorized staff/data preparation or local Dev/Val measurement only.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def config():
    """Read canonical constants without requiring Blender's Python to import pandas."""
    import ast

    tree = ast.parse((Path(__file__).with_name("portable_metric.py")).read_text())
    node = next(
        n.value
        for n in tree.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "PROTOCOL" for t in n.targets)
    )
    return ast.literal_eval(node)


def file_hash(path):
    """Stream a file digest, including large immutable prediction GLBs."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for part in iter(lambda: source.read(1048576), b""):
            digest.update(part)
    return digest.hexdigest()


def scene_views(calibration, settings):
    """Freeze world cameras from the public viewing-ray least-squares center."""
    poses = np.asarray(calibration["camera_to_world_blender"], np.float64)
    origins, directions = poses[:, :3, 3], -poses[:, :3, 2]
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    projectors = np.eye(3)[None] - directions[:, :, None] * directions[:, None, :]
    center = np.linalg.lstsq(
        projectors.sum(0), np.einsum("nij,nj->i", projectors, origins), rcond=None
    )[0]
    radius = max(
        float(np.median(np.linalg.norm(origins - center, axis=1)))
        * settings["scene_radius_multiplier"],
        0.25,
    )
    views = []
    elevations = settings["scene_elevations_degrees"]
    for index in range(settings["scene_views"]):
        elevation = math.radians(elevations[index % len(elevations)])
        azimuth = (
            2
            * math.pi
            * (index // len(elevations))
            / (settings["scene_views"] // len(elevations))
            + math.pi / 12
        )
        direction = np.array(
            [
                math.cos(elevation) * math.cos(azimuth),
                math.cos(elevation) * math.sin(azimuth),
                math.sin(elevation),
            ]
        )
        views.append(
            dict(
                location=(center + radius * direction).tolist(), target=center.tolist()
            )
        )
    return center, radius, views


def render_job(job):
    """Import only declared geometry, replace lighting and produce RGBA views."""
    import bpy
    from mathutils import Vector
    from evaluation.blender_io import load_gt, gt_groups, world_mesh, import_predictions

    protocol = config()
    settings = protocol["render"]
    if bpy.app.version_string != "4.2.0":
        raise RuntimeError("Canonical renders require the provided Blender 4.2.0")
    destination = Path(job["output"])
    destination.mkdir(parents=True, exist_ok=False)
    if job["kind"] == "gt":
        scene = load_gt(job["gt_blend"], 1)
        all_groups = gt_groups(scene)
        floor = (
            world_mesh(all_groups[0])[0]
            if 0 in all_groups
            else np.array([[0.0, 0.0, 0.0]])
        )
        floor_z = float(np.max(floor[:, 2]))
        for obj in all_groups.get(0, []):
            obj.hide_render = True
        groups = {k: v for k, v in all_groups.items() if k > 0}
    else:
        bpy.ops.wm.read_factory_settings(use_empty=True)
        scene = bpy.context.scene
        manifest = json.loads(Path(job["manifest"]).read_text())
        groups = import_predictions(
            dict(prediction_manifest=job["manifest"], manifest_data=manifest)
        )
        floor_z = None
    # Remove scene/imported lights/cameras, not geometry, and never use hidden GT lighting.
    for obj in list(scene.objects):
        if obj.type in {"LIGHT", "CAMERA"}:
            bpy.data.objects.remove(obj, do_unlink=True)
    if scene.rigidbody_world:
        bpy.ops.rigidbody.world_remove()
    scene.use_nodes = False
    scene.render.engine = "CYCLES"
    scene.cycles.samples = settings["samples"]
    scene.cycles.seed = protocol["seed"]
    scene.cycles.use_animated_seed = False
    scene.cycles.use_adaptive_sampling = False
    scene.cycles.use_denoising = True
    scene.cycles.denoiser = "OPENIMAGEDENOISE"
    scene.cycles.denoising_use_gpu = False
    scene.cycles.use_persistent_data = True
    scene.cycles.device = "CPU"
    if job["device"] == "gpu":
        preferences = bpy.context.preferences.addons["cycles"].preferences
        preferences.compute_device_type = "OPTIX"
        preferences.get_devices()
        devices = [device for device in preferences.devices if device.type == "OPTIX"]
        if not devices:
            raise RuntimeError(
                "Requested GPU rendering but no OptiX device is available"
            )
        for device in preferences.devices:
            device.use = device in devices
        scene.cycles.device = "GPU"
    scene.render.threads_mode = "FIXED"
    scene.render.threads = 4
    scene.render.resolution_x = scene.render.resolution_y = settings["resolution"]
    scene.render.resolution_percentage = 100
    scene.render.use_motion_blur = False
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "8"
    scene.render.dither_intensity = 0
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0
    scene.view_settings.gamma = 1
    scene.world = bpy.data.worlds.new("CourseNeutralWorld")
    scene.world.use_nodes = True
    background = scene.world.node_tree.nodes["Background"]
    background.inputs["Color"].default_value = (0.65, 0.65, 0.65, 1)
    background.inputs["Strength"].default_value = 0.7
    camera_data = bpy.data.cameras.new("CourseCamera")
    camera_data.clip_start = 0.0001
    camera_data.clip_end = 1000
    camera = bpy.data.objects.new("CourseCamera", camera_data)
    scene.collection.objects.link(camera)
    scene.camera = camera
    light_data = bpy.data.lights.new("CourseArea", "AREA")
    light = bpy.data.objects.new("CourseArea", light_data)
    scene.collection.objects.link(light)

    def aim(position, target):
        camera.location = position
        camera.rotation_euler = (
            (Vector(target) - camera.location).to_track_quat("-Z", "Y").to_euler()
        )
        bpy.context.view_layer.update()

    def lighting(center, scale):
        light.location = center + np.array([1.0, -1.0, 2.0]) * scale * 2
        light.rotation_euler = (
            (Vector(center) - light.location).to_track_quat("-Z", "Y").to_euler()
        )
        light_data.size = 4 * scale
        light_data.energy = 400 * scale * scale

    def render(path, view):
        path.parent.mkdir(parents=True, exist_ok=True)
        aim(view["location"], view["target"])
        scene.render.filepath = str(path)
        bpy.ops.render.render(write_still=True)
        return dict(
            path=path.relative_to(destination).as_posix(),
            sha256=file_hash(path),
            camera_to_world_blender=np.asarray(camera.matrix_world).tolist(),
            camera_type=camera_data.type,
            ortho_scale=float(camera_data.ortho_scale)
            if camera_data.type == "ORTHO"
            else None,
        )

    records = []
    for identity, objects in sorted(groups.items()):
        vertices, faces = world_mesh(objects)
        geometry_path = destination / "geometry" / f"obj_{identity:03d}.npz"
        geometry_path.parent.mkdir(exist_ok=True)
        np.savez_compressed(geometry_path, vertices=vertices, faces=faces)
        records.append(
            dict(
                id=identity,
                geometry=geometry_path.relative_to(destination).as_posix(),
                geometry_sha256=file_hash(geometry_path),
                bounds=[vertices.min(0).tolist(), vertices.max(0).tolist()],
                views=[],
            )
        )
    with np.load(job["calibration"], allow_pickle=False) as calibration:
        center, radius, fixed_views = scene_views(calibration, settings)
    lighting(center, radius)
    camera_data.type = "PERSP"
    camera_data.lens_unit = "FOV"
    camera_data.sensor_fit = "HORIZONTAL"
    camera_data.angle = math.radians(settings["scene_fov_degrees"])
    scene_images = [
        render(destination / "scene" / f"view_{i:02d}.png", view)
        for i, view in enumerate(fixed_views)
    ]
    for row in records:
        for identity, objects in groups.items():
            for obj in objects:
                obj.hide_render = identity != row["id"]
        bounds = np.asarray(row["bounds"])
        center = bounds.mean(0)
        extent = max(float((bounds[1] - bounds[0]).max()), 0.001)
        lighting(center, extent)
        camera_data.type = "ORTHO"
        camera_data.ortho_scale = settings["object_frame_multiplier"] * extent
        for index, direction in enumerate(settings["object_directions"]):
            view = dict(
                location=(center + np.asarray(direction) * 3 * extent).tolist(),
                target=center.tolist(),
            )
            row["views"].append(
                render(
                    destination / f"object_{row['id']:03d}" / f"view_{index:02d}.png",
                    view,
                )
            )
    receipt = dict(
        schema=protocol["schema"],
        protocol_sha256=hashlib.sha256(
            json.dumps(protocol, sort_keys=True).encode()
        ).hexdigest(),
        scene_id=job["scene_id"],
        input_fingerprint=job["input_fingerprint"],
        objects=records,
        scene_views=scene_images,
        floor_z=floor_z,
        device=job["device"],
        blender_version=bpy.app.version_string,
        simulation=False,
    )
    (destination / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


def prepare_render(
    manifest, calibration, output, device="cpu", gt_blend=None, scene_id=None
):
    """Run an immutable, bounded render job and validate cached source identity."""
    from evaluation.subprocess_utils import run_bounded

    output = Path(output).resolve()
    calibration = Path(calibration).resolve()
    fingerprint = dict(
        render_protocol_sha256=hashlib.sha256(
            json.dumps(
                dict(render=config()["render"], seed=config()["seed"]), sort_keys=True
            ).encode()
        ).hexdigest(),
        calibration_sha256=file_hash(calibration),
        renderer_sha256=file_hash(__file__),
        blender_io_sha256=file_hash(Path(__file__).with_name("blender_io.py")),
        device=device,
    )
    if gt_blend is not None:
        gt_blend = Path(gt_blend).resolve()
        fingerprint["gt_blend_sha256"] = file_hash(gt_blend)
    else:
        manifest = Path(manifest).resolve()
        fingerprint["manifest_sha256"] = file_hash(manifest)
        data = json.loads(manifest.read_text())
        scene_id = data["scene_id"]
        fingerprint["glbs"] = {
            r["glb_path"]: file_hash(manifest.parent / r["glb_path"])
            for r in data["meshes"]
        }
    receipt = output / "receipt.json"
    if receipt.exists():
        record = json.loads(receipt.read_text())
        if record["input_fingerprint"] != fingerprint:
            raise ValueError(
                "Render source/protocol changed; choose a fresh output directory"
            )
        for image in record["scene_views"] + [
            image for row in record["objects"] for image in row["views"]
        ]:
            if file_hash(output / image["path"]) != image["sha256"]:
                raise ValueError("Rendered image hash differs from its receipt")
        for row in record["objects"]:
            if file_hash(output / row["geometry"]) != row["geometry_sha256"]:
                raise ValueError("Cached render geometry differs from its receipt")
        return record
    if output.exists():
        raise FileExistsError(
            "Incomplete render output exists; retain it and choose a new output directory"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    shared = (
        Path(
            os.environ.get(
                "TRACK2_EVALUATION_CACHE", str(ROOT / "outputs/.evaluation_cache")
            )
        )
        / "renders"
        / hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()
    )
    if output != shared.resolve() and (shared / "receipt.json").is_file():
        prepare_render(manifest, calibration, shared, device, gt_blend, scene_id)
        shutil.copytree(shared, output)
        return prepare_render(manifest, calibration, output, device, gt_blend, scene_id)
    job = dict(
        kind="gt" if gt_blend else "prediction",
        gt_blend=str(gt_blend) if gt_blend else None,
        manifest=str(manifest) if manifest else None,
        scene_id=scene_id,
        calibration=str(calibration),
        output=str(output),
        device=device,
        input_fingerprint=fingerprint,
    )
    job_path = output.with_suffix(".job.json")
    job_path.write_text(json.dumps(job, indent=2) + "\n")
    executable = os.environ.get(
        "BLENDER_EXECUTABLE", str(ROOT / "env/blender/blender-4.2.0-linux-x64/blender")
    )
    command = [
        executable,
        "--disable-autoexec",
        "-b",
        "-t",
        "4",
        "--python-exit-code",
        "1",
        "--python",
        str(Path(__file__).resolve()),
        "--",
        "--job",
        str(job_path),
    ]
    with output.with_suffix(".log").open("w") as log:
        run_bounded(
            command,
            output=log,
            timeout=1800,
            status_path=output.with_suffix(".process.json"),
            context=dict(purpose="canonical local RGBA render; no simulation"),
        )
    record = prepare_render(manifest, calibration, output, device, gt_blend, scene_id)
    if output != shared.resolve() and not shared.exists():
        shared.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copytree(output, shared)
        except FileExistsError:
            prepare_render(manifest, calibration, shared, device, gt_blend, scene_id)
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    arguments = (
        sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else sys.argv[1:]
    )
    render_job(json.loads(parser.parse_args(arguments).job.read_text()))
