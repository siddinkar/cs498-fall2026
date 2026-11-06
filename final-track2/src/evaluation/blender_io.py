"""Blender-only GT extraction and final-GLB import; no physics simulation."""

from pathlib import Path
import bpy
import numpy as np
from mathutils import Vector


def load_gt(path, frame):
    bpy.ops.wm.open_mainfile(filepath=str(path))
    scene = bpy.context.scene
    scene.frame_set(frame)
    if scene.rigidbody_world:
        bpy.ops.rigidbody.world_remove()
    for obj in scene.objects:
        obj.animation_data_clear()
    if scene.node_tree:
        for node in list(scene.node_tree.nodes):
            if node.type == "OUTPUT_FILE":
                scene.node_tree.nodes.remove(node)
    return scene


def world_mesh(objects):
    vertices = []
    faces = []
    offset = 0
    graph = bpy.context.evaluated_depsgraph_get()
    for obj in sorted(objects, key=lambda item: item.name):
        evaluated = obj.evaluated_get(graph)
        mesh = evaluated.to_mesh()
        mesh.calc_loop_triangles()
        raw = np.empty(len(mesh.vertices) * 3, np.float32)
        mesh.vertices.foreach_get("co", raw)
        xyz = raw.reshape(-1, 3).astype(np.float64)
        matrix = np.asarray(evaluated.matrix_world, np.float64)
        xyz = xyz @ matrix[:3, :3].T + matrix[:3, 3]
        raw_faces = np.empty(len(mesh.loop_triangles) * 3, np.int32)
        mesh.loop_triangles.foreach_get("vertices", raw_faces)
        triangles = raw_faces.reshape(-1, 3).astype(np.int64)
        vertices.append(xyz)
        faces.append(triangles + offset)
        offset += len(xyz)
        evaluated.to_mesh_clear()
    return np.concatenate(vertices), np.concatenate(faces)


def gt_groups(scene):
    groups = {}
    for obj in sorted(scene.objects, key=lambda item: item.name):
        if obj.type == "MESH" and not obj.hide_render:
            groups.setdefault(obj.pass_index, []).append(obj)
    if not any(identity > 0 for identity in groups):
        raise ValueError(
            "GT .blend needs positive object pass_index IDs; background/floor is 0"
        )
    return groups


def import_predictions(job):
    manifest = job["manifest_data"]
    root = Path(job["prediction_manifest"]).parent
    groups = {}
    for row in manifest["meshes"]:
        previous = set(bpy.context.scene.objects)
        bpy.ops.import_scene.gltf(filepath=str(root / row["glb_path"]))
        objects = sorted(
            [
                obj
                for obj in set(bpy.context.scene.objects) - previous
                if obj.type == "MESH"
            ],
            key=lambda obj: obj.name,
        )
        if not objects:
            raise ValueError("Empty prediction GLB")
        for obj in objects:
            obj.pass_index = row["instance_id"]
        vertices, _ = world_mesh(objects)
        np.testing.assert_allclose(
            [vertices.min(0), vertices.max(0)], row["bbox_dataset_world_m"], atol=2e-6
        )
        groups[row["instance_id"]] = objects
    return groups


def aim(camera, location, target):
    camera.location = location
    camera.rotation_euler = (
        (Vector(target) - camera.location).to_track_quat("-Z", "Y").to_euler()
    )
    bpy.context.view_layer.update()
