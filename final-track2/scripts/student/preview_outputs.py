"""Render qualitative scene/object views from any method's GLBs, without GT.

Run in the tools environment. Blender handles only a validated prediction
manifest; no scene.blend, physics roles, or private rendering setup is opened.
These previews are not official appearance metrics or stability simulations.
"""
import argparse
import html
import json
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap


def render_worker(args):
    """Blender-only worker: embedded predicted materials, neutral lighting."""
    import math
    import bpy
    import numpy as np
    from mathutils import Vector

    manifest = json.loads(args.manifest.read_text())
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'CPU'
    scene.cycles.samples = args.samples
    scene.cycles.use_denoising = True
    scene.render.resolution_x = scene.render.resolution_y = args.resolution
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'
    scene.view_settings.view_transform = 'Standard'
    scene.world = bpy.data.worlds.new('PreviewWorld')
    scene.world.use_nodes = True
    scene.world.node_tree.nodes['Background'].inputs['Color'].default_value = (.65, .65, .65, 1)
    scene.world.node_tree.nodes['Background'].inputs['Strength'].default_value = .7
    groups, bounds = {}, {}
    for entry in manifest['meshes']:
        before = set(scene.objects)
        bpy.ops.import_scene.gltf(filepath=str(args.manifest.parent / entry['glb_path']))
        imported = set(scene.objects) - before
        for obj in imported:
            if obj.type == 'LIGHT':
                obj.hide_render = True
        objects = [o for o in imported if o.type == 'MESH']
        if not objects:
            raise ValueError('Empty prediction GLB')
        xyz = np.concatenate([np.array([o.matrix_world @ v.co for v in o.data.vertices]) for o in objects])
        measured = np.array([xyz.min(0), xyz.max(0)])
        np.testing.assert_allclose(measured, entry['bbox_dataset_world_m'], atol=2e-6)
        groups[entry['instance_id']] = objects
        bounds[entry['instance_id']] = measured
    camera_data = bpy.data.cameras.new('PreviewCamera')
    camera = bpy.data.objects.new('PreviewCamera', camera_data)
    scene.collection.objects.link(camera)
    scene.camera = camera
    camera_data.type = 'ORTHO'
    camera_data.clip_start = .0001
    camera_data.clip_end = 10000
    light_data = bpy.data.lights.new('PreviewArea', 'AREA')
    light = bpy.data.objects.new('PreviewArea', light_data)
    scene.collection.objects.link(light)

    def views(box, folder, angles):
        """Auto-frame the predictions only; preserve their geometry/placement."""
        folder.mkdir(parents=True)
        center = box.mean(0)
        radius = max(float(np.linalg.norm(box[1] - box[0])) * .55, .01)
        camera_data.ortho_scale = 2.2 * radius
        light.location = center + np.array([1, -1, 2]) * radius * 2
        light.rotation_euler = (Vector(center) - light.location).to_track_quat('-Z', 'Y').to_euler()
        light_data.size = 4 * radius
        light_data.energy = 400 * radius * radius
        for index, (azimuth, elevation) in enumerate(angles):
            a, e = math.radians(azimuth), math.radians(elevation)
            camera.location = center + 4 * radius * np.array([math.cos(e)*math.cos(a), math.cos(e)*math.sin(a), math.sin(e)])
            camera.rotation_euler = (Vector(center) - camera.location).to_track_quat('-Z', 'Y').to_euler()
            scene.render.filepath = str(folder / f'view_{index:02d}.png')
            bpy.ops.render.render(write_still=True)

    all_bounds = np.array(list(bounds.values()))
    views(np.array([all_bounds[:, 0].min(0), all_bounds[:, 1].max(0)]),
          args.output / 'scene', [(35, 25), (155, 25), (275, 25), (35, 70)])
    for identity, objects in groups.items():
        for other, collection in groups.items():
            for obj in collection:
                obj.hide_render = other != identity
        views(bounds[identity], args.output / f'object_{identity:03d}', [(35, 25), (215, 25)])


def write_gallery(output, manifest, manifest_path):
    """Make compact labeled montages and a portable gallery for report figures."""
    from PIL import Image, ImageDraw, ImageFont
    from baselines.pi3x_fusion.common import sha256, write_json

    def montage(items, target, columns):
        size = 512
        rows = (len(items) + columns - 1) // columns
        canvas = Image.new('RGB', (columns * size, rows * (size + 36)), 'white')
        draw = ImageDraw.Draw(canvas)
        font = ImageFont.load_default(size=22)
        for index, (path, label) in enumerate(items):
            with Image.open(path) as source:
                image = source.convert('RGB')
                image = image.resize((size, size), Image.Resampling.LANCZOS)
            x, y = index % columns * size, index // columns * (size + 36)
            draw.text((x + 8, y + 7), label, fill='black', font=font)
            canvas.paste(image, (x + (size - image.width)//2, y + 36))
        canvas.save(output / target)

    scene_images = sorted((output / 'scene').glob('view_*.png'))
    if len(scene_images) != 4:
        raise RuntimeError('Incomplete prediction scene preview')
    montage([(p, f'{manifest["scene_id"]}: prediction view {i+1}') for i, p in enumerate(scene_images)], 'scene_views.png', 2)
    objects = []
    cards = []
    for entry in manifest['meshes']:
        identity = entry['instance_id']
        images = sorted((output / f'object_{identity:03d}').glob('view_*.png'))
        if len(images) != 2:
            raise RuntimeError(f'Incomplete prediction object preview: {identity}')
        objects.extend((p, f'Instance {identity}: prediction view {i+1}') for i, p in enumerate(images))
        cards.append(f'<h2>Instance {identity}</h2>' + ''.join(
            f'<img src="{p.relative_to(output).as_posix()}" alt="Instance {identity}">' for p in images))
    montage(objects, 'object_views.png', 2)
    receipt = dict(schema='cs498.track2.prediction_preview.v1', scene_id=manifest['scene_id'],
        uses_gt=False, official_metrics=False, physics_simulation=False,
        manifest_sha256=sha256(manifest_path),
        preview_code_sha256=sha256(__file__),
        glbs={row['glb_path']:row['sha256'] for row in manifest['meshes']},
        scene_montage='scene_views.png',object_montage='object_views.png',
        note='Embedded prediction materials under neutral lighting; not the private evaluation environment.')
    write_json(output / 'preview.json', receipt)
    page = '<!doctype html><html lang="en"><meta charset="utf-8"><title>Prediction-only preview</title><style>body{font:16px system-ui;max-width:1100px;margin:2rem auto;padding:1rem}img{width:48%;vertical-align:top}a{margin-right:1rem}</style>'
    page += f'<h1>{html.escape(manifest["scene_id"])}</h1><p>Prediction only. No GT, official metrics, or physics simulation. Neutral preview lighting differs from official evaluation.</p>'
    page += '<p><a href="scene_views.png">Scene montage</a><a href="object_views.png">Object montage</a><a href="preview.json">Provenance</a></p><h2>Whole scene</h2>'
    page += ''.join(f'<img src="{p.relative_to(output).as_posix()}" alt="Whole-scene prediction">' for p in scene_images)
    (output / 'index.html').write_text(page + ''.join(cards) + '</html>')


def main():
    """Validate a public output, run bounded Blender rendering, and index it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, default=_bootstrap.ROOT / 'data/scenes')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--resolution', type=int, default=512, choices=range(128, 1025), metavar='128..1024')
    parser.add_argument('--samples', type=int, default=16, choices=range(1, 129), metavar='1..128')
    parser.add_argument('--timeout', type=float, default=1800)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else None)
    args.manifest = args.manifest.resolve()
    args.output = (args.output or args.manifest.parent / 'prediction_preview').resolve()
    if args.worker:
        render_worker(args)
        return
    from track2_io import ROOT, validate_scene
    from evaluation.subprocess_utils import run_bounded
    manifest, _ = validate_scene(args.manifest, args.data_root)
    args.output.mkdir(parents=True, exist_ok=False)
    executable = os.environ.get('BLENDER_EXECUTABLE', str(ROOT / 'env/blender/blender-4.2.0-linux-x64/blender'))
    command = [executable, '--disable-autoexec', '-b', '-t', '4', '--python-exit-code', '1',
               '--python', str(Path(__file__).resolve()), '--', '--worker', '--manifest', str(args.manifest),
               '--output', str(args.output), '--resolution', str(args.resolution), '--samples', str(args.samples)]
    with (args.output / 'render.log').open('w') as log:
        run_bounded(command, output=log, timeout=args.timeout,
            status_path=args.output / 'render.process.json', context=dict(purpose='qualitative prediction-only preview'))
    write_gallery(args.output, manifest, args.manifest)
    print(args.output / 'index.html')


if __name__ == '__main__':
    main()
