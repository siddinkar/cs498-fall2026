"""Public textured-mesh scene v1 contract shared by renderers and evaluators.

GLBs use standard right-handed glTF Y-up meters. Their vertices already carry
world placement; no per-instance alignment is needed. DATASET_FROM_GLTF converts
to the dataset's Blender Z-up world used by cameras and future geometry metrics.
"""
import json
import shutil
import struct
from pathlib import Path
import numpy as np
from .pi3x_fusion.common import resolve, read_json, write_json, sha256

SCHEMA = 'tabletop.textured_mesh_scene.v1'
DATASET_FROM_GLTF = np.array([[1., 0, 0, 0], [0, 0, -1., 0], [0, 1., 0, 0], [0, 0, 0, 1.]])
GLTF_FROM_DATASET = DATASET_FROM_GLTF.T


def unpack(blob):
    """Read self-contained, single-buffer GLBs for mesh/UV decoding."""
    if len(blob) < 28:
        raise ValueError('Truncated GLB')
    magic, version, size, count, kind = struct.unpack('<4sIIII', blob[:20])
    if magic != b'glTF' or version != 2 or size != len(blob) or kind != 0x4E4F534A:
        raise ValueError('Expected a glTF 2 GLB')
    data = json.loads(blob[20:20+count])
    offset = 20+count
    length, kind = struct.unpack('<II', blob[offset:offset+8])
    binary = blob[offset+8:offset+8+length]
    if kind != 0x004E4942 or len(binary) != length or len(data.get('buffers', [])) != 1:
        raise ValueError('Mesh decoding requires one embedded binary buffer')
    return data, binary


def pack(data, pieces):
    """Serialize aligned GLB buffer views without changing their IDs."""
    packed = bytearray()
    for view, piece in zip(data['bufferViews'], pieces):
        packed.extend(b'\0'*(-len(packed) % 4))
        view['byteOffset'] = len(packed)
        view['byteLength'] = len(piece)
        packed.extend(piece)
    data['buffers'][0]['byteLength'] = len(packed)
    packed.extend(b'\0'*(-len(packed) % 4))
    encoded = json.dumps(data, separators=(',', ':'), allow_nan=False).encode()
    encoded += b' '*(-len(encoded) % 4)
    return (struct.pack('<4sIII', b'glTF', 2, 12+8+len(encoded)+8+len(packed), len(encoded))
            + struct.pack('<I', 0x4E4F534A)+encoded+struct.pack('<II', len(packed), 0x004E4942)+packed)


def for_trimesh(blob):
    """Decode normalized uint16 UVs to floats for trimesh, in memory only.

    glTF consumers such as Blender support normalized uint16 directly.
    trimesh's integer-UV reader instead flips V before normalization. Keep
    this compatibility conversion beside the fingerprinted mesh reader so
    evaluation receipts cover all position/index/UV interpretation code.
    """
    data, binary = unpack(blob)
    views = data['bufferViews']
    pieces = [binary[v.get('byteOffset', 0):v.get('byteOffset', 0)+v['byteLength']] for v in views]
    changed = False
    for index in {p['attributes']['TEXCOORD_0'] for mesh in data['meshes'] for p in mesh['primitives']
                  if 'TEXCOORD_0' in p['attributes']}:
        accessor = data['accessors'][index]
        if accessor.get('componentType') != 5123 or not accessor.get('normalized'):
            continue
        vi = accessor['bufferView']
        view = views[vi]
        offset = view.get('byteOffset', 0)+accessor.get('byteOffset', 0)
        values = np.ndarray((accessor['count'], 2), dtype='<u2', buffer=binary,
                            offset=offset, strides=(view.get('byteStride', 4), 2)).astype('<f4')/65535
        # Shared/interleaved attributes get a private float view in memory.
        pieces.append(values.tobytes())
        views.append(dict(buffer=0, byteOffset=0, byteLength=values.nbytes, target=34962))
        accessor.update(componentType=5126, normalized=False, bufferView=len(views)-1, byteOffset=0)
        accessor.pop('min', None)
        accessor.pop('max', None)
        changed = True
    return pack(data, pieces) if changed else blob


def publish_textured_meshes(config):
    import trimesh
    from .fire3d_recon.policy import select_conditioned
    if config.get('frozen_profile'):
        from .fire3d_recon.frozen import verify_frozen_profile
        verify_frozen_profile(config)
    from .layout import evaluation_dir
    output = resolve(config['output'])
    public = evaluation_dir(config)
    if (public/'mesh_manifest.json').exists():
        existing=load_mesh_manifest(public/'mesh_manifest.json')
        total=sum((public/row['glb_path']).stat().st_size for row in existing['meshes'])
        if total>=20_000_000:
            raise FileExistsError('Existing export exceeds the current scene budget. Use a new algorithm/run name; '
                                  'completed exports and their measured reports are immutable.')
        # Preserve an existing completed export; never replace its GLBs beneath
        # old metric reports on a resumed run. Changed settings need a new run.
        return existing
    if config.get('frozen_profile'):
        profile = resolve(config['frozen_profile'])
        (public / 'reports').mkdir(parents=True, exist_ok=True)
        shutil.copy2(profile, public / 'reports/inference_profile.json')
        shutil.copy2(profile.with_suffix('.sha256'), public / 'reports/inference_profile.sha256')
    source = resolve(config['source_run'])
    objects, skipped = select_conditioned(read_json(output / 'conditioning.json'), config)
    upstream_skips = read_json(source / 'objects.json').get('skipped_objects', [])
    skipped_by_id = {item['object_id']: dict(item, stage='fusion') for item in upstream_skips}
    skipped_by_id.update({item['object_id']: item for item in skipped})
    skipped = [skipped_by_id[key] for key in sorted(skipped_by_id)]
    entries, blobs = [], []
    for obj in sorted(objects, key=lambda item: item['object_id']):
        original = output / 'objects' / obj['name'] / 'mesh_textured_world.glb'
        if not original.exists():
            raise ValueError(f'Missing supported instance texture mesh: {original}')
        mesh = trimesh.load(original, process=False, force='mesh')
        if mesh.visual.kind != 'texture' or (mesh.visual.material.baseColorTexture is None or mesh.visual.material.metallicRoughnessTexture is None):
            raise ValueError(f'Missing embedded inferred PBR texture: {obj["name"]}')
        bounds = mesh.bounds.copy()
        mesh.apply_transform(GLTF_FROM_DATASET)
        relative = f'meshes/obj_{obj["object_id"]:03d}.glb'
        target = public / relative
        target.parent.mkdir(exist_ok=True, parents=True)
        blob = mesh.export(file_type='glb')
        blobs.append(blob)
        entries.append({'instance_id': int(obj['object_id']), 'category': obj['family'],
                        'glb_path': relative, 'sha256': '',
                        'bbox_dataset_world_m': bounds.tolist(),
                        'vertices': len(mesh.vertices), 'faces': len(mesh.faces)})
    from .glb_budget import fit_scene_glbs
    blobs, budget = fit_scene_glbs(blobs)
    for entry, blob in zip(entries, blobs):
        target = public / entry['glb_path']
        temporary = target.with_suffix('.tmp')
        temporary.write_bytes(blob); temporary.replace(target)
        entry['sha256'] = sha256(target)
    write_json(public / 'reports/export_budget.json', budget)
    inputs = public / 'inputs';inputs.mkdir(exist_ok=True)
    shutil.copy2(source / 'input/cameras.npz', inputs / 'cameras.npz')
    frames = read_json(source / 'input/frames.json')
    write_json(inputs / 'source.json', {'source_scene': frames['source_scene'],
               'fusion_run': str(source), 'frame_indices': [frame['frame_index'] for frame in frames['frames']],
               'input_contract': 'RGB and given calibration; reconstructed meshes use predicted RGBD/masks'})
    manifest = {'schema': SCHEMA, 'scene_id': '_'.join(Path(frames['source_scene']).parts[-2:]),
                'units': 'meters', 'mesh_coordinate_frame': 'gltf_world_y_up_right_handed',
                'dataset_coordinate_frame': 'blender_world_z_up_right_handed',
                'mesh_to_dataset_world': DATASET_FROM_GLTF.tolist(),
                'placement': 'World placement is baked into GLB vertices; no object pose correction',
                'appearance': 'Embedded inferred PBR base-color and metallic/roughness textures',
                'camera_path': 'inputs/cameras.npz', 'source_path': 'inputs/source.json',
                'meshes': entries, 'skipped_instances': skipped,
                'render': {'rgb_directory': 'render/rgb', 'video_path': 'render/reconstruction.mp4',
                           'comparison_path': 'render/comparison.mp4', 'environment_path': 'render/environment.blend'},
                'frozen_profile': config.get('frozen_profile')}
    write_json(public / 'mesh_manifest.json', manifest)
    load_mesh_manifest(public / 'mesh_manifest.json')
    return manifest


def load_mesh_manifest(path, verify_files=True):
    path = Path(path).resolve()
    data = read_json(path)
    required = {'schema', 'scene_id', 'units', 'mesh_coordinate_frame', 'dataset_coordinate_frame',
                'mesh_to_dataset_world', 'placement', 'appearance', 'camera_path', 'source_path',
                'meshes', 'skipped_instances', 'render', 'frozen_profile'}
    if set(data) != required:
        raise ValueError(f'Public v1 manifest fields differ: {sorted(set(data) ^ required)}')
    if data.get('schema') != SCHEMA or data.get('units') != 'meters':
        raise ValueError('Unsupported textured mesh schema or units')
    if data.get('dataset_coordinate_frame') != 'blender_world_z_up_right_handed':
        raise ValueError('Dataset coordinate frame must be Blender world Z-up')
    if data.get('mesh_coordinate_frame') != 'gltf_world_y_up_right_handed':
        raise ValueError('GLB coordinate frame must be explicit glTF world Y-up')
    np.testing.assert_array_equal(data['mesh_to_dataset_world'], DATASET_FROM_GLTF)
    if data['source_path'] != 'inputs/source.json' or not (path.parent / data['source_path']).is_file():
        raise ValueError('Missing standardized source metadata')
    if data['render'] != {'rgb_directory': 'render/rgb', 'video_path': 'render/reconstruction.mp4',
                          'comparison_path': 'render/comparison.mp4', 'environment_path': 'render/environment.blend'}:
        raise ValueError('Public render paths must follow the fixed v1 layout')
    ids = []
    for entry in data['meshes']:
        if set(entry) != {'instance_id', 'category', 'glb_path', 'sha256', 'bbox_dataset_world_m', 'vertices', 'faces'}:
            raise ValueError('Mesh entry fields differ from v1 contract')
        identity = entry['instance_id'];ids.append(identity)
        if type(identity) is not int or identity <= 0 or entry['glb_path'] != f'meshes/obj_{identity:03d}.glb':
            raise ValueError('Mesh names must match positive predicted instance IDs')
        if entry['vertices'] < 3 or entry['faces'] < 1:
            raise ValueError('Empty geometry metadata')
        if not isinstance(entry.get('category'), str) or not entry['category']:
            raise ValueError('Missing predicted instance category')
        bounds = np.asarray(entry['bbox_dataset_world_m'])
        if bounds.shape != (2, 3) or not np.isfinite(bounds).all() or np.any(bounds[1] < bounds[0]):
            raise ValueError('Invalid world-space mesh bounds')
        mesh_path = path.parent / entry['glb_path']
        if verify_files and (not mesh_path.is_file() or sha256(mesh_path) != entry['sha256']):
            raise ValueError(f'Mesh file/hash mismatch: {mesh_path}')
    if not ids or ids != sorted(set(ids)):
        raise ValueError('Mesh list must be nonempty, sorted and have unique instance IDs')
    if data.get('camera_path') != 'inputs/cameras.npz' or not (path.parent / data['camera_path']).is_file():
        raise ValueError('Missing standardized camera artifact')
    return data


def load_textured_meshes(path, dataset_world=True):
    """Return [(instance metadata, Trimesh), ...] for unified evaluation."""
    import trimesh
    from io import BytesIO
    path = Path(path).resolve()
    manifest = load_mesh_manifest(path)
    result = []
    for entry in manifest['meshes']:
        blob = (path.parent / entry['glb_path']).read_bytes()
        mesh = trimesh.load(BytesIO(for_trimesh(blob)), file_type='glb', process=False, force='mesh')
        if mesh.visual.kind != 'texture' or (mesh.visual.material.baseColorTexture is None or mesh.visual.material.metallicRoughnessTexture is None):
            raise ValueError(f'GLB has no embedded PBR appearance: {entry["glb_path"]}')
        if dataset_world:
            mesh.apply_transform(DATASET_FROM_GLTF)
            np.testing.assert_allclose(mesh.bounds, entry['bbox_dataset_world_m'], atol=2e-6)
        result.append((entry, mesh))
    return result
