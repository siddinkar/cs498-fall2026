"""Scene-wide GLB budget without changing positions, topology, or placement.

Only oversized scenes are repacked. Unit-square float UVs may be represented
as normalized uint16 (maximum 1/131070 rounding error), and embedded textures
may be downsampled. This can affect appearance: evaluate the exported files,
never reuse metrics from an earlier full-resolution export.
"""
import copy
import hashlib
from io import BytesIO
import json

import numpy as np
from PIL import Image
from .mesh_output import unpack, pack, for_trimesh

LIMIT = 20_000_000


def geometry_digest(blob):
    """Fingerprint exact position/index arrays and node transforms, not UVs."""
    data, binary = unpack(blob)
    digest = hashlib.sha256(json.dumps(data.get('nodes', []), sort_keys=True).encode())
    for mesh in data['meshes']:
        for primitive in mesh['primitives']:
            for index in [primitive['attributes']['POSITION'], primitive['indices']]:
                accessor = data['accessors'][index]
                view = data['bufferViews'][accessor['bufferView']]
                digest.update(json.dumps(accessor, sort_keys=True).encode())
                digest.update(binary[view.get('byteOffset', 0):view.get('byteOffset', 0)+view['byteLength']])
    return digest.hexdigest()


def repack(blob, max_texture_size):
    """Pack UVs and texture pixels, retaining every other buffer-view byte."""
    source, binary = unpack(blob)
    data = copy.deepcopy(source)
    if data.get('extensionsRequired'):
        raise ValueError('Export a standard embedded GLB before applying the size policy')
    views = data['bufferViews']
    pieces = [binary[v.get('byteOffset', 0):v.get('byteOffset', 0)+v['byteLength']] for v in views]
    uv_indices = {p['attributes']['TEXCOORD_0'] for mesh in data['meshes'] for p in mesh['primitives']
                  if 'TEXCOORD_0' in p['attributes']}
    uv_packed = 0
    for index in uv_indices:
        accessor = data['accessors'][index]
        view_index = accessor.get('bufferView')
        if (accessor.get('componentType') != 5126 or accessor.get('type') != 'VEC2'
                or accessor.get('byteOffset', 0) or accessor.get('sparse') or view_index is None):
            continue
        view = views[view_index]
        users = [a for a in data['accessors'] if a.get('bufferView') == view_index]
        if len(users) != 1 or view.get('byteStride', 8) != 8 or len(pieces[view_index]) != accessor['count']*8:
            continue
        uv = np.frombuffer(pieces[view_index], dtype='<f4').reshape(-1, 2)
        if not np.isfinite(uv).all() or np.any(uv < 0) or np.any(uv > 1):
            continue  # Preserve tiled/out-of-range UVs exactly.
        pieces[view_index] = np.rint(uv.astype(np.float64)*65535).astype('<u2').tobytes()
        accessor.update(componentType=5123, normalized=True)
        accessor.pop('min', None); accessor.pop('max', None)
        view.pop('byteStride', None)
        uv_packed += 1
    resized = 0
    for image in data.get('images', []):
        if 'uri' in image or 'bufferView' not in image:
            raise ValueError('Images must be embedded')
        index = image['bufferView']
        with Image.open(BytesIO(pieces[index])) as original:
            decoded = original.copy()
        if max(decoded.size) <= max_texture_size:
            continue
        decoded.thumbnail((max_texture_size, max_texture_size), Image.Resampling.LANCZOS)
        encoded = BytesIO()
        decoded.save(encoded, format='PNG', optimize=True)
        pieces[index] = encoded.getvalue()
        image['mimeType'] = 'image/png'
        resized += 1
    result = pack(data, pieces)
    if geometry_digest(result) != geometry_digest(blob):
        raise AssertionError('Budget policy must preserve positions, indices and node transforms')
    return result, dict(packed_uv_accessors=uv_packed, resized_images=resized)


def fit_scene_glbs(blobs, limit=LIMIT):
    """Return deterministic, strict-budget GLBs or fail before publishing them."""
    original = sum(map(len, blobs))
    report = dict(schema='cs498.track2.export_budget.v1', limit_bytes=limit,
        original_bytes=original, exported_bytes=original, changed=False,
        geometry_and_placement_unchanged=True, requires_evaluation_of_exported_files=True)
    if original < limit:
        return blobs, report
    for texture_size in [1024, 768, 512, 384, 256, 128, 64]:
        results = [repack(blob, texture_size) for blob in blobs]
        candidate = [blob for blob, _ in results]
        size = sum(map(len, candidate))
        if size < limit:
            report.update(exported_bytes=size, changed=True, max_texture_size=texture_size,
                packed_uv_accessors=sum(item['packed_uv_accessors'] for _, item in results),
                resized_images=sum(item['resized_images'] for _, item in results),
                uv_rounding_max=1/131070,
                note='Appearance may change; re-evaluate these exact exported GLBs.')
            return candidate, report
    raise ValueError(f'Cannot fit scene GLBs below {limit} bytes without changing geometry. '
                     'Reduce mesh complexity in your automatic algorithm and export a new run; never drop objects manually.')
