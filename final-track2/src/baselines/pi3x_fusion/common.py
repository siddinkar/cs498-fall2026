"""Shared IO; stages read RGB and calibration, never GT depth/masks/meshes."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def resolve(path):
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def environment():
    e = os.environ.copy()
    e['PYTHONPATH'] = str(ROOT / 'src') + (os.pathsep + e['PYTHONPATH'] if e.get('PYTHONPATH') else '')
    e['TOKENIZERS_PARALLELISM'] = 'false'
    e['TORCH_HOME'] = str(ROOT / 'cache/torch')
    e['HF_HUB_CACHE'] = str(ROOT / 'cache/models/huggingface/hub')
    e['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    return e


def stage_args():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', required=True)
    p.add_argument('--limit', type=int, default=0)
    return p.parse_args()


def context(args):
    run = resolve(args.run_dir)
    config = read_json(run / 'config.json')
    manifest = read_json(run / 'input/frames.json')
    frames = manifest['frames'][:args.limit] if args.limit else manifest['frames']
    return run, config, manifest, frames


def read_masks(path):
    data = np.load(path)
    return np.unpackbits(data['packed'], axis=1, count=int(np.prod(data['shape']))).reshape(
        -1, *data['shape']).astype(bool)


def write_masks(path, masks):
    masks = np.asarray(masks, bool)
    np.savez_compressed(path, packed=np.packbits(masks.reshape(len(masks), int(np.prod(masks.shape[1:]))), axis=1),
                        shape=np.array(masks.shape[-2:]))


def camera_points(depth, K):
    h, w = depth.shape
    yy, xx = np.mgrid[:h, :w]
    return np.stack([(xx - K[0, 2]) / K[0, 0] * depth,
                     (yy - K[1, 2]) / K[1, 1] * depth, depth], -1)


def world_points(depth, K, camera_to_world):
    local = camera_points(depth, K)
    return local @ camera_to_world[:3, :3].T + camera_to_world[:3, 3]


def write_ply(path, xyz, rgb, **fields):
    from plyfile import PlyData, PlyElement
    xyz = np.asarray(xyz, np.float32)
    rgb = np.asarray(rgb, np.uint8)
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    dtype += [(name, np.asarray(value).dtype.str) for name, value in fields.items()]
    data = np.empty(len(xyz), dtype=dtype)
    for i, name in enumerate(['x', 'y', 'z']):
        data[name] = xyz[:, i]
    for i, name in enumerate(['red', 'green', 'blue']):
        data[name] = rgb[:, i]
    for name, value in fields.items():
        data[name] = value
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(data, 'vertex')], text=False).write(path)
