"""Attach DINOv3/AnyUp features from RGB to measured instance point clouds.

No dataset GT depths, instances, boxes, object poses, or asset meshes are read.
"""
import sys
import time
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
from ..pi3x_fusion.common import ROOT, read_json, write_json, resolve, sha256


def canonical_transform(bounds, margin=1.1, minimum_extent=.01):
    bounds = np.asarray(bounds, np.float64)
    center = bounds.mean(0)
    extent = np.maximum(bounds[1] - bounds[0], minimum_extent) * margin
    scale = float(extent.max())
    local_to_world = np.eye(4)
    local_to_world[:3, :3] = [[scale, 0, 0], [0, 0, -scale], [0, scale, 0]]
    local_to_world[:3, 3] = center
    return local_to_world, np.linalg.inv(local_to_world)


def project_visible(xyz, object_id, data, labels, tolerance):
    inverse = np.linalg.inv(data['camera_to_world_cv'])
    camera = xyz @ inverse[:3, :3].T + inverse[:3, 3]
    K = data['K']
    z = np.maximum(camera[:, 2], .001)
    uv = camera[:, :2] / z[:, None] * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]
    pixels = np.rint(uv).astype(int)
    h, w = labels.shape
    inside = (camera[:, 2] > .01) & (pixels[:, 0] >= 0) & (pixels[:, 0] < w) & (pixels[:, 1] >= 0) & (pixels[:, 1] < h)
    indices = np.flatnonzero(inside)
    x, y = pixels[inside].T
    delta = np.abs(camera[inside, 2] - data['depth'][y, x])
    valid = data['valid'][y, x] & (labels[y, x] == object_id) & (delta < tolerance)
    indices = indices[valid]
    weights = data['confidence'][y[valid], x[valid]].astype(np.float32)**2 * (1 - delta[valid] / tolerance)
    return indices, uv[indices], weights


def prepare(config):
    if config.get('inference_protocol'):
        from .selective_conditioning import prepare as selective_prepare
        return selective_prepare(config)
    from ..resources import gpu_lease
    with gpu_lease('inference:legacy_conditioning'):
        return prepare_dense(config)


def prepare_dense(config):
    import torch
    import torch.nn.functional as F
    source, output = resolve(config['source_run']), resolve(config['output'])
    repo = resolve(config['fire3d_repo'])
    sys.path.insert(0, str(repo))
    if config.get('inference_protocol'):
        from .protocol import load_protocol
        load_protocol(config)
    manifest = read_json(source / 'input/frames.json')
    objects = read_json(source / 'objects.json')['objects']
    from .policy import select_objects
    objects, skipped = select_objects(objects, config)
    write_json(output / 'skipped_objects.json', {'objects': skipped, 'minimum_fused_points': config.get('minimum_fused_points', 100)})
    frames = manifest['frames']
    selected = set(range(len(frames))) if not config['conditioning_views'] else set(
        np.linspace(0, len(frames), config['conditioning_views'], endpoint=False, dtype=int).tolist())
    # Include a few observations of short tracks so low-support instances remain inspectable.
    for obj in objects:
        if obj['low_support']:
            selected.update(obj['observed_frames'][::max(1, len(obj['observed_frames']) // 3)])
    selected = sorted(selected)
    payload = []
    gpu_accumulation = bool(config.get('inference_protocol'))
    for obj in objects:
        data = np.load(source / 'objects' / obj['name'] / 'points.npz')
        xyz, rgb = data['xyz'], data['rgb']
        local_to_world, world_to_local = canonical_transform(obj['bbox_world'], config['bbox_margin'], config['minimum_bbox_extent_m'])
        local = xyz @ world_to_local[:3, :3].T + world_to_local[:3, 3]
        # In release mode pass the complete supplied cloud; the model's
        # default unit-box filter decides which points enter conditioning.
        keep = np.ones(len(xyz), bool) if config.get('inference_protocol') else np.all(np.abs(local) <= .5, axis=1)
        candidates = np.flatnonzero(keep)
        if not len(candidates):
            raise ValueError(f"Empty conditioning after measured bbox filtering: {obj['name']}")
        rng = np.random.default_rng(config['seed'] + obj['object_id'])
        indices = candidates if not config['conditioning_points'] else rng.choice(
            candidates, min(len(candidates), config['conditioning_points']), replace=False)
        payload.append({'object': obj, 'xyz': xyz[indices], 'rgb': rgb[indices],
                        'local_to_world': local_to_world, 'world_to_local': world_to_local,
                        'feature_sum': np.zeros((len(indices), 1024), np.float32),
                        'weight_sum': np.zeros(len(indices), np.float32),
                        'observations': np.zeros(len(indices), np.uint16), 'source_indices': indices})
        if config.get('inference_protocol'):
            payload[-1]['pbr_feature_sum'] = np.zeros((len(indices), 1024), np.float32)
        if gpu_accumulation:
            for key in ['feature_sum', 'pbr_feature_sum']:
                payload[-1][key] = torch.zeros((len(indices), 1024), device='cuda', dtype=torch.float32)
    started = time.perf_counter()
    torch.set_num_threads(4)
    torch.cuda.reset_peak_memory_stats()
    from .image_features import Encoder
    encoder = Encoder(config)
    factor = config['dino_upsample']
    use_natten = config.get('anyup_use_natten', False)
    rgb_mlp = encoder.rgb_mlp
    with torch.inference_mode():
        for position in selected:
            frame = frames[position]
            index = frame['frame_index']
            depth = np.load(source / 'rgbd' / f'frame_{index:04d}.npz')
            labels = np.asarray(Image.open(source / 'instances' / f'frame_{index:04d}.png'))
            rgb = np.asarray(Image.open(frame['rgb_path']).convert('RGB'))
            grid, pbr_grid = encoder(rgb)
            for item in payload:
                indices, uv, weights = project_visible(item['xyz'], item['object']['object_id'], depth, labels, config['depth_tolerance_m'])
                if not len(indices):
                    continue
                # Calibrated Pi3X image -> original RGB normalized pixel coordinates.
                hh, ww = labels.shape
                normalized = 2 * (uv + .5) / [ww, hh] - 1
                # Stream the same independent interpolation queries in small
                # batches; all points and views still participate unchanged.
                for offset in range(0, len(indices), 16384):
                    chosen = indices[offset:offset+16384]
                    query = torch.tensor(normalized[offset:offset+16384], device='cuda', dtype=torch.float32)[None, None]
                    weight = weights[offset:offset+16384, None]
                    samples = F.grid_sample(grid, query, mode='bilinear', padding_mode='border', align_corners=False)[0, :, 0].T.contiguous()
                    if gpu_accumulation:
                        rows = torch.tensor(chosen, device='cuda', dtype=torch.long)
                        precise_weight = torch.tensor(weight, device='cuda', dtype=torch.float64)
                        # Preserve NumPy's original FP64 product/add, then
                        # FP32 storage, exactly; only data movement changes.
                        from .accumulation import accumulate
                        accumulate(item['feature_sum'],samples,rows,precise_weight)
                    else:
                        item['feature_sum'][chosen] += samples.cpu().numpy() * weight
                    if rgb_mlp is not None:
                        pbr_samples = F.grid_sample(pbr_grid, query, mode='bilinear', padding_mode='border', align_corners=False)[0, :, 0].T.contiguous()
                        if gpu_accumulation:
                            accumulate(item['pbr_feature_sum'],pbr_samples,rows,precise_weight)
                        else:
                            item['pbr_feature_sum'][chosen] += pbr_samples.cpu().numpy() * weight
                        del pbr_samples
                    del samples, query
                item['weight_sum'][indices] += weights
                item['observations'][indices] += 1
            print(f'DINO/AnyUp features frame {index:04d} ({selected.index(position)+1}/{len(selected)})', flush=True)
    summary = []
    for item in payload:
        obj = item['object']
        if gpu_accumulation:
            for key in ['feature_sum', 'pbr_feature_sum']:
                item[key] = item[key].cpu().numpy()
        valid = item['weight_sum'] > 0
        if not valid.any():
            raise ValueError(f"No RGB/depth-consistent DINO conditioning for {obj['name']}")
        features = item['feature_sum'] / np.maximum(item['weight_sum'][:, None], 1e-8)
        missing = ~valid
        if missing.any():
            nearest = cKDTree(item['xyz'][valid]).query(item['xyz'][missing], k=1)[1]
            features[missing] = features[np.flatnonzero(valid)[nearest]]
        directory = output / 'objects' / obj['name']
        directory.mkdir(parents=True, exist_ok=True)
        extra = {}
        if rgb_mlp is not None:
            pbr_features = item['pbr_feature_sum'] / np.maximum(item['weight_sum'][:, None], 1e-8)
            if missing.any():
                pbr_features[missing] = pbr_features[np.flatnonzero(valid)[nearest]]
            extra['pbr_features'] = pbr_features
        feature_dtype = np.float32 if config.get('inference_protocol') else np.float16
        from .storage import save_conditioning
        save_conditioning(directory / 'conditioning.npz', xyz=item['xyz'], rgb=item['rgb'], features=features.astype(feature_dtype, copy=False),
                            observations=item['observations'], feature_imputed=missing, source_indices=item['source_indices'],
                            local_to_world=item['local_to_world'], world_to_local=item['world_to_local'], **extra)
        report = {'object_id': obj['object_id'], 'name': obj['name'], 'family': obj['family'], 'low_support': obj['low_support'],
                  'source_points': obj['points'], 'conditioning_points': len(item['xyz']), 'direct_feature_points': int(valid.sum()),
                  'imputed_feature_points': int(missing.sum()), 'mean_observations': float(item['observations'].mean()),
                  'source_pointcloud_sha256': sha256(source / 'objects' / obj['name'] / 'points.npz'),
                  'canonical_normalization': 'uniform_max',
                  'uniform_scale_m': float(np.linalg.norm(item['local_to_world'][:3, 0])),
                  'local_to_world': item['local_to_world'].tolist(), 'world_to_local': item['world_to_local'].tolist()}
        write_json(directory / 'conditioning.json', report)
        summary.append(report)
    write_json(output / 'conditioning.json', {'objects': summary, 'skipped_objects': skipped,
        'feature_views': [frames[p]['frame_index'] for p in selected],
        'features': 'DINOv3 ViT-L/16 + AnyUp multi-backbone; confidence-weighted visible view fusion',
        'canonical_normalization': 'uniform_max',
        'dino_upsample': factor, 'anyup_use_natten': use_natten, 'perception_used': False, 'GT_used': False,
        'storage': 'Lossless ZIP_STORED NumPy archive; unchanged FP32 values, mmap-readable features',
        'seconds': time.perf_counter()-started, 'gpu_peak_GiB': torch.cuda.max_memory_allocated()/2**30})
    print('Conditioning complete', flush=True)
