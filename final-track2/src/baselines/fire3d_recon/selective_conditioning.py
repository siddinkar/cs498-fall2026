"""Fuse full-video features only for rows consumed by native seeded FPS."""
import time
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
from ..pi3x_fusion.common import read_json, write_json, resolve, sha256
from .conditioning import canonical_transform, project_visible
from .preselection import SCHEMA, prepare_plans


def feature_sources(xyz, requested, valid):
    """Preserve dense fusion's nearest-valid-point imputation, including donors."""
    sources = requested.copy()
    missing = ~valid[requested]
    if missing.any():
        nearest = cKDTree(xyz[valid]).query(xyz[requested[missing]], k=1)[1]
        sources[missing] = np.flatnonzero(valid)[nearest]
    return sources, np.unique(sources)


def prepare(config):
    import torch
    from .policy import select_objects
    from ..resources import gpu_lease
    started = time.perf_counter()
    torch.set_num_threads(4)
    torch.cuda.reset_peak_memory_stats()
    source, output = resolve(config['source_run']), resolve(config['output'])
    frames = read_json(source/'input/frames.json')['frames']
    objects, skipped = select_objects(read_json(source/'objects.json')['objects'], config)
    write_json(output/'skipped_objects.json', {'objects': skipped, 'minimum_fused_points': config.get('minimum_fused_points', 100)})
    payload = []
    for obj in objects:
        with np.load(source/'objects'/obj['name']/'points.npz') as data:
            xyz, rgb = data['xyz'], data['rgb']
        forward, inverse = canonical_transform(obj['bbox_world'], config['bbox_margin'], config['minimum_bbox_extent_m'])
        payload.append({'object': obj, 'xyz': xyz, 'rgb': rgb, 'local_to_world': forward, 'world_to_local': inverse,
            'weight_sum': np.zeros(len(xyz), np.float32), 'observations': np.zeros(len(xyz), np.uint16)})
    with gpu_lease(f'inference:{config["input_identity"]}:fps_planning'):
        prepare_plans(config, payload)
    # Scalar visibility over every source point preserves the dense imputation
    # donor pool. No N x 1024 feature matrix is created in this pass.
    for position, frame in enumerate(frames):
        index = frame['frame_index']
        labels = np.asarray(Image.open(source/'instances'/f'frame_{index:04d}.png'))
        with np.load(source/'rgbd'/f'frame_{index:04d}.npz') as depth:
            for item in payload:
                rows, _, weights = project_visible(item['xyz'], item['object']['object_id'], depth, labels, config['depth_tolerance_m'])
                item['weight_sum'][rows] += weights
                item['observations'][rows] += 1
        if (position+1) % 32 == 0 or position == 0:
            print(f'Scalar visibility {position+1}/{len(frames)} (all source points)', flush=True)
    for item in payload:
        valid = item['weight_sum'] > 0
        if not valid.any():
            raise ValueError(f'No RGB/depth-consistent DINO conditioning for {item["object"]["name"]}')
        item['feature_sources'], item['compute_indices'] = feature_sources(item['xyz'], item['feature_indices'], valid)
    with gpu_lease(f'inference:{config["input_identity"]}:dino_fusion'):
        return fuse_and_save(config,payload,frames,skipped,started)


def fuse_and_save(config,payload,frames,skipped,started):
    import torch
    import torch.nn.functional as F
    from .image_features import Encoder
    from .storage import save_conditioning
    source, output = resolve(config['source_run']), resolve(config['output'])
    rows = sum(len(item['compute_indices']) for item in payload)
    feature_bytes = rows*1024*4*2
    # Reserve ample room for the unchanged full-image encoder. Large unions
    # spill to file-backed host accumulators instead of changing sampling.
    use_cuda = feature_bytes <= min(2*2**30, torch.cuda.mem_get_info()[0]//4)
    scratch = output/'conditioning_work'
    if not use_cuda:
        scratch.mkdir(exist_ok=True)
    for item in payload:
        shape = (len(item['compute_indices']), 1024)
        for key in ['feature_sum', 'pbr_feature_sum']:
            if use_cuda:
                item[key] = torch.zeros(shape, device='cuda', dtype=torch.float32)
            elif shape[0]:
                item[key] = np.lib.format.open_memmap(scratch/f'{item["object"]["name"]}_{key}.npy', mode='w+', dtype=np.float32, shape=shape)
                item[key][:] = 0
            else:
                item[key] = np.empty(shape, np.float32)
        item['feature_xyz'] = item['xyz'][item['compute_indices']]
    print(f'Feature accumulators: {rows:,} selected/donor rows, {feature_bytes/2**30:.3f} GiB, {"CUDA" if use_cuda else "host mmap"}', flush=True)
    encoder = Encoder(config)
    with torch.inference_mode():
        for position, frame in enumerate(frames):
            index = frame['frame_index']
            labels = np.asarray(Image.open(source/'instances'/f'frame_{index:04d}.png'))
            rgb = np.asarray(Image.open(frame['rgb_path']).convert('RGB'))
            grid, pbr_grid = encoder(rgb)
            with np.load(source/'rgbd'/f'frame_{index:04d}.npz') as depth:
                for item in payload:
                    indices, uv, weights = project_visible(item['feature_xyz'], item['object']['object_id'], depth, labels, config['depth_tolerance_m'])
                    hh, ww = labels.shape
                    normalized = 2*(uv+.5)/[ww, hh]-1
                    for offset in range(0, len(indices), 16384):
                        chosen = indices[offset:offset+16384]
                        query = torch.tensor(normalized[offset:offset+16384], device='cuda', dtype=torch.float32)[None, None]
                        weight = weights[offset:offset+16384, None]
                        if use_cuda:
                            from .accumulation import accumulate
                            target = torch.tensor(chosen, device='cuda', dtype=torch.long)
                            precise_weight = torch.tensor(weight, device='cuda', dtype=torch.float64)
                        for key, feature_grid in [('feature_sum', grid), ('pbr_feature_sum', pbr_grid)]:
                            samples = F.grid_sample(feature_grid, query, mode='bilinear', padding_mode='border', align_corners=False)[0, :, 0].T.contiguous()
                            if use_cuda:
                                accumulate(item[key], samples, target, precise_weight)
                            else:
                                item[key][chosen] += samples.cpu().numpy()*weight
                            del samples
                        del query
            del grid, pbr_grid
            print(f'DINO/AnyUp selected features frame {index:04d} ({position+1}/{len(frames)})', flush=True)
    summary = []
    for item in payload:
        obj = item['object']
        compute = item['compute_indices']
        gather = np.searchsorted(compute, item['feature_sources'])
        arrays = {}
        for key, name in [('feature_sum', 'features'), ('pbr_feature_sum', 'pbr_features')]:
            sums = item.pop(key)
            if use_cuda:
                sums = sums.cpu().numpy()
                means = sums/np.maximum(item['weight_sum'][compute, None], 1e-8)
                arrays[name] = means[gather]
                del means
            elif len(gather):
                target = np.lib.format.open_memmap(scratch/f'{obj["name"]}_{name}.npy', mode='w+',
                    dtype=np.float32, shape=(len(gather), 1024))
                for offset in range(0, len(gather), 8192):
                    stop = offset+8192
                    denominator = np.maximum(item['weight_sum'][item['feature_sources'][offset:stop], None], 1e-8)
                    target[offset:stop] = sums[gather[offset:stop]]/denominator
                target.flush(); arrays[name] = target
            else:
                arrays[name] = np.empty((0, 1024), np.float32)
            del sums
        valid = item['weight_sum'] > 0
        directory = output/'objects'/obj['name']
        directory.mkdir(parents=True, exist_ok=True)
        save_conditioning(directory/'conditioning.npz', xyz=item['xyz'], rgb=item['rgb'],
            source_indices=np.arange(len(item['xyz'])), observations=item['observations'], feature_imputed=~valid,
            feature_indices=item['feature_indices'], feature_source_indices=item['feature_sources'],
            local_to_world=item['local_to_world'], world_to_local=item['world_to_local'], **arrays)
        report = {'object_id': obj['object_id'], 'name': obj['name'], 'family': obj['family'], 'low_support': obj['low_support'],
            'source_points': obj['points'], 'conditioning_points': len(item['xyz']),
            'direct_feature_points': int(valid.sum()), 'imputed_feature_points': int((~valid).sum()),
            'stored_feature_points': len(item['feature_indices']), 'feature_accumulation_points': len(compute),
            'mean_observations': float(item['observations'].mean()),
            'source_pointcloud_sha256': sha256(source/'objects'/obj['name']/'points.npz'),
            'canonical_normalization': 'uniform_max', 'uniform_scale_m': float(np.linalg.norm(item['local_to_world'][:3, 0])),
            'local_to_world': item['local_to_world'].tolist(), 'world_to_local': item['world_to_local'].tolist()}
        write_json(directory/'conditioning.json', report)
        summary.append(report)
    del encoder
    if scratch.exists():
        import shutil
        shutil.rmtree(scratch)
    write_json(output/'conditioning.json', {'objects': summary, 'skipped_objects': skipped,
        'feature_views': [frame['frame_index'] for frame in frames],
        'features': 'DINOv3 ViT-L/16 + AnyUp multi-backbone; confidence-weighted visible view fusion',
        'canonical_normalization': 'uniform_max', 'dino_upsample': config['dino_upsample'],
        'anyup_use_natten': config.get('anyup_use_natten', False), 'perception_used': False, 'GT_used': False,
        'feature_storage': SCHEMA, 'stored_feature_points': sum(o['stored_feature_points'] for o in summary),
        'feature_accumulator_bytes': feature_bytes, 'accumulation_device': 'cuda' if use_cuda else 'host_mmap',
        'storage': 'Complete XYZ/RGB/scalar visibility; FP32 features only for exact native FPS row unions',
        'seconds': time.perf_counter()-started, 'gpu_peak_GiB': torch.cuda.max_memory_allocated()/2**30})
    print('Selective conditioning complete', flush=True)
