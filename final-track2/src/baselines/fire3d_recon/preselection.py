"""Run native FPS on XYZ/row IDs before allocating any point features.

Each target's seed samples the complete scene. Keep the union of those exact
rows, including the other objects' contexts, so the native forward inputs and
post-sampling RNG states can be reproduced without allocating N x 1024.
"""
import importlib
import random
from types import SimpleNamespace
import numpy as np
from ..pi3x_fusion.common import read_json, write_json, resolve, sha256

SCHEMA = 'native_fps_sparse_v1'


def probe(config, kind, position, seed, points, ids, transforms):
    """Stop an unmodified model forward immediately after its native sampler."""
    import torch
    from .runtime import set_seed
    module = importlib.import_module(f'models.object_gen_{kind}_x2_offline_with_dino')
    native = module.sample_object_feats
    captured = {}

    class Complete(Exception):
        pass

    def capture(**kwargs):
        before_cpu = torch.get_rng_state().numpy().copy()
        before_cuda = torch.cuda.get_rng_state().numpy().copy()
        indices, xyz, mask = native(**kwargs)
        captured.update(global_indices=indices[..., 0].long().cpu().numpy(),
            xyz=xyz.cpu().numpy(), mask=mask.cpu().numpy(),
            candidates=torch.bincount(kwargs['instance_ids'].long(), minlength=len(transforms)).cpu().numpy(),
            rng_cpu_before=before_cpu, rng_cuda_before=before_cuda,
            rng_cpu_after=torch.get_rng_state().numpy().copy(),
            rng_cuda_after=torch.cuda.get_rng_state().numpy().copy())
        raise Complete()

    module.sample_object_feats = capture
    fake = SimpleNamespace(max_cond_len=config['max_cond_len'], profile_step_timing=False, appearance_mode='dino')
    inputs = dict(points=points, colors=torch.arange(points.shape[1], device='cuda', dtype=torch.int64)[None, :, None],
        instance_ids=ids, object_transforms=transforms[None],
        selected_indices=torch.tensor([[position]], device='cuda'), max_num_objects=len(transforms),
        sample_method=config['sample_method'], prune_out_of_range=True, inference_num_steps=config['steps'],
        return_denoised_latents=True, inference_only=True,
        object_coords=torch.empty((0, 4), device='cuda', dtype=torch.int32))
    inputs['object_shape_x2_feats' if kind == 'pbr' else 'object_feats'] = torch.empty((0, 64), device='cuda')
    set_seed(seed)
    try:
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            module.ObjectGen.forward(fake, **inputs)
    except Complete:
        pass
    finally:
        module.sample_object_feats = native
    if not captured:
        raise RuntimeError('Native forward did not reach FPS')
    return captured


def prepare_plans(config, payload):
    import torch
    from .runtime import bootstrap
    from .protocol import object_seed
    if config['sample_method'] != 'fps' or config['conditioning_points'] or config['conditioning_views']:
        raise ValueError('Early native FPS requires the frozen all-point/all-frame protocol')
    bootstrap(config)
    output = resolve(config['output'])
    folder = output/'sampling/plans'
    folder.mkdir(parents=True, exist_ok=True)
    names = [item['object']['name'] for item in payload]
    offsets = np.r_[0, np.cumsum([len(item['xyz']) for item in payload])]
    fingerprint = {'schema': SCHEMA, 'object_order': names, 'offsets': offsets.tolist(),
        'source_cloud_sha256': [sha256(resolve(config['source_run'])/'objects'/name/'points.npz') for name in names],
        'input_identity': config['input_identity'], 'seed': config['seed'], 'max_cond_len': config['max_cond_len'],
        'sample_method': config['sample_method'], 'world_to_local': [item['world_to_local'].tolist() for item in payload],
        'adapter_sha256': sha256(__file__)}
    marker = folder/'manifest.json'
    previous = read_json(marker) if marker.exists() else None
    if previous and previous['fingerprint'] == fingerprint and all((output/row['file']).is_file() and
            sha256(output/row['file']) == row['sha256'] for row in previous['plans']):
        manifest = previous
    else:
        points = torch.cat([torch.from_numpy(item['xyz']) for item in payload]).to('cuda', dtype=torch.float32)[None]
        ids = torch.repeat_interleave(torch.arange(len(payload), device='cuda'),
            torch.tensor(np.diff(offsets), device='cuda'))[None]
        transforms = torch.tensor(np.stack([item['world_to_local'] for item in payload]), device='cuda', dtype=torch.float32)
        plans = []
        python_state, numpy_state = random.getstate(), np.random.get_state()
        try:
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                for position, name in enumerate(names):
                    for stage, kind in [('geometry', 'ss'), ('pbr', 'pbr')]:
                        seed = object_seed(config, name, stage, object_names=names)
                        result = probe(config, kind, position, seed, points, ids, transforms)
                        path = folder/f'{name}_{stage}.npz'
                        np.savez_compressed(path, **result, seed=seed, target_position=position)
                        plans.append({'object': name, 'stage': stage, 'seed': seed,
                            'file': str(path.relative_to(output)), 'sha256': sha256(path)})
                        print(f'Native FPS plan {name}/{stage}: {int(result["mask"].sum())} context rows across {len(names)} objects', flush=True)
        finally:
            random.setstate(python_state); np.random.set_state(numpy_state)
        del points, ids, transforms
        manifest = {'fingerprint': fingerprint, 'plans': plans,
            'algorithm': 'Native model forward preprocessing and FPS on integer row IDs; no point/voxel pre-subsampling',
            'rng': 'Save states immediately before/after native FPS; restore exact post-state in inference'}
        write_json(marker, manifest)
    unions = [[] for _ in payload]
    for row in manifest['plans']:
        with np.load(output/row['file']) as plan:
            for position in range(len(payload)):
                indices = plan['global_indices'][position, plan['mask'][position]]-offsets[position]
                if np.any((indices < 0) | (indices >= offsets[position+1]-offsets[position])):
                    raise ValueError('Native FPS returned rows outside their object')
                unions[position].append(indices)
    for item, parts in zip(payload, unions):
        item['feature_indices'] = np.unique(np.concatenate(parts))
    return manifest


def common(config, feature_key, object_name, flow_kind):
    """Materialize the exact N_objects x 1024 context, never full point features."""
    import torch
    from .memory import CachedSamples
    from .storage import load_feature
    output = resolve(config['output'])
    manifest = read_json(output/'conditioning.json')
    objects = manifest['objects']
    names = [obj['name'] for obj in objects]
    stage = 'pbr' if feature_key == 'pbr_features' else 'geometry'
    plans = read_json(output/'sampling/plans/manifest.json')
    record = next(row for row in plans['plans'] if row['object'] == object_name and row['stage'] == stage)
    path = output/record['file']
    if sha256(path) != record['sha256']:
        raise ValueError('Native FPS plan checksum changed')
    with np.load(path) as data:
        plan = {key: data[key] for key in data.files}
    offsets = np.asarray(plans['fingerprint']['offsets'])
    values = np.zeros((*plan['mask'].shape, 1024), np.float32)
    transforms = []
    for position, obj in enumerate(objects):
        archive = output/'objects'/obj['name']/'conditioning.npz'
        with np.load(archive) as data:
            rows = data['feature_indices']
            transforms.append(data['world_to_local'])
        mask = plan['mask'][position]
        local = plan['global_indices'][position, mask]-offsets[position]
        locations = np.searchsorted(rows, local)
        if np.any(locations >= len(rows)) or not np.array_equal(rows[locations], local):
            raise ValueError('Missing preselected point feature')
        values[position, mask] = load_feature(archive, feature_key)[locations]

    def consumed():
        path = output/'sampling/plans/runtime.json'
        audit = read_json(path) if path.exists() else {'passed': True, 'calls': []}
        audit['calls'].append({'object': object_name, 'flow': flow_kind, 'plan_sha256': record['sha256'],
            'native_pre_rng_matched': True, 'native_post_rng_restored': True,
            'context_shape': list(values.shape)})
        write_json(path, audit)

    context = CachedSamples(torch.from_numpy(values).cuda(), torch.from_numpy(plan['xyz']).cuda(),
        torch.from_numpy(plan['mask']).cuda(),
        *[torch.from_numpy(plan[key]) for key in ['rng_cpu_before', 'rng_cuda_before', 'rng_cpu_after', 'rng_cuda_after']],
        on_use=consumed)
    return {'points': torch.empty((1, 0, 3), device='cuda'), 'colors': context,
        'instance_ids': torch.empty((1, 0), device='cuda', dtype=torch.long),
        'object_transforms': torch.tensor(np.stack(transforms), device='cuda', dtype=torch.float32)[None],
        'max_num_objects': len(objects), '_names': names}
