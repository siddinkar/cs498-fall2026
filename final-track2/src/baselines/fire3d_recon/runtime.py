"""Memory-mapped, staged Fire3D SS/shape inference; no perception import."""
import gc
import os
import sys
import time
from contextlib import contextmanager
import numpy as np
from ..pi3x_fusion.common import resolve, read_json, write_json

_SCENE_INPUTS = {}


def release_scene_inputs():
    _SCENE_INPUTS.clear()
    cleanup()


def scene_inputs(config, feature_key):
    """Match the official common_tensors contract: all scene objects/points.

    Only selected_indices changes with batch size. Sampling all scene objects
    preserves the release RNG consumption and conditioning order.
    """
    import torch
    key = (str(resolve(config['output'])), feature_key)
    if _SCENE_INPUTS.get('key') == key:
        return _SCENE_INPUTS['value']
    release_scene_inputs()
    from .policy import select_objects
    output = resolve(config['output'])
    objects, _ = select_objects(read_json(output / 'conditioning.json')['objects'], config)
    total = sum(o['conditioning_points'] for o in objects)
    points = torch.empty((1, total, 3), device='cuda', dtype=torch.float32)
    # Keep all FP32 feature values on the host. Native CUDA filtering/sampling
    # operates on row tags; only its exact selected rows move to the GPU.
    features = torch.empty((1, total, 1024), device='cpu', dtype=torch.float32)
    ids = torch.empty((1, total), device='cuda', dtype=torch.long)
    transforms, names, start = [], [], 0
    from .storage import load_feature
    for position, obj in enumerate(objects):
        data = np.load(output / 'objects' / obj['name'] / 'conditioning.npz')
        n = len(data['xyz'])
        points[0, start:start+n].copy_(torch.from_numpy(data['xyz']))
        features[0, start:start+n].copy_(torch.from_numpy(load_feature(output / 'objects' / obj['name'] / 'conditioning.npz',feature_key)))
        ids[0, start:start+n] = position
        transforms.append(data['world_to_local'])
        names.append(obj['name'])
        start += n
        data.close()
    value = {'points': points, 'colors': features, 'instance_ids': ids,
             'object_transforms': torch.tensor(np.asarray(transforms), device='cuda', dtype=torch.float32)[None],
             'max_num_objects': len(objects), '_names': names}
    _SCENE_INPUTS.update(key=key, value=value)
    return value


def bootstrap(config):
    if config.get('inference_protocol'):
        # Keep upstream module defaults: dense SDPA, sparse FlashAttention,
        # FlexGEMM convolution. Do not inherit the previous trial's overrides.
        for key in ['ATTN_BACKEND', 'SPARSE_ATTN_BACKEND', 'SPARSE_CONV_BACKEND']:
            os.environ.pop(key, None)
        from .protocol import load_protocol
        load_protocol(config)
    else:
        os.environ.setdefault('ATTN_BACKEND', 'sdpa')
        os.environ.setdefault('SPARSE_ATTN_BACKEND', 'xformers')
        os.environ.setdefault('SPARSE_CONV_BACKEND', 'flex_gemm')
    repo = resolve(config['fire3d_repo'])
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / 'trellis2_x2'))
    import torch
    torch.set_num_threads(4)
    from .memory import install
    install()


def cleanup():
    import torch
    gc.collect()
    torch.cuda.empty_cache()


def set_seed(seed):
    import random
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@contextmanager
def measure(directory, stage):
    import torch
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    try:
        yield
    finally:
        torch.cuda.synchronize()
        write_json(directory / f'{stage}_metrics.json', {'seconds': time.perf_counter()-start,
            'peak_gpu_allocated_GiB': torch.cuda.max_memory_allocated()/2**30,
            'peak_gpu_reserved_GiB': torch.cuda.max_memory_reserved()/2**30})


def load_flow(config, kind):
    import torch
    import yaml
    if kind == 'ss':
        from models.object_gen_ss_x2_offline_with_dino import ObjectGen
    elif kind == 'shape':
        from models.object_gen_shape_x2_offline_with_dino import ObjectGen
    elif kind == 'pbr':
        from models.object_gen_pbr_x2_offline_with_dino import ObjectGen
    else:
        raise ValueError(kind)
    model_root = resolve(config['model_root'])
    directory = model_root / 'reconstruction/flows' / kind
    options = yaml.safe_load((directory / 'config.yaml').read_text())['model']
    options['dino'] = None  # Features already computed from our RGBs, separately.
    options['max_cond_len'] = config['max_cond_len']
    for key in (['ss_x2'] if kind == 'ss' else ['shape_x2'] if kind == 'shape' else ['shape_x2', 'pbr_x2']):
        field = 'latent_stats_path' if key == 'ss_x2' else 'x2_latent_stats_path'
        options[key][field] = str(model_root / 'reconstruction/stats' / f"{key.split('_')[0]}.json")
    print(f'Loading {kind} flow (mmap, without DINO/perception)', flush=True)
    checkpoint = torch.load(directory / 'model.pt', mmap=True, map_location='cpu', weights_only=False)
    key = 'ema_model' if config['use_ema'] else 'model'
    state = checkpoint[key]
    if 'shadow' in state:
        state = state['shadow']
    state = {k: v for k, v in state.items() if not k.startswith('dino_model.')}
    from ..loading import empty_parameters,assign_weights
    with empty_parameters():model=ObjectGen(options)
    assign_weights(model,state)
    del state, checkpoint
    cleanup()
    model = model.eval().requires_grad_(False).cuda()
    print(f'{kind} flow loaded; GPU allocated {torch.cuda.memory_allocated()/2**30:.2f} GiB', flush=True)
    return model


def load_vae(config, kind):
    import torch
    from trellis2 import models
    root = resolve(config['model_root']) / 'reconstruction/vae' / kind
    options = read_json(root / 'config.json')
    decoder_options = options['models']['decoder']
    from ..loading import empty_parameters,assign_weights
    with empty_parameters():decoder=getattr(models,decoder_options['name'])(**decoder_options['args'])
    state = torch.load(root / 'ckpts/decoder.pt', mmap=True, map_location='cpu', weights_only=True)
    assign_weights(decoder,state)
    del state
    cleanup()
    return decoder.eval().requires_grad_(False).cuda(), options


def common(directory, config=None, feature_key='features', flow_kind=None):
    import torch
    if config and config.get('inference_protocol'):
        manifest = read_json(resolve(config['output'])/'conditioning.json')
        from .preselection import SCHEMA
        if manifest.get('feature_storage') == SCHEMA:
            from .preselection import common as preselected_common
            shared = preselected_common(config, feature_key, directory.name, flow_kind)
        else:
            shared = scene_inputs(config, feature_key)
        values = {k: v for k, v in shared.items() if k != '_names'}
        values.update(selected_indices=torch.tensor([[shared['_names'].index(directory.name)]], device='cuda', dtype=torch.long),
            return_denoised_latents=True, sample_method=config['sample_method'], prune_out_of_range=True, inference_only=True)
        return values
    data = np.load(directory / 'conditioning.npz')
    p = torch.tensor(data['xyz'], device='cuda', dtype=torch.float32)[None]
    features = torch.tensor(data['features'], device='cuda', dtype=torch.float32)[None]
    transforms = torch.tensor(data['world_to_local'], device='cuda', dtype=torch.float32)[None, None]
    return {'points': p, 'colors': features, 'instance_ids': torch.zeros(p.shape[:2], device='cuda', dtype=torch.long),
            'object_transforms': transforms, 'selected_indices': torch.zeros((1, 1), device='cuda', dtype=torch.long),
            # Sparse conditioning can be smaller than the CUDA FPS kernel's
            # minimum supported input; use sampling that also handles it.
            # Random sampling below max_cond_len retains every real point once.
            'max_num_objects': 1, 'return_denoised_latents': True,
            'sample_method': (config or {}).get('sample_method', 'fps') if len(data['xyz']) >= 32 else 'random',
            'prune_out_of_range': True, 'inference_only': True}


def reconstruct(config, object_ids=None):
    import torch
    bootstrap(config)
    output = resolve(config['output'])
    from .policy import select_conditioned
    objects, skipped = select_conditioned(read_json(output / 'conditioning.json'), config)
    write_json(output / 'skipped_objects.json', {'objects': skipped, 'minimum_fused_points': config.get('minimum_fused_points', 100)})
    if object_ids:
        objects = [o for o in objects if o['object_id'] in object_ids]
    directories = [output / 'objects' / o['name'] for o in objects]
    if not directories:
        raise ValueError('No matching objects')
    with torch.inference_mode():
        pending = [d for d in directories if not (d / 'ss_latents.npz').exists()]
        if pending:
            model = load_flow(config, 'ss')
            for directory in pending:
                with measure(directory, 'ss_flow'):
                    if config.get('inference_protocol'):
                        from .protocol import object_seed
                        seed = object_seed(config, directory.name, 'geometry')
                    else:
                        seed = config['seed'] + read_json(directory / 'conditioning.json')['object_id']
                    set_seed(seed)
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        result = model(**common(directory, config, flow_kind='ss'), object_feats=torch.zeros((1, 8, 2, 2, 2), device='cuda'),
                                       object_coords=torch.empty((0, 4), device='cuda', dtype=torch.int32),
                                       inference_num_steps=config['steps'])
                    np.savez_compressed(directory / 'ss_latents.npz', latents=result['feat_preds'].float().cpu().numpy())
                    del result
                print(f"{directory.name}: SS flow complete", flush=True)
            del model
            cleanup()
        pending = [d for d in directories if not (d / 'ss_occupancy.npz').exists()]
        if pending:
            decoder, _ = load_vae(config, 'ss')
            for directory in pending:
                with measure(directory, 'ss_decode'):
                    latents = torch.tensor(np.load(directory / 'ss_latents.npz')['latents'], device='cuda')
                    with torch.autocast('cuda', dtype=torch.float16):
                        logits = decoder(latents).float()
                    probabilities = logits.sigmoid()
                    active = probabilities[:, 0] >= config['occupancy_threshold']
                    fallback = False
                    if not active.any():
                        active.flatten()[probabilities.flatten().argmax()] = True
                        fallback = True
                    coords = active.nonzero().to(torch.int32)
                    np.savez_compressed(directory / 'ss_occupancy.npz', coords=coords.cpu().numpy(),
                        probabilities=probabilities.cpu().numpy(), argmax_fallback=fallback)
                    del latents, logits, probabilities, active, coords
                print(f"{directory.name}: SS occupancy complete", flush=True)
            del decoder
            cleanup()
        pending = [d for d in directories if not (d / 'shape_latents.npz').exists()]
        if pending:
            model = load_flow(config, 'shape')
            for directory in pending:
                with measure(directory, 'shape_flow'):
                    if config.get('inference_protocol'):
                        from .protocol import object_seed
                        seed = object_seed(config, directory.name, 'geometry')
                    else:
                        seed = config['seed'] + read_json(directory / 'conditioning.json')['object_id']
                    set_seed(seed)
                    coords = torch.tensor(np.load(directory / 'ss_occupancy.npz')['coords'], device='cuda', dtype=torch.int32).contiguous()
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        result = model(**common(directory, config, flow_kind='shape'), object_feats=torch.zeros((len(coords), 64), device='cuda'),
                                       object_coords=coords, inference_num_steps=config['steps'])
                    raw = result['feat_preds'].float() * model.x2_std.float() + model.x2_mean.float()
                    np.savez_compressed(directory / 'shape_latents.npz', coords=coords.cpu().numpy(), raw_features=raw.cpu().numpy())
                    del coords, result, raw
                print(f"{directory.name}: shape flow complete", flush=True)
            del model
            release_scene_inputs()
            cleanup()
        from trellis2.modules.sparse import SparseTensor
        pending = [d for d in directories if not (d / 'sc_latents.npz').exists()]
        if pending:
            decoder, options = load_vae(config, 'shape')
            normalization = options['dataset']['args']['normalization']
            mean = torch.tensor(normalization['mean'], device='cuda', dtype=torch.float32)[None]
            std = torch.tensor(normalization['std'], device='cuda', dtype=torch.float32)[None]
            for directory in pending:
                with measure(directory, 'shape_hc_decode'):
                    data = np.load(directory / 'shape_latents.npz')
                    sparse = SparseTensor(feats=torch.tensor(data['raw_features'], device='cuda').contiguous(),
                                          coords=torch.tensor(data['coords'], device='cuda', dtype=torch.int32).contiguous())
                    # Native decoder FP16 torso + FP32 heads/norms. Autocast
                    # defeats upstream manual_cast and produces FP32/F16
                    # operand mismatches in FlexGEMM's Triton kernels.
                    with torch.autocast('cuda', enabled=False):
                        normalized = decoder(sparse)
                    raw = normalized.feats.float() * std + mean
                    np.savez_compressed(directory / 'sc_latents.npz', coords=normalized.coords.int().cpu().numpy(), raw_features=raw.cpu().numpy())
                    del sparse, normalized, raw
                print(f"{directory.name}: shape HC decode complete", flush=True)
            del decoder, mean, std
            cleanup()
        pending = [d for d in directories if not (d / 'mesh_world.glb').exists()]
        if pending:
            from trellis2 import models
            from safetensors.torch import load_file
            path = resolve(config['model_root']) / 'external/trellis2/shape_dec_next_dc_f16c32_fp16'
            options = read_json(str(path) + '.json')
            from ..loading import empty_parameters,assign_weights
            with empty_parameters():decoder=getattr(models,options['name'])(**options['args'])
            state = load_file(str(path) + '.safetensors')
            assign_weights(decoder,state)
            del state
            cleanup()
            decoder = decoder.eval().requires_grad_(False).cuda()
            decoder.set_resolution(config['mesh_resolution'])
            for directory in pending:
                with measure(directory, 'shape_sc_decode'):
                    data = np.load(directory / 'sc_latents.npz')
                    sparse = SparseTensor(feats=torch.tensor(data['raw_features'], device='cuda').contiguous(),
                                          coords=torch.tensor(data['coords'], device='cuda', dtype=torch.int32).contiguous())
                    with torch.autocast('cuda', enabled=False):
                        mesh = decoder(sparse)[0]
                    vertices = mesh.vertices.float().cpu().numpy()
                    faces = mesh.faces.cpu().numpy()
                    export_mesh(directory, vertices, faces)
                    del sparse, mesh
                print(f"{directory.name}: mesh exported ({len(vertices)} vertices, {len(faces)} faces)", flush=True)
            del decoder
            cleanup()


def export_mesh(directory, vertices, faces):
    import trimesh
    from scipy.spatial import cKDTree
    if not len(vertices) or not len(faces) or not np.isfinite(vertices).all():
        raise ValueError(f'Empty/nonfinite decoded mesh: {directory.name}')
    conditioning = np.load(directory / 'conditioning.npz')
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.export(directory / 'mesh_canonical.ply')
    matrix = conditioning['local_to_world']
    world = vertices @ matrix[:3, :3].T + matrix[:3, 3]
    indices = cKDTree(conditioning['xyz']).query(world, k=1)[1]
    colors = conditioning['rgb'][indices]
    world_mesh = trimesh.Trimesh(vertices=world, faces=faces, vertex_colors=colors, process=False)
    world_mesh.export(directory / 'mesh_world.ply')
    world_mesh.export(directory / 'mesh_world.glb')
    report = read_json(directory / 'conditioning.json')
    report.update({'status': 'reconstructed', 'vertices': len(vertices), 'faces': len(faces),
        'bbox_world': world_mesh.bounds.tolist(), 'watertight': bool(world_mesh.is_watertight),
        'appearance': 'Nearest source point-cloud RGB vertex colors; not generated PBR texture',
        'conditioning_sampler': 'random (all points)' if len(conditioning['xyz']) < 32 else 'fps',
        'perception_used': False, 'GT_geometry_used': False})
    write_json(directory / 'reconstruction.json', report)
