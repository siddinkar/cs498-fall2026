"""Baseline inference settings and validated reconstruction overrides."""
import copy
import math
from .policy import select_objects
from ..pi3x_fusion.common import read_json, resolve, write_json, sha256

PROTOCOL = 'configs/inference/fire3d_video_v1.json'
BATCH_PATHS = {
    'flow.object_batch_size', 'flow.ss_shape_object_batch_size', 'flow.pbr_object_batch_size',
    'geometry_stage.mesh_decode_batch_size', 'appearance_decode.object_chunk_size',
    'mesh_postprocess.object_batch_size', 'mesh_postprocess.projection_raster_instance_batch_size',
    'conditioning.anyup_frame_batch_size',
}
REMESH_BAND_PATH = 'mesh_postprocess.topology_remesh_band'


def load_protocol(config):
    if config.get('frozen_profile'):
        from .frozen import verify_frozen_profile
        verify_frozen_profile(config)
    path = resolve(config['fire3d_repo']) / config.get('inference_protocol', PROTOCOL)
    original = read_json(path)
    resolved = copy.deepcopy(original)
    for path_key in BATCH_PATHS:
        section, field = path_key.split('.')
        resolved['reconstruction'][section][field] = config.get('inference_batch_size', 1)
    explicit_overrides = {}
    if 'remesh_band' in config.get('pbr', {}):
        band = config['pbr']['remesh_band']
        if isinstance(band, bool) or not isinstance(band, (float, int)) or not math.isfinite(band) or band <= 0:
            raise ValueError('remesh_band must be a finite positive number')
        resolved['reconstruction']['mesh_postprocess']['topology_remesh_band'] = float(band)
        explicit_overrides[REMESH_BAND_PATH] = float(band)
    differences = []
    for section, values in original['reconstruction'].items():
        if not isinstance(values, dict):
            assert values == resolved['reconstruction'][section]
            continue
        for field, value in values.items():
            current = resolved['reconstruction'][section][field]
            if value != current:
                path_key = f'{section}.{field}'
                assert path_key in BATCH_PATHS or path_key in explicit_overrides
                differences.append({'path': path_key, 'official': value, 'current': current})
    output = resolve(config['output'])
    audit = {'passed': True, 'reference': str(path), 'reference_sha256': sha256(path),
        'differences': differences,
        'only_batch_sizes_changed': all(row['path'] in BATCH_PATHS for row in differences),
        'explicit_nonbatch_overrides': explicit_overrides,
        'only_batch_and_explicit_remesh_band_changed': True,
        'input_adapter': 'Existing predicted/fused RGBD instance point clouds and estimated transforms; no Fire3D perception',
        'input_adapter_normalization': config.get('canonical_normalization', 'legacy_anisotropic_cached_inputs'),
        'render_contract': 'Provided camera calibration/trajectory, 192 frames, 512x512, 24 FPS'}
    r = resolved['reconstruction']
    # Numeric/algorithm settings must be derived from the immutable release protocol.
    expected = {'seed': r['flow']['seed'], 'max_cond_len': r['flow']['max_cond_len'],
        'steps': r['flow']['inference_num_steps'], 'sample_method': r['flow']['sample_method'],
        'use_ema': r['flow']['ss']['use_ema'], 'occupancy_threshold': r['occupancy']['threshold'],
        'dino_upsample': r['conditioning']['dino_upsample'], 'mesh_resolution': r['geometry_stage']['shape_decode_resolution']}
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f'Non-batch override forbidden: {key}={config.get(key)!r}, release={value!r}')
    p = r['mesh_postprocess']
    for key, value in {'texture_size': p['texture_size'], 'mesh_faces': p['decimation_target'],
                       'remesh': p['topology_remesh'], 'guidance_strength': r['flow']['pbr_guidance_strength']}.items():
        if config['pbr'].get(key) != value:
            raise ValueError(f'Non-batch PBR override forbidden: {key}')
    if config['conditioning_points'] != 0 or config['conditioning_views'] != 0:
        raise ValueError('Release mode does not allow custom point/view pre-subsampling')
    import yaml
    model_options = yaml.safe_load((resolve(config['model_root']) / 'reconstruction/flows/ss/config.yaml').read_text())['model']['dino']
    if config.get('anyup_use_natten') != model_options.get('anyup_use_natten', True):
        raise ValueError('Non-batch override forbidden: anyup_use_natten')
    if config.get('anyup_q_chunk_size') != model_options.get('anyup_q_chunk_size'):
        raise ValueError('Non-batch override forbidden: anyup_q_chunk_size')
    write_json(output / 'official_protocol.json', original)
    write_json(output / 'resolved_protocol.json', resolved)
    write_json(output / 'protocol_audit.json', audit)
    return resolved


def object_seed(config, object_name, stage, object_names=None):
    from fire3d.runtime.model_loader import stable_seed
    identity = config['input_identity']
    if stage == 'pbr':
        if object_names is None:
            objects, _ = select_objects(read_json(resolve(config['output']) / 'conditioning.json')['objects'], config)
            object_names = [o['name'] for o in objects]
        start = object_names.index(object_name)
        return stable_seed(config['seed'], f'{identity}:pbr:{start}')
    return stable_seed(config['seed'], f'{identity}:{object_name}:conditioning-model-order-v1')
