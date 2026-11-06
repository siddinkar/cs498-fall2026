"""Versioned Fire3D profile: only run locations and input identity may vary."""
import hashlib
import json
import subprocess
from ..pi3x_fusion.common import read_json, resolve, sha256

RUNTIME_FIELDS = {'source_run', 'output', 'input_identity', 'evaluation_output', 'frozen_profile'}


def inference_settings(config):
    return {key: value for key, value in config.items() if key not in RUNTIME_FIELDS}


def verify_frozen_profile(config):
    pointer = config.get('frozen_profile')
    if not pointer:
        raise ValueError('Pipeline requires a frozen Fire3D profile')
    path = resolve(pointer)
    expected_hash = path.with_suffix('.sha256').read_text().strip().split()[0]
    if sha256(path) != expected_hash:
        raise ValueError('Frozen profile checksum mismatch')
    profile = read_json(path)
    if inference_settings(config) != profile['inference_settings']:
        differing = sorted(key for key in inference_settings(config).keys() | profile['inference_settings'].keys()
                           if inference_settings(config).get(key) != profile['inference_settings'].get(key))
        raise ValueError(f'Frozen inference settings changed: {differing}; create a new version explicitly')
    repo = resolve(config['fire3d_repo'])
    revision = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != config['code_revision']:
        raise ValueError('Fire3D source revision differs from frozen profile')
    protocol = repo / config['inference_protocol']
    if sha256(protocol) != profile['provenance']['official_protocol_sha256']:
        raise ValueError('Official protocol hash differs from frozen profile')
    return {'passed': True, 'profile_id': profile['profile_id'], 'profile_path': str(path),
            'profile_sha256': expected_hash, 'runtime_fields': sorted(RUNTIME_FIELDS - {'frozen_profile'}),
            'code_revision': revision, 'model_revision': config['model_revision']}
