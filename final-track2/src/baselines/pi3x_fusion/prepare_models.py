"""Prepare pinned public model repositories and checkpoints."""
import argparse
import os
import subprocess
from .common import read_json, resolve, sha256, write_json


def repository(url, destination, revision):
    if not destination.exists():
        subprocess.run(['git', 'clone', '--filter=blob:none', url, str(destination)], check=True)
        subprocess.run(['git', '-C', str(destination), 'checkout', revision], check=True)
    actual = subprocess.check_output(['git', '-C', str(destination), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != revision:
        raise RuntimeError(f'Expected {revision} in {destination}; found {actual}. Existing repository not modified.')


def main():
    os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
    from huggingface_hub import hf_hub_download, snapshot_download
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/baselines/perception_v2.yaml')
    args = p.parse_args()
    cfg = read_json(resolve(args.config))
    geometry, proposals = cfg['geometry'], cfg['proposals']
    repository('https://github.com/yyfz/Pi3.git', resolve(geometry['repo']), geometry['repo_revision'])
    repository('https://github.com/facebookresearch/sam2.git', resolve(proposals['sam2_repo']),
               '2b90b9f5ceec907a1c18123530e92e794ad901a4')
    snapshot_download(geometry['model'], revision=geometry['model_revision'], token=False,
                      local_dir=resolve(geometry['weights']).parent, allow_patterns=['*.json', '*.safetensors'])
    snapshot_download('microsoft/Florence-2-large-ft', revision=proposals['florence_revision'], token=False,
                      local_dir=resolve(proposals['florence_path']), ignore_patterns=['pytorch_model.bin', '*.md', '.gitattributes'])
    hf_hub_download('facebook/sam2.1-hiera-small', 'sam2.1_hiera_small.pt',
                    revision='ee5bba1d82bb8749febdf90f45e84b687142ba03', token=False,
                    local_dir=resolve(proposals['sam2_checkpoint']).parent)
    write_json(resolve('cache/models/pi3x_fusion_weights.json'), {
        'pi3x': {'model': geometry['model'], 'revision': geometry['model_revision'],
                 'sha256': sha256(resolve(geometry['weights']))},
        'florence2': {'model': 'microsoft/Florence-2-large-ft', 'revision': proposals['florence_revision'],
                     'sha256': sha256(resolve(proposals['florence_path']) / 'model.safetensors')},
        'sam2': {'model': 'facebook/sam2.1-hiera-small',
                 'sha256': sha256(resolve(proposals['sam2_checkpoint']))}})
    print('Pi3X / Florence / SAM2 models ready', flush=True)


if __name__ == '__main__':
    main()
