"""Pin/download public recon-only weights and verify upstream SHA256 values."""
from pathlib import Path
from ..pi3x_fusion.common import read_json, resolve, write_json, sha256


def main():
    from huggingface_hub import snapshot_download
    config = read_json(resolve('configs/baselines/fire3d_recon.yaml'))
    root = resolve(config['model_root'])
    patterns = ['config.json', 'manifest.json', 'checksums.sha256',
        'external/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth', 'external/trellis2/shape*',
        'reconstruction/flows/ss/**', 'reconstruction/flows/shape/**',
        'reconstruction/stats/ss.json', 'reconstruction/stats/shape.json',
        'reconstruction/vae/ss/config.json', 'reconstruction/vae/ss/ckpts/decoder.pt',
        'reconstruction/vae/shape/config.json', 'reconstruction/vae/shape/ckpts/decoder.pt']
    if config.get('pbr', {}).get('enabled', False):
        patterns += ['external/trellis2/tex*', 'reconstruction/flows/pbr/**', 'reconstruction/stats/pbr.json',
                     'reconstruction/vae/pbr/config.json', 'reconstruction/vae/pbr/ckpts/decoder.pt']
    snapshot_download('hongchi/Fire3D', revision=config['model_revision'], local_dir=root,
                      allow_patterns=patterns, token=False, max_workers=2)
    records = []
    for line in (root / 'checksums.sha256').read_text().splitlines():
        expected, relative = line.split(maxsplit=1)
        path = root / relative.strip()
        if not path.is_file() or relative.startswith('perception/'):
            continue
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f'Checksum mismatch: {relative}')
        records.append({'path': str(path), 'sha256': actual, 'bytes': path.stat().st_size})
        print(f'Verified {relative}', flush=True)
    write_json(root / 'recon_weights.json', {'repo': 'hongchi/Fire3D', 'revision': config['model_revision'],
        'code_revision': config['code_revision'], 'weights': records, 'perception_downloaded_or_loaded': False})


if __name__ == '__main__':
    main()
