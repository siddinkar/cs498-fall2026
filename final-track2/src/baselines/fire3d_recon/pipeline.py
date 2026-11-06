#!/usr/bin/env python3
"""Run reconstruction stages and export one GT-free textured mesh scene."""
import argparse
import os
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/baselines/fire3d_recon.yaml')
    parser.add_argument('--stage', choices=['infer', 'conditioning', 'reconstruct', 'texture', 'export'], default='infer')
    parser.add_argument('--objects', type=int, nargs='+', help='Diagnostic model stages only; public output includes all supported instances')
    parser.add_argument('--source-run')
    parser.add_argument('--output')
    parser.add_argument('--input-identity')
    parser.add_argument('--evaluation-output')
    args = parser.parse_args()
    from baselines.pi3x_fusion.common import read_json, resolve, write_json
    from baselines.fire3d_recon.frozen import verify_frozen_profile
    config = read_json(resolve(args.config))
    for key in ['source_run', 'output', 'input_identity', 'evaluation_output']:
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    if any(not config.get(key) for key in ['source_run','output','input_identity','evaluation_output']):
        parser.error('Run scripts/student/run_baseline.py for automatic scene/run paths, or provide all four runtime path/identity flags.')
    # Reject parameter drift BEFORE touching existing artifacts.
    audit = verify_frozen_profile(config)
    out = resolve(config['output'])
    from baselines.layout import evaluation_dir
    public = evaluation_dir(config)
    if (out / 'config.json').exists():
        old = read_json(out / 'config.json')
        old_cmp = {k: v for k, v in old.items() if k != 'frozen_profile'}
        new_cmp = {k: v for k, v in config.items() if k != 'frozen_profile'}
        for comparable in [old_cmp, new_cmp]:
            for key in ['source_run', 'output', 'evaluation_output']:
                comparable[key] = str(resolve(comparable[key]).resolve())
        if old_cmp != new_cmp:
            raise ValueError('Inference configuration/input identity changed; choose a new output directory')
    source = resolve(config['source_run'])
    import numpy as np
    frames = read_json(source / 'input/frames.json')
    with np.load(source / 'input/cameras.npz') as calibration:
        np.testing.assert_array_equal([frame['frame_index'] for frame in frames['frames']], calibration['frame_indices'])
    if args.objects and args.stage not in ['reconstruct', 'texture']:
        raise ValueError('--objects is supported only for diagnostic reconstruct/texture stages')
    write_json(out / 'config.json', config)
    write_json(public / 'reports/frozen_profile.json', audit)
    (out / 'logs').mkdir(exist_ok=True)
    from baselines.fire3d_recon.protocol import load_protocol
    load_protocol(config)
    os.environ.setdefault('TORCH_HOME', str(ROOT / 'cache/torch'))
    from baselines.profiling import Timings
    from baselines.resources import gpu_lease
    timings=Timings(out/'timings.json')
    if args.stage in ['conditioning', 'infer'] and not (out / 'conditioning.json').exists():
        from baselines.fire3d_recon.conditioning import prepare
        with timings.stage('conditioning'):
            prepare(config)
    if args.stage in ['reconstruct', 'infer']:
        from baselines.fire3d_recon.runtime import reconstruct
        with timings.stage('reconstruct'), gpu_lease(f'inference:{config["input_identity"]}:geometry_flows'):
            reconstruct(config, args.objects)
    if args.stage in ['texture', 'infer']:
        from baselines.fire3d_recon.texture import infer_textures
        with timings.stage('texture'), gpu_lease(f'inference:{config["input_identity"]}:pbr_flows_and_bake'):
            infer_textures(config, args.objects)
    if args.stage in ['export', 'infer']:
        from baselines.mesh_output import publish_textured_meshes
        with timings.stage('export'):
            publish_textured_meshes(config)
    if args.stage == 'infer':
        # GT-free release path: no scene.blend, renderer, or GT-side validation.
        from baselines.mesh_output import load_textured_meshes
        load_textured_meshes(public / 'mesh_manifest.json')
    print(f'FIRE3D_STAGE_COMPLETE {args.stage} {out}', flush=True)


if __name__ == '__main__':
    main()
