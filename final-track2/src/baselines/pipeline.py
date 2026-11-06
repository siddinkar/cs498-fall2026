#!/usr/bin/env python3
"""Prepare RGB/calibration inputs and run separate, resumable pipeline stages."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import os
import shutil
import subprocess
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from .pi3x_fusion.common import environment, read_json, resolve, sha256, write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/baselines/perception_v2.yaml')
    p.add_argument('--scene', default='dev/scene_000')
    p.add_argument('--algorithm-name', default='pi3x_sam2_fire3d_v2')
    p.add_argument('--output-root', default='outputs')
    p.add_argument('--stages', nargs='+', choices=['geometry', 'proposals', 'fuse', 'visualize', 'fire3d'],
                   default=['geometry', 'proposals', 'fuse', 'visualize', 'fire3d'])
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--serial',action='store_true',help='Run perception visualization and reconstruction sequentially')
    p.add_argument('--reuse-stages-from', help='Reuse compatible RGB-D and proposals from this scene run')
    p.add_argument('--fusion-run', help='Existing completed fusion run, for --stages fire3d only')
    p.add_argument('--fire3d-config', default='configs/baselines/fire3d_recon.yaml')
    p.add_argument('--fire3d-output', help='Override reconstruction intermediates only')
    p.add_argument('--fire3d-stage', choices=['infer', 'export'], default='infer')
    args = p.parse_args()
    if args.limit and 'fire3d' in args.stages:
        raise ValueError('Fire3D pipeline requires every input frame; --limit is diagnostic frontend-only')
    if args.fusion_run:
        if args.stages != ['fire3d']:
            raise ValueError('--fusion-run requires --stages fire3d')
        run = resolve(args.fusion_run)
        run_fire3d(args, run)
        return
    config = read_json(resolve(args.config))
    if 'fire3d' in args.stages and config['frame_stride'] != 1:
        raise ValueError('Full baseline requires frame_stride=1')
    if 'fire3d' in args.stages and args.stages[-1] != 'fire3d':
        raise ValueError('Fire3D must be the final pipeline stage')
    source = resolve(config['input_root']) / args.scene
    from .layout import OutputLayout
    layout = OutputLayout.for_scene(args.scene, args.algorithm_name, args.output_root)
    run = layout.perception
    run.mkdir(parents=True, exist_ok=True)
    if (run / 'config.json').exists() and read_json(run / 'config.json') != config:
        raise ValueError('Run config changed; use another run ID')
    write_json(run / 'config.json', config)
    calibration = dict(np.load(source / 'cameras.npz'))
    frames = []
    for index in calibration['frame_indices'][::config['frame_stride']]:
        path = source / 'rgb' / f'frame_{int(index):04d}.png'
        frames.append({'frame_index': int(index), 'rgb_path': str(path), 'rgb_sha256': sha256(path)})
    (run / 'input').mkdir(exist_ok=True)
    input_manifest = {'source_scene': str(source), 'frames': frames,
               'calibration_sha256': sha256(source / 'cameras.npz'),
               'input_contract': 'RGB + provided camera poses/intrinsics; no GT depth/masks/meshes'}
    if (run / 'input/frames.json').exists() and read_json(run / 'input/frames.json') != input_manifest:
        raise ValueError('Input RGB/calibration changed; use a new algorithm-name namespace')
    write_json(run / 'input/frames.json', input_manifest)
    np.savez_compressed(run / 'input/cameras.npz', **calibration)
    if args.reuse_stages_from:
        previous = resolve(args.reuse_stages_from)
        old_config = read_json(previous / 'config.json')
        old_input = read_json(previous / 'input/frames.json')
        current_input = read_json(run / 'input/frames.json')
        if old_input != current_input or any(old_config[key] != config[key] for key in ['geometry', 'proposals']):
            raise ValueError('Reused geometry/proposal inputs or model settings differ')
        for folder in ['rgbd', 'proposals']:
            def link_cached(source_path,target_path):
                if Path(target_path).exists():
                    if sha256(source_path)!=sha256(target_path):raise ValueError(f'Reused cache differs: {target_path}')
                    return target_path
                os.link(source_path,target_path);return target_path
            shutil.copytree(previous / folder, run / folder, copy_function=link_cached,dirs_exist_ok=True)
    from .profiling import Timings
    timings=Timings(layout.evaluation/'reports/inference_timings.json')
    py = resolve(config['python'])
    def run_stage(stage):
        if stage == 'fire3d':
            with timings.stage('fire3d'):run_fire3d(args, run)
            return
        command = [str(py), '-m', f'baselines.pi3x_fusion.{stage}', '--run-dir', str(run)]
        if args.limit:
            command += ['--limit', str(args.limit)]
        print(f'STAGE {stage} {run}', flush=True)
        (run / 'logs').mkdir(exist_ok=True)
        with timings.stage(stage), (run / 'logs' / (stage + '.log')).open('a') as log:
            from contextlib import nullcontext
            from .resources import gpu_lease
            lease=gpu_lease(f'inference:{args.scene}:{stage}') if stage in {'geometry','proposals'} else nullcontext()
            with lease:
                subprocess.run(command, cwd=ROOT, env=environment(), stdout=log, stderr=subprocess.STDOUT, check=True)
            if stage=='visualize':
                subprocess.run([str(py),str(ROOT/'scripts/baseline/validate_perception.py'),'--run-dir',str(run)],
                    cwd=ROOT,env=environment(),stdout=log,stderr=subprocess.STDOUT,check=True)
    try:
        if not args.serial and args.stages==['geometry','proposals','fuse','visualize','fire3d']:
            for stage in ['geometry','proposals','fuse']:run_stage(stage)
            with ThreadPoolExecutor(2) as pool:
                futures=[pool.submit(run_stage,stage) for stage in ['visualize','fire3d']]
                for future in futures:future.result()
        else:
            for stage in args.stages:run_stage(stage)
    except BaseException as error:
        if 'fire3d' in args.stages:
            report=layout.evaluation/'reports/pipeline.json'
            failed=read_json(report) if report.exists() else {}
            write_json(report,dict(failed,passed=False,error=str(error),status='failed'))
        raise
    print(f'RUN_COMPLETE {run}', flush=True)


def run_fire3d(args, run):
    frames = read_json(run / 'input/frames.json')
    source_scene = Path(frames['source_scene'])
    if '/'.join(source_scene.parts[-2:]) != args.scene:
        raise ValueError('--scene must match the fusion input scene, including its split')
    with np.load(run / 'input/cameras.npz') as calibration:
        np.testing.assert_array_equal([frame['frame_index'] for frame in frames['frames']], calibration['frame_indices'])
    config = read_json(resolve(args.fire3d_config))
    from .layout import OutputLayout
    layout = OutputLayout.for_scene(args.scene, args.algorithm_name, args.output_root)
    output = resolve(args.fire3d_output) if args.fire3d_output else layout.reconstruction
    identity = 'tabletop:' + Path(frames['source_scene']).name
    config.update(source_run=str(run.resolve()), output=str(output.resolve()), input_identity=identity, evaluation_output=str(layout.evaluation.resolve()))
    perception=read_json(run/'config.json')
    write_json(layout.evaluation/'reports/perception_profile.json',{'algorithm':args.algorithm_name,
        'config':perception,'input_frames':len(frames['frames']),
        'config_sha256':sha256(run/'config.json'),
        'worker_sha256':{name:sha256(ROOT/'src/baselines/pi3x_fusion'/f'{name}.py')
            for name in ['fuse','instance_filter','consensus']},
        'runtime_memory_sha256':sha256(ROOT/'src/baselines/fire3d_recon/memory.py'),
        'execution_implementation_sha256':{str(path.relative_to(ROOT)):sha256(path) for path in sorted((ROOT/'src/baselines').rglob('*.py'))}})
    from .fire3d_recon.frozen import verify_frozen_profile
    verify_frozen_profile(config)
    command = ['bash', str(ROOT / 'scripts/baseline/run_reconstruction.sh'), '--config', str(resolve(args.fire3d_config)),
               '--source-run', str(run.resolve()), '--output', str(output.resolve()),
               '--input-identity', identity, '--evaluation-output', str(layout.evaluation.resolve()), '--stage', args.fire3d_stage]
    (output / 'logs').mkdir(parents=True, exist_ok=True)
    print(f'STAGE fire3d {output}', flush=True)
    with (output / 'logs/pipeline.log').open('a') as log:
        subprocess.run(command, cwd=ROOT, env=environment(), stdout=log, stderr=subprocess.STDOUT, check=True)
    write_json(layout.evaluation / 'reports/pipeline.json', {'passed': True, 'fusion_run': str(run.resolve()),
               'stage': args.fire3d_stage, 'public_manifest': str(layout.evaluation / 'mesh_manifest.json'),
               'all_input_frames': len(frames['frames']), 'command': command})
    print(f'BASELINE_OUTPUT {layout.evaluation / "mesh_manifest.json"}', flush=True)


if __name__ == '__main__':
    main()
