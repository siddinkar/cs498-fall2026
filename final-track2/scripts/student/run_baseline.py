"""Run the complete provided baseline using only released RGB/calibration inputs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import shutil
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap
from track2_io import ROOT, load_scene, scene_path, write_json


def check_media_tools():
    """Fail before model inference if required video executables are absent."""
    missing = [name for name in ['ffmpeg', 'ffprobe'] if shutil.which(name) is None]
    if missing:
        raise FileNotFoundError('Install ffmpeg (including ffprobe) using your system package manager. Missing: '
                                + ', '.join(missing))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', default='dev_scene_000')
    parser.add_argument('--data-root', type=Path, default=ROOT / 'data/scenes')
    parser.add_argument('--algorithm-name', default='baseline')
    parser.add_argument('--output-root', type=Path, default=ROOT / 'outputs')
    parser.add_argument('--check', action='store_true', help='Check inputs, runtimes, and weights without inference')
    args = parser.parse_args()
    if not args.algorithm_name or Path(args.algorithm_name).name != args.algorithm_name or args.algorithm_name in {'.', '..'}:
        raise ValueError('Algorithm name must be a single directory name')
    check_media_tools()
    load_scene(args.data_root, args.scene)
    config = json.loads((ROOT / 'configs/baselines/perception_v2.yaml').read_text())
    config['input_root'] = str(args.data_root.resolve())
    config['python'] = os.environ.get('TRACK2_PERCEPTION_PYTHON', str(ROOT / config['python']))
    reconstruction_python = Path(os.environ.get('TRACK2_RECONSTRUCTION_PYTHON', str(ROOT / 'env/fire3d/.conda/bin/python')))
    missing = [str(p) for p in [Path(config['python']), reconstruction_python,
        ROOT / config['geometry']['weights'], ROOT / config['proposals']['sam2_checkpoint'],
        ROOT / config['proposals']['florence_path'] / 'model.safetensors',
        ROOT / 'cache/models/fire3d/recon_weights.json'] if not p.is_file()]
    if missing:
        raise FileNotFoundError('Complete the setup in README.md. Missing:\n' + '\n'.join(missing))
    if args.check:
        print('INPUTS_AND_RUNTIME_PATHS_OK', args.scene)
        return
    config_path = args.output_root.resolve() / args.algorithm_name / 'perception_config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError('Configuration changed; choose another algorithm name')
    write_json(config_path, config)
    relative_scene = scene_path(args.data_root, args.scene).relative_to(args.data_root.resolve()).as_posix()
    command = [config['python'], str(ROOT / 'scripts/baseline/run.py'), '--config', str(config_path),
        '--scene', relative_scene, '--algorithm-name', args.algorithm_name,
        '--output-root', str(args.output_root.resolve()), '--fire3d-stage', 'infer']
    # Leave the default unset so the shell launcher activates its local CUDA
    # environment. An explicit runtime override inherits the caller's setup.
    environment = dict(os.environ)
    subprocess.run(command, cwd=ROOT, env=environment, check=True)
    from track2_io import validate_scene
    manifest = args.output_root.resolve() / args.algorithm_name / args.scene / 'evaluation/mesh_manifest.json'
    validate_scene(manifest, args.data_root)
    print('BASELINE_COMPLETE', manifest)


if __name__ == '__main__':
    main()
