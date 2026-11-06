"""Call an unrestricted method through the shared RGBScene -> meshes interface."""
import argparse
import importlib
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap
from track2_io import load_scene, export_scene, ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', required=True, help='Python module:function, e.g. methods.my_method:predict')
    parser.add_argument('--scene', required=True)
    parser.add_argument('--data-root', type=Path, default=ROOT / 'data/scenes')
    parser.add_argument('--output-root', type=Path, default=ROOT / 'outputs/my_method')
    args = parser.parse_args()
    module, name = args.method.split(':', 1)
    predict = getattr(importlib.import_module(module), name)
    scene = load_scene(args.data_root, args.scene)
    meshes = predict(scene)
    manifest = export_scene(scene, meshes, args.output_root / args.scene / 'evaluation')
    print('EXPORTED', manifest)


if __name__ == '__main__':
    main()
