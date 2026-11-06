"""Validate one scene, or the entire fixed split, without loading any GT."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap
from track2_io import scene_ids, validate_scene


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--manifest', type=Path)
    group.add_argument('--predictions', type=Path, help='Directory containing <split>_scene_XXX/evaluation/')
    parser.add_argument('--split', choices=['dev', 'val', 'test'], default='test')
    parser.add_argument('--data-root', type=Path, default=Path('data/scenes'))
    args = parser.parse_args()
    if args.manifest:
        manifest, _ = validate_scene(args.manifest, args.data_root)
        print('VALID', manifest['scene_id'])
    else:
        for identity in scene_ids(args.split):
            manifest, _ = validate_scene(args.predictions / identity / 'evaluation/mesh_manifest.json', args.data_root)
            if manifest['scene_id'] != identity:
                raise ValueError(f'Wrong scene in {identity}')
            print('VALID', identity)


if __name__ == '__main__':
    main()
