"""Create a minimal, deterministic Test result ZIP; does not include GT or caches."""
import argparse
import json
import hashlib
from pathlib import Path, PurePosixPath
import re
import stat
import zipfile
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap
from track2_io import scene_ids, validate_scene, sha256


def inspect_result_bundle(bundle, data_root=None):
    """Verify the complete minimal Test inventory, bytes, hashes, and scene cap.

    This read-only check does not execute code, render meshes, or use private
    GT. Official staging still validates meshes and canonical calibration.
    """
    with zipfile.ZipFile(bundle) as zipped:
        infos = zipped.infolist()
        names = [info.filename for info in infos]
        if len(names) != len(set(names)) or len(names) > 10000:
            raise ValueError('Duplicate or excessive bundle entries')
        if sum(info.file_size for info in infos) > 440_000_000:
            raise ValueError('Result bundle exceeds the total size limit')
        if 'bundle_manifest.json' not in names or zipped.getinfo('bundle_manifest.json').file_size > 8_000_000:
            raise ValueError('Missing or oversized bundle manifest')
        receipt = json.loads(zipped.read('bundle_manifest.json'))
        if receipt.get('schema') != 'cs498.track2.result_bundle.v1' or receipt.get('split') != 'test' or receipt.get('scene_ids') != scene_ids('test'):
            raise ValueError('Expected all 20 canonical Test scene IDs')
        declared = {row['path']:row for row in receipt['files']}
        if len(declared) != len(receipt['files']) or set(names) != set(declared) | {'bundle_manifest.json'}:
            raise ValueError('Bundle inventory mismatch')
        for identity in scene_ids('test'):
            required = {f'{identity}/evaluation/{tail}' for tail in ['mesh_manifest.json','inputs/cameras.npz','inputs/source.json']}
            if not required <= set(names):
                raise ValueError(f'{identity}: missing minimal metadata')
            meshes = [info for info in infos if info.filename.startswith(identity+'/evaluation/meshes/')]
            if not meshes or sum(info.file_size for info in meshes) >= 20_000_000:
                raise ValueError(f'{identity}: combined GLBs must be strictly below 20 MB')
        for info in infos:
            path = PurePosixPath(info.filename)
            if (path.is_absolute() or '..' in path.parts or '\\' in info.filename or path.as_posix()!=info.filename
                    or info.is_dir() or stat.S_ISLNK(info.external_attr >> 16)):
                raise ValueError('Unsafe bundle path or link')
            if info.filename == 'bundle_manifest.json':
                continue
            if len(path.parts) < 3 or path.parts[0] not in scene_ids('test') or path.parts[1] != 'evaluation':
                raise ValueError('Unexpected result layout')
            tail = '/'.join(path.parts[2:])
            if tail not in {'mesh_manifest.json','inputs/cameras.npz','inputs/source.json'} and not re.fullmatch(r'meshes/obj_\d{3,}\.glb',tail):
                raise ValueError('Non-minimal artifact in result ZIP')
            if info.file_size != declared[info.filename]['bytes'] or info.file_size > 20_000_000:
                raise ValueError('File size mismatch/limit')
            digest = hashlib.sha256()
            with zipped.open(info) as stream:
                for chunk in iter(lambda:stream.read(1048576), b''):
                    digest.update(chunk)
            if digest.hexdigest() != declared[info.filename]['sha256']:
                raise ValueError('Result checksum mismatch')
        if data_root is not None:
            import tempfile
            import shutil
            with tempfile.TemporaryDirectory(prefix='cs498-track2-result-check-') as folder:
                root=Path(folder)
                for info in infos:
                    target=root.joinpath(*PurePosixPath(info.filename).parts)
                    target.parent.mkdir(parents=True,exist_ok=True)
                    with zipped.open(info) as source,target.open('xb') as sink:
                        shutil.copyfileobj(source,sink)
                for identity in scene_ids('test'):
                    manifest,files=validate_scene(root/identity/'evaluation/mesh_manifest.json',data_root)
                    expected={f'{identity}/evaluation/{p.relative_to(root/identity/"evaluation").as_posix()}' for p in files}
                    actual={name for name in names if name.startswith(identity+'/')}
                    if manifest['scene_id']!=identity or expected!=actual:
                        raise ValueError('Inner Test manifest/inventory does not match its scene')
    return receipt


def package(predictions, data_root, output, split='test'):
    """Validate every required case before creating the submission archive."""
    predictions, output = Path(predictions).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'Refusing to replace {output}')
    entries, sources = [], []
    for identity in scene_ids(split):
        manifest_path = predictions / identity / 'evaluation/mesh_manifest.json'
        manifest, files = validate_scene(manifest_path, data_root)
        if manifest['scene_id'] != identity:
            raise ValueError(f'Scene ID mismatch: {identity}')
        for source in files:
            relative = f'{identity}/evaluation/{source.relative_to(manifest_path.parent).as_posix()}'
            entries.append(dict(path=relative, sha256=sha256(source), bytes=source.stat().st_size))
            sources.append((source, relative))
    receipt = dict(schema='cs498.track2.result_bundle.v1', split=split, scene_ids=scene_ids(split), files=entries)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as raw, zipfile.ZipFile(raw, 'w', compression=zipfile.ZIP_STORED) as archive:
        for source, relative in sources:
            info = zipfile.ZipInfo(relative, date_time=(2026, 1, 1, 0, 0, 0))
            with source.open('rb') as incoming, archive.open(info, 'w') as outgoing:
                import shutil
                shutil.copyfileobj(incoming, outgoing)
        archive.writestr(zipfile.ZipInfo('bundle_manifest.json', date_time=(2026, 1, 1, 0, 0, 0)),
                         json.dumps(receipt, indent=2, allow_nan=False) + '\n')
    print('PACKAGED', output, 'SHA256', sha256(output))
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, default=Path('data/scenes'))
    parser.add_argument('--output', type=Path, default=Path('outputs/test_results.zip'))
    args = parser.parse_args()
    package(args.predictions, args.data_root, args.output)


if __name__ == '__main__':
    main()
