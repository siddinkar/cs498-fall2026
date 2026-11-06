"""Install checksum-verified course archives from a local directory or HTTPS."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
import urllib.request
import zipfile


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def safe_extract(archive, destination, max_bytes=30_000_000_000):
    """Extract only the allowlisted input/evaluation layout; reject unsafe ZIPs."""
    destination = Path(destination).resolve()
    with zipfile.ZipFile(archive) as zipped:
        seen, total = set(), 0
        for info in zipped.infolist():
            path = PurePosixPath(info.filename)
            if path.is_absolute() or '..' in path.parts or '\\' in info.filename or info.filename in seen:
                raise ValueError('Unsafe or duplicate archive path')
            seen.add(info.filename)
            if stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError('Archive links are not allowed')
            parts = path.parts
            if len(parts) < 3 or parts[0] not in {'dev', 'val', 'test'}:
                raise ValueError('Unexpected data archive layout')
            import re
            if not re.fullmatch(r'scene_\d{3}', parts[1]):
                raise ValueError('Invalid scene directory')
            index = int(parts[1][-3:])
            if index >= {'dev': 30, 'val': 10, 'test': 20}[parts[0]]:
                raise ValueError('Scene index outside split')
            tail = '/'.join(parts[2:])
            allowed = tail in {'cameras.npz', 'video.mp4'} or bool(re.fullmatch(r'rgb/frame_\d{4}\.png', tail))
            allowed = allowed or (parts[0] != 'test' and tail in {'scene.blend','physics_eval.json'})
            if not allowed:
                raise ValueError(f'Non-public artifact in data archive: {info.filename}')
            total += info.file_size
            if total > max_bytes:
                raise ValueError('Archive exceeds the uncompressed size limit')
            target = destination.joinpath(*parts)
            if any(parent.is_symlink() for parent in [target, *target.parents]):
                raise ValueError('Destination must not use links')
            if target.exists():
                # Never replace an existing user dataset, even with a matching name.
                with zipped.open(info) as source:
                    expected = hashlib.sha256(source.read()).hexdigest()
                if sha(target) != expected:
                    raise FileExistsError(f'Different existing data: {target}')
        for info in zipped.infolist():
            target = destination.joinpath(*PurePosixPath(info.filename).parts)
            if target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zipped.open(info) as source, target.open('xb') as sink:
                shutil.copyfileobj(source, sink)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=Path(__file__).with_name('dataset_manifest.json'))
    parser.add_argument('--from-local-dir', type=Path)
    parser.add_argument('--split', choices=['dev', 'val', 'test', 'all'], default='all')
    parser.add_argument('--destination', type=Path, default=Path(__file__).parent / 'scenes')
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    records = [r for r in manifest['archives'] if args.split == 'all' or r['split'] == args.split]
    if not records:
        raise SystemExit('No data archives are listed. The release data manifest is not published yet.')
    cache = Path(__file__).parent / 'downloads'
    for record in records:
        name = record['filename']
        if Path(name).name != name:
            raise ValueError('Archive filename must be a single path component')
        if args.from_local_dir:
            archive = args.from_local_dir / name
        else:
            url = record.get('url')
            if not url or not url.startswith('https://'):
                raise SystemExit('Distribution URL is not set. Use --from-local-dir with the provided archives.')
            cache.mkdir(parents=True, exist_ok=True)
            archive = cache / name
            if archive.exists() and sha(archive) != record['sha256']:
                # Keep old cached data intact; a new immutable release gets a
                # content-addressed filename instead of a misleading mismatch.
                archive = cache / (record['sha256'][:16] + '-' + name)
            if not archive.exists():
                with tempfile.NamedTemporaryFile(dir=cache, delete=False) as tmp:
                    temporary = Path(tmp.name)
                    with urllib.request.urlopen(url, timeout=60) as source:
                        shutil.copyfileobj(source, tmp)
                if sha(temporary) != record['sha256']:
                    temporary.unlink()
                    raise ValueError('Downloaded archive checksum mismatch')
                temporary.replace(archive)
        if sha(archive) != record['sha256']:
            raise ValueError(f'Archive checksum mismatch: {name}')
        safe_extract(archive, args.destination)
        print('INSTALLED', name)


if __name__ == '__main__':
    main()
