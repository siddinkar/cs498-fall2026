#!/usr/bin/env bash
# Install only evaluation dependencies and the fixed Blender runtime, no models.
set -euo pipefail
cd "$(dirname "$0")/../.."
python -m pip install -r env/evaluation/requirements.txt
python - <<'PY'
import hashlib
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request
root = Path.cwd()
archive = root / 'cache/blender-4.2.0-linux-x64.tar.xz'
destination = root / 'env/blender'
expected = '4f4fd7646af01f6fee9d420408318381a6e52571268eb7cf9cd5033bd9e7a359'
if not (destination / 'blender-4.2.0-linux-x64/blender').is_file():
    archive.parent.mkdir(exist_ok=True, parents=True)
    if not archive.exists():
        request = urllib.request.Request(
            'https://download.blender.org/release/Blender4.2/' + archive.name,
            headers={'User-Agent': 'CS498-Track2-Installer/1.0'})
        # The official CDN rejects its default Python-urllib user agent.
        # Interrupted downloads never become a valid-looking cached archive.
        with tempfile.NamedTemporaryFile(dir=archive.parent, delete=False) as sink:
            temporary = Path(sink.name)
            with urllib.request.urlopen(request, timeout=60) as source:
                shutil.copyfileobj(source, sink)
        actual = hashlib.sha256()
        with temporary.open('rb') as source:
            for chunk in iter(lambda: source.read(1048576), b''):
                actual.update(chunk)
        if actual.hexdigest() != expected:
            raise ValueError('Blender download checksum mismatch; temporary file retained for diagnosis')
        temporary.replace(archive)
    digest = hashlib.sha256()
    with archive.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    if digest.hexdigest() != expected:
        raise ValueError('Blender archive checksum mismatch')
    destination.mkdir(exist_ok=True, parents=True)
    with tarfile.open(archive) as source:
        # The archive has an official pinned digest. No untrusted tar is accepted.
        source.extractall(destination)
executable = destination / 'blender-4.2.0-linux-x64/blender'
subprocess.run([str(executable), '--disable-autoexec', '-b', '--factory-startup',
                '--python-exit-code', '1', '--python-expr',
                "import bpy; print('Blender Python ready:', bpy.app.version_string)"],
               check=True, timeout=90)
print('BLENDER_EXECUTABLE=' + str(executable))
PY
