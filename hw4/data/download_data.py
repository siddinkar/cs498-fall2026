"""Download, verify, and unpack the compact public HW4 data package."""

from __future__ import annotations

import argparse
import hashlib
import time
import urllib.request
import zipfile
from pathlib import Path


DATA_ROOT = Path(__file__).resolve().parent
ARCHIVE_NAME = "hw4_data.zip"
ARCHIVE_URL = (
    "https://huggingface.co/datasets/hongchi/cs498/resolve/main/"
    "hw4_data.zip?download=true"
)
MAX_MEMBER_BYTES = 10_000_000


def sha256(path: Path) -> str:
    """Return the hexadecimal SHA-256 digest used to verify one local file."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_hashes() -> dict[str, str]:
    """Load the release manifest as a mapping from relative path to digest."""
    entries = {}
    for line in (DATA_ROOT / "SHA256SUMS").read_text(encoding="ascii").splitlines():
        digest, name = line.split(maxsplit=1)
        entries[name.strip()] = digest
    return entries


def download_resumable(url: str, target: Path) -> None:
    """Download ``url`` with bounded retries and resume a partial transfer."""
    partial = target.with_name(target.name + ".partial")
    for attempt in range(5):
        offset = partial.stat().st_size if partial.exists() else 0
        request = urllib.request.Request(url, headers={"Range": f"bytes={offset}-"} if offset else {})
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                continued = bool(offset and response.status == 206)
                mode = "ab" if continued else "wb"
                with partial.open(mode) as output:
                    while chunk := response.read(4 * 1024 * 1024):
                        output.write(chunk)
            partial.replace(target)
            return
        except OSError as error:
            if attempt == 4:
                raise RuntimeError(f"Failed to download {url}") from error
            print(f"Download interrupted; retrying ({attempt + 1}/5)...", flush=True)
            time.sleep(min(2**attempt, 15))


def verify_installed(root: Path, hashes: dict[str, str]) -> None:
    """Verify every extracted dataset member against the release manifest."""
    for name, expected in hashes.items():
        if name == ARCHIVE_NAME:
            continue
        path = root / name
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Missing or invalid {path}")
    print("Verified 50 train, 5 validation, and 12 test images.", flush=True)


def install(archive_path: Path, output: Path, hashes: dict[str, str]) -> None:
    """Safely extract the expected archive members and verify their hashes."""
    expected_members = set(hashes) - {ARCHIVE_NAME}
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        if set(names) != expected_members or len(names) != len(expected_members):
            raise ValueError("Unexpected ZIP contents")
        for info in archive.infolist():
            relative = Path(info.filename)
            if relative.is_absolute() or ".." in relative.parts or info.file_size > MAX_MEMBER_BYTES:
                raise ValueError(f"Unsafe ZIP member: {info.filename}")
            target = output / relative
            if target.is_file() and sha256(target) == hashes[info.filename]:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_name(target.name + ".partial")
            digest = hashlib.sha256()
            with archive.open(info) as source, partial.open("wb") as sink:
                while chunk := source.read(4 * 1024 * 1024):
                    sink.write(chunk)
                    digest.update(chunk)
            if digest.hexdigest() != hashes[info.filename]:
                partial.unlink(missing_ok=True)
                raise ValueError(f"SHA-256 mismatch for {info.filename}")
            partial.replace(target)
    verify_installed(output, hashes)


def main() -> None:
    """Run the command-line download, extraction, or verification workflow."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DATA_ROOT)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    hashes = expected_hashes()
    if args.verify_only:
        verify_installed(output, hashes)
        return

    archive = output / ARCHIVE_NAME
    if not archive.is_file():
        print(f"Downloading {ARCHIVE_URL}", flush=True)
        download_resumable(ARCHIVE_URL, archive)
    if sha256(archive) != hashes[ARCHIVE_NAME]:
        archive.unlink()
        raise ValueError("Downloaded ZIP failed SHA-256; rerun to retry")
    install(archive, output, hashes)
    archive.unlink()
    print("HW4 data ready. Removed the ZIP after verified extraction.", flush=True)


if __name__ == "__main__":
    main()
