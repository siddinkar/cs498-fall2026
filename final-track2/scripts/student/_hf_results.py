"""Pinned HF result transport; never upload code, reports or credentials here.

The student packer uses only the standard library. Downloading on the grading
machine optionally uses huggingface_hub and its normal local authentication.
These receipts freeze file identity, not prediction authenticity or ownership.
"""

import hashlib
import json
from pathlib import Path
import re
import shutil
import stat
import zipfile

MAX_BYTES = 8_000_000_000
MANIFEST = "results_manifest.json"
LOCATION = "results_location.json"


def is_result(name):
    """Separate large generated outputs/weights from the submitted source."""
    return (name.startswith("evaluation/")
            or name in {"test_results.zip", "kaggle_submission.csv", "kaggle_receipt.json"}
            or (name.startswith("models/") and name != "models/manifest.json"))


def _digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_archive(output, files, manifest_name, manifest):
    """Write a fresh deterministic ZIP64 archive with no enclosing directory."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as raw, zipfile.ZipFile(raw, "w") as zipped:
        for path, name in files:
            info = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            info.file_size = path.stat().st_size
            info.compress_type = (zipfile.ZIP_STORED if path.suffix.lower() in
                {".zip", ".png", ".mp4", ".glb", ".pt", ".pth", ".safetensors"}
                else zipfile.ZIP_DEFLATED)
            with path.open("rb") as incoming, zipped.open(info, "w", force_zip64=True) as outgoing:
                shutil.copyfileobj(incoming, outgoing)
        zipped.writestr(zipfile.ZipInfo(manifest_name, date_time=(2026, 1, 1, 0, 0, 0)),
                       json.dumps(manifest, indent=2, allow_nan=False) + "\n")


def prepare(root, output, netid, without_test=False, data_root=None):
    """Validate the normal full staging tree and package only its large outputs."""
    from project_submission import validate_project
    from track2_io import ROOT

    root, output = Path(root).absolute(), Path(output).absolute()
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", netid):
        raise ValueError("Invalid NetID")
    if output.exists() or root == output or root in output.parents:
        raise ValueError("Use a fresh results ZIP outside the staging directory")
    files, selection, warnings = validate_project(
        root, without_test, data_root=data_root or ROOT / "data/scenes")
    files = [(p, n) for p, n in files if is_result(n)]
    manifest = dict(schema="cs498.track2.hf_results.v1", netid=netid,
        final_method=selection["final_method"], final_config=selection["final_config"],
        selected_scenes={s: selection[s] for s in ["dev", "val", "test"]},
        missing_test=without_test,
        test_results_sha256=None if without_test else _digest(root / "test_results.zip"),
        files=[dict(path=n, bytes=p.stat().st_size, sha256=_digest(p)) for p, n in files],
        warnings=warnings)
    write_archive(output, files, MANIFEST, manifest)
    if output.stat().st_size > MAX_BYTES:
        raise ValueError("HF results ZIP exceeds the 8 GB archive safety bound")
    return manifest


def inspect(path):
    """Validate every result ZIP member before any extraction or model parsing."""
    from project_submission import safe_name

    with zipfile.ZipFile(path) as zipped:
        infos = zipped.infolist()
        names = [i.filename for i in infos]
        if (len(names) != len(set(names)) or len(names) > 100000
                or sum(i.file_size for i in infos) > MAX_BYTES
                or MANIFEST not in names
                or zipped.getinfo(MANIFEST).file_size > 16_000_000):
            raise ValueError("Unsafe or missing HF results inventory")
        manifest = json.loads(zipped.read(MANIFEST))
        if (not isinstance(manifest, dict)
                or manifest.get("schema") != "cs498.track2.hf_results.v1"
                or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", manifest.get("netid", ""))
                or type(manifest.get("missing_test")) is not bool):
            raise ValueError("Invalid HF results identity")
        rows = manifest.get("files")
        if not isinstance(rows, list) or any(not isinstance(r, dict) for r in rows):
            raise ValueError("Invalid HF results inventory")
        declared = {r["path"]: r for r in rows}
        if len(rows) != len(declared) or set(names) != set(declared) | {MANIFEST}:
            raise ValueError("HF results inventory mismatch")
        expected_test = {"test_results.zip", "kaggle_submission.csv", "kaggle_receipt.json"}
        if (not manifest["missing_test"] and not expected_test <= set(names)) or (
                manifest["missing_test"] and expected_test & set(names)):
            raise ValueError("HF results Test-omission flag mismatch")
        if not {"evaluation/val_summary.json", "evaluation/evidence_selection.json"} <= set(names):
            raise ValueError("HF results missing Val summary/selection")
        for info in infos:
            if info.is_dir() or stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError("HF results links/directories are not allowed")
            if info.filename == MANIFEST:
                continue
            safe_name(info.filename)
            if not is_result(info.filename):
                raise ValueError("HF results must not contain report, source, AI record or credentials")
            row = declared[info.filename]
            if (type(row.get("bytes")) is not int or row["bytes"] != info.file_size
                    or not re.fullmatch(r"[a-f0-9]{64}", row.get("sha256", ""))):
                raise ValueError("HF result file size/hash declaration mismatch")
            h = hashlib.sha256()
            with zipped.open(info) as incoming:
                for chunk in iter(lambda: incoming.read(1024 * 1024), b""):
                    h.update(chunk)
            if h.hexdigest() != row["sha256"]:
                raise ValueError("HF result file checksum mismatch")
    return manifest


def validate_location(record):
    """Allow only one dataset file pinned to a full commit, never a moving main."""
    if (not isinstance(record, dict)
            or record.get("schema") != "cs498.track2.results_location.v1"
            or record.get("repo_type") != "dataset"
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}/[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", record.get("repo_id", ""))
            or ".." in record["repo_id"] or record["repo_id"].endswith(".")
            or not re.fullmatch(r"[a-f0-9]{40}", record.get("revision", ""))
            or record.get("filename") != "results.zip"
            or not re.fullmatch(r"[a-f0-9]{64}", record.get("sha256", ""))
            or type(record.get("bytes")) is not int or not 0 < record["bytes"] <= MAX_BYTES):
        raise ValueError("HF result location needs dataset ID, full 40-character commit, results.zip, bytes and SHA-256")
    return record


def location(path, repo_id, revision):
    """Create the small immutable pointer carried inside the course ZIP."""
    return validate_location(dict(schema="cs498.track2.results_location.v1",
        repo_type="dataset", repo_id=repo_id, revision=revision, filename="results.zip",
        sha256=_digest(path), bytes=Path(path).stat().st_size))


def verify_file(path, record):
    """A cached/local/downloaded archive must match the submitted exact bytes."""
    validate_location(record)
    if Path(path).stat().st_size != record["bytes"] or _digest(path) != record["sha256"]:
        raise ValueError("HF results archive differs from the submitted bytes/SHA-256")
    return Path(path)


def download(record, cache_dir=None):
    """Fetch only the pinned dataset file; never execute repository code.

    This parsing workflow downloads anonymously, never inheriting a staff token.
    For private repos, fetch separately with legitimate read access first, then
    pass the verified local file to the isolated unpacker. Never request tokens
    from students or give a native mesh parser access to staff credentials.
    """
    validate_location(record)
    try:
        from huggingface_hub import get_hf_file_metadata, hf_hub_download, hf_hub_url
    except ImportError as error:
        raise RuntimeError("Grading download needs huggingface_hub; install it in the staff environment") from error
    url = hf_hub_url(record["repo_id"], record["filename"],
                     repo_type="dataset", revision=record["revision"])
    metadata = get_hf_file_metadata(url, token=False)
    if metadata.size != record["bytes"] or metadata.commit_hash != record["revision"]:
        raise ValueError("Pinned HF metadata size/revision mismatch; do not download another version")
    path = hf_hub_download(record["repo_id"], record["filename"], repo_type="dataset",
                           revision=record["revision"], cache_dir=cache_dir, token=False)
    return verify_file(path, record)


def restore(path, destination, course_manifest):
    """Restore the normal grading tree from a verified result archive."""
    record = inspect(path)
    for key in ("netid", "final_method", "final_config", "selected_scenes", "missing_test", "test_results_sha256"):
        if record.get(key) != course_manifest.get(key):
            raise ValueError("HF results and course ZIP identity/selection mismatch")
    destination = Path(destination)
    with zipfile.ZipFile(path) as zipped:
        for name in (r["path"] for r in record["files"]):
            target = destination.joinpath(*Path(name).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with zipped.open(name) as incoming, target.open("xb") as outgoing:
                shutil.copyfileobj(incoming, outgoing)
    return record
