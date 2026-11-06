"""Validate a course submission; host large results in your own HF dataset.

Prepare the documented folder first. The packer never collects your working
repository automatically: keep datasets, caches, environments and credentials
outside the staging folder. Missing visual evidence is recorded for partial
credit rather than being mistaken for successful evidence.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import stat
import zipfile
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap

from track2_io import scene_ids, sha256
from package_submission import inspect_result_bundle
from course_metrics import summarize

MAX_PROJECT_BYTES = (
    8_000_000_000  # Archive safety bound, NOT the per-scene mesh budget.
)
ROOT_FILES = {
    "report.pdf",
    "README.md",
    "ai_collaboration.jsonl",
    "test_results.zip",
    "kaggle_submission.csv",
    "kaggle_receipt.json",
    "results_location.json",
}
ROOT_DIRS = {"code", "models", "evaluation"}
BLOCKED_PARTS = {
    ".git",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    "node_modules",
    "credentials",
    "credential",
    "site-packages",
}


def safe_name(name):
    """Require one canonical relative archive path, with no link-like aliases."""
    path = PurePosixPath(name)
    if (
        not name
        or not path.parts
        or path.is_absolute()
        or ".." in path.parts
        or "\\" in name
        or path.as_posix() != name
    ):
        raise ValueError(f"Unsafe submission path: {name}")
    if any(part in BLOCKED_PARTS for part in path.parts) or path.name in {
        ".env",
        "id_rsa",
        "id_ed25519",
    }:
        raise ValueError(f"Environment/cache/credential file is not allowed: {name}")
    if (
        path.parts[0] not in ROOT_DIRS
        and name not in ROOT_FILES
        and name != "submission_manifest.json"
    ):
        raise ValueError(f"Unexpected submission entry: {name}")
    return path


def check_models(root):
    """Check model reproducibility declarations without downloading anything."""
    path = root / "models/manifest.json"
    record = json.loads(path.read_text())
    if type(record.get("required")) is not bool or not isinstance(
        record.get("checkpoints"), list
    ):
        raise ValueError(
            "models/manifest.json needs required (boolean) and checkpoints (list)"
        )
    if record["required"] != bool(record["checkpoints"]):
        raise ValueError(
            "List checkpoints for a model-based method; use required:false and [] only when none are needed"
        )
    for checkpoint in record["checkpoints"]:
        if not re.fullmatch("[a-fA-F0-9]{64}", checkpoint.get("sha256", "")):
            raise ValueError(
                "Each checkpoint needs its file SHA-256, not a repository revision"
            )
        if bool(checkpoint.get("path")) == bool(checkpoint.get("url")):
            raise ValueError(
                "Each checkpoint needs exactly one local path or stable HTTPS download URL"
            )
        if checkpoint.get("path"):
            relative = safe_name(checkpoint["path"])
            if (
                relative.parts[0] != "models"
                or sha256(root.joinpath(*relative.parts))
                != checkpoint["sha256"].lower()
            ):
                raise ValueError("Local checkpoint path/hash mismatch")
        else:
            from urllib.parse import urlparse

            url = urlparse(checkpoint["url"])
            if url.scheme != "https" or not url.netloc or url.username or url.password:
                raise ValueError(
                    "Use a stable HTTPS checkpoint URL without embedded credentials"
                )
    return record


def validate_project(root, without_test=False, received=False, data_root=None):
    """Validate essential files, selected IDs, Val means and inner Test ZIP."""
    root = Path(root).absolute()
    if any(path.is_symlink() for path in [root, *root.parents]):
        raise ValueError("Submission staging directory must not use symlinks")
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("Submission staging must not contain symlinks")
        if path.is_file():
            relative = path.relative_to(root).as_posix()
            safe_name(relative)
            if relative == "submission_manifest.json":
                if not received:
                    raise ValueError(
                        "The packer creates submission_manifest.json; do not stage a stale copy"
                    )
                continue
            files.append((path, relative))
    required = {
        "report.pdf",
        "README.md",
        "ai_collaboration.jsonl",
        "models/manifest.json",
        "evaluation/val_summary.json",
        "evaluation/evidence_selection.json",
        "code/README.md",
    }
    names = {relative for _, relative in files}
    if not required <= names:
        raise ValueError(
            "Missing submission files: " + ", ".join(sorted(required - names))
        )
    with (root / "report.pdf").open("rb") as stream:
        if stream.read(5) != b"%PDF-":
            raise ValueError("report.pdf must be a PDF, not a renamed source document")
    if not any(
        (root / "code" / name).is_file()
        for name in [
            "requirements.txt",
            "environment.yml",
            "pyproject.toml",
            "DEPENDENCIES.md",
        ]
    ):
        raise ValueError(
            "code/ needs requirements.txt, environment.yml, pyproject.toml or DEPENDENCIES.md"
        )
    for name in ["README.md", "code/README.md"]:
        if not (root / name).read_text().strip():
            raise ValueError(
                f"{name} must describe the exact final reproduction commands"
            )
    check_models(root)
    records = [
        json.loads(line)
        for line in (root / "ai_collaboration.jsonl").read_text().splitlines()
        if line.strip()
    ]
    if not records or any(
        not isinstance(row, dict)
        or row.get("type") in {"example_only", "template_instructions"}
        for row in records
    ):
        raise ValueError(
            "Replace AI-record sample entries with your actual meaningful collaboration record"
        )
    selection = json.loads((root / "evaluation/evidence_selection.json").read_text())
    if (
        selection.get("schema") != "cs498.track2.evidence_selection.v1"
        or not selection.get("final_method", "").strip()
    ):
        raise ValueError("Evidence selection needs schema and final_method")
    configuration = safe_name(selection.get("final_config", ""))
    if (
        configuration.parts[0] != "code"
        or not root.joinpath(*configuration.parts).is_file()
    ):
        raise ValueError(
            "final_config must name the included final configuration or entrypoint under code/"
        )
    for split in ["dev", "val", "test"]:
        values = selection.get(split, [])
        if split == "test" and without_test and values == []:
            continue
        if (
            not isinstance(values, list)
            or len(values) != 3
            or len(set(values)) != 3
            or not set(values) <= set(scene_ids(split))
        ):
            raise ValueError(
                f"Select exactly three distinct canonical {split} scene IDs"
            )
    measured = summarize(root / "evaluation/val", "val")
    claimed = json.loads((root / "evaluation/val_summary.json").read_text())
    if not isinstance(claimed, dict) or any(
        claimed.get(key) != measured[key]
        for key in [
            "schema",
            "split",
            "expected",
            "completed",
            "missing",
            "cases",
            "protocol_sha256",
        ]
    ):
        raise ValueError("Val summary inventory differs from included per-case reports")
    for key, value in measured["means"].items():
        proposed = float(claimed["means"][key])
        if not math.isfinite(proposed) or abs(proposed - value) > 1e-10:
            raise ValueError(
                "Val means do not match all ten fixed cases, including missing-case zeros"
            )
    warnings = []
    for split in ["dev", "val"]:
        for identity in selection[split]:
            folder = (
                root
                / "evaluation"
                / split
                / identity
                / "evaluation/reports/evaluation/default"
            )
            for filename in [
                "metrics.json",
                "matching.json",
                "visualizations.json",
                "index.html",
            ]:
                if not (folder / filename).is_file():
                    warnings.append(
                        f"{identity}: missing {filename}; visual-evidence credit needs review"
                    )
            visual_index = folder / "visualizations.json"
            if visual_index.is_file():

                def linked(value):
                    if isinstance(value, dict):
                        for key, item in value.items():
                            if key in {"image", "video", "overlays"} and isinstance(
                                item, str
                            ):
                                yield item
                            elif key == "images" and isinstance(item, list):
                                yield from item
                            else:
                                yield from linked(item)
                    elif isinstance(value, list):
                        for item in value:
                            yield from linked(item)

                try:
                    for relative in linked(json.loads(visual_index.read_text())):
                        path = PurePosixPath(relative)
                        if path.is_absolute() or ".." in path.parts or "\\" in relative:
                            warnings.append(
                                f"{identity}: nonportable visual link; evidence is invalid: {relative}"
                            )
                        elif not folder.joinpath(*path.parts).exists():
                            warnings.append(
                                f"{identity}: missing linked artifact {relative}"
                            )
                except (ValueError, TypeError):
                    warnings.append(
                        f"{identity}: malformed visual index; evidence needs review"
                    )
    test = root / "test_results.zip"
    if without_test:
        if test.exists():
            raise ValueError(
                "--without-test requires an intentionally absent test_results.zip"
            )
        warnings.append(
            "No Test submission: entire base score halved once; neither bonus eligible"
        )
    else:
        if not test.is_file():
            raise ValueError(
                "Missing test_results.zip; use --without-test only for intentional omission and the stated penalty"
            )
        inspect_result_bundle(test, data_root)
        csv_path = root / "kaggle_submission.csv"
        receipt_path = root / "kaggle_receipt.json"
        if not csv_path.is_file() or not receipt_path.is_file():
            raise ValueError(
                "Include the exact kaggle_submission.csv and kaggle_receipt.json for the final Test version"
            )
        from kaggle_receipt import validate_csv

        validate_csv(csv_path, bundle=test)
        receipt = json.loads(receipt_path.read_text())
        if (
            receipt.get("schema") != "cs498.track2.kaggle_receipt.v1"
            or receipt.get("competition") != "cs498-fall-2026-track2-reconstruction"
            or type(receipt.get("submission_id")) is not int
            or receipt["submission_id"] <= 0
            or receipt.get("csv_sha256") != sha256(csv_path)
        ):
            raise ValueError(
                "Kaggle receipt needs the competition, actual submission ID and exact CSV SHA-256"
            )
        with zipfile.ZipFile(test) as zipped:
            test_manifests = {
                identity: json.loads(
                    zipped.read(f"{identity}/evaluation/mesh_manifest.json")
                )
                for identity in selection["test"]
            }
        for identity in selection["test"]:
            folder = root / "evaluation/test" / identity / "prediction_preview"
            for filename in [
                "index.html",
                "preview.json",
                "scene_views.png",
                "object_views.png",
            ]:
                if not (folder / filename).is_file():
                    warnings.append(f"{identity}: missing prediction-only {filename}")
            if (folder / "preview.json").is_file():
                try:
                    preview = json.loads((folder / "preview.json").read_text())
                    if not isinstance(preview, dict):
                        raise ValueError("Preview receipt must be a JSON object")
                    measured = test_manifests[identity]
                    with zipfile.ZipFile(test) as zipped:
                        digest = hashlib.sha256(
                            zipped.read(f"{identity}/evaluation/mesh_manifest.json")
                        ).hexdigest()
                    if (
                        preview.get("scene_id") != identity
                        or preview.get("manifest_sha256") != digest
                        or preview.get("glbs")
                        != {
                            row["glb_path"]: row["sha256"] for row in measured["meshes"]
                        }
                        or preview.get("uses_gt") is not False
                        or preview.get("official_metrics") is not False
                    ):
                        warnings.append(
                            f"{identity}: stale or invalid prediction-preview identity; do not award that evidence"
                        )
                except (ValueError, KeyError, TypeError):
                    warnings.append(
                        f"{identity}: malformed prediction-preview receipt; evidence needs review"
                    )
    if sum(path.stat().st_size for path, _ in files) > MAX_PROJECT_BYTES:
        raise ValueError(
            "Project exceeds the 8 GB packaging safety bound; use checkpoint URLs and exclude caches/data"
        )
    return files, selection, warnings


def package_project(root, output, netid, without_test=False, data_root=None,
                    hf_results=None, hf_repo=None, hf_revision=None):
    """Create one deterministic outer ZIP; never replace an existing archive."""
    root, output = Path(root).absolute(), Path(output).absolute()
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", netid):
        raise ValueError(
            "NetID must contain only letters, digits, underscore or hyphen"
        )
    if output.exists() or output == root or root in output.parents:
        raise ValueError("Use a new output ZIP outside the staging directory")
    if data_root is None:
        from track2_io import ROOT

        data_root = ROOT / "data/scenes"
    files, selection, warnings = validate_project(
        root, without_test, data_root=data_root
    )
    hosted = any(x is not None for x in (hf_results, hf_repo, hf_revision))
    pointer = None
    if hosted:
        if not all(x is not None for x in (hf_results, hf_repo, hf_revision)):
            raise ValueError("Provide --hf-results-zip, --hf-repo and --hf-revision together")
        from _hf_results import inspect, is_result, location
        remote = inspect(hf_results)
        expected = {n: (p.stat().st_size, sha256(p)) for p, n in files if is_result(n)}
        declared = {r["path"]: (r["bytes"], r["sha256"]) for r in remote["files"]}
        if (declared != expected or remote["netid"] != netid
                or remote["final_method"] != selection["final_method"]
                or remote["final_config"] != selection["final_config"]
                or remote["selected_scenes"] != {s: selection[s] for s in ["dev", "val", "test"]}
                or remote["missing_test"] is not without_test
                or remote["test_results_sha256"] != (None if without_test else sha256(root / "test_results.zip"))):
            raise ValueError("HF results are stale or belong to another method/student; prepare a new archive")
        pointer = location(hf_results, hf_repo, hf_revision)
        files = [(p, n) for p, n in files if not is_result(n) and n != "results_location.json"]
    entries = [
        dict(path=name, bytes=path.stat().st_size, sha256=sha256(path))
        for path, name in files
    ]
    pointer_text = json.dumps(pointer, indent=2) + "\n" if hosted else None
    if hosted:
        entries.append(dict(path="results_location.json", bytes=len(pointer_text.encode()),
                            sha256=hashlib.sha256(pointer_text.encode()).hexdigest()))
    manifest = dict(
        schema="cs498.track2.project_submission.v2" if hosted else "cs498.track2.project_submission.v1",
        netid=netid,
        final_method=selection["final_method"],
        final_config=selection["final_config"],
        selected_scenes={split: selection[split] for split in ["dev", "val", "test"]},
        missing_test=without_test,
        test_results_sha256=None if without_test else sha256(root / "test_results.zip"),
        files=entries,
        warnings=warnings,
        geometry_limit="All GLBs in each scene combined, including textures/table, must be strictly below 20,000,000 bytes",
    )
    receipt_text = json.dumps(manifest, indent=2, allow_nan=False) + "\n"
    if (
        sum(row["bytes"] for row in entries) + len(receipt_text.encode())
        > MAX_PROJECT_BYTES
    ):
        raise ValueError(
            "Project including its generated receipt exceeds the 8 GB packaging safety bound"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with (
        output.open("xb") as stream,
        zipfile.ZipFile(
            stream, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
        ) as zipped,
    ):
        for path, name in files:
            info = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            info.file_size = path.stat().st_size
            info.compress_type = (
                zipfile.ZIP_STORED
                if path.suffix.lower()
                in {".zip", ".png", ".mp4", ".glb", ".pt", ".pth", ".safetensors"}
                else zipfile.ZIP_DEFLATED
            )
            import shutil

            with (
                path.open("rb") as incoming,
                zipped.open(info, "w", force_zip64=True) as outgoing,
            ):
                shutil.copyfileobj(incoming, outgoing)
        if hosted:
            zipped.writestr(zipfile.ZipInfo("results_location.json", date_time=(2026, 1, 1, 0, 0, 0)), pointer_text)
        zipped.writestr(
            zipfile.ZipInfo(
                "submission_manifest.json", date_time=(2026, 1, 1, 0, 0, 0)
            ),
            receipt_text,
        )
    return manifest


def inspect_project_archive(path):
    """Verify outer inventory before safe staff extraction; never run code."""
    with zipfile.ZipFile(path) as zipped:
        infos = zipped.infolist()
        names = [info.filename for info in infos]
        if (
            len(names) != len(set(names))
            or len(names) > 100000
            or sum(info.file_size for info in infos) > MAX_PROJECT_BYTES
        ):
            raise ValueError(
                "Duplicate/excessive project archive entries or byte limit"
            )
        if (
            "submission_manifest.json" not in names
            or zipped.getinfo("submission_manifest.json").file_size > 16_000_000
        ):
            raise ValueError("Missing or oversized project inventory")
        manifest = json.loads(zipped.read("submission_manifest.json"))
        if manifest.get("schema") not in {"cs498.track2.project_submission.v1", "cs498.track2.project_submission.v2"}:
            raise ValueError("Not a Track 2 outer submission ZIP")
        if type(manifest.get("missing_test")) is not bool or not re.fullmatch(
            r"[A-Za-z0-9_-]{1,64}", manifest.get("netid", "")
        ):
            raise ValueError("Invalid submission identity or Test-omission flag")
        if manifest["schema"] == "cs498.track2.project_submission.v2":
            from _hf_results import is_result, validate_location
            if ("results_location.json" not in names
                    or zipped.getinfo("results_location.json").file_size > 16384
                    or any(is_result(n) for n in names)):
                raise ValueError("HF-mode course ZIP must contain a small pointer, not large results")
            validate_location(json.loads(zipped.read("results_location.json")))
            if not {"report.pdf", "README.md", "code/README.md", "models/manifest.json", "ai_collaboration.jsonl"} <= set(names):
                raise ValueError("HF-mode course ZIP still requires report, code, models declaration and AI record")
        elif ("test_results.zip" in names) == manifest["missing_test"]:
            raise ValueError("Test-omission flag disagrees with archive contents")
        declared = {row["path"]: row for row in manifest["files"]}
        if len(declared) != len(manifest["files"]) or set(names) != set(declared) | {
            "submission_manifest.json"
        }:
            raise ValueError("Project inventory mismatch")
        for info in infos:
            safe_name(info.filename)
            if info.is_dir() or stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError("Archive directory/link entries are not allowed")
            if info.filename == "submission_manifest.json":
                continue
            row = declared[info.filename]
            if info.file_size != row["bytes"]:
                raise ValueError("Project file size mismatch")
            digest = hashlib.sha256()
            with zipped.open(info) as incoming:
                for chunk in iter(lambda: incoming.read(1048576), b""):
                    digest.update(chunk)
            if digest.hexdigest() != row["sha256"]:
                raise ValueError("Project file checksum mismatch")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--netid", required=True)
    parser.add_argument("--prepare-hf-results", action="store_true",
                        help="First step: create results.zip locally; upload it using your HF account")
    parser.add_argument("--hf-results-zip", type=Path, help="Exact prepared local results.zip uploaded to HF")
    parser.add_argument("--hf-repo", help="Your HF dataset ID: USER/REPO (not a URL)")
    parser.add_argument("--hf-revision", help="Full 40-character commit containing results.zip; never main")
    parser.add_argument(
        "--data-root",
        type=Path,
        help="Released canonical RGB/calibration data; defaults to handout/data/scenes",
    )
    parser.add_argument(
        "--without-test",
        action="store_true",
        help="Intentional omission: base halved, no bonuses",
    )
    args = parser.parse_args()
    if args.prepare_hf_results:
        if any(x is not None for x in (args.hf_results_zip, args.hf_repo, args.hf_revision)):
            parser.error("Use preparation and final HF-pointer packaging as separate steps")
        from _hf_results import prepare
        manifest = prepare(args.submission_dir, args.output, args.netid, args.without_test, args.data_root)
        print("PREPARED HF RESULTS", args.output, "SHA256", sha256(args.output))
        print("Upload as results.zip to your own HF dataset; keep the full commit hash.")
        for warning in manifest["warnings"]:
            print("WARNING", warning)
        return
    manifest = package_project(
        args.submission_dir, args.output, args.netid, args.without_test, args.data_root,
        args.hf_results_zip, args.hf_repo, args.hf_revision
    )
    print("PACKAGED", args.output, "SHA256", sha256(args.output))
    for warning in manifest["warnings"]:
        print("WARNING", warning)


if __name__ == "__main__":
    main()
