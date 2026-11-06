"""Record the exact final CSV and its actual Kaggle submission ID, without a token."""

import argparse
import json
from pathlib import Path
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap
from track2_io import scene_ids, sha256
from evaluation.portable_metric import (
    CHUNKS_PER_SCENE,
    PROTOCOL_SHA256,
    decode_payload,
    payloads_from_frame,
    validate_record,
    _decoded_image,
)


def bundle_sources(bundle):
    """Read source identities from a checksum-verified ZIP, without parsing meshes.

    This checks archive/source consistency only. It does not certify that local
    images, points or voxels were generated from those GLBs.
    """
    from package_submission import inspect_result_bundle

    receipt = inspect_result_bundle(bundle)
    files = {row["path"]: row for row in receipt["files"]}
    sources = {}
    with zipfile.ZipFile(bundle) as archive:
        for identity in scene_ids("test"):
            prefix = f"{identity}/evaluation/"
            manifest = json.loads(archive.read(prefix + "mesh_manifest.json"))
            if not isinstance(manifest, dict):
                raise ValueError("Inner Test manifest must be a JSON object")
            meshes = manifest.get("meshes")
            if manifest.get("scene_id") != identity or not isinstance(meshes, list):
                raise ValueError("Inner Test manifest has the wrong scene identity")
            ids, hashes, paths, size = [], [], [], 0
            for row in meshes:
                if not isinstance(row, dict) or not {
                    "instance_id", "glb_path", "sha256"
                } <= set(row):
                    raise ValueError("Inner Test manifest has an invalid object record")
                object_id = row["instance_id"]
                if (
                    type(object_id) is not int
                    or not 1 <= object_id <= 32767
                    or row["glb_path"] != f"meshes/obj_{object_id:03d}.glb"
                ):
                    raise ValueError("Inner Test manifest has invalid object IDs/paths")
                path = prefix + row["glb_path"]
                if path not in files or row["sha256"] != files[path]["sha256"]:
                    raise ValueError("Inner Test manifest differs from its actual GLBs")
                ids.append(object_id)
                hashes.append(files[path]["sha256"])
                paths.append(path)
                size += files[path]["bytes"]
            actual = {
                path for path in files if path.startswith(prefix + "meshes/")
            }
            if not ids or ids != sorted(set(ids)) or set(paths) != actual:
                raise ValueError("Inner Test manifest has the wrong GLB inventory")
            sources[identity] = dict(ids=ids, hashes=hashes, bytes=size)
    return sources


def validate_csv(csv_path, bundle=None):
    """Decode every canonical Test payload; optionally bind its source to the ZIP."""
    import pandas as pd

    if csv_path.suffix != ".csv" or not csv_path.is_file():
        raise ValueError("Use the exact plain CSV uploaded to Kaggle")
    sources = bundle_sources(bundle) if bundle is not None else None
    identities = scene_ids("test")
    count = 0
    for index, frame in enumerate(pd.read_csv(csv_path, chunksize=CHUNKS_PER_SCENE)):
        if list(frame.columns) != ["Id", "Prediction"]:
            raise ValueError("Expected exactly Id,Prediction CSV columns")
        if index >= len(identities) or frame.Id.tolist() != [
            f"{identities[index]}__{part:04d}" for part in range(CHUNKS_PER_SCENE)
        ]:
            raise ValueError(
                "The final CSV needs every Test chunk row in the canonical order"
            )
        identity, encoded = next(payloads_from_frame(frame))
        prediction = decode_payload(encoded)
        try:
            rows = validate_record(prediction, identity)
        finally:
            # Match evaluate_scene's per-case cache lifetime during validation.
            _decoded_image.cache_clear()
        if sources is not None:
            expected = sources[identity]
            if (
                prediction["source"]["glb_sha256"] != expected["hashes"]
                or prediction["source"]["glb_bytes"] != expected["bytes"]
                or [row["id"] for row in rows] != expected["ids"]
            ):
                raise ValueError(
                    f"{identity}: CSV/GLB source mismatch in hashes, bytes or object IDs"
                )
        count += 1
    if count != len(identities):
        raise ValueError(
            "The final CSV needs every Test chunk row in the canonical order"
        )


def create(csv_path, submission_id, output):
    """Write traceability metadata; official submission status is checked on Kaggle."""
    validate_csv(csv_path)
    if type(submission_id) is not int or submission_id <= 0:
        raise ValueError(
            "Use the positive numeric ID on the Kaggle submission details page"
        )
    if output.exists():
        raise FileExistsError(
            "Receipt already exists; do not silently replace a final submission identity"
        )
    record = dict(
        schema="cs498.track2.kaggle_receipt.v1",
        competition="cs498-fall-2026-track2-reconstruction",
        submission_id=submission_id,
        csv_sha256=sha256(csv_path),
        csv_bytes=csv_path.stat().st_size,
        protocol_sha256=PROTOCOL_SHA256,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        stream.write(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--submission-id", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    create(args.csv, args.submission_id, args.output)
