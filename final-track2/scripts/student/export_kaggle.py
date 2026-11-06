"""Render and export inputs-only predictions to the portable Kaggle CSV.

All geometry is exported from final GLBs. This command never opens Test GT.
Partial --scenes exports are diagnostic only, not accepted by the final board.
"""

import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap
from track2_io import ROOT, validate_scene, scene_ids
from baselines.mesh_output import load_textured_meshes
from evaluation.portable_metric import (
    PROTOCOL,
    PROTOCOL_SHA256,
    array_record,
    image_record,
    geometry_hash,
    encode_payload,
    validate_record,
    payload_rows,
    read_array,
    features_hash,
)
from evaluation.portable_render import prepare_render
from evaluation.local_features import prepare_features


def prediction_record(manifest_path, render_output, data_root, device="cpu"):
    """Validate immutable final meshes, render them, then encode complete geometry."""
    manifest, _ = validate_scene(manifest_path, data_root)
    receipt = prepare_render(
        manifest_path,
        manifest_path.parent / manifest["camera_path"],
        render_output,
        device,
    )
    rendered = {r["id"]: r for r in receipt["objects"]}
    arrays = []
    for entry, mesh in load_textured_meshes(manifest_path):
        identity = entry["instance_id"]
        arrays.append(dict(id=identity, vertices=mesh.vertices, faces=mesh.faces))
    features = prepare_features(arrays, render_output / "features_v3")
    objects = features["objects"]
    parsed = []
    for row in objects:
        row["views"] = [
            image_record(render_output / image["path"])
            for image in rendered[row["id"]]["views"]
        ]
        parsed.append(
            dict(
                id=row["id"],
                bounds=read_array(row["bounds"], "<f8", 2),
                points=read_array(row["points"], "<f4", PROTOCOL["object_points"]),
                collision=dict(
                    status=row["collision"]["status"],
                    occupied=read_array(
                        row["collision"]["occupied"],
                        "<i4",
                        PROTOCOL["collision"]["max_runs_per_object"],
                        4,
                    ),
                    core=read_array(
                        row["collision"]["core"],
                        "<i4",
                        PROTOCOL["collision"]["max_runs_per_object"],
                        4,
                    ),
                ),
            )
        )
    record = dict(
        schema=PROTOCOL["schema"],
        scene_id=manifest["scene_id"],
        protocol_sha256=PROTOCOL_SHA256,
        objects=objects,
        scene_points=features["scene_points"],
        scene_views=[
            image_record(render_output / image["path"])
            for image in receipt["scene_views"]
        ],
        source=dict(
            geometry_sha256=geometry_hash(arrays),
            features_sha256=features_hash(
                parsed,
                read_array(features["scene_points"], "<f4", PROTOCOL["scene_points"]),
            ),
            glb_bytes=sum(
                (manifest_path.parent / r["glb_path"]).stat().st_size
                for r in manifest["meshes"]
            ),
            glb_sha256=[r["sha256"] for r in manifest["meshes"]],
        ),
    )
    validate_record(record, manifest["scene_id"])
    return record


def export(predictions, data_root, output, render_root, identities, device="cpu"):
    """Write one row per scene after a complete validated local export."""
    if output.exists():
        raise FileExistsError("Refusing to replace a previous submission CSV")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Atomic finalization: interrupted rendering never leaves a valid-looking CSV.
    partial = output.with_suffix(output.suffix + ".partial")
    if partial.exists():
        raise FileExistsError(
            "Previous partial export exists; choose a fresh output name"
        )
    if output.suffix != ".csv":
        raise ValueError("Use an uncompressed .csv output for Kaggle submission")
    try:
        with open(partial, "wt", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["Id", "Prediction"])
            for identity in identities:
                manifest = predictions / identity / "evaluation/mesh_manifest.json"
                record = prediction_record(
                    manifest, render_root / identity, data_root, device
                )
                writer.writerows(payload_rows(identity, encode_payload(record)))
                stream.flush()
                print("EXPORTED", identity, flush=True)
        partial.replace(output)
    except BaseException:
        # Keep the partial file for diagnosis; never replace a completed CSV.
        raise
    print("CSV", output, "BYTES", output.stat().st_size, flush=True)


def main():
    """Use the canonical scene list unless an explicit diagnostic subset is selected."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/scenes")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=["dev", "val", "test"], default="test")
    parser.add_argument(
        "--render-root",
        type=Path,
        help="Optional immutable render snapshots; default is <output stem>_cache beside the CSV",
    )
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument(
        "--scenes", nargs="+", help="Diagnostic subset; not a complete final submission"
    )
    args = parser.parse_args()
    identities = args.scenes or scene_ids(args.split)
    if identities != sorted(set(identities)) or not set(identities) <= set(
        scene_ids(args.split)
    ):
        raise ValueError("Scene selection must be canonical, unique and ordered")
    export(
        args.predictions.resolve(),
        args.data_root.resolve(),
        args.output.resolve(),
        (
            args.render_root or args.output.parent / (args.output.stem + "_cache")
        ).resolve(),
        identities,
        args.device,
    )


if __name__ == "__main__":
    main()
