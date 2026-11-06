"""Rehearse the Kaggle CSV contract locally using public Val ground truth.

Use a one-scene diagnostic first, then all ten Val scenes. This exercises the
same decoder and numerical scorer as Kaggle without using a Test submission.
It is a format/numeric rehearsal, not certification of algorithm provenance.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap
from track2_io import ROOT, scene_ids, validate_scene
from evaluation.portable_local import target_record
from evaluation.portable_metric import (
    PROTOCOL_SHA256,
    decode_payload,
    payloads_from_frame,
    evaluate_scene,
    CHUNKS_PER_SCENE,
    ParticipantVisibleError,
)
from evaluation.portable_render import file_hash


def check(csv_path, data_root, output, identities, device="cpu", predictions=None):
    """Validate every row and final artifact identity before public-Val scoring."""
    if (
        not identities
        or identities != sorted(set(identities))
        or not set(identities) <= set(scene_ids("val"))
    ):
        raise ValueError("This local rehearsal accepts ordered Val scenes only")
    if csv_path.suffix != ".csv":
        raise ValueError("Use the generated plain .csv, not .gz or .partial")
    if output.exists():
        raise FileExistsError("Keep the previous check; choose a fresh --output")
    frame = pd.read_csv(csv_path)
    expected = [
        f"{name}__{i:04d}" for name in identities for i in range(CHUNKS_PER_SCENE)
    ]
    if list(frame.columns) != ["Id", "Prediction"] or frame.Id.tolist() != expected:
        raise ParticipantVisibleError(
            "CSV columns/row IDs differ from the selected Val inventory"
        )
    metric = ROOT / "src/evaluation/portable_metric.py"
    started = time.perf_counter()
    cases = []
    for identity, encoded in payloads_from_frame(frame):
        prediction = decode_payload(encoded)
        if predictions is not None:
            manifest_path = predictions / identity / "evaluation/mesh_manifest.json"
            manifest, _ = validate_scene(manifest_path, data_root)
            hashes = [row["sha256"] for row in manifest["meshes"]]
            size = sum(
                (manifest_path.parent / row["glb_path"]).stat().st_size
                for row in manifest["meshes"]
            )
            if (
                prediction.get("source", {}).get("glb_sha256") != hashes
                or prediction["source"].get("glb_bytes") != size
            ):
                raise ParticipantVisibleError(
                    "CSV differs from the final validated Val GLBs"
                )
        # Public Val targets are available through the documented downloader.
        # The shared content-addressed cache avoids re-rendering an unchanged GT.
        target = target_record(
            data_root,
            data_root,
            identity,
            output.parent / (output.stem + "_cache") / identity,
            device,
        )
        result = evaluate_scene(target, prediction, identity)
        result["submission_format"] = dict(
            decoded_keys=sorted(prediction),
            scene_points=prediction["scene_points"]["shape"],
            scene_views=len(prediction["scene_views"]),
            objects=[
                dict(
                    id=row["id"],
                    bounds_shape=row["bounds"]["shape"],
                    points_shape=row["points"]["shape"],
                    object_views=len(row["views"]),
                    occupied_runs=row["collision"]["occupied"]["shape"][0],
                    core_runs=row["collision"]["core"]["shape"][0],
                    voxel_preprocessing_status=row["collision"]["status"],
                )
                for row in prediction["objects"]
            ],
        )
        cases.append(result)
        print(identity, f"score={result['final_score']:.5f}", "FORMAT OK", flush=True)
    receipt = dict(
        schema="cs498.track2.val_csv_check.v1",
        passed=True,
        protocol_sha256=PROTOCOL_SHA256,
        metric_sha256=file_hash(metric),
        csv_sha256=file_hash(csv_path),
        csv_bytes=csv_path.stat().st_size,
        scenes=identities,
        rows=len(frame),
        csv_columns=["Id", "Prediction"],
        rows_per_scene=CHUNKS_PER_SCENE,
        payload_encoding="JSON -> zlib -> Base64 -> fixed-size chunk rows (transport, not encryption)",
        complete_val=identities == scene_ids("val"),
        score=float(np.mean([row["final_score"] for row in cases])),
        cases=cases,
        wall_seconds=time.perf_counter() - started,
        test_gt_accessed=False,
        note="Same CSV decoder and numerical scorer as Kaggle; Val only. Not proof of feature provenance.",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2) + "\n")
    print("VALID VAL CSV", f"mean={receipt['score']:.5f}", "receipt:", output)
    return receipt


def main():
    """Use only Val IDs; Test scoring must never be simulated using local GT."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/scenes")
    parser.add_argument(
        "--predictions",
        type=Path,
        help="Also validate the final GLBs and compare their source hashes",
    )
    parser.add_argument(
        "--scenes", nargs="+", help="Diagnostic Val subset; omit for all ten Val scenes"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    args = parser.parse_args()
    identities = args.scenes or scene_ids("val")
    if identities != sorted(set(identities)) or not set(identities) <= set(
        scene_ids("val")
    ):
        raise ValueError("This local rehearsal accepts ordered Val scenes only")
    check(
        args.csv.resolve(),
        args.data_root.resolve(),
        args.output.resolve(),
        identities,
        args.device,
        args.predictions.resolve() if args.predictions else None,
    )


if __name__ == "__main__":
    main()
