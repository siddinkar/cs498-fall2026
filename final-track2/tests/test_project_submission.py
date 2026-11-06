"""One outer ZIP, immutable inner Test bundle, truthful partial evidence."""

import copy
import csv
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import pytest
import trimesh

from course_metrics import summarize, performance_points
from package_submission import package, inspect_result_bundle
from project_submission import package_project, inspect_project_archive, safe_name
from track2_io import export_scene, scene_ids, write_json, sha256
from test_student_interface import input_scene
from evaluation.portable_metric import (
    PROTOCOL_SHA256,
    PROTOCOL,
    CHUNKS_PER_SCENE,
    array_record,
    decode_payload,
    empty_prediction,
    encode_payload,
    features_hash,
    geometry_hash,
    payload_rows,
    payloads_from_frame,
    read_array,
    validate_record,
)
from kaggle_receipt import create as create_kaggle_receipt, validate_csv


def inert_prediction(identity, manifest_path):
    """Valid schema/source fixture; no claim of renderer or surface provenance."""
    manifest = json.loads(manifest_path.read_text())
    mesh = manifest["meshes"][0]
    bounds = np.asarray(mesh["bbox_dataset_world_m"], dtype="<f8")
    points = np.tile(bounds[0], (PROTOCOL["object_points"], 1)).astype("<f4")
    scene_points = np.tile(bounds[0], (PROTOCOL["scene_points"], 1)).astype("<f4")
    runs = np.empty((0, 4), "<i4")
    parsed = dict(
        id=mesh["instance_id"],
        bounds=bounds,
        points=points,
        collision=dict(status="complete", occupied=runs, core=runs),
    )
    record = empty_prediction(identity)
    record["objects"] = [
        dict(
            id=parsed["id"],
            bounds=array_record(bounds, "<f8"),
            points=array_record(points, "<f4"),
            collision=dict(
                status="complete",
                occupied=array_record(runs, "<i4"),
                core=array_record(runs, "<i4"),
            ),
            views=record["scene_views"][:6],
        )
    ]
    record["scene_points"] = array_record(scene_points, "<f4")
    box = trimesh.creation.box()
    record["source"] = dict(
        geometry_sha256=geometry_hash(
            [dict(id=parsed["id"], vertices=box.vertices, faces=box.faces)]
        ),
        features_sha256=features_hash([parsed], scene_points),
        glb_bytes=(manifest_path.parent / mesh["glb_path"]).stat().st_size,
        glb_sha256=[mesh["sha256"]],
    )
    return record


def staged_project(tmp_path, without_test=False):
    root = tmp_path / "submission"
    (root / "code").mkdir(parents=True)
    (root / "report.pdf").write_bytes(b"%PDF-1.4\nformat-fixture\n")
    (root / "README.md").write_text(
        "Final method: example. Run the command in code/README.md."
    )
    (root / "code/README.md").write_text("python method.py --config final_config.json")
    (root / "code/method.py").write_text('"""Fixture source snapshot."""\n')
    (root / "code/requirements.txt").write_text("numpy\n")
    write_json(root / "code/final_config.json", dict(seed=498))
    write_json(root / "models/manifest.json", dict(required=False, checkpoints=[]))
    (root / "ai_collaboration.jsonl").write_text(
        json.dumps(
            dict(
                task="Investigate duplicate instances",
                verification="Compared all Val cases",
            )
        )
        + "\n"
    )
    write_json(
        root / "evaluation/val_summary.json", summarize(root / "evaluation/val", "val")
    )
    selection = dict(
        schema="cs498.track2.evidence_selection.v1",
        final_method="example",
        final_config="code/final_config.json",
        dev=scene_ids("dev")[:3],
        val=scene_ids("val")[:3],
        test=[] if without_test else scene_ids("test")[:3],
    )
    write_json(root / "evaluation/evidence_selection.json", selection)
    data = tmp_path / "data"
    if not without_test:
        for identity in scene_ids("test"):
            scene = input_scene(data, identity)
            export_scene(
                scene,
                {1: trimesh.creation.box()},
                tmp_path / "predictions" / identity / "evaluation",
            )
        package(tmp_path / "predictions", data, root / "test_results.zip")
        with (root / "kaggle_submission.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["Id", "Prediction"])
            for identity in scene_ids("test"):
                writer.writerows(
                    payload_rows(
                        identity,
                        encode_payload(
                            inert_prediction(
                                identity,
                                tmp_path / "predictions" / identity / "evaluation/mesh_manifest.json",
                            )
                        ),
                    )
                )
        create_kaggle_receipt(
            root / "kaggle_submission.csv", 1, root / "kaggle_receipt.json"
        )
    return root, data


def test_outer_deterministic_receipts_and_missing_evidence_warnings(tmp_path):
    root, data = staged_project(tmp_path)
    record = package_project(root, tmp_path / "a.zip", "netid", data_root=data)
    package_project(root, tmp_path / "b.zip", "netid", data_root=data)
    assert (tmp_path / "a.zip").read_bytes() == (tmp_path / "b.zip").read_bytes()
    checked = inspect_project_archive(tmp_path / "a.zip")
    assert checked["test_results_sha256"] == record["test_results_sha256"]
    assert checked["warnings"] and not checked["missing_test"]
    with zipfile.ZipFile(tmp_path / "a.zip") as zipped:
        assert {
            "report.pdf",
            "code/method.py",
            "test_results.zip",
            "submission_manifest.json",
        } <= set(zipped.namelist())
        assert all(
            zipped.getinfo(name).date_time == (2026, 1, 1, 0, 0, 0)
            for name in zipped.namelist()
        )


@pytest.mark.parametrize("without_test", [False, True])
def test_hf_results_split_keeps_source_private_and_pins_exact_results(tmp_path, without_test):
    from _hf_results import prepare, inspect, validate_location

    root, data = staged_project(tmp_path, without_test)
    results = tmp_path / "results.zip"
    manifest = prepare(root, results, "netid", without_test, data)
    assert inspect(results) == manifest
    with zipfile.ZipFile(results) as zipped:
        assert "evaluation/val_summary.json" in zipped.namelist()
        assert not {"report.pdf", "code/method.py", "ai_collaboration.jsonl", "models/manifest.json"} & set(zipped.namelist())
    small = tmp_path / "course.zip"
    receipt = package_project(root, small, "netid", without_test, data,
                              results, "student/track2-results", "a" * 40)
    assert receipt["schema"] == "cs498.track2.project_submission.v2"
    assert inspect_project_archive(small) == receipt
    with zipfile.ZipFile(small) as zipped:
        assert {"report.pdf", "code/method.py", "models/manifest.json", "results_location.json"} <= set(zipped.namelist())
        assert not {"test_results.zip", "kaggle_submission.csv", "evaluation/val_summary.json"} & set(zipped.namelist())
        pointer = validate_location(json.loads(zipped.read("results_location.json")))
        assert pointer["sha256"] == sha256(results)
        assert pointer["revision"] == "a" * 40


@pytest.mark.parametrize("field,value", [("revision", "main"), ("revision", "abcd"),
    ("repo_id", "https://example.com/data"), ("repo_id", "student/../secret"),
    ("filename", "../results.zip"), ("bytes", True), ("sha256", "oops")])
def test_hf_location_rejects_moving_revisions_urls_and_unsafe_fields(field, value):
    from _hf_results import validate_location
    record = dict(schema="cs498.track2.results_location.v1", repo_type="dataset",
        repo_id="student/results", revision="a" * 40, filename="results.zip",
        bytes=123, sha256="b" * 64)
    record[field] = value
    with pytest.raises(ValueError, match="HF result location"):
        validate_location(record)


def test_hf_stale_source_and_wrong_student_rejected(tmp_path):
    from _hf_results import prepare
    root, data = staged_project(tmp_path, True)
    results = tmp_path / "results.zip"
    prepare(root, results, "netid", True, data)
    with pytest.raises(ValueError, match="stale"):
        package_project(root, tmp_path / "wrong.zip", "another_student", True, data,
                        results, "student/results", "a" * 40)
    write_json(root / "evaluation/extra.json", {"new_output": True})
    with pytest.raises(ValueError, match="stale"):
        package_project(root, tmp_path / "stale.zip", "netid", True, data,
                        results, "student/results", "a" * 40)


def test_hf_result_archive_cannot_smuggle_source_or_traversal(tmp_path):
    from _hf_results import prepare, inspect
    root, data = staged_project(tmp_path, True)
    results = tmp_path / "results.zip"
    prepare(root, results, "netid", True, data)
    for forbidden in ["code/secret.py", "../escape", "models/manifest.json"]:
        with zipfile.ZipFile(results) as source:
            blobs = {n: source.read(n) for n in source.namelist()}
        record = json.loads(blobs["results_manifest.json"])
        import hashlib
        record["files"].append(dict(path=forbidden, bytes=1, sha256=hashlib.sha256(b"x").hexdigest()))
        blobs["results_manifest.json"] = json.dumps(record).encode()
        blobs[forbidden] = b"x"
        bad = tmp_path / ("bad_" + str(len(forbidden)) + ".zip")
        with zipfile.ZipFile(bad, "w") as target:
            for n, content in blobs.items():
                target.writestr(n, content)
        with pytest.raises(ValueError):
            inspect(bad)


def test_intentional_without_test_uses_empty_selection(tmp_path):
    root, data = staged_project(tmp_path, True)
    record = package_project(root, tmp_path / "no_test.zip", "netid", True, data)
    assert record["missing_test"] and record["test_results_sha256"] is None
    assert record["selected_scenes"]["test"] == []
    inspect_project_archive(tmp_path / "no_test.zip")
    empty = summarize(root / "evaluation/val", "val")
    reference = dict(protocol_sha256=PROTOCOL_SHA256)
    assert all(
        value == 0 for value in performance_points(empty, None, reference).values()
    )


def test_nonfinite_val_summary_and_staging_symlink_rejected(tmp_path):
    root, data = staged_project(tmp_path, True)
    path = root / "evaluation/val_summary.json"
    record = json.loads(path.read_text())
    record["means"]["matching"] = float("nan")
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="Val means"):
        package_project(root, tmp_path / "bad.zip", "netid", True, data)
    assert not (tmp_path / "bad.zip").exists()
    write_json(path, summarize(root / "evaluation/val", "val"))
    (root / "code/linked.py").symlink_to(root / "code/method.py")
    with pytest.raises(ValueError, match="symlinks"):
        package_project(root, tmp_path / "bad.zip", "netid", True, data)


def test_stale_test_preview_is_invalid_evidence_not_result_rejection(tmp_path):
    root, data = staged_project(tmp_path)
    write_json(
        root / "evaluation/test/test_scene_000/prediction_preview/preview.json",
        dict(scene_id="test_scene_001"),
    )
    record = package_project(root, tmp_path / "warning.zip", "netid", data_root=data)
    assert any("stale or invalid" in warning for warning in record["warnings"])


@pytest.mark.parametrize("receipt", [[], None, 42])
def test_malformed_preview_is_warning_not_packaging_failure(tmp_path, receipt):
    root, data = staged_project(tmp_path)
    write_json(
        root / "evaluation/test/test_scene_000/prediction_preview/preview.json", receipt
    )
    record = package_project(root, tmp_path / "warning.zip", "netid", data_root=data)
    assert any(
        "malformed prediction-preview" in warning for warning in record["warnings"]
    )


def test_inner_receipts_do_not_replace_canonical_geometry_validation(tmp_path):
    root, data = staged_project(tmp_path)
    source = root / "test_results.zip"
    with zipfile.ZipFile(source) as zipped:
        files = {info.filename: zipped.read(info) for info in zipped.infolist()}
    files["test_scene_000/evaluation/meshes/obj_001.glb"] = b"not a glb"
    receipt = json.loads(files["bundle_manifest.json"])
    import hashlib

    for row in receipt["files"]:
        row.update(
            bytes=len(files[row["path"]]),
            sha256=hashlib.sha256(files[row["path"]]).hexdigest(),
        )
    files["bundle_manifest.json"] = json.dumps(receipt).encode()
    with zipfile.ZipFile(source, "w") as zipped:
        for name, blob in files.items():
            zipped.writestr(name, blob)
    inspect_result_bundle(source)  # Container integrity alone is not model validity.
    with pytest.raises(ValueError):
        package_project(root, tmp_path / "bad.zip", "netid", data_root=data)


def rewrite_last_prediction(root, fault):
    """Keep the outer CSV checksum honest while changing a late-scene payload."""
    path = root / "kaggle_submission.csv"
    frame = pd.read_csv(path)
    start = 19 * CHUNKS_PER_SCENE
    identity, encoded = next(payloads_from_frame(frame.iloc[start:]))
    record = decode_payload(encoded)
    if fault == "empty":
        record = empty_prediction(identity)
    elif fault == "source":
        record["source"] = {}
    elif fault == "hash":
        record["source"]["glb_sha256"] = ["0" * 64]
    elif fault == "bytes":
        record["source"]["glb_bytes"] += 1
    elif fault == "object_ids":
        parsed = validate_record(record, identity)
        record["objects"][0]["id"] = parsed[0]["id"] = 2
        record["source"]["features_sha256"] = features_hash(
            parsed, read_array(record["scene_points"], "<f4", PROTOCOL["scene_points"])
        )
    encoded = "not-base64" if fault == "malformed" else encode_payload(record)
    frame.loc[start:, "Prediction"] = [v for _, v in payload_rows(identity, encoded)]
    frame.to_csv(path, index=False)
    receipt_path = root / "kaggle_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt.update(csv_sha256=sha256(path), csv_bytes=path.stat().st_size)
    write_json(receipt_path, receipt)


@pytest.mark.parametrize("fault", ["empty", "source", "hash", "bytes", "object_ids", "malformed"])
def test_final_packer_rejects_payload_or_source_mismatch_on_all20(tmp_path, fault):
    root, data = staged_project(tmp_path)
    rewrite_last_prediction(root, fault)
    expected = (
        "source mismatch" if fault in {"empty", "hash", "bytes", "object_ids"}
        else "Missing final-geometry" if fault == "source"
        else "Malformed bounded scene payload"
    )
    with pytest.raises(ValueError, match=expected):
        package_project(root, tmp_path / "bad.zip", "netid", data_root=data)
    assert not (tmp_path / "bad.zip").exists()


def test_receipt_rejects_malformed_payload_before_writing(tmp_path):
    root, _ = staged_project(tmp_path)
    rewrite_last_prediction(root, "malformed")
    output = tmp_path / "bad_receipt.json"
    with pytest.raises(ValueError, match="Malformed bounded scene payload"):
        create_kaggle_receipt(root / "kaggle_submission.csv", 1, output)
    assert not output.exists()


@pytest.mark.parametrize(
    "path",
    [
        "../report.pdf",
        "/report.pdf",
        "code/../report.pdf",
        "code/.git/config",
        "code/credentials/hf",
        "code/.env",
        "evaluation//bad",
    ],
)
def test_unsafe_archive_names_rejected(path):
    with pytest.raises(ValueError):
        safe_name(path)
