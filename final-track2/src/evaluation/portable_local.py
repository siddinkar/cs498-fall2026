"""Canonical Dev/Val evaluation with local rendering and the Kaggle CPU scorer.

Test inputs have no GT. This module never fetches GT or a remote metric result.
The staff target-preparation command uses the same renderer and target schema.
"""

import copy
import hashlib
import html
import json
from pathlib import Path
import sys
import time

import numpy as np

from evaluation.portable_metric import (
    PROTOCOL,
    PROTOCOL_SHA256,
    array_record,
    evaluate_scene,
    geometry_hash,
    image_record,
    validate_record,
    read_array,
    features_hash,
)
from evaluation.local_features import prepare_features
from evaluation.portable_render import prepare_render, file_hash
from track2_io import ROOT, scene_path, validate_scene


def target_record(private_gt, public_inputs, identity, output, device):
    """Build a trusted target from an available Dev/Val GT scene (staff: Test)."""
    scene = scene_path(private_gt, identity)
    calibration = scene_path(public_inputs, identity) / "cameras.npz"
    receipt = prepare_render(
        None,
        calibration,
        output,
        device=device,
        gt_blend=scene / "scene.blend",
        scene_id=identity,
    )
    arrays = []
    for row in receipt["objects"]:
        with np.load(output / row["geometry"], allow_pickle=False) as geometry:
            vertices, faces = geometry["vertices"].copy(), geometry["faces"].copy()
        arrays.append(dict(id=row["id"], vertices=vertices, faces=faces))
    features = prepare_features(arrays, output / "features_v3")
    objects, parsed = features["objects"], []
    rendered = {r["id"]: r for r in receipt["objects"]}
    for row in objects:
        row["views"] = [
            image_record(output / image["path"])
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
        scene_id=identity,
        protocol_sha256=PROTOCOL_SHA256,
        objects=objects,
        scene_points=features["scene_points"],
        scene_views=[
            image_record(output / image["path"]) for image in receipt["scene_views"]
        ],
        support_ids=[1],
        floor_z=receipt["floor_z"],
        source=dict(
            geometry_sha256=geometry_hash(arrays),
            glb_bytes=0,
            glb_sha256=[],
            features_sha256=features_hash(
                parsed,
                read_array(features["scene_points"], "<f4", PROTOCOL["scene_points"]),
            ),
        ),
    )
    validate_record(record, identity, target=True)
    return record


def evaluate(manifest_path, data_root, run_name="default", device="cpu"):
    """Write immutable numeric and side-by-side image evidence for one public case."""
    # Import the student exporter rather than duplicate the final-GLB adapter.
    sys.path.insert(0, str(ROOT / "scripts/student"))
    from export_kaggle import prediction_record

    manifest_path, data_root = Path(manifest_path).resolve(), Path(data_root).resolve()
    manifest, _ = validate_scene(manifest_path, data_root)
    identity = manifest["scene_id"]
    if identity.startswith("test_"):
        raise ValueError(
            "Test GT is private. Export predictions and submit them to Kaggle."
        )
    if not run_name or any(
        char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
        for char in run_name
    ):
        raise ValueError(
            "run-name must contain only letters, digits, underscore or hyphen"
        )
    output = manifest_path.parent / "reports/evaluation" / run_name
    if (output / "metrics.json").exists():
        raise FileExistsError(
            "Completed evaluation exists; use a new run name for a new result"
        )
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    target = target_record(
        data_root, data_root, identity, output / "gt_renders", device
    )
    prediction = prediction_record(
        manifest_path, output / "pred_renders", data_root, device
    )
    result = evaluate_scene(target, prediction, identity)
    result.update(
        schema="cs498.track2.case_metrics.v2",
        protocol_sha256=PROTOCOL_SHA256,
        passed=True,
        score=dict(
            groups=copy.deepcopy(result["groups"]), final_score=result["final_score"]
        ),
        fingerprint=dict(
            protocol_sha256=PROTOCOL_SHA256,
            portable_metric_sha256=file_hash(
                Path(__file__).with_name("portable_metric.py")
            ),
            prediction_manifest_sha256=file_hash(manifest_path),
            pred_glb_sha256=[row["sha256"] for row in manifest["meshes"]],
        ),
        wall_seconds=time.perf_counter() - started,
    )
    result["signature"] = hashlib.sha256(
        json.dumps(result["fingerprint"], sort_keys=True).encode()
    ).hexdigest()
    (output / "matching.json").write_text(
        json.dumps(result.get("matching", {}), indent=2) + "\n"
    )
    from evaluation.portable_visuals import make_visuals

    visualizations = make_visuals(target, prediction, result, output)
    visualizations.update(
        dict(
            scene_appearance=dict(
                images=[f"pred_renders/scene/view_{i:02d}.png" for i in range(12)]
            ),
            object_appearance=dict(
                images=[
                    f"pred_renders/object_{p:03d}/view_{i:02d}.png"
                    for _, p in result.get("matching", {}).get("pairs", [])
                    for i in range(6)
                ]
            ),
        )
    )
    (output / "visualizations.json").write_text(
        json.dumps(visualizations, indent=2) + "\n"
    )
    # Arrays/images are evidence, not student-entered scalar metrics. The report
    # links full mesh/collision details and original normalized view PNGs.
    pairs = result.get("matching", {}).get("pairs", [])
    sections = []
    for key in ["matching", "scene_geometry", "collision_free"]:
        image = visualizations[key]["image"]
        sections.append(
            f'<details open><summary>{html.escape(key.replace("_", " ").title())}: {result["groups"][key]:.3f}</summary><img class="metric" src="{image}" loading="lazy"></details>'
        )
    sections.append("<details><summary>Scene appearance: fixed world cameras</summary>")

    def view_pair(left, right, label):
        return f'<figure><figcaption>{html.escape(label)}</figcaption><div class="pair"><img src="{left}" alt="Ground truth" loading="lazy"><img src="{right}" alt="Prediction" loading="lazy"></div></figure>'

    for index in range(PROTOCOL["render"]["scene_views"]):
        sections.append(
            view_pair(
                f"gt_renders/scene/view_{index:02d}.png",
                f"pred_renders/scene/view_{index:02d}.png",
                f"Scene view {index}",
            )
        )
    sections.append(
        "</details><h2>Objects: world-space geometry and independently AABB-framed appearance</h2>"
    )
    for gid, pid in pairs:
        sections.append(f"<details><summary>GT {gid} / prediction {pid}</summary>")
        geometry = next(
            row
            for row in visualizations["objects"]
            if row["gt_id"] == gid and row["pred_id"] == pid
        )
        sections.append(
            f'<img class="metric" src="{geometry["geometry"]["image"]}" loading="lazy">'
        )
        for index in range(6):
            sections.append(
                view_pair(
                    f"gt_renders/object_{gid:03d}/view_{index:02d}.png",
                    f"pred_renders/object_{pid:03d}/view_{index:02d}.png",
                    f"GT {gid} / prediction {pid}, view {index}",
                )
            )
        sections.append("</details>")
    matched_gt = {gid for gid, _ in pairs}
    matched_pred = {pid for _, pid in pairs}
    sections.append(
        "<h2>Unmatched instances</h2><p>GT misses: "
        + str([row["id"] for row in target["objects"] if row["id"] not in matched_gt])
        + "; prediction extras: "
        + str(
            [
                row["id"]
                for row in prediction["objects"]
                if row["id"] not in matched_pred
            ]
        )
        + ". Missing objects retain their coverage penalty.</p>"
    )
    (output / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Track 2 evaluation</title>'
        "<style>body{font:16px system-ui;max-width:1100px;margin:30px auto}img{width:49%;background:white}.metric{width:100%}.pair{display:flex;gap:1%}figure{margin:20px 0}pre{overflow:auto}details{margin:16px 0;border-top:1px solid #ddd}summary{padding:10px;cursor:pointer}</style>"
        f"<h1>{html.escape(identity)}</h1><p>Left: GT. Right: prediction. No physics simulation.</p>"
        '<p><a href="metrics.json">Complete numeric, instance-matching and collision evidence</a></p>'
        "<pre>"
        + html.escape(json.dumps(result["groups"], indent=2))
        + "</pre>"
        + "".join(sections)
    )
    result["visualizations"] = visualizations
    # Commit marker comes last: incomplete plotting can resume cached renders.
    (output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
