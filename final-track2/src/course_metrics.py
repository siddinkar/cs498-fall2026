"""Course aggregation for the canonical offline/Kaggle reconstruction protocol."""

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from track2_io import scene_ids, ROOT
from evaluation.portable_metric import PROTOCOL, PROTOCOL_SHA256

GROUPS = (
    "matching",
    "scene_geometry",
    "object_geometry",
    "scene_appearance",
    "object_appearance",
)
BASE_POINTS = 20
REPORT_POINTS = 8
REPORT_CRITERION_POINTS = 2
RANK_BONUS_POINTS = 2
NOVELTY_BONUS_POINTS = 2
RESULTS_EVIDENCE_POINTS = 8
EVIDENCE_GROUP_POINTS = 2
EVIDENCE_SCENES_PER_SPLIT = 3
EVIDENCE_SPLITS = ("dev", "val", "test")
EVIDENCE_GROUPS = dict(
    matching="Instances / matching",
    scene="Scene geometry and appearance",
    objects="Object geometry and appearance",
    collision_free="Collision-free evidence",
)
AUTOMATIC_MAXIMA = dict(perception=1.5, reconstruction=1.5, collision_free=1.0)
AUTOMATIC_POINTS = 4
POINT_MAXIMA = dict(results_evidence=RESULTS_EVIDENCE_POINTS, **AUTOMATIC_MAXIMA)


def official_metric_identity():
    """Identify canonical numerical settings and source, not an authenticity signature."""
    folder = ROOT / "src/evaluation"
    return dict(
        protocol_sha256=PROTOCOL_SHA256,
        portable_metric_sha256=hashlib.sha256(
            (folder / "portable_metric.py").read_bytes()
        ).hexdigest(),
    )


def summarize(predictions, split, run_name="default", allow_reference_import=False):
    """Macro-average every fixed scene; missing/failed reports contribute zero.

    The old simulation report schema is never imported as collision-free data.
    Staff preview imports may skip source-receipt checking but not protocol checks.
    """
    rows, missing = [], []
    for identity in scene_ids(split):
        path = (
            Path(predictions)
            / identity
            / "evaluation/reports/evaluation"
            / run_name
            / "metrics.json"
        )
        if not path.is_file():
            missing.append(identity)
            continue
        report = json.loads(path.read_text())
        if (
            report.get("schema") != "cs498.track2.case_metrics.v2"
            or report.get("protocol_sha256") != PROTOCOL_SHA256
        ):
            raise ValueError(
                f"Noncanonical or obsolete evaluation in {path}; re-evaluate with the released protocol"
            )
        if report.get("passed") is not True or report.get("status") not in {
            "complete",
            "empty_prediction",
        }:
            missing.append(identity)
            continue
        if report["scene_id"] != identity:
            raise ValueError(f"Wrong scene in {path}")
        fingerprint = report.get("fingerprint", {})
        if not allow_reference_import:
            if any(
                fingerprint.get(key) != value
                for key, value in official_metric_identity().items()
            ):
                raise ValueError(
                    f"Scorer identity differs in {path}; diagnostic/stale reports cannot be graded"
                )
            signature = hashlib.sha256(
                json.dumps(fingerprint, sort_keys=True).encode()
            ).hexdigest()
            if report.get("signature") != signature:
                raise ValueError(f"Invalid metric receipt in {path}")
        manifest = Path(predictions) / identity / "evaluation/mesh_manifest.json"
        if manifest.is_file():
            from baselines.mesh_output import load_mesh_manifest
            from track2_io import sha256

            current = load_mesh_manifest(manifest)
            if fingerprint.get("prediction_manifest_sha256") != sha256(
                manifest
            ) or fingerprint.get("pred_glb_sha256") != [
                row["sha256"] for row in current["meshes"]
            ]:
                raise ValueError(
                    f"Stale exported GLBs in {path}; evaluate the exact final files"
                )
        groups = report["groups"]
        values = {key: float(groups[key]) for key in (*GROUPS, "collision_free")}
        values["benchmark"] = float(report["final_score"])
        if any(
            not np.isfinite(v) or not 0 <= v <= (100 if k == "benchmark" else 1)
            for k, v in values.items()
        ):
            raise ValueError(f"Out-of-range evaluation in {path}")
        rows.append(dict(scene_id=identity, **values))
    total = len(scene_ids(split))
    keys = (*GROUPS, "collision_free", "benchmark")
    means = {key: math.fsum(row[key] for row in rows) / total for key in keys}
    return dict(
        schema="cs498.track2.split_summary.v2",
        split=split,
        expected=total,
        completed=len(rows),
        missing=missing,
        means=means,
        cases=rows,
        protocol_sha256=PROTOCOL_SHA256,
        collision_definition=PROTOCOL["collision"]["score"],
    )


def performance_points(val, test, reference):
    """Four metric points, split equally across full Val/Test macro-averages.

    Evidence/report points are independent of baseline attainment. With no Test,
    assess Val, then the outer grader halves the whole base once.
    """
    for summary in [val, test]:
        if summary is not None and (
            summary.get("schema") != "cs498.track2.split_summary.v2"
            or summary.get("protocol_sha256") != PROTOCOL_SHA256
        ):
            raise ValueError(
                "Course points require the current collision-free protocol"
            )
    if reference.get("protocol_sha256") != PROTOCOL_SHA256:
        raise ValueError("Reference is not from the current collision-free protocol")

    def credit(value):
        return min(1.0, max(0.0, float(value) / 0.75))

    parts = dict.fromkeys(AUTOMATIC_MAXIMA, 0.0)
    for summary in [val, test if test is not None else val]:
        means = summary["means"]
        parts["perception"] += 0.75 * credit(means["matching"])
        parts["reconstruction"] += math.fsum(
            0.1875 * credit(means[key]) for key in GROUPS if key != "matching"
        )
        parts["collision_free"] += 0.5 * credit(means["collision_free"])
    return parts
