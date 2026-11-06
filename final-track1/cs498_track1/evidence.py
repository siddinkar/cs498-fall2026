"""Bind report images to the final method and canonical validation summary.

These receipts prevent accidental stale packaging, not deliberate falsification.
Students remain responsible for the report; staff independently rerun methods.
"""

import json
from pathlib import Path

from .protocol import sha256, tree_hash, write_json

FIGURES = ("inputs.png", "completion.png", "comparison.png", "class_iou.png")


def figure_state(source, evaluation, output):
    """Hash the exact method, measured Val summary and four required images."""
    summary_path = Path(evaluation) / "summary.json"
    summary = json.loads(summary_path.read_text())
    digest = tree_hash(source)
    if summary["submission_tree_sha256"] != digest:
        raise ValueError("Stale validation: rerun evaluate after code or weight edits")
    images = {}
    for filename in FIGURES:
        path = Path(output) / filename
        if not path.is_file() or not path.stat().st_size:
            raise FileNotFoundError(f"Missing report figure: {path}")
        images[filename] = sha256(path)
    return dict(
        schema_version=1,
        submission_tree_sha256=digest,
        validation_summary_sha256=sha256(summary_path),
        figures_sha256=images,
    )


def record_figures(source, evaluation, output):
    """Record freshly rendered figures (provided or authored by the student)."""
    state = figure_state(source, evaluation, output)
    write_json(Path(output) / "evidence_manifest.json", state)
    return state


def validate_figures(source, evaluation, output):
    """Reject modified images or evidence from an earlier method/evaluation."""
    path = Path(output) / "evidence_manifest.json"
    if not path.is_file():
        raise ValueError("Missing figure receipt: run python run.py figures")
    if json.loads(path.read_text()) != figure_state(source, evaluation, output):
        raise ValueError("Stale report figures: regenerate figures and rebuild report")
