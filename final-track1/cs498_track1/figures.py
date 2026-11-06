"""Render required evidence from the final method's canonical Val predictions."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from .dataset import load_inputs, load_target
from .protocol import CLASSES, tree_hash
from .evidence import record_figures
from matplotlib.patches import Patch
from .visuals import COLORS, comparison_images, save, style


def comparison(fig, values, mask, completion, y, height, titles):
    """Use equal-sized, tightly framed images rather than padded 3D subplots."""
    images = comparison_images(values, mask, completion)
    for column, (image, title) in enumerate(zip(images, titles)):
        x = 0.012 + column * 0.5
        ax = fig.add_axes([x, y, 0.476, height])
        ax.imshow(image)
        ax.set_axis_off()
        fig.text(
            x + 0.238, y + height + 0.025, title, ha="center", va="bottom", fontsize=9
        )


def semantic_key(fig):
    """Keep the full class-ID color key and visibility explanation readable."""
    names = [
        "Floor",
        "Walls/ceiling",
        "Opening",
        "Furniture",
        "Appliance",
        "Plumbing fixture",
        "Electronics",
        "Small object",
        "Other",
    ]
    fig.legend(
        [Patch(facecolor=COLORS[c]) for c in range(1, 10)],
        names,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=5,
        frameon=False,
        fontsize=7,
        handlelength=1.0,
        columnspacing=1.3,
    )


def generate(root, evaluation, output, source):
    """Render a fixed illustrative Val sample and measured high/low-scoring examples."""
    style()
    summary = json.loads((evaluation / "summary.json").read_text())
    if summary["submission_tree_sha256"] != tree_hash(source):
        raise ValueError("Stale validation: rerun evaluate after code or weight edits")
    candidates = [
        s for s in summary["samples"] if sum(s["metrics"]["class_support"]) > 0
    ]
    ordered = sorted(candidates, key=lambda s: s["metrics"]["completion_mean_iou"])
    sample = next(
        (
            s
            for s in candidates
            if s["house_id"] == "val_house_009" and s["sample_id"] == "sample_00002"
        ),
        ordered[len(ordered) // 2],
    )

    def get(s):
        """Load context/GT separately from the prediction previously produced by inference."""
        path = root / "samples" / s["house_id"] / s["sample_id"]
        inputs = load_inputs(path)
        target = load_target(path)
        with np.load(
            evaluation / "predictions" / s["house_id"] / (s["sample_id"] + ".npz")
        ) as f:
            prediction = f["prediction"]
        return inputs, target, prediction

    inputs, target, pred = get(sample)
    fig, axes = plt.subplots(1, 3, figsize=(8, 2.7))
    for ax, image, name in zip(
        axes, inputs["rgb"], ["Front-left", "Front", "Front-right"]
    ):
        ax.imshow(image.transpose(1, 2, 0))
        ax.axis("off")
        ax.set_title(name, fontsize=10)
    fig.suptitle(sample["house_id"] + " / " + sample["sample_id"], fontsize=10)
    fig.tight_layout()
    save(fig, output / "inputs.png")
    fig = plt.figure(figsize=(8, 3.35))
    context_prediction = target["labels"].copy()
    context_prediction[target["completion_mask"]] = pred[target["completion_mask"]]
    comparison(
        fig,
        [target["labels"], context_prediction],
        target["output_mask"],
        target["completion_mask"],
        0.21,
        0.65,
        ["Ground-truth semantics", "Predicted hidden semantics + shared GT context"],
    )
    fig.text(
        0.5,
        0.145,
        "Colors: semantic classes. Strong outlines: hidden cells. GT context is display-only.",
        fontsize=8,
        ha="center",
    )
    semantic_key(fig)
    save(fig, output / "completion.png")
    fig = plt.figure(figsize=(8, 6))
    # Avoid a trivial one-class, one-cell perfect example. The complete metrics
    # remain in the summary; this filter chooses informative illustrations only.
    substantial = [
        s
        for s in ordered
        if sum(s["metrics"]["class_support"]) >= 100
        and sum(n > 0 for n in s["metrics"]["class_support"]) >= 3
    ] or ordered
    for row, s in enumerate([substantial[-1], substantial[0]]):
        _, t, p = get(s)
        hidden_prediction = t["labels"].copy()
        hidden_prediction[t["completion_mask"]] = p[t["completion_mask"]]
        y = 0.58 if row == 0 else 0.15
        rank = "Higher-scoring" if row == 0 else "Lower-scoring"
        comparison(
            fig,
            [t["labels"], hidden_prediction],
            t["output_mask"],
            t["completion_mask"],
            y,
            0.33,
            [
                "Ground truth",
                f"Hidden prediction: mIoU {s['metrics']['completion_mean_iou']:.3f}",
            ],
        )
        fig.text(
            0.5,
            y - 0.025,
            f"{rank} example: {s['house_id']} / {s['sample_id']}",
            fontsize=8,
            ha="center",
        )
    semantic_key(fig)
    save(fig, output / "comparison.png")
    houses = list(summary["evaluation"]["by_house"].values())
    values = [
        np.mean([h["class_iou"][c] for h in houses if h["class_support"][c] > 0])
        if any(h["class_support"][c] > 0 for h in houses)
        else 0
        for c in range(10)
    ]
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.bar(range(10), values, color=COLORS)
    ax.set_xticks(range(10), CLASSES, rotation=35, ha="right", fontsize=8)
    ax.set(ylim=(0, 1), ylabel="Mean IoU over target-present houses")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save(fig, output / "class_iou.png")
    record_figures(source, evaluation, output)
    print("Wrote four report figures and their evidence manifest to", output)
