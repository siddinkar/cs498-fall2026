"""Local, reviewable matching/geometry/collision evidence for the canonical metric."""

from pathlib import Path
import itertools

import numpy as np
from scipy.spatial import cKDTree

from evaluation.portable_metric import (
    PROTOCOL,
    bbox_iou,
    validate_record,
    read_array,
)


def make_visuals(target, prediction, result, output):
    """Draw measured evidence, not a physics replay or independent metric definition."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folder = output / "visualizations"
    folder.mkdir(exist_ok=True)
    gt = validate_record(target, target["scene_id"], target=True)
    pred = validate_record(prediction, prediction["scene_id"])
    visuals = {}
    pairs = result.get("matching", {}).get("pairs", [])
    fig, ax = plt.subplots(
        figsize=(max(5, len(pred) * 0.55), max(3.5, len(gt) * 0.35)),
        layout="constrained",
    )
    matrix = np.array([[bbox_iou(g["bounds"], p["bounds"]) for p in pred] for g in gt])
    if pred:
        plot = ax.imshow(matrix, vmin=0, vmax=1, cmap="Blues", aspect="auto")
        ax.set_xticks(range(len(pred)), [str(row["id"]) for row in pred])
        ax.set_yticks(range(len(gt)), [str(row["id"]) for row in gt])
        for gid, pid in pairs:
            i = next(i for i, row in enumerate(gt) if row["id"] == gid)
            j = next(i for i, row in enumerate(pred) if row["id"] == pid)
            ax.add_patch(
                plt.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="#b94a43", lw=1.5
                )
            )
        fig.colorbar(plot, ax=ax, label="World AABB IoU")
    else:
        ax.text(
            0.5,
            0.5,
            "No predicted instances",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    ax.set_xlabel("Predicted instance ID")
    ax.set_ylabel("GT instance ID")
    ax.set_title(
        f"Class-agnostic matching; instance F1 = {result['groups']['matching']:.3f}"
    )
    fig.savefig(folder / "matching.png", dpi=160)
    plt.close(fig)
    visuals["matching"] = dict(image="visualizations/matching.png")

    def distance_views(a, b, name, title, metrics):
        path = folder / name
        fig = plt.figure(figsize=(10, 7), layout="constrained")
        all_points = np.concatenate([a, b]) if len(b) else a
        center = (all_points.min(0) + all_points.max(0)) * 0.5
        radius = max(float(np.ptp(all_points, axis=0).max()) * 0.55, 0.005)
        errors = [
            cKDTree(b).query(a, workers=1)[0] if len(b) else np.full(len(a), np.inf),
            cKDTree(a).query(b, workers=1)[0] if len(b) else np.array([]),
        ]
        plot = None
        for column, angle in enumerate([35, 155]):
            for row, points in enumerate([a, b]):
                ax = fig.add_subplot(2, 2, row * 2 + column + 1, projection="3d")
                choice = np.arange(len(points))[:: max(1, len(points) // 3500)][:3500]
                p = points[choice]
                if len(p):
                    plot = ax.scatter(
                        p[:, 0],
                        p[:, 1],
                        p[:, 2],
                        c=np.minimum(errors[row][choice], 0.05) * 1000,
                        s=1,
                        cmap="magma",
                        vmin=0,
                        vmax=50,
                        depthshade=False,
                    )
                else:
                    ax.text2D(0.2, 0.5, "No predicted surface", transform=ax.transAxes)
                ax.set(
                    xlim=(center[0] - radius, center[0] + radius),
                    ylim=(center[1] - radius, center[1] + radius),
                    zlim=(center[2] - radius, center[2] + radius),
                    xlabel="X (m)",
                    ylabel="Y (m)",
                    zlabel="Z (m)",
                )
                ax.set_box_aspect((1, 1, 1))
                ax.view_init(25, angle)
                ax.set_title(
                    "GT → prediction" if row == 0 else "Prediction → GT", fontsize=10
                )
        if plot is not None:
            fig.colorbar(
                plot,
                ax=fig.axes,
                shrink=0.6,
                label="Nearest sampled-surface distance (mm; clipped at 50)",
            )
        suffix = (
            f"; CD={metrics['cd_m'] * 1000:.2f} mm, F1={metrics['f1']:.3f}"
            if metrics
            else "; no prediction"
        )
        fig.suptitle(title + suffix, fontsize=11)
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return dict(image="visualizations/" + name)

    gpoints = read_array(target["scene_points"], "<f4", PROTOCOL["scene_points"])
    ppoints = (
        read_array(prediction["scene_points"], "<f4", PROTOCOL["scene_points"])
        if pred
        else np.zeros((0, 3))
    )
    visuals["scene_geometry"] = distance_views(
        gpoints,
        ppoints,
        "scene_geometry.png",
        "Whole-scene geometry",
        result.get("scene_geometry"),
    )
    visuals["objects"] = []
    for row in result.get("object_geometry", []):
        gid, pid = row["gt_id"], row["pred_id"]
        g = next(r for r in gt if r["id"] == gid)
        p = next(r for r in pred if r["id"] == pid)
        a, b = g["points"], p["points"]
        name = f"object_gt_{gid:03d}_pred_{pid:03d}.png"
        visuals["objects"].append(
            dict(
                gt_id=gid,
                pred_id=pid,
                geometry=distance_views(
                    a, b, name, f"GT {gid} / prediction {pid}", row
                ),
            )
        )

    collision = result.get("collision", {})
    bad = set(collision.get("nonfree_ids", []))
    fig = plt.figure(figsize=(10, 4.6), layout="constrained")
    bounds = (
        np.concatenate([r["bounds"] for r in pred])
        if pred
        else np.array([[0, 0, 0], [1, 1, 1]])
    )
    center = (bounds.min(0) + bounds.max(0)) * 0.5
    radius = max(float(np.ptp(bounds, axis=0).max()) * 0.55, 0.01)
    for column, azimuth in enumerate([35, 155]):
        ax = fig.add_subplot(1, 2, column + 1, projection="3d")
        for row in pred:
            corners = np.array(
                list(itertools.product(*zip(row["bounds"][0], row["bounds"][1])))
            )
            color = "#b94a43" if row["id"] in bad else "#287d62"
            for i in range(8):
                for bit in [1, 2, 4]:
                    j = i ^ bit
                    if i < j:
                        ax.plot(*corners[[i, j]].T, color=color, lw=0.8)
            c = row["bounds"].mean(0)
            ax.text(*c, str(row["id"]), color=color, fontsize=9)
        for witness in collision.get("collisions", []):
            q = witness.get("witness_world_m")
            if q is not None:
                ax.scatter(*q, color="#b94a43", s=45, marker="x")
        ax.set(
            xlim=(center[0] - radius, center[0] + radius),
            ylim=(center[1] - radius, center[1] + radius),
            zlim=(center[2] - radius, center[2] + radius),
            xlabel="X (m)",
            ylabel="Y (m)",
            zlabel="Z (m)",
        )
        ax.view_init(25, azimuth)
        ax.set_box_aspect((1, 1, 1))
    state = (
        "timed out"
        if collision.get("timed_out")
        else ("flagged IDs " + str(sorted(bad)) if bad else "no voxel-core overlap")
    )
    if not pred:
        state = "no predictions; no collision credit"
    fig.suptitle(
        f"Static collision evidence: {state}; C={result['groups']['collision_free']:.3f}",
        fontsize=11,
    )
    fig.savefig(folder / "collision_free.png", dpi=160)
    plt.close(fig)
    visuals["collision_free"] = dict(
        image="visualizations/collision_free.png", details="metrics.json"
    )
    return visuals


def write_evidence(target, prediction, result, output):
    """Create a GT-dependent, side-by-side report from already-scored records.

    Local Dev/Val may use this publicly; private Test batch outputs are for
    authorized graders only. No renderer, GLB parser or student code is run.
    """
    import base64
    import html

    output.mkdir(parents=True, exist_ok=True)
    visualizations = make_visuals(target, prediction, result, output)
    sections = []

    def pair(a, b, name, title):
        paths = []
        for side, encoded in (("gt", a), ("pred", b)):
            path = output / "images" / side / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(base64.b64decode(encoded, validate=True))
            paths.append(path.relative_to(output).as_posix())
        return (
            f'<figure><figcaption>{html.escape(title)}</figcaption><div class="pair">'
            f'<img src="{paths[0]}" alt="GT"><img src="{paths[1]}" alt="Prediction"></div></figure>'
        )

    for key in ("matching", "scene_geometry", "collision_free"):
        sections.append(
            f'<figure><img class="metric" src="{visualizations[key]["image"]}" alt="{key}"></figure>'
        )
    sections.append(
        "<details open><summary>Whole-scene appearance: GT left, prediction right</summary>"
    )
    for index, (a, b) in enumerate(
        zip(target["scene_views"], prediction["scene_views"])
    ):
        sections.append(
            pair(a, b, f"scene_{index:02d}.png", f"Shared world view {index}")
        )
    sections.append("</details>")
    gt_objects = {row["id"]: row for row in target["objects"]}
    pred_objects = {row["id"]: row for row in prediction["objects"]}
    for row in visualizations["objects"]:
        gid, pid = row["gt_id"], row["pred_id"]
        sections.append(
            f"<details><summary>Complete object: GT {gid}, prediction {pid}</summary>"
        )
        sections.append(
            f'<img class="metric" src="{row["geometry"]["image"]}" alt="World-space object geometry">'
        )
        for index, (a, b) in enumerate(
            zip(gt_objects[gid]["views"], pred_objects[pid]["views"])
        ):
            sections.append(
                pair(
                    a,
                    b,
                    f"object_gt_{gid:03d}_pred_{pid:03d}_{index:02d}.png",
                    f"AABB-relative direction {index}",
                )
            )
        sections.append("</details>")
    sections.append(
        '<p>Unmatched objects and collision details remain in <a href="metrics.json">metrics.json</a>.</p>'
    )
    (output / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Track 2 metric evidence</title>'
        "<style>body{font:16px system-ui;max-width:1100px;margin:30px auto}img{width:49%}.metric{width:100%}.pair{display:flex;gap:1%}figure{margin:20px 0}summary{padding:10px;cursor:pointer}</style>"
        f"<h1>{html.escape(target['scene_id'])}</h1><p>GT left; prediction right. Static collision checks, no simulation.</p>"
        + "".join(sections)
    )
    return visualizations
