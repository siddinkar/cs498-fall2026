"""Public CPU/Kaggle protocol, collision, transport and adversarial-input gates."""

import base64
import copy
import io
import zlib
import time
import builtins
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import trimesh
from PIL import Image

from evaluation import portable_metric as metric
from evaluation import local_features
from evaluation.local_geometry import sample_surface, union_samples
from evaluation.local_features import voxelize


def with_features(row):
    """Mimic local export; Kaggle itself never samples or voxelizes a mesh."""
    points = sample_surface(
        row["vertices"],
        row["faces"],
        metric.PROTOCOL["object_points"],
        metric.stable_seed("object"),
    ).astype("<f4")
    collision = voxelize(row, points)
    collision.pop("diagnostic", None)
    return dict(id=row["id"], bounds=row["bounds"], points=points, collision=collision)


def collision(rows, floor_z):
    return metric.collision_free([with_features(row) for row in rows], floor_z)


def geometry(identity, center, extents):
    mesh = trimesh.creation.box(extents=extents)
    mesh.apply_translation(center)
    return dict(
        id=identity,
        vertices=np.asarray(mesh.vertices),
        faces=np.asarray(mesh.faces),
        bounds=mesh.bounds,
    )


def png():
    pixels = np.zeros((512, 512, 4), np.uint8)
    pixels[128:384, 128:384] = [128, 180, 210, 255]
    output = io.BytesIO()
    Image.fromarray(pixels).save(output, format="PNG")
    return base64.b64encode(output.getvalue()).decode()


def record(rows, target=False):
    image = png()
    parsed = [with_features(row) for row in rows]
    scene_points = (
        union_samples(rows, metric.PROTOCOL["scene_points"])[0].astype("<f4")
        if rows
        else np.empty((0, 3), "<f4")
    )
    result = dict(
        schema=metric.PROTOCOL["schema"],
        scene_id="test_scene_000",
        protocol_sha256=metric.PROTOCOL_SHA256,
        objects=[
            dict(
                id=row["id"],
                bounds=metric.array_record(row["bounds"], "<f8"),
                points=metric.array_record(row["points"], "<f4"),
                collision=dict(
                    status=row["collision"]["status"],
                    occupied=metric.array_record(row["collision"]["occupied"], "<i4"),
                    core=metric.array_record(row["collision"]["core"], "<i4"),
                ),
                views=[image] * 6,
            )
            for row in parsed
        ],
        scene_points=metric.array_record(scene_points, "<f4"),
        scene_views=[image] * 12,
        source=dict(
            geometry_sha256=metric.geometry_hash(rows),
            glb_bytes=100,
            glb_sha256=[],
            features_sha256=metric.features_hash(parsed, scene_points),
        ),
    )
    if target:
        result.update(support_ids=[1], floor_z=0.0)
    return result


def fixtures():
    return [
        geometry(1, [0, 0, 0.5], [1, 1, 0.1]),
        geometry(2, [0, 0, 0.65], [0.2, 0.2, 0.2]),
    ]


def test_perfect_copy_and_local_kaggle_entrypoint_agree():
    rows = fixtures()
    target, prediction = record(rows, target=True), record(rows)
    local = metric.evaluate_scene(target, prediction, "test_scene_000")
    assert local["final_score"] == pytest.approx(100)
    assert local["collision"]["nonfree_ids"] == []
    solution = pd.DataFrame(
        metric.payload_rows("test_scene_000", metric.encode_payload(target)),
        columns=["Id", "Prediction"],
    )
    submission = pd.DataFrame(
        metric.payload_rows("test_scene_000", metric.encode_payload(prediction)),
        columns=["Id", "Prediction"],
    )
    assert metric.score(solution, submission, "Id") == local["final_score"]


@pytest.mark.parametrize(
    "position, collides",
    [(1.01, False), (1.0, False), (0.999, False), (0.98, True), (0.0, True)],
)
def test_contact_tolerance_and_coincident_solids(position, collides):
    rows = [geometry(1, [0, 0, 1], [1, 1, 1]), geometry(2, [position, 0, 1], [1, 1, 1])]
    result = collision(rows, 0)
    assert bool(result["nonfree_ids"]) == collides


def test_complete_containment_cannot_hide_from_surface_intersection():
    rows = [geometry(1, [0, 0, 1], [1, 1, 1]), geometry(2, [0, 0, 1], [0.1, 0.1, 0.1])]
    assert collision(rows, 0)["nonfree_ids"] == [1, 2]
    thin = geometry(2, [0, 0, 1], [0.001, 0.1, 0.1])
    assert collision([rows[0], thin], 0)["nonfree_ids"] == [1, 2]


def test_closed_concavity_and_stock_kaggle_dependencies(monkeypatch):
    ring = trimesh.creation.annulus(r_min=0.2, r_max=0.4, height=0.2)
    ring.apply_translation([0, 0, 1])
    row = dict(id=1, vertices=ring.vertices, faces=ring.faces, bounds=ring.bounds)
    center = geometry(2, [0, 0, 1], [0.1, 0.1, 0.1])
    crossing = geometry(2, [0.35, 0, 1], [0.1, 0.1, 0.1])
    original = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name.split(".")[0] in {"trimesh", "rtree", "embreex", "fcl"}:
            raise ImportError("Not in the Kaggle scoring runtime")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    assert collision([row, center], 0)["nonfree_ids"] == []
    assert collision([row, crossing], 0)["nonfree_ids"] == [1, 2]


def test_planar_leaf_is_a_surface_not_a_filled_or_invalid_volume():
    vertices = np.array([[-0.1, 0, 0.8], [0.1, 0, 0.8], [0, 0.1, 0.8]])
    leaf = dict(
        id=2,
        vertices=vertices,
        faces=np.array([[0, 1, 2]]),
        bounds=np.array([vertices.min(0), vertices.max(0)]),
    )
    table = geometry(1, [0, 0, 0.5], [1, 1, 0.1])
    result = collision([table, leaf], 0)
    assert result["nonfree_ids"] == []
    assert voxelize(leaf, sample_surface(vertices, leaf["faces"], 20000, 498))[
        "diagnostic"
    ]["surface_only_indices"] == [0]
    container = geometry(1, [0, 0, 0.8], [1, 1, 1])
    assert collision([container, leaf], 0)["nonfree_ids"] == [1, 2]


def test_compound_shelf_does_not_fill_empty_compartments():
    # Closed boards touch along welded edges, so the aggregate can be non-manifold.
    boards = [
        trimesh.creation.box(extents=[1, 1, 0.1]),
        trimesh.creation.box(extents=[0.1, 1, 1]),
    ]
    boards[0].apply_translation([0, 0, 0.55])
    boards[1].apply_translation([-0.45, 0, 1])
    shelf = trimesh.util.concatenate(boards)
    row = dict(id=1, vertices=shelf.vertices, faces=shelf.faces, bounds=shelf.bounds)
    cup = geometry(2, [0, 0, 0.75], [0.1, 0.1, 0.1])
    result = collision([row, cup], 0)
    assert result["nonfree_ids"] == []
    assert not voxelize(row, sample_surface(row["vertices"], row["faces"], 20000, 498))[
        "diagnostic"
    ]["convex_fallback"]


def test_collision_deadline_affects_only_collision_group(monkeypatch):
    def slow(row, points):
        time.sleep(10)

    target, prediction = record(fixtures(), True), record(fixtures())
    monkeypatch.setattr(local_features, "voxelize", slow)
    failed = local_features.bounded_voxelize(fixtures(), [np.zeros((20, 3))] * 2, 0.02)
    assert all(row["status"] == "timed_out" for row in failed)
    for row, value in zip(prediction["objects"], failed):
        row["collision"] = dict(
            status=value["status"],
            occupied=metric.array_record(value["occupied"], "<i4"),
            core=metric.array_record(value["core"], "<i4"),
        )
    parsed = [
        dict(
            id=r["id"],
            bounds=metric.read_array(r["bounds"], "<f8", 2),
            points=metric.read_array(r["points"], "<f4", 20000),
            collision=dict(
                status=r["collision"]["status"],
                occupied=np.empty((0, 4), "<i4"),
                core=np.empty((0, 4), "<i4"),
            ),
        )
        for r in prediction["objects"]
    ]
    prediction["source"]["features_sha256"] = metric.features_hash(
        parsed, metric.read_array(prediction["scene_points"], "<f4", 100000)
    )
    result = metric.evaluate_scene(target, prediction, "test_scene_000")
    assert result["groups"]["collision_free"] == 0
    assert result["final_score"] == pytest.approx(90)


def test_floor_penetration_and_missing_support():
    assert collision([geometry(1, [0, 0, 0.49], [1, 1, 1])], 0)["nonfree_ids"] == [1]
    target = record(fixtures(), target=True)
    prediction = record(fixtures()[1:])
    assert (
        metric.evaluate_scene(target, prediction, "test_scene_000")["groups"][
            "collision_free"
        ]
        == 0
    )


def test_missing_instance_does_not_inflate_collision_credit():
    rows = fixtures() + [geometry(3, [0.3, 0, 0.65], [0.2, 0.2, 0.2])]
    result = metric.evaluate_scene(
        record(rows, target=True), record(rows[:2]), "test_scene_000"
    )
    assert result["groups"]["collision_free"] == 0.5


def test_empty_prediction_is_valid_zero_not_background_credit():
    result = metric.evaluate_scene(
        record(fixtures(), target=True), record([]), "test_scene_000"
    )
    assert result["final_score"] == 0
    empty = metric.empty_prediction("test_scene_000")
    assert metric.validate_record(empty, "test_scene_000") == []
    assert all(not metric.read_image(image)[1].any() for image in empty["scene_views"])
    assert (
        metric.evaluate_scene(record(fixtures(), target=True), empty, "test_scene_000")[
            "final_score"
        ]
        == 0
    )


def test_decompression_limit_and_trailing_bytes(monkeypatch):
    raw = base64.b64encode(zlib.compress(b"a" * 5000)).decode()
    monkeypatch.setattr(metric, "MAX_PAYLOAD_BYTES", 1000)
    with pytest.raises(metric.ParticipantVisibleError):
        metric.decode_payload(raw)
    with pytest.raises(metric.ParticipantVisibleError):
        metric.decode_payload(
            base64.b64encode(zlib.compress(b"{}") + b"extra").decode()
        )


def test_array_shape_bytes_nonfinite_and_bad_indices():
    original = record(fixtures())
    for mutator in [
        lambda r: r["objects"][0]["points"].update(shape=[10**12, 3]),
        lambda r: r["objects"][0].update(
            points=metric.array_record(np.full((20000, 3), np.nan), "<f4")
        ),
        lambda r: r["objects"][0].update(
            bounds=metric.array_record(np.array([[1, 1, 1], [-1, -1, -1]]), "<f8")
        ),
        lambda r: r["objects"][0].update(id=True),
        lambda r: r.update(support_ids=[1]),
        lambda r: r["source"].update(glb_bytes=20_000_000),
    ]:
        broken = copy.deepcopy(original)
        mutator(broken)
        with pytest.raises(metric.ParticipantVisibleError):
            metric.validate_record(broken, "test_scene_000")


@pytest.mark.parametrize("status", [[], {}, None, 0, True])
def test_collision_status_requires_a_string(status):
    prediction = record(fixtures())
    prediction["objects"][0]["collision"]["status"] = status
    with pytest.raises(
        metric.ParticipantVisibleError, match="Invalid collision preprocessing record"
    ):
        metric.validate_record(prediction, "test_scene_000")


def test_kaggle_rejects_missing_duplicate_or_reordered_ids():
    solution = pd.DataFrame({"Id": ["a", "b"], "Prediction": ["", ""]})
    for identities in [["a"], ["a", "a"], ["b", "a"]]:
        with pytest.raises(metric.ParticipantVisibleError):
            metric.score(
                solution,
                pd.DataFrame({"Id": identities, "Prediction": [""] * len(identities)}),
                "Id",
            )


def test_chunked_transport_padding_boundaries_and_missing_rows():
    value = "A" * (metric.CHUNK_CHARACTERS + 4)
    frame = pd.DataFrame(
        metric.payload_rows("test_scene_000", value), columns=["Id", "Prediction"]
    )
    assert list(metric.payloads_from_frame(frame)) == [("test_scene_000", value)]
    assert frame.Prediction.str.len().max() <= metric.CHUNK_CHARACTERS
    for broken in [frame.iloc[:-1], frame.iloc[::-1], pd.concat([frame, frame])]:
        with pytest.raises(metric.ParticipantVisibleError):
            list(metric.payloads_from_frame(broken))
    broken = frame.copy()
    broken.loc[4, "Prediction"] = "BAD"
    with pytest.raises(metric.ParticipantVisibleError):
        list(metric.payloads_from_frame(broken))


def test_sparse_run_intersections_and_subset_match_expanded_cells():
    """The online interval shortcut must agree with explicit small-grid sets."""
    from evaluation.local_features import merge_runs

    rng = np.random.default_rng(498)
    for _ in range(100):
        values = []
        for side in range(2):
            rows = []
            for i in range(20):
                x, y, z = rng.integers(-3, 4, 3)
                rows.append([x, y, z, z + int(rng.integers(1, 5))])
            runs = merge_runs(rows)
            cells = {
                (int(x), int(y), z)
                for x, y, lo, hi in runs
                for z in range(int(lo), int(hi))
            }
            values.append((runs, cells))
        a, b = values
        witness = metric.overlap_witness(a[0], b[0])
        assert (witness is not None) == bool(a[1] & b[1])
        if witness is not None:
            cell = tuple(
                np.floor(
                    np.array(witness) / metric.PROTOCOL["collision"]["pitch_m"]
                ).astype(int)
            )
            assert cell in a[1] & b[1]
        assert metric.runs_subset(a[0], b[0]) == (a[1] <= b[1])


def test_numeric_kaggle_core_has_no_local_geometry_dependency(monkeypatch):
    target, prediction = record(fixtures(), True), record(fixtures())
    original = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name.startswith(
            ("evaluation.local_", "evaluation.portable_render")
        ) or name.split(".")[0] in {"trimesh", "bpy", "rtree", "fcl"}:
            raise AssertionError("Kaggle attempted heavy mesh work")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    assert (
        metric.evaluate_scene(target, prediction, "test_scene_000")["final_score"]
        == 100
    )
    assert metric._decoded_image.cache_info().currsize == 0


def test_local_feature_cache_reuse_and_tamper_check(tmp_path, monkeypatch):
    monkeypatch.setenv("TRACK2_EVALUATION_CACHE", str(tmp_path / "shared"))
    first = local_features.prepare_features(fixtures(), tmp_path / "first")

    def forbidden(*args):
        raise AssertionError("Cached result should not voxelize again")

    monkeypatch.setattr(local_features, "bounded_voxelize", forbidden)
    assert local_features.prepare_features(fixtures(), tmp_path / "second") == first
    path = tmp_path / "second/features.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="cache changed"):
        local_features.prepare_features(fixtures(), tmp_path / "second")


def test_val_csv_rehearsal_uses_actual_decoder_and_scorer(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts/student/check_kaggle_submission.py"
    spec = importlib.util.spec_from_file_location("val_checker", script)
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    target, prediction = record(fixtures(), True), record(fixtures())
    identity = "val_scene_000"
    target["scene_id"] = prediction["scene_id"] = identity
    frame = pd.DataFrame(
        metric.payload_rows(identity, metric.encode_payload(prediction)),
        columns=["Id", "Prediction"],
    )
    csv_path = tmp_path / "val.csv"
    frame.to_csv(csv_path, index=False)
    monkeypatch.setattr(checker, "target_record", lambda *a, **k: target)
    result = checker.check(csv_path, tmp_path, tmp_path / "check.json", [identity])
    assert result["passed"] and result["score"] == 100 and result["rows"] == 2048
    assert not result["complete_val"] and not result["test_gt_accessed"]
    with pytest.raises(FileExistsError):
        checker.check(csv_path, tmp_path, tmp_path / "check.json", [identity])
    with pytest.raises(metric.ParticipantVisibleError, match="Val inventory"):
        checker.check(csv_path, tmp_path, tmp_path / "wrong.json", ["val_scene_001"])
