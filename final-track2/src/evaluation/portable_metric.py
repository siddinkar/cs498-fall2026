"""Bounded, offline CPU scoring for local exports and Kaggle.

All heavy mesh processing runs locally before export. Scoring compares fixed
point arrays, canonical PNGs and sparse voxel runs; it never processes triangles,
renders, samples surfaces or runs physics. Hashes are consistency receipts, not
proof of the submitted arrays' origin. Final code/model/GLB review is required.
"""

import base64
import functools
import hashlib
import io
import json
import math
import zlib

import numpy as np
import pandas as pd
from PIL import Image
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from skimage.metrics import structural_similarity


PROTOCOL = {
    "schema": "cs498.track2.kaggle.v3",
    "seed": 4982026,
    "scene_points": 100000,
    "object_points": 20000,
    "matching_iou": 0.25,
    "f1_distance_m": 0.01,
    "cd_scale_fraction": 0.05,
    "psnr_ceiling_db": 40.0,
    "render": {
        "resolution": 512,
        "samples": 32,
        "scene_views": 12,
        "scene_elevations_degrees": [25, 55],
        "scene_fov_degrees": 60,
        "scene_radius_multiplier": 1.35,
        "object_directions": [
            [1, 0, 0],
            [-1, 0, 0],
            [0, 1, 0],
            [0, -1, 0],
            [0, 0, 1],
            [0, 0, -1],
        ],
        "object_camera": "ORTHO",
        "object_frame_multiplier": 1.4,
        "background": "white_composite_from_RGBA",
        "lighting": "neutral_world_and_AABB_relative_area_v1",
        "view_transform": "Standard",
    },
    "collision": {
        "method": "sparse_world_voxel_core_overlap_v1",
        "pitch_m": 0.005,
        "origin_m": [0, 0, 0],
        "centers": "(integer_index + 0.5) * pitch",
        "occupied": "component_center_interiors_union_quantized_object_surface_samples",
        "core": "union_of_component_interiors_eroded_by_3x3x3_cells",
        "pair_rule": "core_A_intersects_occupied_B_or_core_B_intersects_occupied_A",
        "tolerance_m": 0.002,
        "floor_rule": "original_minimum_z_below_floor_minus_tolerance",
        "local_timeout_seconds": 600,
        "max_runs_per_object": 500000,
        "max_runs_per_scene": 1000000,
        "open_mesh_policy": "convex_hull_collision_proxy_only",
        "score": "matched_collision_free_foreground_GT_recall; support_required",
    },
    "weights": {
        "matching": 0.20,
        "scene_geometry": 0.20,
        "object_geometry": 0.20,
        "scene_appearance": 0.15,
        "object_appearance": 0.15,
        "collision_free": 0.10,
    },
}
PROTOCOL_SHA256 = hashlib.sha256(
    json.dumps(PROTOCOL, sort_keys=True).encode()
).hexdigest()
MAX_PAYLOAD_BYTES = 80_000_000
MAX_IMAGE_BYTES = 1_500_000
MAX_OBJECTS = 64
MAX_VERTICES = 2_000_000
MAX_FACES = 2_000_000
# CSV transport is separate from the numerical/rendering protocol. Fixed slots
# let prediction lengths differ from GT without exposing private target sizes.
# Kaggle's block CSV parser rejects multi-megabyte individual fields.
CHUNK_CHARACTERS = 65_536
CHUNKS_PER_SCENE = 2_048
PADDING_CHUNK = "-"


class ParticipantVisibleError(ValueError):
    """A bounded contract error that is safe to expose without private GT."""


def payload_rows(identity, value):
    """Split one bounded scene payload into fixed, short, uniquely addressed rows."""
    if (
        not isinstance(value, str)
        or not value
        or len(value) > CHUNK_CHARACTERS * CHUNKS_PER_SCENE
    ):
        raise ParticipantVisibleError("Scene exceeds the fixed CSV transport budget")
    for index in range(CHUNKS_PER_SCENE):
        chunk = value[index * CHUNK_CHARACTERS : (index + 1) * CHUNK_CHARACTERS]
        yield f"{identity}__{index:04d}", chunk or PADDING_CHUNK


def payloads_from_frame(frame, row_id_column_name="Id"):
    """Reassemble complete scene blocks; reject holes, duplicates and oversized cells."""
    if len(frame) == 0 or len(frame) % CHUNKS_PER_SCENE:
        raise ParticipantVisibleError("Need every fixed CSV chunk row for every scene")
    ids = frame[row_id_column_name].tolist()
    values = frame["Prediction"].tolist()
    seen = set()
    for start in range(0, len(ids), CHUNKS_PER_SCENE):
        first = ids[start]
        if not isinstance(first, str) or "__" not in first:
            raise ParticipantVisibleError("Invalid chunked scene ID")
        identity = first.rsplit("__", 1)[0]
        if not identity or identity in seen:
            raise ParticipantVisibleError("Duplicate or empty scene block")
        seen.add(identity)
        pieces, ended = [], False
        for index in range(CHUNKS_PER_SCENE):
            if ids[start + index] != f"{identity}__{index:04d}":
                raise ParticipantVisibleError(
                    "Missing, duplicate or reordered scene chunk"
                )
            value = values[start + index]
            if not isinstance(value, str) or not value or len(value) > CHUNK_CHARACTERS:
                raise ParticipantVisibleError("Invalid or oversized CSV chunk")
            if value == PADDING_CHUNK:
                ended = True
            else:
                if ended:
                    raise ParticipantVisibleError(
                        "Non-padding data follows the last scene chunk"
                    )
                pieces.append(value)
                if len(value) < CHUNK_CHARACTERS:
                    ended = True
        if not pieces:
            raise ParticipantVisibleError("Scene payload cannot be entirely padding")
        yield identity, "".join(pieces)


def stable_seed(name):
    """Derive repeatable local seeds without Python's randomized hash()."""
    return int.from_bytes(
        hashlib.sha256(f"{PROTOCOL['seed']}:{name}".encode()).digest()[:8], "little"
    )


def encode_payload(record):
    """Encode a self-contained record; this is compression/transport, not encryption."""
    raw = json.dumps(record, separators=(",", ":"), allow_nan=False).encode()
    if len(raw) > MAX_PAYLOAD_BYTES:
        raise ParticipantVisibleError(
            "Scene payload exceeds the uncompressed byte limit"
        )
    return base64.b64encode(zlib.compress(raw, 6)).decode("ascii")


def decode_payload(value):
    """Decode with explicit compressed/decompressed limits, no pickle or filesystem."""
    if not isinstance(value, str) or len(value) > 4 * MAX_PAYLOAD_BYTES // 3 + 8:
        raise ParticipantVisibleError("Invalid or oversized scene payload")
    try:
        compressed = base64.b64decode(value, validate=True)
        decoder = zlib.decompressobj()
        raw = decoder.decompress(compressed, MAX_PAYLOAD_BYTES + 1)
        if len(raw) > MAX_PAYLOAD_BYTES or not decoder.eof or decoder.unused_data:
            raise ValueError("Oversized, truncated or concatenated compressed payload")
        result = json.loads(raw)
    except (ValueError, zlib.error, UnicodeError) as error:
        raise ParticipantVisibleError("Malformed bounded scene payload") from error
    if not isinstance(result, dict):
        raise ParticipantVisibleError("Scene record must be an object")
    return result


def array_record(array, dtype):
    """Store a numeric array without executable pickle metadata."""
    array = np.ascontiguousarray(array, dtype=dtype)
    return dict(
        shape=list(array.shape), data=base64.b64encode(array.tobytes()).decode("ascii")
    )


def read_array(record, dtype, limit, width=3):
    """Validate dimensions and byte counts before constructing an array."""
    if not isinstance(record, dict) or set(record) != {"shape", "data"}:
        raise ParticipantVisibleError("Invalid numeric array record")
    shape = record["shape"]
    if (
        not isinstance(shape, list)
        or len(shape) != 2
        or shape[1] != width
        or any(type(n) is not int or n < 0 for n in shape)
        or shape[0] > limit
    ):
        raise ParticipantVisibleError("Numeric array shape exceeds the contract")
    size = math.prod(shape) * np.dtype(dtype).itemsize
    if not isinstance(record["data"], str) or len(record["data"]) != 4 * (
        (size + 2) // 3
    ):
        raise ParticipantVisibleError("Numeric array byte count differs from its shape")
    try:
        raw = base64.b64decode(record["data"], validate=True)
    except ValueError as error:
        raise ParticipantVisibleError("Invalid array encoding") from error
    if len(raw) != size:
        raise ParticipantVisibleError("Incorrect numeric array byte count")
    return np.frombuffer(raw, dtype=dtype).reshape(shape)


def image_record(path):
    """Embed one lossless prediction RGBA PNG, including its foreground mask."""
    return base64.b64encode(path.read_bytes()).decode("ascii")


def empty_prediction(identity):
    """Construct a runnable zero prediction without copying any GT image or label."""
    output = io.BytesIO()
    size = PROTOCOL["render"]["resolution"]
    Image.new("RGBA", (size, size), (0, 0, 0, 0)).save(output, format="PNG")
    blank = base64.b64encode(output.getvalue()).decode("ascii")
    return dict(
        schema=PROTOCOL["schema"],
        scene_id=identity,
        protocol_sha256=PROTOCOL_SHA256,
        objects=[],
        scene_points=array_record(np.empty((0, 3)), "<f4"),
        scene_views=[blank] * PROTOCOL["render"]["scene_views"],
        source=dict(
            geometry_sha256=geometry_hash([]),
            glb_bytes=0,
            glb_sha256=[],
            features_sha256=features_hash([], np.empty((0, 3), "<f4")),
        ),
    )


def read_image(value):
    """Decode only a fixed-resolution RGBA PNG and composite on white."""
    if not isinstance(value, str) or len(value) > 4 * MAX_IMAGE_BYTES // 3 + 8:
        raise ParticipantVisibleError("Missing or oversized PNG")
    return _decoded_image(value)


@functools.lru_cache(maxsize=256)
def _decoded_image(value):
    """Bounded per-case reuse: validation and scoring must not decode PNGs twice."""
    try:
        raw = base64.b64decode(value, validate=True)
        with Image.open(io.BytesIO(raw)) as image:
            size = PROTOCOL["render"]["resolution"]
            if (
                image.format != "PNG"
                or image.size != (size, size)
                or image.mode != "RGBA"
            ):
                raise ValueError("Expected fixed-resolution RGBA PNG")
            rgba = np.asarray(image, dtype=np.uint8).copy()
    except (ValueError, OSError, Image.DecompressionBombError) as error:
        raise ParticipantVisibleError("Invalid fixed-resolution RGBA PNG") from error
    alpha = rgba[:, :, 3:4].astype(np.float64) / 255
    rgb = np.rint(rgba[:, :, :3] * alpha + 255 * (1 - alpha)).astype(np.uint8)
    return rgb, rgba[:, :, 3] >= 128


def validate_record(record, identity, target=False):
    """Validate each case before matching; GT-only fields are forbidden in predictions."""
    keys = {
        "schema",
        "scene_id",
        "protocol_sha256",
        "objects",
        "scene_points",
        "scene_views",
        "source",
    }
    if target:
        keys |= {"support_ids", "floor_z"}
    if (
        not isinstance(record, dict)
        or set(record) != keys
        or record["schema"] != PROTOCOL["schema"]
    ):
        raise ParticipantVisibleError("Scene schema differs from the released protocol")
    if record["scene_id"] != identity or record["protocol_sha256"] != PROTOCOL_SHA256:
        raise ParticipantVisibleError("Wrong scene or evaluation protocol")
    rows = record["objects"]
    if not isinstance(rows, list) or len(rows) > MAX_OBJECTS:
        raise ParticipantVisibleError("Invalid instance list")
    ids = []
    parsed = []
    runs_total = 0
    for row in rows:
        if (
            not isinstance(row, dict)
            or set(row) != {"id", "bounds", "points", "collision", "views"}
            or type(row["id"]) is not int
            or not 0 < row["id"] <= 32767
        ):
            raise ParticipantVisibleError("Invalid predicted instance ID or record")
        ids.append(row["id"])
        bounds = read_array(row["bounds"], "<f8", 2)
        points = read_array(row["points"], "<f4", PROTOCOL["object_points"])
        if (
            bounds.shape != (2, 3)
            or not np.isfinite(bounds).all()
            or np.max(np.abs(bounds)) > 100
            or np.any(bounds[1] < bounds[0])
            or len(points) != PROTOCOL["object_points"]
            or not np.isfinite(points).all()
            or np.any(points < bounds[0] - 1e-5)
            or np.any(points > bounds[1] + 1e-5)
        ):
            raise ParticipantVisibleError("Invalid world-space point samples or bounds")
        collision = row["collision"]
        if (
            not isinstance(collision, dict)
            or set(collision) != {"status", "occupied", "core"}
            or not isinstance(collision["status"], str)
            or collision["status"]
            not in {"complete", "timed_out", "budget_exceeded", "invalid_solid"}
        ):
            raise ParticipantVisibleError("Invalid collision preprocessing record")
        occupied, core = [read_runs(collision[key]) for key in ("occupied", "core")]
        runs_total += len(occupied) + len(core)
        if runs_total > PROTOCOL["collision"]["max_runs_per_scene"]:
            raise ParticipantVisibleError("Sparse voxel scene budget exceeded")
        if collision["status"] != "complete" and (len(occupied) or len(core)):
            raise ParticipantVisibleError(
                "Failed preprocessing must not include voxel runs"
            )
        if not runs_subset(core, occupied):
            raise ParticipantVisibleError(
                "Collision core must be a subset of occupancy"
            )
        for runs in (occupied, core):
            if len(runs):
                low = (
                    np.column_stack((runs[:, :2], runs[:, 2]))
                    * PROTOCOL["collision"]["pitch_m"]
                )
                high = (
                    np.column_stack((runs[:, :2] + 1, runs[:, 3]))
                    * PROTOCOL["collision"]["pitch_m"]
                )
                if np.any(
                    low < bounds[0] - PROTOCOL["collision"]["pitch_m"] - 1e-5
                ) or np.any(high > bounds[1] + PROTOCOL["collision"]["pitch_m"] + 1e-5):
                    raise ParticipantVisibleError(
                        "Voxel cells lie outside declared geometry bounds"
                    )
        if not isinstance(row["views"], list) or len(row["views"]) != 6:
            raise ParticipantVisibleError(
                "Each instance needs six canonical object views"
            )
        # Validate unmatched views too; matching must not hide malformed images.
        for view in row["views"]:
            read_image(view)
        parsed.append(
            dict(
                id=row["id"],
                bounds=bounds,
                points=points,
                collision=dict(
                    status=collision["status"], occupied=occupied, core=core
                ),
                views=row["views"],
            )
        )
    if ids != sorted(set(ids)):
        raise ParticipantVisibleError("Instance IDs must be sorted and unique")
    scene_points = read_array(record["scene_points"], "<f4", PROTOCOL["scene_points"])
    if (
        len(scene_points) != (PROTOCOL["scene_points"] if rows else 0)
        or not np.isfinite(scene_points).all()
    ):
        raise ParticipantVisibleError("Invalid complete scene point sample count")
    if rows:
        bounds = np.array(
            [
                np.min([r["bounds"][0] for r in parsed], axis=0),
                np.max([r["bounds"][1] for r in parsed], axis=0),
            ]
        )
        if np.any(scene_points < bounds[0] - 1e-5) or np.any(
            scene_points > bounds[1] + 1e-5
        ):
            raise ParticipantVisibleError("Scene samples lie outside scene bounds")
    if not isinstance(record["scene_views"], list) or len(record["scene_views"]) != 12:
        raise ParticipantVisibleError("Need exactly twelve fixed scene views")
    for view in record["scene_views"]:
        read_image(view)
    if target and (
        not rows
        or not set(record["support_ids"]) <= set(ids)
        or not np.isfinite(record["floor_z"])
    ):
        raise ValueError("Invalid private target metadata")
    source = record["source"]
    if not isinstance(source, dict) or set(source) != {
        "geometry_sha256",
        "features_sha256",
        "glb_bytes",
        "glb_sha256",
    }:
        raise ParticipantVisibleError("Missing final-geometry source receipt")
    if not target and (
        type(source["glb_bytes"]) is not int
        or not 0 <= source["glb_bytes"] < 20_000_000
    ):
        raise ParticipantVisibleError(
            "Combined scene GLBs must be strictly below 20 MB"
        )
    if source["features_sha256"] != features_hash(parsed, scene_points):
        raise ParticipantVisibleError("Feature receipt differs from submitted arrays")
    for key in ("geometry_sha256", "features_sha256"):
        if (
            not isinstance(source[key], str)
            or len(source[key]) != 64
            or any(c not in "0123456789abcdef" for c in source[key])
        ):
            raise ParticipantVisibleError("Invalid source digest")
    if (
        not isinstance(source["glb_sha256"], list)
        or len(source["glb_sha256"]) > MAX_OBJECTS
        or any(
            not isinstance(v, str)
            or len(v) != 64
            or any(c not in "0123456789abcdef" for c in v)
            for v in source["glb_sha256"]
        )
    ):
        raise ParticipantVisibleError("Invalid GLB source digests")
    return parsed


def features_hash(rows, scene_points):
    """Hash submitted features for consistency only, not trusted provenance."""
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(scene_points, dtype="<f4").tobytes())
    for row in rows:
        digest.update(str(row["id"]).encode() + b"\0")
        for key, dtype in [("bounds", "<f8"), ("points", "<f4")]:
            digest.update(np.ascontiguousarray(row[key], dtype=dtype).tobytes())
        digest.update(row["collision"]["status"].encode())
        for key in ("occupied", "core"):
            array = np.ascontiguousarray(row["collision"][key], dtype="<i4")
            digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
            digest.update(array.tobytes())
    return digest.hexdigest()


def run_keys(runs, high=False):
    """Pack bounded signed grid coordinates in a sortable unsigned integer."""
    values = runs.astype(np.int64)
    return (
        ((values[:, 0] + 32768) << 32)
        | ((values[:, 1] + 32768) << 16)
        | (values[:, 3 if high else 2] + 32768)
    )


def read_runs(record):
    """Validate canonical disjoint half-open z intervals, without voxel expansion."""
    runs = read_array(record, "<i4", PROTOCOL["collision"]["max_runs_per_object"], 4)
    if not len(runs):
        return runs
    if np.any(np.abs(runs.astype(np.int64)) > 20002) or np.any(
        runs[:, 3] <= runs[:, 2]
    ):
        raise ParticipantVisibleError("Invalid sparse voxel coordinates")
    keys = run_keys(runs)
    same_column = np.all(runs[1:, :2] == runs[:-1, :2], axis=1)
    if np.any(keys[1:] <= keys[:-1]) or np.any(
        same_column & (runs[1:, 2] <= runs[:-1, 3])
    ):
        raise ParticipantVisibleError("Voxel runs must be sorted, merged and disjoint")
    return runs


def runs_subset(a, b):
    """Verify every core interval is wholly contained in one occupied interval."""
    if not len(a):
        return True
    if not len(b):
        return False
    indices = np.searchsorted(run_keys(b), run_keys(a), side="right") - 1
    valid = indices >= 0
    indices = np.maximum(indices, 0)
    return bool(
        np.all(
            valid
            & np.all(a[:, :2] == b[indices, :2], axis=1)
            & (a[:, 3] <= b[indices, 3])
        )
    )


def overlap_witness(a, b):
    """Find a shared grid cell by interval search, never by dense voxel expansion."""
    if not len(a) or not len(b):
        return None
    indices = np.searchsorted(run_keys(b), run_keys(a, high=True), side="left") - 1
    valid = indices >= 0
    indices = np.maximum(indices, 0)
    hit = valid & np.all(a[:, :2] == b[indices, :2], axis=1) & (b[indices, 3] > a[:, 2])
    selected = np.flatnonzero(hit)
    if not len(selected):
        return None
    i = selected[0]
    cell = np.array([a[i, 0], a[i, 1], max(a[i, 2], b[indices[i], 2])])
    return ((cell + 0.5) * PROTOCOL["collision"]["pitch_m"]).tolist()


def collision_free(rows, floor_z):
    """Lightweight sparse-grid comparison; no triangle queries or simulation."""
    bad = {
        r["id"]
        for r in rows
        if r["bounds"][0, 2] < floor_z - PROTOCOL["collision"]["tolerance_m"]
        or r["collision"]["status"] != "complete"
    }
    pairs = []
    for i, a in enumerate(rows):
        for b in rows[i + 1 :]:
            if np.any(
                np.minimum(a["bounds"][1], b["bounds"][1])
                < np.maximum(a["bounds"][0], b["bounds"][0])
            ):
                continue
            witness = overlap_witness(
                a["collision"]["core"], b["collision"]["occupied"]
            )
            if witness is None:
                witness = overlap_witness(
                    b["collision"]["core"], a["collision"]["occupied"]
                )
            if witness is not None:
                bad.update((a["id"], b["id"]))
                pairs.append(
                    dict(first=a["id"], second=b["id"], witness_world_m=witness)
                )
    return dict(
        nonfree_ids=sorted(bad),
        collisions=pairs,
        simulation=False,
        backend=PROTOCOL["collision"]["method"],
        pitch_m=PROTOCOL["collision"]["pitch_m"],
    )


def geometry_hash(rows):
    """Hash ordered complete geometry; does not certify the rendering process."""
    digest = hashlib.sha256()
    for row in rows:
        digest.update(str(row["id"]).encode() + b"\0")
        for key, dtype in [("vertices", "<f8"), ("faces", "<i4")]:
            array = np.ascontiguousarray(row[key], dtype=dtype)
            digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
            digest.update(array.tobytes())
    return digest.hexdigest()


def bbox_iou(a, b):
    """Compute world AABB IoU; never normalize geometry before matching."""
    intersection = np.maximum(0, np.minimum(a[1], b[1]) - np.maximum(a[0], b[0])).prod()
    union = (
        np.maximum(0, a[1] - a[0]).prod()
        + np.maximum(0, b[1] - b[0]).prod()
        - intersection
    )
    return float(intersection / union) if union > 0 else 0.0


def match_instances(gt, pred, support_ids):
    """Maximize eligible match cardinality, then total IoU, without class labels."""
    n, m = len(gt), len(pred)
    costs = np.zeros((n + m, n + m))
    costs[:n, :m] = 1e6
    ious = np.array(
        [[bbox_iou(g["bounds"], p["bounds"]) for p in pred] for g in gt]
    ).reshape(n, m)
    for i in range(n):
        for j in range(m):
            if ious[i, j] >= PROTOCOL["matching_iou"]:
                costs[i, j] = -(min(n, m) + 1 + ious[i, j]) + 1e-12 * (i * m + j)
    ii, jj = linear_sum_assignment(costs)
    pairs = [
        (gt[i]["id"], pred[j]["id"])
        for i, j in zip(ii, jj)
        if i < n and j < m and ious[i, j] >= PROTOCOL["matching_iou"]
    ]
    foreground_pairs = [(g, p) for g, p in pairs if g not in support_ids]
    support_predictions = {p for g, p in pairs if g in support_ids}
    count = sum(g["id"] not in support_ids for g in gt)
    tp = len(foreground_pairs)
    precision = (
        tp / (m - len(support_predictions)) if m > len(support_predictions) else 0.0
    )
    recall = tp / count if count else 0.0
    return dict(
        pairs=pairs,
        foreground_pairs=foreground_pairs,
        recall=recall,
        instance_f1=2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0,
        gt_objects=count,
        tp=tp,
        fp=m - len(support_predictions) - tp,
        fn=count - tp,
        support_complete=set(support_ids) <= {g for g, _ in pairs},
    )


def geometry_metrics(gt, pred, diagonal):
    """Unsquared symmetric Chamfer distance in meters and F1 at 10 mm."""
    if np.array_equal(gt, pred):
        return dict(cd_m=0.0, f1=1.0, precision=1.0, recall=1.0, score=1.0)
    pg = cKDTree(gt).query(pred, workers=1)[0]
    gp = cKDTree(pred).query(gt, workers=1)[0]
    cd = float((pg.mean() + gp.mean()) * 0.5)
    precision, recall = (
        float(np.mean(pg <= PROTOCOL["f1_distance_m"])),
        float(np.mean(gp <= PROTOCOL["f1_distance_m"])),
    )
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    value = 0.5 * (
        np.exp(-cd / max(diagonal * PROTOCOL["cd_scale_fraction"], 1e-9)) + f1
    )
    return dict(cd_m=cd, f1=f1, precision=precision, recall=recall, score=float(value))


def appearance(gt_images, pred_images):
    """Average full-image and foreground-union PSNR/SSIM in fixed normalized views."""
    if len(gt_images) != len(pred_images):
        raise ParticipantVisibleError("Appearance view counts differ")
    values = []
    for gt_value, pred_value in zip(gt_images, pred_images):
        gt, gm = read_image(gt_value)
        pred, pm = read_image(pred_value)
        if np.array_equal(gt, pred) and np.array_equal(gm, pm):
            values.append(1.0 if (gm | pm).any() else 0.5)
            continue
        a, b = gt.astype(np.float64) / 255, pred.astype(np.float64) / 255
        mse = np.mean((a - b) ** 2, axis=2)
        full, image_map = structural_similarity(
            a,
            b,
            data_range=1.0,
            channel_axis=2,
            gaussian_weights=True,
            sigma=1.5,
            use_sample_covariance=False,
            full=True,
        )
        union = gm | pm

        def psnr(error):
            return (
                1.0
                if error == 0
                else float(
                    np.clip(-10 * np.log10(error) / PROTOCOL["psnr_ceiling_db"], 0, 1)
                )
            )

        global_score = 0.5 * (psnr(float(mse.mean())) + np.clip(full, 0, 1))
        foreground = (
            0.5
            * (psnr(float(mse[union].mean())) + np.clip(image_map[union].mean(), 0, 1))
            if union.any()
            else 0.0
        )
        values.append(float(0.5 * (global_score + foreground)))
    return float(np.mean(values))


def evaluate_scene(target, prediction, identity):
    """Compare a case with bounded temporary caches, never cross-user state."""
    try:
        return _evaluate_scene(target, prediction, identity)
    finally:
        _decoded_image.cache_clear()


def _evaluate_scene(target, prediction, identity):
    """Evaluate complete case evidence without opening Blender or external files."""
    gt = validate_record(target, identity, target=True)
    pred = validate_record(prediction, identity)
    groups = {name: 0.0 for name in PROTOCOL["weights"]}
    if not pred:
        return dict(
            scene_id=identity, final_score=0.0, groups=groups, status="empty_prediction"
        )
    support = set(target["support_ids"])
    matching = match_instances(gt, pred, support)
    groups["matching"] = matching["instance_f1"]
    gpoints = read_array(target["scene_points"], "<f4", PROTOCOL["scene_points"])
    ppoints = read_array(prediction["scene_points"], "<f4", PROTOCOL["scene_points"])
    diagonal = float(
        np.linalg.norm(
            np.max([r["bounds"][1] for r in gt], axis=0)
            - np.min([r["bounds"][0] for r in gt], axis=0)
        )
    )
    scene_geometry = geometry_metrics(gpoints, ppoints, diagonal)
    groups["scene_geometry"] = scene_geometry["score"]
    groups["scene_appearance"] = appearance(
        target["scene_views"], prediction["scene_views"]
    )
    by_gt = {r["id"]: r for r in gt}
    by_pred = {r["id"]: r for r in pred}
    object_geometry, object_appearance = [], []
    for gid, pid in matching["pairs"]:
        g, p = by_gt[gid], by_pred[pid]
        row = geometry_metrics(
            g["points"],
            p["points"],
            float(np.linalg.norm(g["bounds"][1] - g["bounds"][0])),
        )
        object_geometry.append(dict(gt_id=gid, pred_id=pid, **row))
        object_appearance.append(appearance(g["views"], p["views"]))
    coverage = matching["recall"]
    groups["object_geometry"] = (
        coverage * float(np.mean([r["score"] for r in object_geometry]))
        if object_geometry
        else 0.0
    )
    groups["object_appearance"] = (
        coverage * float(np.mean(object_appearance)) if object_appearance else 0.0
    )
    collisions = collision_free(pred, float(target["floor_z"]))
    # Missing collision representations must not make neighboring instances look
    # clean: any failed local preprocessing makes this scene's C zero only.
    collision_ready = all(r["collision"]["status"] == "complete" for r in pred)
    good = sum(
        pid not in collisions["nonfree_ids"]
        for gid, pid in matching["foreground_pairs"]
    )
    groups["collision_free"] = (
        good / matching["gt_objects"]
        if matching["gt_objects"] and matching["support_complete"] and collision_ready
        else 0.0
    )
    score_value = float(
        np.clip(
            100 * math.fsum(PROTOCOL["weights"][key] * groups[key] for key in groups),
            0,
            100,
        )
    )
    return dict(
        scene_id=identity,
        final_score=float(score_value),
        groups=groups,
        matching=matching,
        scene_geometry=scene_geometry,
        object_geometry=object_geometry,
        collision=collisions,
        protocol_sha256=PROTOCOL_SHA256,
        status="complete",
    )


def score(
    solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str
) -> float:
    """Kaggle entry point: equal scene weight over the complete private solution."""
    expected = [row_id_column_name, "Prediction"]
    if list(submission.columns) != expected or len(submission) != len(solution):
        raise ParticipantVisibleError(
            "Expected exactly Id,Prediction and every required scene"
        )
    ids = submission[row_id_column_name].tolist()
    if ids != solution[row_id_column_name].tolist() or len(ids) != len(set(ids)):
        raise ParticipantVisibleError("Scene IDs must match the required unique order")
    if not ids:
        raise ValueError("Private solution is empty")
    values = []
    for (identity, gt), (pred_identity, pred) in zip(
        payloads_from_frame(solution, row_id_column_name),
        payloads_from_frame(submission, row_id_column_name),
    ):
        if identity != pred_identity:
            raise ParticipantVisibleError("Wrong scene block")
        values.append(
            evaluate_scene(decode_payload(gt), decode_payload(pred), identity)[
                "final_score"
            ]
        )
    return float(np.mean(values))
