"""Cache local surface samples and sparse collision volumes from final meshes.

Only this local module handles triangles. Kaggle receives fixed point sets,
bounds, canonical images and run-length encoded voxel columns, never a score.
The grid is world anchored: cell (x,y,z) has center (index + 0.5) * pitch.
"""

import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time

import numpy as np
from scipy.ndimage import binary_erosion

from evaluation.local_geometry import (
    _SolidIndex,
    _solid_components,
    sample_surface,
    union_samples,
)
from evaluation.portable_metric import (
    PROTOCOL,
    PROTOCOL_SHA256,
    array_record,
    geometry_hash,
    stable_seed,
)


MAX_COMPONENT_CELLS = 100_000_000
MAX_COMPONENT_COLUMNS = 1_000_000


def merge_runs(runs):
    """Canonicalize half-open integer z intervals without expanding voxel cells."""
    if not len(runs):
        return np.empty((0, 4), dtype="<i4")
    rows = np.asarray(runs, dtype=np.int32).reshape(-1, 4)
    rows = rows[np.lexsort((rows[:, 2], rows[:, 1], rows[:, 0]))]
    output = []
    x, y, lo, hi = map(int, rows[0])
    for nx, ny, nl, nh in rows[1:]:
        if nx == x and ny == y and nl <= hi:
            hi = max(hi, int(nh))
        else:
            output.append((x, y, lo, hi))
            x, y, lo, hi = map(int, (nx, ny, nl, nh))
    output.append((x, y, lo, hi))
    return np.asarray(output, dtype="<i4")


def _runs(volume, origin):
    """Encode occupied dense local columns; no global grid allocation is needed."""
    starts = np.diff(
        volume.astype(np.int8), axis=2, prepend=np.int8(0), append=np.int8(0)
    )
    a, b = np.argwhere(starts == 1), np.argwhere(starts == -1)
    if not len(a):
        return np.empty((0, 4), dtype="<i4")
    return np.column_stack(
        (a[:, :2] + origin[:2], a[:, 2] + origin[2], b[:, 2] + origin[2])
    ).astype("<i4")


def _component_volume(triangles, pitch):
    """Fill grid-center occupancy by deterministic +Z triangle scanlines locally."""
    index = _SolidIndex(triangles)
    origin = np.ceil(index.bounds[0] / pitch - 0.5 - 1e-9).astype(int) - 1
    upper = np.floor(index.bounds[1] / pitch - 0.5 + 1e-9).astype(int) + 1
    shape = upper - origin + 1
    if (
        math.prod(map(int, shape)) > MAX_COMPONENT_CELLS
        or int(shape[0]) * int(shape[1]) > MAX_COMPONENT_COLUMNS
    ):
        raise ValueError("Collision voxelization exceeds the local component budget")
    volume = np.zeros(tuple(shape), dtype=bool)
    xy = np.indices(tuple(shape[:2])).reshape(2, -1).T
    world_xy = (xy + origin[:2] + 0.5) * pitch
    in_box = np.all(
        (world_xy >= index.bounds[0, :2] - 1e-10)
        & (world_xy <= index.bounds[1, :2] + 1e-10),
        axis=1,
    )
    xy, world_xy = xy[in_box], world_xy[in_box]
    bins = np.clip(np.floor((world_xy - index.origin) / index.pitch).astype(int), 0, 31)
    bins = bins[:, 0] + 32 * bins[:, 1]
    for cell in np.unique(bins):
        selected = np.flatnonzero(bins == cell)
        tri = index.triangles[index._faces(cell)]
        ab, ac = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
        det = ab[:, 0] * ac[:, 1] - ab[:, 1] * ac[:, 0]
        good = np.abs(det) > 1e-18
        tri, ab, ac, det = tri[good], ab[good], ac[good], det[good]
        if not len(tri):
            continue
        batch = max(1, min(256, 200_000 // len(tri)))
        for start in range(0, len(selected), batch):
            ids = selected[start : start + batch]
            delta = world_xy[ids, None, :] - tri[None, :, 0, :2]
            u = (
                delta[:, :, 0] * ac[None, :, 1] - delta[:, :, 1] * ac[None, :, 0]
            ) / det
            v = (
                ab[None, :, 0] * delta[:, :, 1] - ab[None, :, 1] * delta[:, :, 0]
            ) / det
            hit = (u >= -1e-10) & (v >= -1e-10) & (u + v <= 1 + 1e-10)
            z = tri[None, :, 0, 2] + u * ab[None, :, 2] + v * ac[None, :, 2]
            z = np.sort(np.where(hit, z, np.inf), axis=1)
            for row, ray in zip(ids, z):
                ray = ray[np.isfinite(ray)]
                ray = ray[np.r_[True, np.diff(ray) > 1e-9]] if len(ray) else ray
                # At shared-edge tangencies odd rays can occur. Do not invent
                # volume on an unpaired boundary intersection.
                for low, high in zip(
                    ray[0 : len(ray) // 2 * 2 : 2], ray[1 : len(ray) // 2 * 2 : 2]
                ):
                    lo = max(0, int(np.ceil(low / pitch - 0.5 + 1e-8)) - origin[2])
                    hi = min(
                        int(shape[2]),
                        int(np.ceil(high / pitch - 0.5 - 1e-8)) - origin[2],
                    )
                    if hi > lo:
                        volume[xy[row, 0], xy[row, 1], lo:hi] = True
    return volume, origin


def voxelize(row, surface_points):
    """Build occupied/core column runs, preserving separate concave components.

    Occupied = component grid-center interiors plus quantized sampled surface.
    Core = union of each component interior eroded by a 3x3x3 cell neighborhood.
    A collision is core(A) intersect occupied(B), or vice versa. This intentionally
    ignores sub-grid penetration/contact; it is not exact mesh intersection.
    """
    pitch = PROTOCOL["collision"]["pitch_m"]
    parts, info = _solid_components(row)
    if info["invalid_collision_solid"]:
        return dict(
            status="invalid_solid",
            occupied=np.empty((0, 4), "<i4"),
            core=np.empty((0, 4), "<i4"),
            diagnostic=info,
        )
    cells = np.unique(np.floor(surface_points / pitch).astype(np.int32), axis=0)
    occupied = [np.column_stack((cells, cells[:, 2] + 1))]
    core = []
    for i, triangles in enumerate(parts):
        if i in info.get("surface_only_indices", []):
            continue
        volume, origin = _component_volume(triangles, pitch)
        occupied.append(_runs(volume, origin))
        core.append(
            _runs(binary_erosion(volume, structure=np.ones((3, 3, 3), bool)), origin)
        )
    return dict(
        status="complete",
        occupied=merge_runs(np.concatenate(occupied)),
        core=merge_runs(np.concatenate(core) if core else []),
        diagnostic=info,
    )


def _voxel_worker(rows, points, sender):
    """Keep only local geometry preprocessing in a bounded child process."""
    try:
        sender.send((True, [voxelize(row, point) for row, point in zip(rows, points)]))
    except ValueError as error:
        sender.send((False, str(error)))
    finally:
        sender.close()


def bounded_voxelize(rows, points, timeout=None):
    """A local collision preprocessing timeout affects only collision credit."""
    import multiprocessing

    timeout = timeout or PROTOCOL["collision"]["local_timeout_seconds"]
    methods = multiprocessing.get_all_start_methods()
    context = multiprocessing.get_context("fork" if "fork" in methods else "spawn")
    receiver, sender = context.Pipe(False)
    process = context.Process(target=_voxel_worker, args=(rows, points, sender))
    process.start()
    sender.close()
    try:
        if not receiver.poll(timeout):
            status, values = "timed_out", None
        else:
            try:
                success, values = receiver.recv()
            except EOFError as error:
                raise RuntimeError(
                    "Local voxel worker failed; fix installation before submission"
                ) from error
            status = "complete" if success else "budget_exceeded"
        if status == "complete":
            return values
        return [
            dict(
                status=status,
                occupied=np.empty((0, 4), "<i4"),
                core=np.empty((0, 4), "<i4"),
                diagnostic={},
            )
            for row in rows
        ]
    finally:
        receiver.close()
        process.join(0.2)
        if process.is_alive():
            process.terminate()
            process.join(2)
        if process.is_alive():
            process.kill()
            process.join(2)


def prepare_features(rows, cache):
    """Reuse immutable source/version-addressed samples and sparse voxel caches."""
    cache = Path(cache)
    fingerprint = dict(
        protocol_sha256=PROTOCOL_SHA256,
        geometry_sha256=geometry_hash(rows),
        preprocessor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        mesh_helper_sha256=hashlib.sha256(
            Path(__file__).with_name("local_geometry.py").read_bytes()
        ).hexdigest(),
    )
    path = cache / "features.json"
    receipt = cache / "receipt.json"
    if receipt.exists():
        old = json.loads(receipt.read_text())
        if (
            old["fingerprint"] != fingerprint
            or hashlib.sha256(path.read_bytes()).hexdigest() != old["features_sha256"]
        ):
            raise ValueError(
                "Local feature cache changed; select a fresh cache directory"
            )
        return json.loads(path.read_text())
    if cache.exists():
        raise FileExistsError(
            "Incomplete local feature cache; select a fresh directory"
        )
    shared = (
        Path(
            os.environ.get(
                "TRACK2_EVALUATION_CACHE",
                str(Path(__file__).resolve().parents[2] / "outputs/.evaluation_cache"),
            )
        )
        / "features"
        / hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()
    )
    if shared != cache.resolve() and (shared / "receipt.json").is_file():
        # Copy a verified immutable snapshot, not a symlink into a report. This
        # keeps exported evidence self-contained when packaged for another user.
        prepare_features(rows, shared)
        cache.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(shared, cache)
        return json.loads(path.read_text())
    started = time.perf_counter()
    points = [
        sample_surface(
            row["vertices"],
            row["faces"],
            PROTOCOL["object_points"],
            stable_seed("object"),
        )
        for row in rows
    ]
    collisions = bounded_voxelize(rows, points) if rows else []
    objects = []
    for row, point, collision in zip(rows, points, collisions):
        if (
            len(collision["occupied"]) > PROTOCOL["collision"]["max_runs_per_object"]
            or len(collision["core"]) > PROTOCOL["collision"]["max_runs_per_object"]
        ):
            collision = dict(
                status="budget_exceeded",
                occupied=np.empty((0, 4), "<i4"),
                core=np.empty((0, 4), "<i4"),
            )
        objects.append(
            dict(
                id=row["id"],
                bounds=array_record(
                    np.array([row["vertices"].min(0), row["vertices"].max(0)]), "<f8"
                ),
                points=array_record(point, "<f4"),
                collision=dict(
                    status=collision["status"],
                    occupied=array_record(collision["occupied"], "<i4"),
                    core=array_record(collision["core"], "<i4"),
                ),
            )
        )
    if (
        sum(
            len(collision["occupied"]) + len(collision["core"])
            for collision in collisions
        )
        > PROTOCOL["collision"]["max_runs_per_scene"]
    ):
        for row in objects:
            row["collision"] = dict(
                status="budget_exceeded",
                occupied=array_record(np.empty((0, 4)), "<i4"),
                core=array_record(np.empty((0, 4)), "<i4"),
            )
    scene_points = (
        union_samples(rows, PROTOCOL["scene_points"])[0] if rows else np.empty((0, 3))
    )
    result = dict(objects=objects, scene_points=array_record(scene_points, "<f4"))
    cache.mkdir(parents=True)
    path.write_text(json.dumps(result, separators=(",", ":")))
    receipt.write_text(
        json.dumps(
            dict(
                fingerprint=fingerprint,
                features_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                collision_diagnostics=[
                    dict(
                        id=row["id"],
                        status=value["status"],
                        **value.get("diagnostic", {}),
                    )
                    for row, value in zip(rows, collisions)
                ],
                wall_seconds=time.perf_counter() - started,
            ),
            indent=2,
        )
        + "\n"
    )
    if shared != cache.resolve() and not shared.exists():
        shared.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copytree(cache, shared)
        except FileExistsError:
            # Concurrent local jobs may have finished the same immutable key.
            prepare_features(rows, shared)
    return result
