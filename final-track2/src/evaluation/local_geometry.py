"""Local-only mesh preprocessing. Never imported by Kaggle scoring."""

import numpy as np
from evaluation.portable_metric import ParticipantVisibleError, stable_seed


def sample_surface(vertices, faces, count, seed):
    """Area-uniform triangle samples with a fixed seed, in original world placement."""
    triangles = vertices[faces]
    area = (
        np.linalg.norm(
            np.cross(
                triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
            ),
            axis=1,
        )
        * 0.5
    )
    total = area.sum(dtype=np.float64)
    if not np.isfinite(total) or total <= 0:
        raise ParticipantVisibleError("Mesh has no nondegenerate surface")
    rng = np.random.default_rng(seed)
    indices = np.minimum(
        np.searchsorted(np.cumsum(area), rng.random(count) * total, side="right"),
        len(faces) - 1,
    )
    r, t = np.sqrt(rng.random(count)), rng.random(count)
    return np.einsum(
        "ni,nij->nj", np.stack([1 - r, r * (1 - t), r * t], axis=1), triangles[indices]
    )


def _solid_components(row):
    """Weld only coincident vertices; preserve closed concave components."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import ConvexHull, QhullError

    original = row["vertices"]
    _, first, inverse = np.unique(
        np.round(original, 8), axis=0, return_index=True, return_inverse=True
    )
    vertices, faces = original[first], inverse[row["faces"]]
    triangles = vertices[faces]
    valid = (
        np.linalg.norm(
            np.cross(
                triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
            ),
            axis=1,
        )
        > 1e-14
    )
    faces = faces[valid]
    if not len(faces):
        return [], dict(component_count=0, invalid_collision_solid=True)
    edges = np.sort(
        np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1
    )
    _, inverse_edges, counts = np.unique(
        edges, axis=0, return_inverse=True, return_counts=True
    )
    order = np.argsort(inverse_edges, kind="stable")
    starts = np.cumsum(counts) - counts
    paired = starts[counts == 2]
    face_index = np.tile(np.arange(len(faces)), 3)
    a, b = face_index[order[paired]], face_index[order[paired + 1]]
    graph = coo_matrix(
        (np.ones(len(a), dtype=np.uint8), (a, b)), shape=(len(faces), len(faces))
    ).tocsr()
    number, labels = connected_components(graph, directed=False)
    # Excessive fragmented triangle soup is not a usable closed collision solid.
    if number > 512:
        return [], dict(
            component_count=number,
            invalid_collision_solid=True,
            component_budget_exceeded=True,
        )
    parts, fallback, invalid, surface_only = [], 0, False, []
    for label in range(number):
        component = faces[labels == label]
        closed = bool(np.all(counts == 2))
        if not closed:
            boundary = np.sort(
                np.vstack(
                    [component[:, [0, 1]], component[:, [1, 2]], component[:, [2, 0]]]
                ),
                axis=1,
            )
            _, edge_counts = np.unique(boundary, axis=0, return_counts=True)
            closed = bool(np.all(edge_counts == 2))
        if closed:
            parts.append(vertices[component])
        else:
            points = vertices[np.unique(component)]
            try:
                hull = ConvexHull(points)
                parts.append(points[hull.simplices])
                fallback += 1
            except QhullError:
                # Decorative leaves/paper can be truly planar, not broken solids.
                # Keep their surface probes, but do not fill an imaginary volume.
                if np.linalg.matrix_rank(points - points.mean(0), tol=1e-8) <= 2:
                    surface_only.append(len(parts))
                    parts.append(vertices[component])
                else:
                    invalid = True
    return parts, dict(
        component_count=number,
        convex_fallback=bool(fallback),
        convex_fallback_count=fallback,
        invalid_collision_solid=invalid or not parts,
        surface_only_indices=surface_only,
    )


class _SolidIndex:
    """Bounded XY triangle bins for exact ray parity and contact-distance tests.

    No optional native collision/ray library is required in Kaggle's scorer.
    Closed components are queried independently, so a shelf's gaps stay empty.
    """

    def __init__(self, triangles, solid=True):
        self.solid = solid
        self.triangles = np.asarray(triangles, dtype=np.float64)
        self.minimum = self.triangles.min(axis=1)
        self.maximum = self.triangles.max(axis=1)
        self.bounds = np.array([self.minimum.min(0), self.maximum.max(0)])
        self.origin = self.bounds[0, :2]
        self.pitch = np.maximum((self.bounds[1, :2] - self.origin) / 32, 1e-12)
        lower = np.clip(
            np.floor((self.minimum[:, :2] - self.origin) / self.pitch).astype(int),
            0,
            31,
        )
        upper = np.clip(
            np.floor((self.maximum[:, :2] - self.origin) / self.pitch).astype(int),
            0,
            31,
        )
        widths = upper[:, 0] - lower[:, 0] + 1
        sizes = widths * (upper[:, 1] - lower[:, 1] + 1)
        if int(sizes.sum()) > 2_000_000:
            raise ValueError("Collision spatial-index budget exceeded")
        face_ids = np.repeat(np.arange(len(triangles)), sizes)
        starts = np.cumsum(sizes) - sizes
        offset = np.arange(len(face_ids)) - np.repeat(starts, sizes)
        x = lower[face_ids, 0] + offset % widths[face_ids]
        y = lower[face_ids, 1] + offset // widths[face_ids]
        cells = x + 32 * y
        order = np.argsort(cells, kind="stable")
        self.face_ids = face_ids[order]
        self.starts = np.searchsorted(cells[order], np.arange(1025))

    def _faces(self, cell):
        return self.face_ids[self.starts[cell] : self.starts[cell + 1]]

    def contains(self, points):
        """Odd +Z ray intersections, deduplicating shared-edge hits."""
        points = np.asarray(points, dtype=np.float64)
        result = np.zeros(len(points), dtype=bool)
        if not self.solid:
            return result
        eligible = np.all(
            (points >= self.bounds[0]) & (points <= self.bounds[1]), axis=1
        )
        indices = np.flatnonzero(eligible)
        if not len(indices):
            return result
        xy = np.clip(
            np.floor((points[indices, :2] - self.origin) / self.pitch).astype(int),
            0,
            31,
        )
        cells = xy[:, 0] + 32 * xy[:, 1]
        for cell in np.unique(cells):
            point_ids = indices[cells == cell]
            candidates = self.triangles[self._faces(cell)]
            if not len(candidates):
                continue
            ab, ac = (
                candidates[:, 1] - candidates[:, 0],
                candidates[:, 2] - candidates[:, 0],
            )
            determinant = ab[:, 0] * ac[:, 1] - ab[:, 1] * ac[:, 0]
            valid = np.abs(determinant) > 1e-18
            candidates, ab, ac, determinant = (
                candidates[valid],
                ab[valid],
                ac[valid],
                determinant[valid],
            )
            if not len(candidates):
                continue
            # Keep each temporary point/triangle product small, including huge faces.
            batch_size = max(1, min(128, 200_000 // len(candidates)))
            for start in range(0, len(point_ids), batch_size):
                batch = point_ids[start : start + batch_size]
                delta = points[batch, None, :2] - candidates[None, :, 0, :2]
                u = (
                    delta[:, :, 0] * ac[None, :, 1] - delta[:, :, 1] * ac[None, :, 0]
                ) / determinant
                v = (
                    ab[None, :, 0] * delta[:, :, 1] - ab[None, :, 1] * delta[:, :, 0]
                ) / determinant
                z = candidates[None, :, 0, 2] + u * ab[None, :, 2] + v * ac[None, :, 2]
                hit = (
                    (u >= -1e-10)
                    & (v >= -1e-10)
                    & (u + v <= 1 + 1e-10)
                    & (z > points[batch, None, 2] + 1e-10)
                )
                z = np.sort(np.where(hit, z, np.inf), axis=1)
                finite = np.isfinite(z)
                distinct = finite.copy()
                differences = np.zeros_like(z[:, 1:])
                np.subtract(
                    z[:, 1:],
                    z[:, :-1],
                    out=differences,
                    where=finite[:, 1:] & finite[:, :-1],
                )
                distinct[:, 1:] &= differences > 1e-9
                result[batch] = distinct.sum(axis=1) % 2 == 1
        return result

    def farther_than_surface(self, point, tolerance):
        """Whether a contained point lies farther than tolerance from every face."""
        lower = np.clip(
            np.floor((point[:2] - tolerance - self.origin) / self.pitch).astype(int),
            0,
            31,
        )
        upper = np.clip(
            np.floor((point[:2] + tolerance - self.origin) / self.pitch).astype(int),
            0,
            31,
        )
        candidates = [
            self._faces(x + 32 * y)
            for y in range(lower[1], upper[1] + 1)
            for x in range(lower[0], upper[0] + 1)
        ]
        face_ids = np.unique(np.concatenate(candidates))
        near = np.all(
            (point >= self.minimum[face_ids] - tolerance)
            & (point <= self.maximum[face_ids] + tolerance),
            axis=1,
        )
        triangles = self.triangles[face_ids[near]]
        if not len(triangles):
            return True
        ab, ac, delta = (
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
            point - triangles[:, 0],
        )
        aa, cc, cross = (
            np.einsum("ij,ij->i", ab, ab),
            np.einsum("ij,ij->i", ac, ac),
            np.einsum("ij,ij->i", ab, ac),
        )
        da, dc = np.einsum("ij,ij->i", delta, ab), np.einsum("ij,ij->i", delta, ac)
        denominator = aa * cc - cross * cross
        u, v = (
            (cc * da - cross * dc) / denominator,
            (aa * dc - cross * da) / denominator,
        )
        normal = np.cross(ab, ac)
        squared = np.full(len(triangles), np.inf)
        inside = (u >= 0) & (v >= 0) & (u + v <= 1)
        squared[inside] = np.einsum(
            "ij,ij->i", delta[inside], normal[inside]
        ) ** 2 / np.einsum("ij,ij->i", normal[inside], normal[inside])
        for first, second in [(0, 1), (1, 2), (2, 0)]:
            edge = triangles[:, second] - triangles[:, first]
            length = np.einsum("ij,ij->i", edge, edge)
            projection = np.clip(
                np.einsum("ij,ij->i", point - triangles[:, first], edge) / length, 0, 1
            )
            difference = point - triangles[:, first] - projection[:, None] * edge
            squared = np.minimum(squared, np.einsum("ij,ij->i", difference, difference))
        return bool(np.all(squared > tolerance * tolerance + 1e-16))


def union_samples(rows, count):
    """Uniform sampling of the entire surface union, including table, not floor."""
    vertices, faces, offset = [], [], 0
    for row in rows:
        vertices.append(row["vertices"])
        faces.append(row["faces"] + offset)
        offset += len(row["vertices"])
    v = np.concatenate(vertices)
    return sample_surface(v, np.concatenate(faces), count, stable_seed("scene")), float(
        np.linalg.norm(v.max(0) - v.min(0))
    )
