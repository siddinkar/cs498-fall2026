"""Depth-gated cross-view instance voting using known cameras and predicted depth."""
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
from .common import read_json, world_points, write_json


def refine(run, frames, xyz, initial_ids, settings):
    maximum_id = int(initial_ids.max())
    votes = np.zeros((len(xyz), maximum_id + 1), np.float32)
    counts = np.zeros((len(xyz), maximum_id + 1), np.uint16)
    selected = np.linspace(0, len(frames), min(len(frames), settings['consensus_views']), endpoint=False, dtype=int)
    for position in selected:
        index = frames[position]['frame_index']
        data = np.load(run / 'rgbd' / f'frame_{index:04d}.npz')
        labels = np.asarray(Image.open(run / 'instances' / f'frame_{index:04d}.png'), dtype=np.int32)
        inverse = np.linalg.inv(data['camera_to_world_cv']).astype(np.float32)
        points = xyz @ inverse[:3, :3].T + inverse[:3, 3]
        K = data['K']
        positive = points[:, 2] > .01
        z = np.maximum(points[:, 2], .01)
        x = np.rint(points[:, 0] / z * K[0, 0] + K[0, 2]).astype(int)
        y = np.rint(points[:, 1] / z * K[1, 1] + K[1, 2]).astype(int)
        h, w = labels.shape
        inside = positive & (x >= 0) & (x < w) & (y >= 0) & (y < h)
        source = np.flatnonzero(inside)
        xx, yy = x[inside], y[inside]
        difference = np.abs(points[inside, 2] - data['depth'][yy, xx])
        ids = labels[yy, xx]
        valid = data['valid'][yy, xx] & (difference < settings['consensus_depth_tolerance_m']) & (ids > 0)
        source, ids = source[valid], ids[valid]
        confidence = data['confidence'][yy[valid], xx[valid]].astype(np.float32)
        weights = confidence ** 2 * (1 - difference[valid] / settings['consensus_depth_tolerance_m'])
        votes[source, ids] += weights
        counts[source, ids] += 1
    best = votes.argmax(1)
    supported = counts[np.arange(len(xyz)), best] >= settings['consensus_minimum_votes']
    refined = initial_ids.copy()
    refined[supported] = best[supported]
    write_json(run / 'consensus.json', {'views': [frames[i]['frame_index'] for i in selected],
        'depth_tolerance_m': settings['consensus_depth_tolerance_m'],
        'supported_points': int(supported.sum()), 'relabeled_points': int((refined != initial_ids).sum()),
        'total_points': len(xyz), 'method': 'Confidence-weighted instance votes, only for depth-consistent visible surfaces',
        'GT_depth_or_masks_used': False})
    return refined


def update_masks(run, frames, xyz, object_ids, settings, remap=None):
    """Feed fused labels back to measured mask pixels; no unseen masks invented."""
    tree = cKDTree(xyz)
    present = set(object_ids.tolist())
    for frame in frames:
        index = frame['frame_index']
        data = np.load(run / 'rgbd' / f'frame_{index:04d}.npz')
        path = run / 'instances' / f'frame_{index:04d}.png'
        labels = np.array(Image.open(path), dtype=np.uint16)
        if remap:
            original=labels.copy()
            for source_id,target_id in remap.items():labels[original==source_id]=target_id
        labels[~np.isin(labels, list(present))] = 0
        measured = data['valid'] & (labels > 0)
        cloud = world_points(data['depth'], data['K'], data['camera_to_world_cv'])[measured]
        distances, nearest = tree.query(cloud, k=1, distance_upper_bound=settings['consensus_depth_tolerance_m'], workers=4)
        valid = np.isfinite(distances)
        current = labels[measured]
        current[valid] = object_ids[nearest[valid]]
        labels[measured] = current
        labels[~np.isin(labels, list(present))] = 0
        Image.fromarray(labels).save(path)
