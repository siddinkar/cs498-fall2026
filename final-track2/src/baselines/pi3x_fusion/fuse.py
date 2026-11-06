"""Lift masks using predicted RGB-D; associate and fuse static 3D instances."""
from collections import Counter
import gc
import time
import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from .common import context, read_json, read_masks, sha256, stage_args, world_points, write_json, write_ply
from .instance_filter import required_track_frames,nms_instances


def family(label):
    text = label.lower()
    for group, tokens in [('table', ['table']), ('cup', ['cup', 'mug']), ('bowl', ['bowl']),
                          ('box', ['box', 'block', 'package', 'tissue']), ('ball', ['ball', 'sphere']),
                          ('plate', ['plate', 'dish']), ('bottle', ['bottle'])]:
        if any(token in text for token in tokens):
            return group
    return 'unknown'


def make_owner(masks):
    """Small nested proposals own their pixels; parent masks keep the rest."""
    owner = np.full(masks.shape[1:], -1, np.int16)
    for index in np.argsort(masks.sum((1, 2)))[::-1]:
        owner[masks[index]] = index
    return owner


def bbox_iou(a, b):
    intersection = np.maximum(0, np.minimum(a[1], b[1]) - np.maximum(a[0], b[0])).prod()
    volume_a = np.maximum(a[1] - a[0], .001).prod()
    volume_b = np.maximum(b[1] - b[0], .001).prod()
    return float(intersection / max(volume_a + volume_b - intersection, 1e-10))


def score_match(candidate, track, settings):
    known = candidate['family'] != 'unknown' and track['family'] != 'unknown'
    if known and candidate['family'] != track['family']:
        return -1.
    ca = np.mean(candidate['bbox'], axis=0)
    cb = np.mean(track['bbox'], axis=0)
    distance = np.linalg.norm(ca - cb)
    max_extent = max(np.max(candidate['bbox'][1] - candidate['bbox'][0]),
                     np.max(track['bbox'][1] - track['bbox'][0]))
    if distance > max(.05, max_extent * .7):
        return -1.
    tree = track['tree']
    close = tree.query(candidate['sample'], k=1, distance_upper_bound=settings['association_radius_m'])[0]
    overlap = float(np.isfinite(close).mean())
    iou = bbox_iou(candidate['bbox'], track['bbox'])
    if overlap < settings['association_overlap']:
        if not (known and iou > .35 and distance < .035):
            return -1.
    return overlap * .7 + iou * .2 + (.1 if known else 0)


def update_track(track, candidate, frame_index, voxel):
    keys = np.floor(candidate['sample'] / voxel).astype(np.int64)
    track['keys'].update(map(tuple, keys))
    points = (np.asarray(list(track['keys']), np.float32) + .5) * voxel
    track['tree'] = cKDTree(points)
    track['bbox'] = np.stack([np.minimum(track['bbox'][0], candidate['bbox'][0]),
                              np.maximum(track['bbox'][1], candidate['bbox'][1])])
    track['frames'].add(frame_index)
    track['labels'][candidate['label']] += 1
    if track['family'] == 'unknown' and candidate['family'] != 'unknown':
        track['family'] = candidate['family']


def associate_frames(observations, settings):
    tracks, associations = [], []
    for frame in observations:
        proposals = frame['proposals']
        scores = np.full((len(proposals), len(tracks)), -1., dtype=np.float64)
        for i, candidate in enumerate(proposals):
            for j, track in enumerate(tracks):
                scores[i, j] = score_match(candidate, track, settings)
        matched = {}
        if scores.size:
            rows, columns = linear_sum_assignment(-scores)
            matched = {int(i): int(j) for i, j in zip(rows, columns) if scores[i, j] >= 0}
        assignments = {}
        for i, candidate in enumerate(proposals):
            if i in matched:
                track_id = matched[i]
            else:
                track_id = len(tracks)
                tracks.append({'keys': set(), 'tree': None, 'bbox': candidate['bbox'].copy(),
                               'family': candidate['family'], 'frames': set(), 'labels': Counter()})
            update_track(tracks[track_id], candidate, frame['frame_index'], settings['association_voxel_m'])
            assignments[candidate['proposal_id']] = track_id
        associations.append({'frame_index': frame['frame_index'], 'proposal_to_track': assignments})
    # Consolidate reappearance fragments only when they never coexist. This is
    # essential for touching boxes of the same class: coexistence forbids merging.
    parent = list(range(len(tracks)))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i in range(len(tracks)):
        for j in range(i + 1, len(tracks)):
            a, b = root(i), root(j)
            if a == b or tracks[a]['frames'] & tracks[b]['frames']:
                continue
            if tracks[a]['family'] != tracks[b]['family']:
                continue
            candidate = {'sample': tracks[b]['tree'].data, 'bbox': tracks[b]['bbox'],
                         'family': tracks[b]['family']}
            if score_match(candidate, tracks[a], settings) < .35:
                continue
            parent[b] = a
            tracks[a]['frames'].update(tracks[b]['frames'])
            tracks[a]['labels'].update(tracks[b]['labels'])
            tracks[a]['keys'].update(tracks[b]['keys'])
            p = (np.asarray(list(tracks[a]['keys']), np.float32) + .5) * settings['association_voxel_m']
            tracks[a]['tree'] = cKDTree(p)
            tracks[a]['bbox'] = np.stack([np.minimum(tracks[a]['bbox'][0], tracks[b]['bbox'][0]),
                                          np.maximum(tracks[a]['bbox'][1], tracks[b]['bbox'][1])])
    minimum_frames=required_track_frames(settings,len({frame['frame_index'] for frame in observations}))
    active = [i for i in range(len(tracks)) if root(i) == i and len(tracks[i]['frames']) >= minimum_frames]
    final_id = {track: k + 1 for k, track in enumerate(active)}
    for frame in associations:
        frame['proposal_to_object'] = {proposal: final_id.get(root(track), 0)
                                      for proposal, track in frame['proposal_to_track'].items()}
    return tracks, active, associations


def voxel_keys(points, voxel):
    q = np.floor(points / voxel).astype(np.int64)
    bias = 1 << 20
    if np.any(np.abs(q) >= bias):
        raise ValueError('Point outside supported +/- voxel*2^20 metric range')
    q += bias
    return (q[:, 0] << 42) | (q[:, 1] << 21) | q[:, 2]


def reduce_points(xyz, rgb, confidence, voxel):
    keys, inverse = np.unique(voxel_keys(xyz, voxel), return_inverse=True)
    weight = np.maximum(confidence.astype(np.float64), .001) ** 2
    weights = np.bincount(inverse, weights=weight, minlength=len(keys))
    position = np.stack([np.bincount(inverse, weights=xyz[:, i] * weight, minlength=len(keys)) for i in range(3)], -1)
    color = np.stack([np.bincount(inverse, weights=rgb[:, i] * weight, minlength=len(keys)) for i in range(3)], -1)
    return keys, position, color, weights


def merge_reductions(chunks, minimum_views):
    if not chunks:
        return None
    keys = np.concatenate([x[0] for x in chunks])
    unique, inverse = np.unique(keys, return_inverse=True)
    views = np.bincount(inverse, minlength=len(unique)).astype(np.int32)
    weight = np.bincount(inverse, weights=np.concatenate([x[3] for x in chunks]), minlength=len(unique))
    xyzsum = np.concatenate([x[1] for x in chunks])
    rgbsum = np.concatenate([x[2] for x in chunks])
    xyz = np.stack([np.bincount(inverse, weights=xyzsum[:, i], minlength=len(unique)) for i in range(3)], -1) / weight[:, None]
    rgb = np.stack([np.bincount(inverse, weights=rgbsum[:, i], minlength=len(unique)) for i in range(3)], -1) / weight[:, None]
    valid = views >= minimum_views
    return {'keys': unique[valid], 'xyz': xyz[valid].astype(np.float32),
            'rgb': np.clip(rgb[valid], 0, 255).astype(np.uint8), 'views': views[valid],
            'weight': weight[valid].astype(np.float32)}


def load_frame(run, frame):
    index = frame['frame_index']
    data = dict(np.load(run / 'rgbd' / f'frame_{index:04d}.npz'))
    shape = data['depth'].shape
    rgb = np.array(Image.open(frame['rgb_path']).convert('RGB').resize(shape[::-1], Image.Resampling.LANCZOS))
    proposal = read_json(run / 'proposals' / f'frame_{index:04d}.json')
    masks = read_masks(run / 'proposals' / proposal['mask_file'])
    if masks.shape[1:] != shape:
        masks = np.stack([np.array(Image.fromarray(x).resize(shape[::-1], Image.Resampling.NEAREST)) for x in masks]) if len(masks) else np.empty((0, *shape), bool)
    owner = make_owner(masks)
    points = world_points(data['depth'], data['K'], data['camera_to_world_cv'])
    return data, rgb, proposal, masks, owner, points


def main():
    args = stage_args()
    run, config, manifest, frames = context(args)
    cfg = config['fusion']
    start = time.monotonic()
    rng = np.random.default_rng(config['seed'])
    observations = []
    for frame in frames:
        data, rgb, proposal, masks, owner, points = load_frame(run, frame)
        candidates = []
        for obj in proposal['objects']:
            mask = owner == obj['proposal_id']
            if cfg['mask_erosion_pixels']:
                mask = binary_erosion(mask, iterations=cfg['mask_erosion_pixels'])
            valid = mask & data['valid'] & np.isfinite(points).all(-1)
            cloud = points[valid]
            if len(cloud) < cfg['minimum_lifted_points']:
                continue
            bounds = np.quantile(cloud, [.01, .99], axis=0)
            cloud = cloud[np.all((cloud >= bounds[0]) & (cloud <= bounds[1]), -1)]
            if not len(cloud):
                continue
            sample = cloud[rng.choice(len(cloud), min(cfg['sample_points_per_proposal'], len(cloud)), replace=False)]
            sample = np.unique(np.floor(sample / cfg['association_voxel_m']), axis=0) * cfg['association_voxel_m']
            candidates.append({'proposal_id': obj['proposal_id'], 'label': obj['label'], 'family': family(obj['label']),
                               'bbox': bounds, 'sample': sample, 'pixels': int(valid.sum())})
        observations.append({'frame_index': frame['frame_index'], 'proposals': candidates})
    tracks, active, associations = associate_frames(observations, cfg)
    minimum_frames=required_track_frames(cfg,len({f['frame_index'] for f in frames}))
    write_json(run/'track_support.json',{'input_frames':len({f['frame_index'] for f in frames}),
        'minimum_distinct_frames':minimum_frames,'minimum_fraction':cfg.get('minimum_track_fraction',0),
        'tracks':[{'raw_track_id':i,'distinct_frames':len(track['frames']),
            'family':track['family'],'selected_for_fusion':i in active,
            'label_votes':dict(track['labels'])} for i,track in enumerate(tracks)],
        'counting':'Unique input frame IDs, not proposal count; merged reappearance roots count each frame once'})
    write_json(run / 'association.json', {'method': 'Metric 3D surface overlap + label compatibility + one-to-one assignment; coexistence forbids merging',
        'frames': associations, 'raw_tracks': len(tracks), 'persistent_objects': len(active),
        'minimum_distinct_frames':required_track_frames(cfg,len({f['frame_index'] for f in frames})),
        'worker_sha256': sha256(__file__), 'settings': cfg})
    chunks = {i + 1: [] for i in range(len(active))}
    frame_lookup = {x['frame_index']: x for x in associations}
    (run / 'instances').mkdir(exist_ok=True)
    for frame in frames:
        data, rgb, proposal, masks, owner, points = load_frame(run, frame)
        labels = np.zeros(owner.shape, np.uint16)
        assignment = frame_lookup[frame['frame_index']]['proposal_to_object']
        for proposal_id, object_id in assignment.items():
            object_id = int(object_id)
            if not object_id:
                continue
            labels[owner == int(proposal_id)] = object_id
        # Keep complete 2D instance masks for visual review; point fusion uses
        # conservative eroded interiors + model confidence/edge filtering.
        Image.fromarray(labels).save(run / 'instances' / f"frame_{frame['frame_index']:04d}.png")
        for object_id in np.unique(labels):
            if object_id == 0:
                continue
            mask = labels == object_id
            if cfg['mask_erosion_pixels']:
                mask = binary_erosion(mask, iterations=cfg['mask_erosion_pixels'])
            valid = mask & data['valid'] & np.isfinite(points).all(-1)
            cloud, color, confidence = points[valid], rgb[valid], data['confidence'][valid]
            if len(cloud):
                bounds = np.quantile(cloud, [.005, .995], axis=0)
                keep = np.all((cloud >= bounds[0]) & (cloud <= bounds[1]), -1)
                chunks[int(object_id)].append(reduce_points(cloud[keep], color[keep], confidence[keep], cfg['voxel_m']))
        if frame['frame_index'] % 24 == 0:
            print(f'FUSE_POINTS {frame["frame_index"]:04d}', flush=True)
    fused = {object_id: merge_reductions(values, cfg['minimum_voxel_views']) for object_id, values in chunks.items()}
    del chunks
    gc.collect()
    # A voxel belongs to one object. Resolve overlapping object measurements by
    # multi-view confidence-weighted evidence instead of emitting duplicates.
    populated = {k: v for k, v in fused.items() if v is not None and len(v['keys'])}
    if not populated:
        raise RuntimeError('No persistent fused points; inspect depth/proposal quality')
    all_keys = np.concatenate([v['keys'] for v in populated.values()])
    all_weight = np.concatenate([v['weight'] for v in populated.values()])
    all_ids = np.concatenate([np.full(len(v['keys']), k, np.int32) for k, v in populated.items()])
    order = np.lexsort((-all_weight, all_keys))
    unique_mask = np.r_[True, np.diff(all_keys[order]) != 0]
    winning = order[unique_mask]
    xyz_all = np.concatenate([v['xyz'] for v in populated.values()])[winning]
    rgb_all = np.concatenate([v['rgb'] for v in populated.values()])[winning]
    views_all = np.concatenate([v['views'] for v in populated.values()])[winning]
    ids_all = all_ids[winning]
    from .consensus import refine, update_masks
    ids_all = refine(run, frames, xyz_all, ids_all, cfg)
    minimum_points = cfg.get('minimum_instance_points', 100)
    low_support_ids = {int(i) for i in np.unique(ids_all) if np.count_nonzero(ids_all == i) < minimum_points}
    skipped = []
    if cfg.get('skip_low_support', True):
        for i in sorted(low_support_ids):
            skipped.append({'object_id': i, 'points': int(np.count_nonzero(ids_all == i)), 'reason': f'fewer than {minimum_points} fused points'})
        supported = ~np.isin(ids_all, list(low_support_ids))
        xyz_all, rgb_all, views_all, ids_all = xyz_all[supported], rgb_all[supported], views_all[supported], ids_all[supported]
        if not len(ids_all):
            raise RuntimeError('All instances are low support; inspect depth/proposals or lower minimum_instance_points')
    candidates=[]
    for object_id in np.unique(ids_all):
        object_id=int(object_id);keep=ids_all==object_id;track=tracks[active[object_id-1]]
        candidates.append({'object_id':object_id,'family':track['family'],
            'observed_frames':sorted(track['frames']),'points':int(keep.sum()),
            'median_view_count':float(np.median(views_all[keep])),
            'bbox_world':np.quantile(xyz_all[keep],[cfg['bbox_quantile'],1-cfg['bbox_quantile']],axis=0).tolist()})
    remap,nms_report=nms_instances(candidates,cfg)
    write_json(run/'nms_3d.json',nms_report)
    for source_id,target_id in remap.items():ids_all[ids_all==source_id]=target_id
    skipped.extend(dict(row,stage='nms_3d',merged_into=row['kept_object_id']) for row in nms_report['suppressed'])
    write_json(run / 'skipped_objects.json', {'objects': skipped, 'minimum_instance_points': minimum_points})
    update_masks(run, frames, xyz_all, ids_all, cfg, remap=remap)
    objects, scene_xyz, scene_rgb, scene_ids = [], [], [], []
    for object_id in populated:
        keep = ids_all == object_id
        xyz, color, views = xyz_all[keep], rgb_all[keep], views_all[keep]
        if not len(xyz):
            continue
        track = tracks[active[object_id - 1]]
        sources=[object_id]+sorted(source for source,target in remap.items() if target==object_id)
        observed_frames=set();label_votes=Counter()
        for source_id in sources:
            observed_frames.update(tracks[active[source_id-1]]['frames'])
            label_votes.update(tracks[active[source_id-1]]['labels'])
        folder = run / 'objects' / f'obj_{object_id:03d}'
        folder.mkdir(parents=True, exist_ok=True)
        q = cfg['bbox_quantile']
        bbox = np.quantile(xyz, [q, 1 - q], axis=0)
        full_bbox = np.stack([xyz.min(0), xyz.max(0)])
        write_ply(folder / 'rgb.ply', xyz, color, view_count=views)
        np.savez_compressed(folder / 'points.npz', xyz=xyz, rgb=color, view_count=views,
                            bbox_world=bbox, bbox_world_full=full_bbox)
        record = {'object_id': object_id, 'name': f'obj_{object_id:03d}', 'family': track['family'],
                  'label_votes': dict(label_votes), 'observed_frames': sorted(observed_frames),
                  'points': len(xyz), 'bbox_world': bbox.tolist(), 'bbox_world_full': full_bbox.tolist(),
                  'bbox_quantile': q, 'units': 'meters', 'voxel_m': cfg['voxel_m'],
                  'median_view_count': float(np.median(views)), 'pointcloud': f'objects/obj_{object_id:03d}/rgb.ply'}
        record['low_support'] = len(xyz) < minimum_points
        if remap:record['merged_source_ids']=sources
        record['support_note'] = f'Fewer than {minimum_points} fused points; weak proposal' if record['low_support'] else 'Persistent multi-view surface measurements'
        write_json(folder / 'object.json', record)
        objects.append(record)
        scene_xyz.append(xyz)
        scene_rgb.append(color)
        scene_ids.append(np.full(len(xyz), object_id, np.int32))
    xyz, rgb, ids = np.concatenate(scene_xyz), np.concatenate(scene_rgb), np.concatenate(scene_ids)
    write_ply(run / 'scene_segmented_rgb.ply', xyz, rgb, object_id=ids)
    np.savez_compressed(run / 'scene_segmented.npz', xyz=xyz, rgb=rgb, object_id=ids)
    write_json(run / 'objects.json', {'objects': objects, 'skipped_objects': skipped, 'frames': len(frames), 'points': len(xyz),
               'coordinate_frame': 'Provided camera world, meters',
               'input_contract': manifest['input_contract'], 'fusion_settings': cfg,
               'seconds': time.monotonic() - start, 'raw_tracks': len(tracks)})
    print(f'FUSED {len(objects)} objects {len(xyz)} points {time.monotonic() - start:.1f}s', flush=True)


if __name__ == '__main__':
    main()
