#!/usr/bin/env python3
"""Check calibration, RGB-D/mask contracts, fused clouds and visualization timing."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
from PIL import Image
from plyfile import PlyData

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from baselines.pi3x_fusion.common import read_json, read_masks, resolve, sha256, write_json
from baselines.pi3x_fusion.fuse import voxel_keys


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', required=True)
    args = p.parse_args()
    run = resolve(args.run_dir)
    manifest = read_json(run / 'input/frames.json')
    cfg = read_json(run / 'config.json')
    objects = read_json(run / 'objects.json')['objects']
    object_ids = {x['object_id'] for x in objects}
    calibration = np.load(run / 'input/cameras.npz')
    positions = {int(index): i for i, index in enumerate(calibration['frame_indices'])}
    expected_K = calibration['K'].copy()
    expected_K[:2] *= cfg['geometry']['resolution'] / calibration['image_size_wh'][0]
    annotations = read_json(run / 'visualizations/frame_annotations.json')['frames']
    assert len(annotations) == len(manifest['frames'])
    assert sha256(Path(manifest['source_scene']) / 'cameras.npz') == manifest['calibration_sha256']
    for frame in manifest['frames']:
        index = frame['frame_index']
        assert sha256(frame['rgb_path']) == frame['rgb_sha256']
        data = np.load(run / 'rgbd' / f'frame_{index:04d}.npz')
        np.testing.assert_array_equal(data['camera_to_world_cv'], calibration['camera_to_world_cv'][positions[index]])
        np.testing.assert_allclose(data['K'], expected_K, atol=1e-9)
        assert data['depth'].shape == (cfg['geometry']['resolution'],) * 2
        assert np.all(np.isfinite(data['depth'][data['valid']]))
        assert np.all(data['depth'][data['valid']] > 0)
        assert np.all((data['confidence'] >= 0) & (data['confidence'] <= 1))
        proposal = read_json(run / 'proposals' / f'frame_{index:04d}.json')
        masks = read_masks(run / 'proposals' / proposal['mask_file'])
        assert len(masks) == len(proposal['objects'])
        assert masks.shape[1:] == tuple(calibration['image_size_wh'][::-1])
        labels = np.asarray(Image.open(run / 'instances' / f'frame_{index:04d}.png'))
        assert set(np.unique(labels)) <= {0, *object_ids}
    data = np.load(run / 'scene_segmented.npz')
    assert np.isfinite(data['xyz']).all()
    assert len(data['xyz']) == len(data['rgb']) == len(data['object_id'])
    assert len(np.unique(voxel_keys(data['xyz'], cfg['fusion']['voxel_m']))) == len(data['xyz'])
    assert set(np.unique(data['object_id'])) == object_ids
    assert len(PlyData.read(run / 'scene_segmented_rgb.ply')['vertex']) == len(data['xyz'])
    for obj in objects:
        cloud = np.load(run / 'objects' / obj['name'] / 'points.npz')
        assert len(cloud['xyz']) == obj['points']
        assert cloud['view_count'].min() >= cfg['fusion']['minimum_voxel_views']
        bounds = np.asarray(obj['bbox_world_full'])
        assert np.all(cloud['xyz'] >= bounds[0] - 1e-6) and np.all(cloud['xyz'] <= bounds[1] + 1e-6)
        from baselines.pi3x_fusion.instance_filter import required_track_frames
        assert len(set(obj['observed_frames']))>=required_track_frames(cfg['fusion'],len(manifest['frames']))
    if cfg['fusion'].get('nms_3d',{}).get('enabled'):
        nms=read_json(run/'nms_3d.json')
        assert object_ids==set(nms['kept_object_ids'])
        assert not object_ids & {row['object_id'] for row in nms['suppressed']}
        assert (run/'visualizations/nms_3d_multiview.png').is_file()
    videos = {}
    for name in ['rgbd_masks_bbox.mp4', 'projected_3d_bbox.mp4']:
        result = subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height,nb_frames,r_frame_rate,duration', '-of', 'json',
            str(run / 'visualizations' / name)], text=True)
        stream = json.loads(result)['streams'][0]
        assert int(stream['nb_frames']) == len(manifest['frames'])
        assert stream['r_frame_rate'] == '24/1'
        assert abs(float(stream['duration']) - 8.) < 1e-6
        videos[name] = stream
    assert (run / 'visualizations/pointcloud.html').is_file()
    report = {'passed': True, 'frames': len(manifest['frames']), 'objects': len(objects),
        'points': len(data['xyz']), 'low_support_object_ids': [x['object_id'] for x in objects if x['low_support']],
        'input_contract': manifest['input_contract'], 'calibration_and_input_rgb_unchanged': True,
        'finite_metric_pointcloud': True, 'unique_fused_voxels': True, 'videos': videos,
        'script_sha256': sha256(__file__)}
    report['minimum_distinct_frames']=required_track_frames(cfg['fusion'],len(manifest['frames']))
    report['nms_3d']=cfg['fusion'].get('nms_3d',{'enabled':False})
    write_json(run / 'validation.json', report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
