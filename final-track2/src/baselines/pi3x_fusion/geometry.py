"""Pi3X RGB + pose/K conditioning; metric RGB-D maps for every input frame."""
import sys
from pathlib import Path
import time
import numpy as np
from PIL import Image
from .common import context, read_json, resolve, sha256, stage_args, write_json


def fit_similarity(source, target):
    """Source->target Sim(3), fitted only to provided camera centers."""
    source, target = np.asarray(source, float), np.asarray(target, float)
    a, b = source - source.mean(0), target - target.mean(0)
    covariance = b.T @ a / len(a)
    u, singular, vt = np.linalg.svd(covariance)
    sign = np.ones(3)
    sign[-1] = np.linalg.det(u @ vt)
    rotation = u @ np.diag(sign) @ vt
    variance = np.mean(np.sum(a * a, 1))
    if variance < 1e-10:
        raise ValueError('Insufficient camera motion to recover metric scale')
    scale = float(np.dot(singular, sign) / variance)
    if scale <= 0 or not np.isfinite(scale):
        raise ValueError('Invalid camera-derived metric scale')
    translation = target.mean(0) - scale * rotation @ source.mean(0)
    residual = np.linalg.norm(scale * source @ rotation.T + translation - target, axis=1)
    return scale, rotation, translation, residual


def load_model(cfg):
    import torch
    from safetensors.torch import load_file
    sys.path.insert(0, str(resolve(cfg['repo'])))
    from pi3.models.pi3x import Pi3X
    from ..loading import empty_parameters,assign_weights
    with empty_parameters():model=Pi3X(use_multimodal=True).eval()
    # No prior depth is supplied. Avoid allocating/initializing its encoder.
    model.depth_encoder=None
    state={key:value for key,value in load_file(str(resolve(cfg['weights'])),device='cpu').items() if not key.startswith('depth_encoder.')}
    assign_weights(model,state)
    del state
    model.to(dtype=torch.bfloat16)
    # Upstream heads explicitly disable autocast and promote features to FP32.
    for name in ['point_head', 'conf_head', 'camera_head', 'metric_head']:
        getattr(model, name).float()
    return model.to('cuda')


def main(args=None):
    import torch
    torch.set_num_threads(4)
    args = stage_args() if args is None else args
    run, config, manifest, targets = context(args)
    cfg = config['geometry']
    folder = run / 'rgbd'
    folder.mkdir(exist_ok=True)
    signature = {'settings': cfg, 'worker_sha256': sha256(__file__),
                 'calibration_sha256': manifest['calibration_sha256'],
                 'dependencies_sha256':{name:sha256(Path(__file__).parent/name) for name in ['encoder_cache.py','rope.py']},
                 'loading_sha256':sha256(Path(__file__).parents[1]/'loading.py')}
    if (folder / 'settings.json').exists() and read_json(folder / 'settings.json') != signature:
        raise ValueError('Geometry settings changed; use another run ID')
    write_json(folder / 'settings.json', signature)
    calibration = np.load(run / 'input/cameras.npz')
    frame_to_position = {int(index): i for i, index in enumerate(calibration['frame_indices'])}
    images = {x['frame_index']: x for x in manifest['frames']}
    anchors = [manifest['frames'][i]['frame_index'] for i in np.linspace(0, len(manifest['frames']),
               cfg['anchor_views'], endpoint=False, dtype=int)]
    resolution = cfg['resolution']
    width, height = calibration['image_size_wh']
    K = calibration['K'].copy()
    K[0] *= resolution / width
    K[1] *= resolution / height
    cached_images = {}
    def image_tensor(index):
        if index not in cached_images:
            rgb = np.array(Image.open(images[index]['rgb_path']).convert('RGB').resize(
                (resolution, resolution), Image.Resampling.LANCZOS), dtype=np.float32) / 255
            cached_images[index] = torch.from_numpy(rgb).permute(2, 0, 1)
        return cached_images[index]
    pending = [x['frame_index'] for x in targets if not (folder / f"frame_{x['frame_index']:04d}.npz").exists()]
    batches = []
    start = time.monotonic()
    if pending:
        torch.manual_seed(config['seed'])
        torch.cuda.reset_peak_memory_stats()
        model = load_model(cfg)
        from .rope import install
        install(model,resolution)
        from .encoder_cache import AnchorEncoderCache
        model.encoder=AnchorEncoderCache(model.encoder,anchors)
        from pi3.utils.geometry import depth_normal_edge
        for offset in range(0, len(pending), cfg['target_batch']):
            batch_targets = pending[offset:offset + cfg['target_batch']]
            indices = list(dict.fromkeys(anchors + batch_targets))
            model.encoder.indices=indices
            poses = calibration['camera_to_world_cv'][[frame_to_position[i] for i in indices]]
            imgs = torch.stack([image_tensor(i) for i in indices])[None].to('cuda')
            ks = torch.tensor(np.repeat(K[None], len(indices), 0), dtype=torch.float32, device='cuda')[None]
            pose_tensor = torch.tensor(poses, dtype=torch.float32, device='cuda')[None]
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                output = model(imgs=imgs, intrinsics=ks, poses=pose_tensor)
                confidence = output['conf'][0, ..., 0].sigmoid()
                valid = confidence > cfg['confidence_threshold']
                edges = depth_normal_edge(output['local_points'][0], rtol=cfg['depth_edge_rtol'], mask=valid)
            predicted_poses = output['camera_poses'][0].float().cpu().numpy()
            # Pose conditioning is relative; Pi3X still predicts its own camera
            # poses/metric factor. Calibrate scale using the distributed anchors,
            # then unproject predicted depth with the exact provided K/poses.
            scale, rotation, translation, residual = fit_similarity(
                predicted_poses[:len(anchors), :3, 3], poses[:len(anchors), :3, 3])
            depth = output['local_points'][0, ..., 2].float().cpu().numpy() * scale
            confidence_np = confidence.float().cpu().numpy()
            valid_np = (valid & ~edges).cpu().numpy() & np.isfinite(depth) & (depth > .02) & (depth < 30)
            batch_record = {'indices': indices, 'targets': batch_targets, 'scale_from_camera_centers': scale,
                'anchor_center_rmse_m': float(np.sqrt(np.mean(residual ** 2))),
                'rotation_source_to_world': rotation.tolist(), 'translation_source_to_world': translation.tolist(),
                'predicted_camera_to_world': predicted_poses.tolist(), 'metric_head': float(output['metric'][0]),
                'peak_allocated_bytes': torch.cuda.max_memory_allocated()}
            batch_id = f'batch_{batch_targets[0]:04d}'
            write_json(folder / (batch_id + '.json'), batch_record)
            batches.append(batch_record)
            for index in batch_targets:
                j = indices.index(index)
                np.savez_compressed(folder / f'frame_{index:04d}.npz', depth=depth[j].astype(np.float32),
                    confidence=confidence_np[j].astype(np.float16), valid=valid_np[j],
                    K=K, camera_to_world_cv=calibration['camera_to_world_cv'][frame_to_position[index]],
                    frame_index=index, camera_scale=scale)
                if index not in anchors:
                    cached_images.pop(index, None)
            del output, imgs, ks, pose_tensor, confidence, valid, edges
            print(f'RGBD {batch_targets[0]:04d}-{batch_targets[-1]:04d} context={len(indices)} scale={scale:.4f} camera_RMSE={batch_record["anchor_center_rmse_m"]:.4f}m', flush=True)
    write_json(folder / 'summary.json', {'frames': len(targets), 'resolution': [resolution, resolution],
        'anchors': anchors, 'confidence_threshold': cfg['confidence_threshold'],
        'seconds': time.monotonic() - start, 'new_batches': len(batches),
        'input_contract': 'RGB + provided OpenCV camera-to-world poses and pixel intrinsics; no GT depth',
        'depth_convention': 'Camera z depth in meters after camera-center scale calibration',
        'world_convention': 'Provided camera world; exact given K and poses used for depth unprojection',
        'conditioning': 'Poses and intrinsics; depths=None; unused depth encoder removed',
        'precision': 'BF16 weights/attention, FP32 output heads'})


if __name__ == '__main__':
    main()
