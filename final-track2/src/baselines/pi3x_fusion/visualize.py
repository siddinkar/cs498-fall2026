"""2D masks/boxes, RGB-D review video, 3D point clouds and interactive bboxes."""
import itertools
import subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from .common import context, read_json, read_masks, stage_args, write_json

from ..colors import PALETTE, color


def bbox_corners(bounds):
    return np.asarray(list(itertools.product(*zip(bounds[0], bounds[1]))), float)


def bbox_edges(bounds):
    corners = bbox_corners(bounds)
    return [(corners[i], corners[i ^ bit]) for i in range(8) for bit in [1, 2, 4] if i < (i ^ bit)]


def font(size=14):
    try:
        return ImageFont.truetype('DejaVuSans.ttf', size)
    except OSError:
        return ImageFont.load_default()


def overlay(rgb, labels, names):
    pixels = np.asarray(rgb).copy()
    canvas = pixels.copy()
    for object_id in np.unique(labels):
        if not object_id:
            continue
        mask = labels == object_id
        canvas[mask] = (.58 * canvas[mask] + .42 * np.array(color(int(object_id)))).astype(np.uint8)
    image = Image.fromarray(canvas)
    draw = ImageDraw.Draw(image)
    annotations = []
    for object_id in np.unique(labels):
        if not object_id:
            continue
        yy, xx = np.nonzero(labels == object_id)
        box = [int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1]
        c = color(int(object_id))
        draw.rectangle(box, outline=c, width=2)
        text = names.get(int(object_id), str(object_id))
        draw.rectangle([box[0], max(0, box[1] - 19), min(image.width, box[0] + len(text) * 8 + 5), box[1]], fill=c)
        draw.text((box[0] + 2, max(0, box[1] - 18)), text, fill=(15, 15, 15), font=font(13))
        annotations.append({'object_id': int(object_id), 'bbox_xyxy': box, 'pixels': len(xx)})
    return image, annotations


def projected_boxes(rgb, objects, K, camera_to_world):
    image = Image.fromarray(np.asarray(rgb).copy())
    draw = ImageDraw.Draw(image)
    inverse = np.linalg.inv(camera_to_world)
    for obj in objects:
        corners = bbox_corners(obj['bbox_world'])
        camera = corners @ inverse[:3, :3].T + inverse[:3, 3]
        projected = camera @ K.T
        projected = projected[:, :2] / projected[:, 2:3]
        c = color(obj['object_id'])
        for i in range(8):
            for bit in [1, 2, 4]:
                j = i ^ bit
                if i < j and camera[i, 2] > 0 and camera[j, 2] > 0:
                    draw.line([tuple(projected[i]), tuple(projected[j])], fill=c, width=2)
        anchor = projected.mean(0)
        if np.isfinite(anchor).all():
            draw.text(tuple(anchor), f'{obj["object_id"]}: {obj["family"]}', fill=c, font=font(14), stroke_width=1, stroke_fill=(0, 0, 0))
    return image


def sampled_clouds(run, objects, limit=25000):
    rng = np.random.default_rng(498)
    result = []
    for obj in objects:
        data = np.load(run / 'objects' / obj['name'] / 'points.npz')
        xyz, rgb = data['xyz'], data['rgb']
        bounds = np.asarray(obj['bbox_world'])
        keep = np.all((xyz >= bounds[0] - .01) & (xyz <= bounds[1] + .01), axis=-1)
        xyz, rgb = xyz[keep], rgb[keep]
        if len(xyz) > limit:
            selected = rng.choice(len(xyz), limit, replace=False)
            xyz, rgb = xyz[selected], rgb[selected]
        result.append((obj, xyz, rgb))
    return result


def static_clouds(run, objects, viz):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    clouds = sampled_clouds(run, objects)
    for mode in ['rgb', 'instances']:
        fig = plt.figure(figsize=(10, 8), dpi=140)
        ax = fig.add_subplot(111, projection='3d')
        for obj, xyz, rgb in clouds:
            colors = rgb / 255 if mode == 'rgb' else np.tile(np.array(color(obj['object_id'])) / 255, (len(xyz), 1))
            ax.scatter(*xyz.T, c=colors, s=.8, depthshade=False, rasterized=True,
                       label=f'{obj["object_id"]}: {obj["family"]} ({obj["points"]:,} points)')
            for a, b in bbox_edges(obj['bbox_world']):
                ax.plot(*np.stack([a, b]).T, color=np.array(color(obj['object_id'])) / 255, linewidth=.9)
        bounds = np.array([x['bbox_world'] for x in objects])
        low, high = bounds[:, 0].min(0), bounds[:, 1].max(0)
        ax.set_xlim(low[0] - .03, high[0] + .03)
        ax.set_ylim(low[1] - .03, high[1] + .03)
        ax.set_zlim(low[2] - .03, high[2] + .03)
        ax.set_box_aspect(np.maximum(high - low, .01))
        ax.view_init(elev=27, azim=-65)
        ax.set(xlabel='X (m)', ylabel='Y (m)', zlabel='Z (m)', title=f'Pi3X + 3D instance fusion | {mode} | measured surfaces')
        ax.legend(loc='upper left', fontsize=8)
        fig.tight_layout()
        fig.savefig(viz / f'pointcloud_{mode}.png')
        plt.close(fig)
    columns = 3
    rows = (len(objects) + columns - 1) // columns
    fig = plt.figure(figsize=(15, 4.4 * rows), dpi=110)
    for i, (obj, xyz, rgb) in enumerate(clouds):
        ax = fig.add_subplot(rows, columns, i + 1, projection='3d')
        ax.scatter(*xyz.T, c=rgb / 255, s=1.8, depthshade=False, rasterized=True)
        for a, b in bbox_edges(obj['bbox_world']):
            ax.plot(*np.stack([a, b]).T, color=np.array(color(obj['object_id'])) / 255, linewidth=.8)
        bounds = np.asarray(obj['bbox_world'])
        center = bounds.mean(0)
        extent = max((bounds[1] - bounds[0]).max(), .02) * .6
        ax.set_xlim(center[0] - extent, center[0] + extent)
        ax.set_ylim(center[1] - extent, center[1] + extent)
        ax.set_zlim(center[2] - extent, center[2] + extent)
        ax.set_box_aspect([1, 1, 1])
        ax.view_init(elev=25, azim=-65)
        ax.set_title(f'{obj["name"]}: {obj["family"]}\n{obj["points"]:,} points, {len(obj["observed_frames"])} frames', fontsize=10)
        ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(viz / 'objects_rgb.png')
    plt.close(fig)


def interactive_clouds(run, objects, viz):
    import plotly.graph_objects as go
    clouds = sampled_clouds(run, objects)
    traces = []
    for mode in ['rgb', 'instances']:
        for obj, xyz, rgb in clouds:
            colors = [f'rgb({r},{g},{b})' for r, g, b in rgb] if mode == 'rgb' else 'rgb(%d,%d,%d)' % color(obj['object_id'])
            traces.append(go.Scatter3d(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode='markers',
                name=f'{obj["name"]} {obj["family"]}', visible=mode == 'rgb',
                marker=dict(size=2, color=colors), hovertemplate=f'{obj["name"]}: {obj["family"]}<br>X=%{{x:.3f}} Y=%{{y:.3f}} Z=%{{z:.3f}}<extra></extra>'))
    for obj in objects:
        lines = []
        for a, b in bbox_edges(obj['bbox_world']):
            lines.extend([a, b, [None, None, None]])
        x, y, z = zip(*lines)
        traces.append(go.Scatter3d(x=x, y=y, z=z, mode='lines', name=f'BBox {obj["object_id"]}',
            line=dict(color='rgb(%d,%d,%d)' % color(obj['object_id']), width=4), showlegend=False))
    n = len(objects)
    buttons = [dict(label='All: RGB', method='update', args=[{'visible': [True] * n + [False] * n + [True] * n}]),
               dict(label='All: instance colors', method='update', args=[{'visible': [False] * n + [True] * n + [True] * n}])]
    for i, obj in enumerate(objects):
        visible = [False] * (3 * n)
        visible[i] = visible[2 * n + i] = True
        buttons.append(dict(label=f'{obj["name"]}: {obj["family"]}', method='update', args=[{'visible': visible}]))
    fig = go.Figure(traces)
    fig.update_layout(title='Pi3X RGB-D + Florence/SAM2 + 3D fusion (provided camera world)',
        scene=dict(aspectmode='data', xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='Z (m)',
                   camera=dict(up=dict(x=0, y=0, z=1), eye=dict(x=1.5, y=-1.8, z=1.2))),
        updatemenus=[dict(buttons=buttons, x=0, y=1.08)], margin=dict(l=0, r=0, b=0, t=100))
    fig.write_html(viz / 'pointcloud.html', include_plotlyjs=True, full_html=True)


def filter_review(run,viz):
    import matplotlib.pyplot as plt
    nms=read_json(run/'nms_3d.json');support=read_json(run/'track_support.json')
    candidates=nms['candidates'];suppressed={row['object_id']:row for row in nms['suppressed']}
    counts=[row['distinct_frames'] for row in support['tracks']]
    fig,ax=plt.subplots(figsize=(10,4),dpi=140)
    ax.hist(counts,bins=np.arange(0,support['input_frames']+9,8),color='#528ba7')
    ax.axvline(support['minimum_distinct_frames'],color='#bc3838',label=f"Required: {support['minimum_distinct_frames']} / {support['input_frames']} distinct frames")
    ax.set(xlabel='Distinct observed input frames',ylabel='Raw tracks',title='Track support gate (before fused instance generation)')
    ax.legend();fig.tight_layout();fig.savefig(viz/'track_support.png');plt.close(fig)
    bounds=np.asarray([row['bbox_world'] for row in candidates]);low=bounds[:,0].min(0);high=bounds[:,1].max(0)
    fig=plt.figure(figsize=(15,9),dpi=140)
    for row in range(2):
        for column,azimuth in enumerate([-65,55,175]):
            ax=fig.add_subplot(2,3,row*3+column+1,projection='3d')
            for obj in candidates:
                identity=obj['object_id'];rejected=identity in suppressed
                if row and rejected:continue
                c='#c23939' if rejected else np.asarray(color(identity))/255
                for a,b in bbox_edges(obj['bbox_world']):ax.plot(*np.stack([a,b]).T,color=c,linewidth=1.2,linestyle='--' if rejected else '-')
                center=np.asarray(obj['bbox_world']).mean(0)
                suffix=f" -> {suppressed[identity]['kept_object_id']}" if rejected else ''
                ax.text(*center,f'{identity}: {obj["family"]}{suffix}',fontsize=7,color=c)
            ax.set_xlim(low[0]-.03,high[0]+.03);ax.set_ylim(low[1]-.03,high[1]+.03);ax.set_zlim(low[2]-.03,high[2]+.03)
            ax.set_box_aspect(np.maximum(high-low,.01));ax.view_init(25,azimuth)
            ax.set_title(('Before NMS' if not row else 'Retained instances')+f' | azimuth {azimuth}°')
            ax.set(xlabel='X (m)',ylabel='Y (m)',zlabel='Z (m)')
    fig.suptitle(f"3D bbox NMS: IoU >= {nms['iou_threshold']} | dashed red boxes merged into retained instances")
    fig.tight_layout();fig.savefig(viz/'nms_3d_multiview.png');plt.close(fig)


def main():
    import matplotlib
    args = stage_args()
    run, config, manifest, frames = context(args)
    objects = read_json(run / 'objects.json')['objects']
    names = {x['object_id']: f'{x["object_id"]}: {x["family"]}' + (' (weak)' if x.get('low_support') else '') for x in objects}
    viz = run / 'visualizations'
    for folder in ['overview', 'proposals', 'fused', 'projected_bbox']:
        (viz / folder).mkdir(parents=True, exist_ok=True)
    calibration = np.load(run / 'input/cameras.npz')
    frame_to_position = {int(x): i for i, x in enumerate(calibration['frame_indices'])}
    sampled_depth = []
    for frame in frames[::max(1, len(frames) // 16)]:
        data = np.load(run / 'rgbd' / f"frame_{frame['frame_index']:04d}.npz")
        sampled_depth.append(data['depth'][data['valid']][::20])
    low, high = np.quantile(np.concatenate(sampled_depth), [.02, .98])
    annotations, overview_files = [], []
    cmap = matplotlib.colormaps['turbo']
    for position, frame in enumerate(frames):
        index = frame['frame_index']
        rgb = np.array(Image.open(frame['rgb_path']).convert('RGB'))
        labels_image = Image.open(run / 'instances' / f'frame_{index:04d}.png')
        labels = np.asarray(labels_image.resize((rgb.shape[1], rgb.shape[0]), Image.Resampling.NEAREST))
        fused, frame_annotations = overlay(rgb, labels, names)
        proposal = read_json(run / 'proposals' / f'frame_{index:04d}.json')
        masks = read_masks(run / 'proposals' / proposal['mask_file'])
        from .fuse import make_owner
        proposal_labels = make_owner(masks) + 1
        proposed, _ = overlay(rgb, proposal_labels, {x['proposal_id'] + 1: f'p{x["proposal_id"]}: {x["label"][:28]}' for x in proposal['objects']})
        data = np.load(run / 'rgbd' / f'frame_{index:04d}.npz')
        normalized = np.clip((data['depth'] - low) / max(high - low, .01), 0, 1)
        depth_rgb = (cmap(normalized)[..., :3] * 255).astype(np.uint8)
        depth_rgb[~data['valid']] = 0
        depth = Image.fromarray(depth_rgb).resize((rgb.shape[1], rgb.shape[0]))
        bbox = projected_boxes(rgb, objects, calibration['K'], calibration['camera_to_world_cv'][frame_to_position[index]])
        filename = f'frame_{position:04d}'
        proposed.save(viz / 'proposals' / (filename + '.jpg'), quality=92)
        fused.save(viz / 'fused' / (filename + '.jpg'), quality=92)
        bbox.save(viz / 'projected_bbox' / (filename + '.jpg'), quality=92)
        w, h = rgb.shape[1], rgb.shape[0]
        combined = Image.new('RGB', (4 * w, h + 32), (20, 20, 20))
        draw = ImageDraw.Draw(combined)
        for column, (image, title) in enumerate([(Image.fromarray(rgb), 'RGB'), (depth, f'Pi3X depth: {low:.2f}-{high:.2f} m'),
                (proposed, 'Florence + SAM2 proposals'), (fused, '3D-fused masks + 2D boxes')]):
            combined.paste(image, (column * w, 32))
            draw.text((column * w + 8, 7), f'Frame {index} | {title}', fill=(245, 245, 245), font=font(14))
        target = viz / 'overview' / (filename + '.png')
        combined.save(target)
        overview_files.append(target)
        annotations.append({'source_frame_index': index, 'visualization_frame_index': position,
                            'objects': frame_annotations})
    fps = config['fps'] / config['frame_stride']
    for folder, ext, target in [('overview', 'png', 'rgbd_masks_bbox.mp4'),
                                 ('projected_bbox', 'jpg', 'projected_3d_bbox.mp4')]:
        subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-framerate', str(fps),
            '-i', str(viz / folder / f'frame_%04d.{ext}'), '-frames:v', str(len(frames)), '-an',
            '-c:v', 'libx264', '-threads', '4', '-crf', '18', '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart', str(viz / target)], check=True)
    chosen = np.linspace(0, len(frames) - 1, 6, dtype=int)
    sheet = Image.new('RGB', (1536, 408 * len(chosen)))
    for row, position in enumerate(chosen):
        sheet.paste(Image.open(overview_files[position]).resize((1536, 408)), (0, row * 408))
    sheet.save(viz / 'masks_bbox_contact_sheet.png')
    write_json(viz / 'frame_annotations.json', {'frames': annotations, 'image_size_wh': calibration['image_size_wh'].tolist(),
        'bbox_2d_convention': 'Tight visible fused-mask box in original RGB pixels; x2/y2 exclusive',
        'bbox_3d': 'objects.json bbox_world in provided camera world, meters',
        'depth_preview_range_m': [float(low), float(high)]})
    static_clouds(run, objects, viz)
    interactive_clouds(run, objects, viz)
    if (run/'nms_3d.json').exists() and (run/'track_support.json').exists():filter_review(run,viz)
    print(f'VISUALIZED {len(frames)} frames {len(objects)} objects', flush=True)


if __name__ == '__main__':
    main()
