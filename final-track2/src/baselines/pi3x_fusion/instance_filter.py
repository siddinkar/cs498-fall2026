"""Distinct-frame support and deterministic class-agnostic 3D AABB NMS."""
import math
import numpy as np


def required_track_frames(settings,total_frames):
    fraction=float(settings.get('minimum_track_fraction',0))
    if not np.isfinite(fraction) or not 0<=fraction<=1:
        raise ValueError('minimum_track_fraction must be in [0, 1]')
    return max(int(settings['minimum_track_views']),math.ceil(fraction*total_frames))


def bbox_iou_3d(a,b):
    a,b=np.asarray(a,dtype=np.float64),np.asarray(b,dtype=np.float64)
    intersection=np.maximum(0,np.minimum(a[1],b[1])-np.maximum(a[0],b[0])).prod()
    volume_a=np.maximum(a[1]-a[0],0).prod();volume_b=np.maximum(b[1]-b[0],0).prod()
    return float(intersection/max(volume_a+volume_b-intersection,1e-12))


def nms_instances(objects,settings):
    options=settings.get('nms_3d',{})
    enabled=bool(options.get('enabled',False));threshold=float(options.get('iou_threshold',.65))
    if not np.isfinite(threshold) or not 0<threshold<=1:raise ValueError('3D NMS IoU must be in (0, 1]')
    ranked=sorted(objects,key=lambda o:(-len(set(o['observed_frames'])),-o['median_view_count'],-o['points'],o['object_id']))
    kept=[];suppressed=[];mapping={}
    for candidate in ranked:
        duplicate=None
        if enabled:
            for winner in kept:
                overlap=bbox_iou_3d(candidate['bbox_world'],winner['bbox_world'])
                if overlap>=threshold:
                    duplicate={'object_id':candidate['object_id'],'kept_object_id':winner['object_id'],
                        'bbox_3d_iou':overlap,'reason':'3D bbox NMS: duplicate instance; points merged into retained instance'}
                    break
        if duplicate:
            suppressed.append(duplicate);mapping[candidate['object_id']]=duplicate['kept_object_id']
        else:kept.append(candidate)
    return mapping,{'enabled':enabled,'class_agnostic':True,'iou_threshold':threshold,
        'bbox_source':'Quantile bounds of fused predicted RGB-D points, in dataset world meters',
        'ranking':'Distinct observed frames, median voxel view count, fused points; instance ID breaks ties',
        'candidates':objects,'kept_object_ids':sorted(o['object_id'] for o in kept),
        'suppressed':suppressed,'points_of_suppressed_instances':'Merged into retained instance; no rescaling or GT input'}
