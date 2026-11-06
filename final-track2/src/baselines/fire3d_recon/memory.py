"""Bit-exact native unit-box filtering/sampling with host feature storage.

The native sampler selects XYZ/instance IDs independently of feature values.
Run that unchanged sampler on integer row tags, then gather only the selected
real rows. This avoids two full N x 1024 filtered/sorted temporary tensors;
all points, FPS calls, RNG consumption, masks, feature values and dtypes remain.
"""
import importlib
from dataclasses import dataclass


@dataclass
class DeferredRows:
    source: object
    rows: object


@dataclass
class CachedSamples:
    values: object
    positions: object
    mask: object
    cpu_before: object
    cuda_before: object
    cpu_after: object
    cuda_after: object
    on_use: object = None

    def __getitem__(self, index):
        if index != 0:
            raise IndexError('Cached conditioning requires dataloader batch size 1')
        return self

    def sample(self, max_num_objects, max_feats_len, sample_method):
        import torch
        if sample_method != 'fps' or self.mask.shape != (max_num_objects, max_feats_len):
            raise ValueError('Cached conditioning does not match native sampler arguments')
        if not torch.equal(torch.get_rng_state(), self.cpu_before) or not torch.equal(torch.cuda.get_rng_state(), self.cuda_before):
            raise ValueError('Native FPS entry RNG changed; refusing to alter inference noise')
        torch.set_rng_state(self.cpu_after)
        torch.cuda.set_rng_state(self.cuda_after)
        if self.on_use:
            self.on_use()
        return self.values, self.positions, self.mask


def feature_rows(feats,device):
    import torch
    if isinstance(feats,DeferredRows):return feats.source,feats.rows
    return feats,torch.arange(len(feats),device=device,dtype=torch.int64)


def deferred_unit_box_filter(native):
    def filter_points(coords,feats,instance_ids,object_transforms,max_num_objects):
        if isinstance(feats,CachedSamples):return coords,feats,instance_ids
        source,rows=feature_rows(feats,coords.device)
        positions,tags,ids=native(coords,rows[:,None],instance_ids,object_transforms,max_num_objects)
        return positions,DeferredRows(source,tags[:,0]),ids
    filter_points._native_unit_box_filter=native
    return filter_points


def deferred_feature_gather(native):
    def sample(coords,feats,instance_ids,max_num_objects,max_feats_len,sample_method='random'):
        import torch
        if isinstance(feats,CachedSamples):return feats.sample(max_num_objects,max_feats_len,sample_method)
        source,rows=feature_rows(feats,coords.device)
        tags=rows[:,None]
        indices,positions,mask=native(coords,tags,instance_ids,max_num_objects,max_feats_len,sample_method)
        if not len(source):
            values=torch.zeros((*mask.shape,source.shape[-1]),device=coords.device,dtype=source.dtype)
        else:
            selected=indices[...,0].to(source.device)
            values=source[selected].to(coords.device).masked_fill(~mask[...,None],0)
        return values,positions,mask
    sample._native_sampler=native
    return sample


def install():
    for kind in ['ss','shape','pbr']:
        module=importlib.import_module(f'models.object_gen_{kind}_x2_offline_with_dino')
        if not hasattr(module.sample_object_feats,'_native_sampler'):
            module.sample_object_feats=deferred_feature_gather(module.sample_object_feats)
        if not hasattr(module.filter_points_in_object_unit_box,'_native_unit_box_filter'):
            module.filter_points_in_object_unit_box=deferred_unit_box_filter(module.filter_points_in_object_unit_box)
