"""Fire3D learned PBR inference and UV texture baking, with staged decoders.

RGB is input conditioning for the checkpoint's learned RGB MLP. Texture maps
come from PBR flow/VAEs; no nearest-RGB transfer or source asset material is used.
"""
import numpy as np
from ..pi3x_fusion.common import resolve, read_json, write_json
from .runtime import bootstrap, load_flow, load_vae, cleanup, common, measure, set_seed, release_scene_inputs
from .policy import select_objects, select_conditioned


def sparse_from(path, feature_key='raw_features'):
    import torch
    from trellis2.modules.sparse import SparseTensor
    data = np.load(path)
    return SparseTensor(feats=torch.tensor(data[feature_key], device='cuda', dtype=torch.float32).contiguous(),
                        coords=torch.tensor(data['coords'], device='cuda', dtype=torch.int32).contiguous())


def save_guides(path, guides):
    data = {}
    for index, guide in enumerate(guides):
        data[f'coords_{index}'] = guide.coords.int().cpu().numpy()
        # Both guided decoders consume only the exact >0 subdivision decisions.
        data[f'feats_{index}'] = (guide.feats > 0).to(dtype=__import__('torch').uint8).cpu().numpy()
    np.savez_compressed(path, count=len(guides), **data)


def load_guides(path):
    import torch
    from trellis2.modules.sparse import SparseTensor
    data = np.load(path)
    return [SparseTensor(feats=torch.tensor(data[f'feats_{i}'], device='cuda', dtype=torch.float32).contiguous(),
                         coords=torch.tensor(data[f'coords_{i}'], device='cuda', dtype=torch.int32).contiguous())
            for i in range(int(data['count']))]


def load_sc_decoder(config, kind):
    from trellis2 import models
    from safetensors.torch import load_file
    root = resolve(config['model_root']) / 'external/trellis2'
    path = root / f'{kind}_dec_next_dc_f16c32_fp16'
    options = read_json(str(path) + '.json')
    from ..loading import empty_parameters,assign_weights
    with empty_parameters():decoder=getattr(models,options['name'])(**options['args'])
    state = load_file(str(path) + '.safetensors')
    assign_weights(decoder,state)
    del state
    cleanup()
    return decoder.eval().requires_grad_(False).cuda()


def infer_textures(config, object_ids=None):
    import torch
    bootstrap(config)
    output = resolve(config['output'])
    objects, skipped = select_conditioned(read_json(output / 'conditioning.json'), config)
    if object_ids:
        objects = [o for o in objects if o['object_id'] in object_ids]
    write_json(output / 'skipped_objects.json', {'objects': skipped, 'minimum_fused_points': config.get('minimum_fused_points', 100)})
    directories = [output / 'objects' / o['name'] for o in objects]
    settings = config['pbr']
    with torch.inference_mode():
        pending = [d for d in directories if not (d / 'pbr_latents.npz').exists()]
        if pending:
            model = load_flow(config, 'pbr')
            if model.appearance_mode != 'dino_rgb' or model.rgb_mlp is None:
                raise ValueError('Expected the release dino_rgb PBR checkpoint')
            for directory in pending:
                with measure(directory, 'pbr_flow'):
                    if config.get('inference_protocol'):
                        inputs = common(directory, config, feature_key='pbr_features', flow_kind='pbr')
                        rgb = None
                    else:
                        inputs = common(directory, config)
                        observed = np.load(directory / 'conditioning.npz')
                        rgb = torch.tensor(observed['rgb'], device='cuda', dtype=torch.float32) / 255.
                        with torch.autocast('cuda', dtype=torch.bfloat16):
                            inputs['colors'] = inputs['colors'] + model.rgb_mlp(rgb * 2. - 1.)[None].float()
                    # The upstream 'colors' gateway accepts precomposed features only in
                    # 'dino' mode. This flag bypasses encoder recomputation; the checkpoint
                    # RGB MLP above remains active and its learned weights are preserved.
                    shape = np.load(directory / 'shape_latents.npz')
                    coords = torch.tensor(shape['coords'], device='cuda', dtype=torch.int32).contiguous()
                    features = torch.tensor(shape['raw_features'], device='cuda', dtype=torch.float32).contiguous()
                    if config.get('inference_protocol'):
                        from .protocol import object_seed
                        set_seed(object_seed(config, directory.name, 'pbr'))
                        from eval.reconstruction.lc64_benchmark_appearance import predict_pbr_sparse
                        for key in ['return_denoised_latents', 'sample_method', 'prune_out_of_range', 'inference_only']:
                            inputs.pop(key)
                        predicted = predict_pbr_sparse(model=model, common=inputs, shape_features=features, coords=coords,
                            selected_indices=inputs.pop('selected_indices'), sample_method=config['sample_method'],
                            inference_num_steps=config['steps'], device=torch.device('cuda'), guidance_strength=settings['guidance_strength'])
                        result = {'pbr_x2_pred_feats_raw': predicted}
                        del predicted
                    else:
                        set_seed(config['seed'] + read_json(directory / 'conditioning.json')['object_id'])
                        mode = model.appearance_mode
                        model.appearance_mode = 'dino'
                        with torch.autocast('cuda', dtype=torch.bfloat16):
                            result = model(**inputs, object_shape_x2_feats=features, object_coords=coords,
                                           inference_num_steps=config['steps'], guidance_strength=settings['guidance_strength'])
                        model.appearance_mode = mode
                    np.savez_compressed(directory / 'pbr_latents.npz', coords=coords.cpu().numpy(),
                                        raw_features=result['pbr_x2_pred_feats_raw'].float().cpu().numpy())
                    del result, inputs, rgb, coords, features
                print(f'{directory.name}: learned PBR flow complete', flush=True)
            del model
            release_scene_inputs()
            cleanup()
        pending = [d for d in directories if not (d / 'shape_hc_subdivisions.npz').exists()]
        if pending:
            decoder, _ = load_vae(config, 'shape')
            for directory in pending:
                with measure(directory, 'shape_hc_guides'):
                    sparse = sparse_from(directory / 'shape_latents.npz')
                    normalized, guides = decoder(sparse, return_subs=True)
                    existing = np.load(directory / 'sc_latents.npz')
                    if not np.array_equal(existing['coords'], normalized.coords.cpu().numpy()):
                        raise ValueError('Shape HC topology changed during texture inference')
                    save_guides(directory / 'shape_hc_subdivisions.npz', guides)
                    del sparse, normalized, guides
                print(f'{directory.name}: shape HC guides saved', flush=True)
            del decoder
            cleanup()
        pending = [d for d in directories if not (d / 'pbr_sc_latents.npz').exists()]
        if pending:
            decoder, options = load_vae(config, 'pbr')
            stats = options['dataset']['args']['pbr_slat_normalization']
            mean = torch.tensor(stats['mean'], device='cuda', dtype=torch.float32)[None]
            std = torch.tensor(stats['std'], device='cuda', dtype=torch.float32)[None]
            for directory in pending:
                with measure(directory, 'pbr_hc_decode'):
                    sparse = sparse_from(directory / 'pbr_latents.npz')
                    guides = load_guides(directory / 'shape_hc_subdivisions.npz')
                    normalized = decoder(sparse, guide_subs=guides)
                    raw = normalized.feats.float() * std + mean
                    expected = np.load(directory / 'sc_latents.npz')['coords']
                    if not np.array_equal(expected, normalized.coords.cpu().numpy()):
                        raise ValueError('PBR HC coordinates differ from inferred shape')
                    np.savez_compressed(directory / 'pbr_sc_latents.npz', coords=normalized.coords.cpu().numpy(), raw_features=raw.cpu().numpy())
                    del sparse, guides, normalized, raw
                print(f'{directory.name}: PBR HC decode complete', flush=True)
            del decoder, mean, std
            cleanup()
        pending = [d for d in directories if not (d / 'shape_sc_subdivisions.npz').exists()]
        if pending:
            decoder = load_sc_decoder(config, 'shape')
            decoder.set_resolution(config['mesh_resolution'])
            for directory in pending:
                with measure(directory, 'shape_sc_guides'):
                    sparse = sparse_from(directory / 'sc_latents.npz')
                    meshes, guides = decoder(sparse, return_subs=True)
                    save_guides(directory / 'shape_sc_subdivisions.npz', guides)
                    del sparse, meshes, guides
                print(f'{directory.name}: shape SC guides saved', flush=True)
            del decoder
            cleanup()
        pending = [d for d in directories if not (d / 'pbr_volume.npz').exists()]
        if pending:
            decoder = load_sc_decoder(config, 'tex')
            for directory in pending:
                with measure(directory, 'pbr_sc_decode'):
                    sparse = sparse_from(directory / 'pbr_sc_latents.npz')
                    guides = load_guides(directory / 'shape_sc_subdivisions.npz')
                    decoded = decoder(sparse, guide_subs=guides)
                    volume = (decoded.feats.float() * .5 + .5).clamp(0, 1)
                    dtype = np.float32 if config.get('inference_protocol') else np.float16
                    np.savez_compressed(directory / 'pbr_volume.npz', coords=decoded.coords[:, 1:].int().cpu().numpy(),
                                        attrs=volume.cpu().numpy().astype(dtype, copy=False))
                    del sparse, guides, decoded, volume
                print(f'{directory.name}: PBR SC material volume complete', flush=True)
            del decoder
            cleanup()
        for directory in directories:
            if not (directory / 'mesh_textured_world.glb').exists():
                with measure(directory, 'texture_bake'):
                    bake(directory, config)
                cleanup()
    compose_scene(output, objects, skipped, config)


def bake(directory, config):
    import torch
    import trimesh
    import o_voxel
    canonical = trimesh.load(directory / 'mesh_canonical.ply', force='mesh', process=False)
    data = np.load(directory / 'pbr_volume.npz')
    if config.get('inference_protocol'):
        textured = bake_release(directory, config, canonical, data)
    else:
        textured = o_voxel.postprocess.to_glb(
        vertices=torch.tensor(canonical.vertices, device='cuda', dtype=torch.float32).contiguous(),
        faces=torch.tensor(canonical.faces, device='cuda', dtype=torch.int32).contiguous(),
        attr_volume=torch.tensor(data['attrs'], device='cuda', dtype=torch.float32).contiguous(),
        coords=torch.tensor(data['coords'], device='cuda', dtype=torch.int32).contiguous(),
        attr_layout={'base_color': slice(0, 3), 'metallic': slice(3, 4), 'roughness': slice(4, 5), 'alpha': slice(5, 6)},
        aabb=[[-.5, -.5, -.5], [.5, .5, .5]], grid_size=[config['mesh_resolution']]*3,
        decimation_target=config['pbr']['mesh_faces'], texture_size=config['pbr']['texture_size'],
            remesh=config['pbr']['remesh'], verbose=False)
    textured.export(directory / 'mesh_textured_canonical.glb')
    material = textured.visual.material
    texture_dir = directory / 'textures'
    texture_dir.mkdir(exist_ok=True)
    material.baseColorTexture.save(texture_dir / 'base_color.png')
    material.metallicRoughnessTexture.save(texture_dir / 'metallic_roughness.png')
    textured.apply_transform(np.load(directory / 'conditioning.npz')['local_to_world'])
    textured.export(directory / 'mesh_textured_world.glb')
    report = {'object_id': read_json(directory / 'conditioning.json')['object_id'], 'status': 'textured',
        'texture_source': 'Fire3D learned dino_rgb PBR flow + PBR-HC/SC VAE decoders',
        'conditioning': 'Fused predicted point positions, DINOv3/AnyUp features, observed RGB; inferred shape latents',
        'RGB_vertex_color_transfer_used': False, 'source_asset_textures_or_GT_used': False,
        'vertices': len(textured.vertices), 'faces': len(textured.faces), 'texture_size': list(material.baseColorTexture.size),
        'bbox_world': textured.bounds.tolist(), 'geometry_postprocess': 'Official CuMesh cleanup/decimation and UV unwrap',
        'base_color': 'textures/base_color.png', 'metallic_roughness': 'textures/metallic_roughness.png'}
    write_json(directory / 'texture.json', report)
    print(f'{directory.name}: learned texture GLB exported', flush=True)


def bake_release(directory, config, canonical, data):
    """Use the reference batched remesh/UV/baking path with batch size one."""
    import torch
    import trimesh
    from eval.reconstruction.lc64_shape_pbr_decode import postprocess_and_bake_scene
    from eval.reconstruction.batched_ovoxel_postprocess import PostprocessConfig
    from .protocol import load_protocol
    protocol = load_protocol(config)['reconstruction']
    options = protocol['mesh_postprocess']
    raw = {'vertices': torch.tensor(canonical.vertices, dtype=torch.float32).contiguous(),
           'faces': torch.tensor(canonical.faces, dtype=torch.int32).contiguous(),
           'coords': torch.tensor(data['coords'], dtype=torch.int32).contiguous(),
           'attrs': torch.tensor(data['attrs'], dtype=torch.float32).contiguous()}
    summary = postprocess_and_bake_scene(
        decode_summary={'objects': [{'object_position': 0}]}, output_dir=directory / 'release_postprocess',
        device=torch.device('cuda'), resolution=config['mesh_resolution'], raw_objects=[raw],
        decimation_target=options['decimation_target'], texture_size=options['texture_size'],
        mesh_batch_size=options['object_batch_size'], backend=options['backend'],
        topology_config=PostprocessConfig(
            initial_target_multiplier=options['topology_initial_target_multiplier'],
            simplify_threshold=options['topology_simplify_threshold'], chart_area_penalty_weight=options['topology_chart_area_penalty'],
            remesh=options['topology_remesh'], remesh_band=options['topology_remesh_band'],
            remesh_project=options['topology_remesh_project'], remesh_resolution=config['mesh_resolution'],
            xatlas_block_align=options['xatlas_block_align'], uv_cpu_workers=options['uv_cpu_workers']),
        surface_mapping=options['surface_mapping'], sparse_query_mode=options['sparse_query_mode'],
        texture_fill_mode=options['texture_fill_mode'], texture_dilation_pixels=options['texture_dilation_pixels'],
        texture_erode_iterations=options['texture_erode_iterations'],
        background_decimation_multiplier=options['background_decimation_multiplier'])
    return trimesh.load(summary['objects'][0]['canonical_glb'], force='mesh', process=False)


def compose_scene(output, objects, skipped, config):
    import trimesh
    scene = trimesh.Scene()
    reports = []
    for obj in objects:
        directory = output / 'objects' / obj['name']
        if not (directory / 'mesh_textured_world.glb').exists():
            continue
        mesh = trimesh.load(directory / 'mesh_textured_world.glb', force='mesh', process=False)
        if mesh.visual.kind != 'texture':
            raise ValueError(f'Missing generated PBR texture in {obj["name"]}')
        scene.add_geometry(mesh, node_name=obj['name'], geom_name=obj['name'])
        reports.append(read_json(directory / 'texture.json'))
    scene.export(output / 'scene_textured.glb')
    write_json(output / 'textures.json', {'objects': reports, 'skipped_objects': skipped,
        'source': 'Fire3D PBR inference', 'perception_loaded': False, 'GT_used': False, 'config': config['pbr']})
