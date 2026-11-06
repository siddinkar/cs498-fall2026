"""The frozen full-image DINO/AnyUp and learned RGB feature encoder."""
from ..pi3x_fusion.common import resolve


class Encoder:
    def __init__(self, config):
        import torch
        from models.dino_anyup import load_anyup_upsampler
        self.config = config
        root, repo = resolve(config['model_root']), resolve(config['fire3d_repo'])
        self.model = torch.hub.load(str(repo/'third_party/dinov3'), 'dinov3_vitl16', source='local',
            weights=str(root/'external/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth')).cuda().eval().requires_grad_(False)
        self.upsampler = load_anyup_upsampler({'anyup_repo': str(repo/'third_party/anyup'), 'anyup_source': 'local',
            'anyup_model': 'anyup_multi_backbone', 'anyup_use_natten': config.get('anyup_use_natten', False)}, 'cuda') if config['dino_upsample'] > 1 else None
        self.rgb_mlp = None
        if config.get('inference_protocol'):
            import yaml
            options = yaml.safe_load((root/'reconstruction/flows/pbr/config.yaml').read_text())['model']
            hidden = options['appearance_condition']['hidden_channels']
            self.rgb_mlp = torch.nn.Sequential(torch.nn.Linear(3, hidden), torch.nn.SiLU(), torch.nn.Linear(hidden, 1024))
            checkpoint = torch.load(root/'reconstruction/flows/pbr/model.pt', map_location='cpu', mmap=True, weights_only=False)
            self.rgb_mlp.load_state_dict({k.removeprefix('rgb_mlp.'): v for k, v in checkpoint['model'].items() if k.startswith('rgb_mlp.')}, strict=True)
            del checkpoint
            self.rgb_mlp = self.rgb_mlp.cuda().eval().requires_grad_(False)
        self.mean = torch.tensor([.485, .456, .406], device='cuda')[None, :, None, None]
        self.std = torch.tensor([.229, .224, .225], device='cuda')[None, :, None, None]

    def __call__(self, rgb):
        import torch
        import torch.nn.functional as F
        from models.dino_anyup import run_dino_anyup_features
        h, w = rgb.shape[:2]
        if h % 16 or w % 16:
            raise ValueError('DINO inputs must have dimensions divisible by 16')
        factor = self.config['dino_upsample']
        image = torch.as_tensor(rgb.copy(), device='cuda').permute(2, 0, 1)[None].float()/255
        image = (image-self.mean)/self.std
        if self.rgb_mlp is not None:
            raw_rgb = torch.as_tensor(rgb.copy(), device='cuda').permute(2, 0, 1)[None].float()/255
            rgb_cells = F.avg_pool2d(raw_rgb, kernel_size=16//factor, stride=16//factor)
            del raw_rgb
        with torch.autocast('cuda', dtype=torch.bfloat16):
            if factor > 1:
                feats = run_dino_anyup_features(dino_model=self.model, anyup_upsampler=self.upsampler, rgbs_tensor=image,
                    dino_downsample=16, dino_upsample=factor, anyup_q_chunk_size=self.config['anyup_q_chunk_size'],
                    anyup_tile_grid=0, anyup_frame_batch_size=1)
                grid = feats.reshape(h//(16//factor), w//(16//factor), 1024).permute(2, 0, 1)[None].float()
            else:
                feats = self.model(image, is_training=True)['x_norm_patchtokens']
                grid = feats.reshape(h//16, w//16, 1024).permute(2, 0, 1)[None].float()
            pbr_grid = None
            if self.rgb_mlp is not None:
                rgb_grid = self.rgb_mlp(rgb_cells.permute(0, 2, 3, 1)*2-1).permute(0, 3, 1, 2).float()
                pbr_grid = grid+rgb_grid
        return grid, pbr_grid
