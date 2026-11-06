"""One stable output layout; public evaluation artifacts never contain caches."""
from dataclasses import dataclass
from pathlib import Path
from .pi3x_fusion.common import resolve


def scene_uid(scene):
    parts = Path(scene).parts
    if len(parts) != 2 or parts[0] not in {'dev', 'val', 'test'} or not parts[1].startswith('scene_'):
        raise ValueError('Scene must be <dev|val|test>/scene_XXX')
    return '_'.join(parts)


@dataclass(frozen=True)
class OutputLayout:
    root: Path

    @classmethod
    def for_scene(cls, scene, algorithm='pi3x_sam2_fire3d', output_root='outputs'):
        if not algorithm or Path(algorithm).name != algorithm or algorithm in {'.', '..'}:
            raise ValueError('Algorithm name must be a single directory name')
        return cls(resolve(output_root) / algorithm / scene_uid(scene))

    @property
    def evaluation(self): return self.root / 'evaluation'
    @property
    def perception(self): return self.root / 'intermediates/perception'
    @property
    def reconstruction(self): return self.root / 'intermediates/reconstruction'


def evaluation_dir(config):
    return resolve(config.get('evaluation_output', config['output']))
