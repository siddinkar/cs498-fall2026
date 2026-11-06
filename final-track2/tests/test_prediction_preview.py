"""The report preview must work for input-only Test and never require GT."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import numpy as np
from PIL import Image
import trimesh

import preview_outputs
from track2_io import export_scene
from test_student_interface import input_scene


def test_gt_free_gallery_and_provenance(tmp_path):
    scene = input_scene(tmp_path / 'data')
    manifest = export_scene(scene, {1:trimesh.creation.box()}, tmp_path / 'evaluation')
    output = manifest.parent / 'prediction_preview'
    for folder, count in [('scene',4),('object_001',2)]:
        (output / folder).mkdir(parents=True)
        for i in range(count):
            Image.new('RGB',(128,128),(100+i*20,120,150)).save(output / folder / f'view_{i:02d}.png')
    preview_outputs.write_gallery(output,json.loads(manifest.read_text()),manifest)
    receipt = json.loads((output / 'preview.json').read_text())
    assert receipt['uses_gt'] is receipt['official_metrics'] is receipt['physics_simulation'] is False
    assert set(receipt['glbs']) == {'meshes/obj_001.glb'}
    assert (output / 'scene_views.png').is_file() and (output / 'object_views.png').is_file()
    assert 'Instance 1' in (output / 'index.html').read_text()
    assert not (tmp_path / 'data/test/scene_000/scene.blend').exists()
    (output / 'object_001/view_01.png').unlink()
    with pytest.raises(RuntimeError,match='Incomplete prediction object'):
        preview_outputs.write_gallery(output,json.loads(manifest.read_text()),manifest)


@pytest.mark.skipif(not os.environ.get('BLENDER_EXECUTABLE'),reason='Set BLENDER_EXECUTABLE for actual rendering')
def test_real_blender_preview_of_input_only_test(tmp_path):
    scene = input_scene(tmp_path / 'data')
    mesh = trimesh.creation.box(extents=[.2,.3,.4])
    mesh.apply_translation([.1,.2,.8])
    manifest = export_scene(scene,{7:mesh},tmp_path / 'evaluation')
    command = [sys.executable,str(Path(preview_outputs.__file__)), '--manifest',str(manifest),
               '--data-root',str(tmp_path / 'data'),'--resolution','128','--samples','1','--timeout','120']
    result = subprocess.run(command,capture_output=True,text=True,timeout=140)
    log = manifest.parent / 'prediction_preview/render.log'
    assert result.returncode == 0, result.stdout+result.stderr+(log.read_text() if log.exists() else '')
    assert (manifest.parent / 'prediction_preview/index.html').is_file()
    pixels = np.asarray(Image.open(manifest.parent / 'prediction_preview/scene/view_00.png'))
    assert pixels[:,:,:3].std() > 3, 'Render must show a nonempty prediction, not a blank image'
    assert not (tmp_path / 'data/test/scene_000/scene.blend').exists()
    assert not (manifest.parent / 'reports/evaluation').exists()
