"""Portable configuration and early dependency errors in the student package."""
import hashlib
import json
from pathlib import Path

import pytest

import run_baseline
from track2_io import ROOT


def test_reusable_reconstruction_config_has_no_run_paths():
    config = json.loads((ROOT/'configs/baselines/fire3d_recon.yaml').read_text())
    assert not {'source_run','output','evaluation_output','input_identity'} & config.keys()
    profile = ROOT/config['frozen_profile']
    assert hashlib.sha256(profile.read_bytes()).hexdigest() == profile.with_suffix('.sha256').read_text().strip().split()[0]
    assert 'reference_run' not in json.loads(profile.read_text()).get('provenance',{})


def test_clean_student_root_and_media_preflight(monkeypatch):
    assert not list(ROOT.glob('*.py'))
    assert {path.name for path in ROOT.iterdir() if path.is_file()} == {
        'README.md','report.tex','references.bib','requirements.txt','final_track2_handout.pdf'}
    monkeypatch.setattr(run_baseline.shutil,'which',lambda name: None if name=='ffprobe' else '/usr/bin/ffmpeg')
    with pytest.raises(FileNotFoundError,match='ffprobe'):
        run_baseline.check_media_tools()
