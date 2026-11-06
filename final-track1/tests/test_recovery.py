"""Provided training recovery tests use a tiny toy model, not a student solution."""

import copy
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from cs498_track1 import workflow
from cs498_track1.training_state import atomic_save, random_state, restore_random_state


def test_rng_checkpoint_safe_load(tmp_path):
    before = random_state()
    try:
        random.seed(71)
        np.random.seed(72)
        torch.manual_seed(73)
        path = tmp_path / "state.pt"
        atomic_save(random_state(), path)
        expected = (random.random(), np.random.rand(), torch.rand(3))
        restore_random_state(torch.load(path, weights_only=True))
        assert random.random() == expected[0]
        assert np.random.rand() == expected[1]
        assert torch.equal(torch.rand(3), expected[2])
    finally:
        restore_random_state(before)


def test_failed_save_preserves_old_checkpoint(tmp_path, monkeypatch):
    path = tmp_path / "last.pt"
    atomic_save({"epoch": 1}, path)
    before = path.read_bytes()

    def fail(value, temporary):
        Path(temporary).write_bytes(b"interrupted")
        raise RuntimeError("Interrupted")

    monkeypatch.setattr(torch, "save", fail)
    with pytest.raises(RuntimeError, match="Interrupted"):
        atomic_save({"epoch": 2}, path)
    assert path.read_bytes() == before


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_interrupted_training_matches_uninterrupted(tmp_path, monkeypatch, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA recovery is tested when a GPU is available")

    class Dataset(torch.utils.data.Dataset):
        def __init__(self, paths, use_depth):
            self.paths = paths

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, index):
            return dict(
                rgb=torch.full((3, 3, 2, 2), float(np.random.rand())),
                depth_m=torch.ones(3, 1, 2, 2),
                intrinsics=torch.eye(3),
                t_world_camera=torch.eye(4).repeat(3, 1, 1),
                t_world_anchor=torch.eye(4),
                labels=torch.full((1, 1, 1), 1 + index % 2),
                completion_mask=torch.ones((1, 1, 1), dtype=torch.bool),
                output_mask=torch.ones((1, 1, 1), dtype=torch.bool),
                house_id=self.paths[index].parent.name,
            )

    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(1, 10)

        def forward(self, rgb, *args):
            x = rgb.mean(dim=(1, 2, 3, 4))[:, None]
            return self.linear(torch.nn.functional.dropout(x, 0.25, self.training))[
                :, :, None, None, None
            ]

    def loss(logits, labels, *args, **kwargs):
        value = torch.nn.functional.cross_entropy(logits, labels)
        return value, {}

    def paths(root, split):
        return [tmp_path / f"{split}_house_000" / f"sample_{i:05}" for i in range(4)]

    monkeypatch.setattr(workflow, "load_adapter", lambda source: None)
    monkeypatch.setattr(workflow, "ThreeViewDataset", Dataset)
    monkeypatch.setattr(workflow, "sample_paths", paths)
    monkeypatch.setattr(
        workflow,
        "expected_sample_ids",
        lambda split: [p.parent.name + "__" + p.name for p in paths(None, split)],
    )
    monkeypatch.setattr(
        workflow,
        "load_target",
        lambda path: dict(
            labels=np.ones((1, 1, 1), np.int8), completion_mask=np.ones((1, 1, 1), bool)
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "semocc.learning",
        SimpleNamespace(
            completion_loss=loss, class_weights_from_histogram=lambda h: torch.ones(10)
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "semocc.model",
        SimpleNamespace(build_triview_model=lambda *args: Toy()),
    )
    monkeypatch.setattr(workflow, "plot_training", lambda *args: None)

    def arguments(name):
        source = tmp_path / name / "student_submission"
        source.mkdir(parents=True)
        (source / "submission.py").write_text("# toy fixture\n")
        return SimpleNamespace(
            smoke=False,
            submission_dir=source,
            device=device,
            rgb_only=False,
            data_root=tmp_path,
            batch_size=2,
            workers=0,
            epochs=2,
            output=tmp_path / name / "outputs",
            resume=None,
        )

    complete = arguments("complete")
    receipts = tmp_path / "download_receipts"
    receipts.mkdir()
    (receipts / "val.json").write_text(json.dumps({"revision": "fixture-original"}))
    workflow.train(complete)
    interrupted = arguments("interrupted")

    def disconnect(history, path):
        if len(history) == 1:
            raise RuntimeError("simulated disconnect after completed epoch")

    monkeypatch.setattr(workflow, "plot_training", disconnect)
    with pytest.raises(RuntimeError, match="simulated disconnect"):
        workflow.train(interrupted)
    monkeypatch.setattr(workflow, "plot_training", lambda *args: None)
    interrupted.resume = interrupted.output / "checkpoint_last.pt"
    workflow.train(interrupted)
    full = torch.load(complete.output / "checkpoint_last.pt", weights_only=True)
    resumed = torch.load(interrupted.output / "checkpoint_last.pt", weights_only=True)
    for key in full["model_state"]:
        assert torch.equal(full["model_state"][key], resumed["model_state"][key])
        assert torch.equal(
            full["best_checkpoint"]["model_state"][key],
            resumed["best_checkpoint"]["model_state"][key],
        )
    assert full["scheduler"] == resumed["scheduler"]
    assert full["scaler"] == resumed["scaler"]
    assert full["optimizer"]["param_groups"] == resumed["optimizer"]["param_groups"]
    for key, values in full["optimizer"]["state"].items():
        for name, value in values.items():
            other = resumed["optimizer"]["state"][key][name]
            assert (
                torch.equal(value, other)
                if isinstance(value, torch.Tensor)
                else value == other
            )
    assert full["sampler_epoch"] == resumed["sampler_epoch"] == 2
    assert [x["loss"] for x in full["history"]] == [
        x["loss"] for x in resumed["history"]
    ]
    assert [x["validation"] for x in full["history"]] == [
        x["validation"] for x in resumed["history"]
    ]
    # Already complete recovery recreates missing best weights without more epochs.
    best = interrupted.submission_dir / "weights/checkpoint_best.pt"
    best.unlink()
    workflow.train(interrupted)
    assert best.is_file()
    bad = copy.copy(interrupted)
    bad.batch_size = 1
    with pytest.raises(ValueError, match="Resume configuration changed"):
        workflow.train(bad)
    (receipts / "val.json").write_text(json.dumps({"revision": "fixture-changed"}))
    with pytest.raises(ValueError, match="Resume configuration changed"):
        workflow.train(interrupted)
