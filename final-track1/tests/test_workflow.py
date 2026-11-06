"""Packaging guards use small fixtures; model correctness is tested separately."""

import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

from cs498_track1 import workflow as course
import numpy as np
import pytest
from cs498_track1.download import DEFAULT_REVISION, save_receipt
from cs498_track1.evidence import record_figures
from cs498_track1.diagnostics import storage_path
from cs498_track1.protocol import SHAPE, encode_labels, sha256, tree_hash


def test_replacement_adapter_can_define_dataclass(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "path", sys.path.copy())
    monkeypatch.setitem(sys.modules, "final_method", None)
    (tmp_path / "submission.py").write_text(
        "from __future__ import annotations\n"
        "from dataclasses import dataclass\n"
        "@dataclass\nclass Model:\n    value: int = 7\n"
        "def load_model(model_dir, device): return Model()\n"
        "def predict_sample(model, inputs, device): return None\n"
    )
    adapter = course.load_adapter(tmp_path)
    assert adapter.load_model(None, "cpu").value == 7


def test_preflight_works_without_installed_packages():
    entry = Path(__file__).resolve().parents[1] / "run.py"
    result = subprocess.run(
        [sys.executable, "-S", str(entry), "doctor"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 1
    assert "Fix:" in result.stdout
    assert "Traceback" not in result.stderr


def test_disk_check_targets_data_filesystem(tmp_path):
    assert storage_path(tmp_path / "not_yet_downloaded" / "data") == tmp_path.resolve()


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    """Build complete synthetic evidence without downloading a dataset."""
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "student_submission"
    source.mkdir()
    (source / "method.json").write_text(json.dumps({"student_id": "audit"}))
    (source / "submission.py").write_text("# Test fixture, not a model\n")
    (source / "requirements.txt").write_text("")
    # A legitimate method module is not an API token or credential file.
    (source / "tokenizer.py").write_text("# Tokenization implementation fixture\n")
    root = tmp_path / "data"
    for split in ["val", "test"]:
        receipt = dict(
            repo_id="hongchi/cs498",
            revision=DEFAULT_REVISION,
            split=split,
            shards=[str(i) for i in range(10)],
        )
        p = root / "download_receipts" / f"{split}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(receipt))
    ids = []
    for h in range(10):
        for s in range(15):
            house, sample = f"test_house_{h:03}", f"sample_{s:05}"
            p = root / "samples" / house / sample / "manifest.json"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("{}")
            ids.append(house + "__" + sample)
    csv = Path("outputs/kaggle/submission.csv")
    csv.parent.mkdir(parents=True)
    encoded = encode_labels(np.zeros(SHAPE, np.int8))
    csv.write_text("Id,Prediction\n" + "".join(f"{i},{encoded}\n" for i in ids))
    digest = tree_hash(source)
    runtime = dict(
        submission_tree_sha256=digest,
        csv_sha256=sha256(csv),
        complete=True,
        data_receipt=course.receipts(root, "test"),
    )
    csv.with_name("submission_runtime.json").write_text(json.dumps(runtime))
    val = Path("outputs/evaluation/val/summary.json")
    val.parent.mkdir(parents=True)
    val.write_text(
        json.dumps(
            dict(
                submission_tree_sha256=digest,
                complete=True,
                data_receipt=course.receipts(root, "val"),
            )
        )
    )
    Path("report.pdf").write_bytes(b"report fixture")
    for filename in ["inputs.png", "completion.png", "comparison.png", "class_iou.png"]:
        p = Path("outputs/report") / filename
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"image fixture")
    record_figures(source, val.parent, Path("outputs/report"))
    monkeypatch.setattr(course, "probe", lambda args: None)
    return SimpleNamespace(
        submission_dir=source,
        data_root=root,
        student_id="audit",
        kaggle_username="audit",
        device="cpu",
        output=Path("packed"),
    )


def test_package_structure(bundle):
    course.package(bundle)
    with zipfile.ZipFile("packed/cs498-track1-audit.zip") as z:
        assert "student_submission/method.json" in z.namelist()
        assert "report.pdf" in z.namelist()
        assert "outputs/report/evidence_manifest.json" in z.namelist()
        assert not any(n.startswith("data/") for n in z.namelist())
    manifest = json.loads(Path("packed/package_manifest.json").read_text())
    assert manifest["zip_sha256"] == sha256("packed/cs498-track1-audit.zip")


@pytest.mark.parametrize("command", ["validate", "package"])
def test_cli_cpu_checks_cannot_be_overridden(command):
    from cs498_track1.cli import main

    options = (
        ["--student-id", "audit", "--kaggle-username", "audit"]
        if command == "package"
        else []
    )
    with pytest.raises(SystemExit) as failure:
        main([command, "--device", "cuda", *options])
    assert failure.value.code == 2


def test_package_always_probes_cpu(bundle, monkeypatch):
    bundle.device = "cuda"
    seen = []
    monkeypatch.setattr(course, "probe", lambda args: seen.append(args.device))
    course.package(bundle)
    assert seen == ["cpu"]
    assert bundle.device == "cuda"


def test_stale_method_rejected(bundle):
    (bundle.submission_dir / "submission.py").write_text("# changed after inference\n")
    with pytest.raises(ValueError, match="Stale"):
        course.package(bundle)


@pytest.mark.parametrize("changed", ["figure", "summary"])
def test_stale_figures_rejected(bundle, changed):
    path = (
        Path("outputs/report/completion.png")
        if changed == "figure"
        else Path("outputs/evaluation/val/summary.json")
    )
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="Stale report figures"):
        course.package(bundle)


@pytest.mark.parametrize("module_name", ["data", "dataset"])
def test_custom_module_names_work_in_fresh_cli(bundle, module_name):
    """Both actual CLI paths must work without inheriting a test process's imports."""
    from PIL import Image
    from cs498_track1.protocol import POLICY, VIEWS

    source = bundle.submission_dir
    (source / f"{module_name}.py").write_text("def marker(): return 7\n")
    (source / "submission.py").write_text(
        f"from {module_name} import marker\n"
        "import numpy as np\n"
        "def load_model(model_dir, device):\n"
        "    assert marker() == 7\n"
        "    assert model_dir.name == 'weights'\n"
        "    return None\n"
        "def predict_sample(model, inputs, device):\n"
        "    return np.zeros((40,14,80), dtype=np.int8)\n"
    )
    for index in range(2):
        sample = bundle.data_root / "samples/test_house_000" / f"sample_{index:05}"
        (sample / "manifest.json").write_text(
            json.dumps(dict(target_policy=POLICY, view_names=VIEWS))
        )
        (sample / "rgb").mkdir()
        for view in VIEWS:
            Image.new("RGB", (16, 16)).save(sample / "rgb" / f"{view}.png")
        np.save(sample / "K.npy", np.eye(3, dtype=np.float32))
        np.save(
            sample / "T_world_camera.npy",
            np.tile(np.eye(4, dtype=np.float32), (3, 1, 1)),
        )
        np.save(sample / "T_world_anchor.npy", np.eye(4, dtype=np.float32))
    digest = tree_hash(source)
    for path in [
        Path("outputs/evaluation/val/summary.json"),
        Path("outputs/kaggle/submission_runtime.json"),
    ]:
        item = json.loads(path.read_text())
        item["submission_tree_sha256"] = digest
        path.write_text(json.dumps(item))
    record_figures(source, Path("outputs/evaluation/val"), Path("outputs/report"))
    entry = Path(__file__).resolve().parents[1] / "run.py"
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    for command, extra in [
        ("validate", []),
        ("package", ["--student-id", "audit", "--kaggle-username", "audit"]),
    ]:
        result = subprocess.run(
            [
                sys.executable,
                str(entry),
                command,
                "--data-root",
                str(bundle.data_root),
                *extra,
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, result.stdout + result.stderr


def test_wrong_test_house_rejected(bundle):
    samples = bundle.data_root / "samples"
    (samples / "test_house_009").rename(samples / "test_house_010")
    with pytest.raises(ValueError, match="10 Test houses"):
        course.package(bundle)


def test_modified_csv_rejected(bundle):
    p = Path("outputs/kaggle/submission.csv")
    p.write_text(p.read_text() + "\n")
    with pytest.raises(ValueError):
        course.package(bundle)


def test_wrong_data_revision_rejected(bundle):
    p = Path("outputs/evaluation/val/summary.json")
    summary = json.loads(p.read_text())
    summary["data_receipt"]["revision"] = "old"
    p.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="receipt"):
        course.package(bundle)


def test_subset_receipts_merge_and_reset_on_revision_change(tmp_path):
    official = [
        dict(role="val", house_id=f"val_house_{i:03}", sha256=str(i)) for i in range(3)
    ]
    path = tmp_path / "download_receipts/val.json"
    save_receipt(tmp_path, "val", DEFAULT_REVISION, official[:2], official)
    save_receipt(tmp_path, "val", DEFAULT_REVISION, official[2:], official)
    before = path.read_bytes()
    save_receipt(tmp_path, "val", DEFAULT_REVISION, official[:1], official)
    assert path.read_bytes() == before
    assert json.loads(before)["shards"] == official
    save_receipt(tmp_path, "val", "another-revision", official[:1], official)
    assert json.loads(path.read_text())["shards"] == official[:1]
    path.write_text('{"incomplete":')
    save_receipt(tmp_path, "val", DEFAULT_REVISION, official, official)
    assert json.loads(path.read_text())["shards"] == official


@pytest.mark.parametrize("fail", [True, False])
def test_csv_generation_is_atomic(tmp_path, monkeypatch, fail):
    source = tmp_path / "method"
    source.mkdir()
    output = tmp_path / "submission.csv"
    output.write_text("previous complete CSV\n")
    paths = [tmp_path / "test_house_000" / f"sample_{i:05}" for i in range(2)]
    calls = []

    def predict(*args):
        calls.append(None)
        if fail and len(calls) == 2:
            raise RuntimeError("Interrupted inference")
        return np.zeros(SHAPE, dtype=np.int8)

    monkeypatch.setattr(
        course,
        "load_adapter",
        lambda p: SimpleNamespace(
            load_model=lambda *args: None, predict_sample=predict
        ),
    )
    monkeypatch.setattr(course, "sample_paths", lambda *args: paths)
    monkeypatch.setattr(course, "load_inputs", lambda p: {})
    args = SimpleNamespace(
        submission_dir=source,
        output=output,
        device="cpu",
        data_root=tmp_path,
        max_samples=None,
    )
    if fail:
        with pytest.raises(RuntimeError, match="Interrupted"):
            course.generate_csv(args)
        assert output.read_text() == "previous complete CSV\n"
        assert not output.with_name("submission_runtime.json").exists()
    else:
        receipt = course.generate_csv(args)
        assert receipt["csv_sha256"] == sha256(output)
        assert not output.with_suffix(".csv.partial").exists()


def test_mutating_interface_probe_rejected(bundle, monkeypatch):
    def mutate(args):
        (args.submission_dir / "state.txt").write_text(
            "Unexpected inference side effect"
        )

    monkeypatch.setattr(course, "probe", mutate)
    with pytest.raises(ValueError, match="changed.*probe"):
        course.package(bundle)
