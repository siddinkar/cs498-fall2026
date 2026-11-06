"""Provided workflow shared by the command-line tools. No student TODOs here."""

from __future__ import annotations

import copy
import csv
import importlib.util
import json
import math
import random
import sys
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from .dataset import (
    HouseBatchSampler,
    ThreeViewDataset,
    load_inputs,
    load_target,
    sample_paths,
)
from .protocol import (
    CLASSES,
    COMPETITION,
    MODEL_CONFIG,
    POLICY,
    SHAPE,
    TASK,
    confusion_metrics,
    decode_labels,
    encode_labels,
    expected_sample_ids,
    sha256,
    summarize,
    tree_hash,
    validate_prediction,
    write_json,
)
from torch.utils.data import DataLoader
from .download import DEFAULT_REVISION
from .evidence import FIGURES, validate_figures
from .training_state import atomic_save, random_state, restore_random_state, signature


def device_name(name):
    """Select CUDA when available, while retaining CPU smoke-test support."""
    return (
        "cuda"
        if name == "auto" and torch.cuda.is_available()
        else ("cpu" if name == "auto" else name)
    )


def load_adapter(source):
    """Import the selected final method and its own semocc implementation."""
    source = Path(source).resolve()
    sys.path.insert(0, str(source))
    spec = importlib.util.spec_from_file_location(
        "final_method",
        source / "submission.py",
        submodule_search_locations=[str(source)],
    )
    module = importlib.util.module_from_spec(spec)
    # Standard import registration is needed by dataclasses and class metadata
    # in replacement adapters, not just plain function-only baseline modules.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if not all(
        callable(getattr(module, k, None)) for k in ("load_model", "predict_sample")
    ):
        raise ValueError("Missing interface functions")
    return module


def receipts(root, split):
    """Read the pinned download receipt; never include credentials."""
    p = Path(root) / "download_receipts" / f"{split}.json"
    return json.loads(p.read_text()) if p.is_file() else None


def evaluate(args, split="val"):
    """Evaluate the same inputs-only adapter used for Kaggle, with external GT."""
    torch.set_num_threads(2)
    source = args.submission_dir.resolve()
    before = tree_hash(source)
    adapter = load_adapter(source)
    device = device_name(args.device)
    start = time.perf_counter()
    model = adapter.load_model(source / "weights", device)
    paths = sample_paths(args.data_root, split)
    if args.max_samples:
        paths = paths[: args.max_samples]
    by_house = defaultdict(lambda: np.zeros((10, 10), dtype=np.int64))
    rows = []
    for i, p in enumerate(paths):
        prediction = validate_prediction(
            adapter.predict_sample(model, load_inputs(p), device)
        )
        target = load_target(p)
        mask = target["completion_mask"] & (target["labels"] >= 0)
        confusion = np.bincount(
            target["labels"][mask].astype(np.int64) * 10 + prediction[mask],
            minlength=100,
        ).reshape(10, 10)
        by_house[p.parent.name] += confusion
        predpath = args.output / "predictions" / p.parent.name / f"{p.name}.npz"
        predpath.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(predpath, prediction=prediction)
        rows.append(
            dict(
                house_id=p.parent.name,
                sample_id=p.name,
                metrics=confusion_metrics(confusion),
            )
        )
        if (i + 1) % 15 == 0:
            print(f"evaluated {i + 1}/{len(paths)} samples", flush=True)
    if tree_hash(source) != before:
        raise ValueError("Method changed its packaged files during validation")
    result = dict(
        task=TASK,
        target_policy=POLICY,
        split=split,
        shape_zyx=list(SHAPE),
        num_samples=len(paths),
        num_houses=len(by_house),
        complete=(
            [p.parent.name + "__" + p.name for p in paths] == expected_sample_ids(split)
        ),
        submission_tree_sha256=before,
        data_receipt=receipts(args.data_root, split),
        wall_seconds=time.perf_counter() - start,
        evaluation=summarize(by_house),
        samples=rows,
    )
    write_json(args.output / "summary.json", result)
    print(json.dumps(result["evaluation"]["metrics"], indent=2), flush=True)
    return result


def train(args):
    """Train the baseline at most 15 epochs; select weights using validation only."""
    if args.smoke:
        args.output = args.output / "smoke"
        print(
            f"Smoke-only artifacts: {args.output}; final method weights are unchanged",
            flush=True,
        )
    source = args.submission_dir.resolve()
    load_adapter(source)
    from semocc.learning import class_weights_from_histogram, completion_loss
    from semocc.model import build_triview_model

    random.seed(498)
    np.random.seed(498)
    torch.manual_seed(498)
    torch.set_num_threads(2)
    device = device_name(args.device)
    use_amp = device == "cuda"
    cfg = copy.deepcopy(MODEL_CONFIG)
    cfg["triview_model"]["use_depth"] = not args.rgb_only
    train_paths = sample_paths(args.data_root, "train")
    val_paths = sample_paths(args.data_root, "val")
    if not args.smoke:
        if [p.parent.name + "__" + p.name for p in train_paths] != expected_sample_ids(
            "train"
        ) or [p.parent.name + "__" + p.name for p in val_paths] != expected_sample_ids(
            "val"
        ):
            raise ValueError(
                "Full training requires 30000 Train and 150 Val samples; use --smoke for a subset"
            )
    else:
        train_paths = train_paths[:32]
        val_paths = val_paths[:15]
    train_data = ThreeViewDataset(train_paths, not args.rgb_only)
    val_data = ThreeViewDataset(val_paths, not args.rgb_only)
    sampler = HouseBatchSampler(train_paths, args.batch_size)
    loader = DataLoader(
        train_data, batch_sampler=sampler, num_workers=args.workers, pin_memory=use_amp
    )
    validation = DataLoader(
        val_data,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=use_amp,
    )
    histogram = np.zeros(10, dtype=np.int64)
    print(
        f"Training inputs: {len(train_paths)} Train / {len(val_paths)} Val; computing class weights",
        flush=True,
    )
    for index, p in enumerate(train_paths, 1):
        t = load_target(p)
        histogram += np.bincount(t["labels"][t["completion_mask"]], minlength=10)
        if index % 5000 == 0:
            print(f"Class histogram {index}/{len(train_paths)}", flush=True)
    weights = class_weights_from_histogram(histogram).to(device)
    model = build_triview_model(cfg, 9).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.0001)
    epochs = 1 if args.smoke else min(max(args.epochs, 1), 15)
    run_signature = signature(source, cfg, train_paths, val_paths, args, epochs, device)
    total = epochs * len(loader)
    warmup = min(1000, max(total - 1, 1))

    def lr_scale(step):
        """Use 1000-step linear warmup then cosine decay, as in the source run."""
        if step < warmup:
            return max(1e-6, (step + 1) / warmup)
        return 0.5 * (
            1 + math.cos(math.pi * min(1, (step - warmup) / max(1, total - warmup)))
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    args.output.mkdir(parents=True, exist_ok=True)
    history = []
    best = -1.0
    best_checkpoint = None
    first_epoch = 1
    resume = getattr(args, "resume", None)
    if resume:
        state = torch.load(resume, map_location="cpu", weights_only=True)
        if state.get("resume_format") != 1:
            raise ValueError(
                "Not a resumable checkpoint from this trainer; start a new run"
            )
        if state["run_signature"] != run_signature:
            raise ValueError(
                "Resume configuration changed: keep code, data, epochs, batch size, device and Torch version unchanged"
            )
        model.load_state_dict(state["model_state"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        sampler.epoch = state["sampler_epoch"]
        history, best = state["history"], state["best"]
        best_checkpoint = state["best_checkpoint"]
        first_epoch = state["epoch"] + 1
        restore_random_state(state["random_state"])
        best_path = (
            args.output / "checkpoint_best.pt"
            if args.smoke
            else source / "weights/checkpoint_best.pt"
        )
        atomic_save(best_checkpoint, best_path)
        write_json(args.output / "training_curve.json", history)
        plot_training(history, args.output / "training_curve.png")
        message = (
            f"Restored completed epoch {state['epoch']}; next epoch {first_epoch}/{epochs}"
            if first_epoch <= epochs
            else f"Already completed {epochs} epochs; no additional optimizer steps"
        )
        print(message, flush=True)
    started = time.perf_counter()
    for epoch in range(first_epoch, epochs + 1):
        model.train()
        losses = []
        epoch_start = time.perf_counter()
        for batch_index, batch in enumerate(loader, 1):
            rgb = batch["rgb"].to(device)
            depth = batch.get("depth_m")
            depth = None if depth is None else depth.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device, dtype=torch.float16, enabled=use_amp
            ):
                logits = model(
                    rgb,
                    depth,
                    batch["intrinsics"].to(device),
                    batch["t_world_camera"].to(device),
                    batch["t_world_anchor"].to(device),
                )
                loss, _ = completion_loss(
                    logits,
                    batch["labels"].long().to(device),
                    batch["completion_mask"].to(device),
                    batch["output_mask"].to(device),
                    class_weights=weights,
                )
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            losses.append(float(loss.detach()))
            if batch_index % 100 == 0:
                print(
                    f"Epoch {epoch}/{epochs}, batch {batch_index}/{len(loader)}, loss {np.mean(losses):.4f}",
                    flush=True,
                )
        model.eval()
        confusions = defaultdict(lambda: np.zeros((10, 10), dtype=np.int64))
        with torch.inference_mode():
            for batch in validation:
                with torch.autocast(
                    device_type=device, dtype=torch.float16, enabled=use_amp
                ):
                    depth = batch.get("depth_m")
                    depth = None if depth is None else depth.to(device)
                    pred = (
                        model(
                            batch["rgb"].to(device),
                            depth,
                            batch["intrinsics"].to(device),
                            batch["t_world_camera"].to(device),
                            batch["t_world_anchor"].to(device),
                        )
                        .argmax(1)
                        .cpu()
                        .numpy()
                    )
                for i, h in enumerate(batch["house_id"]):
                    mask = batch["completion_mask"][i].numpy()
                    target = batch["labels"][i].numpy()
                    confusions[h] += np.bincount(
                        target[mask].astype(np.int64) * 10 + pred[i][mask],
                        minlength=100,
                    ).reshape(10, 10)
        metrics = summarize(confusions)["metrics"]
        score = metrics["completion_mean_iou"]
        row = dict(
            epoch=epoch,
            loss=float(np.mean(losses)),
            validation=metrics,
            seconds=time.perf_counter() - epoch_start,
        )
        history.append(row)
        write_json(args.output / "training_curve.json", history)
        ckpt = dict(
            model_state=model.state_dict(),
            model_config=cfg,
            epoch=epoch,
            validation=metrics,
            target_policy=POLICY,
        )
        if score > best:
            best = score
            best_checkpoint = copy.deepcopy(ckpt)
            best_checkpoint["model_state"] = {
                k: v.detach().cpu().clone() for k, v in ckpt["model_state"].items()
            }
        # One recovery checkpoint also retains the historical best weights.
        # A disconnected/incomplete epoch is replayed, never counted as finished.
        recovery = dict(
            **ckpt,
            resume_format=1,
            run_signature=run_signature,
            optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(),
            scaler=scaler.state_dict(),
            sampler_epoch=sampler.epoch,
            history=history,
            best=best,
            best_checkpoint=best_checkpoint,
            random_state=random_state(cuda=use_amp),
        )
        atomic_save(recovery, args.output / "checkpoint_last.pt")
        best_path = (
            args.output / "checkpoint_best.pt"
            if args.smoke
            else source / "weights/checkpoint_best.pt"
        )
        atomic_save(best_checkpoint, best_path)
        plot_training(history, args.output / "training_curve.png")
        print(json.dumps(row), flush=True)
    write_json(
        args.output / "summary.json",
        dict(
            epochs=epochs,
            maximum_epochs=15,
            smoke=args.smoke,
            best_validation_miou=best,
            wall_seconds=time.perf_counter() - started,
            resumed_from=None if resume is None else str(resume),
        ),
    )


def plot_training(history, path):
    """Refresh a legible loss/validation curve after every epoch."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(8, 2.5))
    x = [r["epoch"] for r in history]
    axes[0].plot(x, [r["loss"] for r in history])
    axes[0].set(xlabel="Epoch", ylabel="Training loss")
    axes[1].plot(x, [r["validation"]["completion_mean_iou"] for r in history])
    axes[1].set(xlabel="Epoch", ylabel="Val house-macro mIoU", ylim=(0, 1))
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def generate_csv(args):
    """Run each public Test sample independently and write a canonical CSV."""
    torch.set_num_threads(2)
    start = time.perf_counter()
    source = args.submission_dir.resolve()
    before = tree_hash(source)
    adapter = load_adapter(source)
    device = device_name(args.device)
    model = adapter.load_model(source / "weights", device)
    paths = sample_paths(args.data_root, "test")
    if args.max_samples:
        paths = paths[: args.max_samples]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    with temporary.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Id", "Prediction"])
        for i, p in enumerate(paths):
            prediction = validate_prediction(
                adapter.predict_sample(model, load_inputs(p), device)
            )
            writer.writerow([p.parent.name + "__" + p.name, encode_labels(prediction)])
            if (i + 1) % 15 == 0:
                print(f"predicted {i + 1}/{len(paths)} samples", flush=True)
    if tree_hash(source) != before:
        raise ValueError("Method changed its packaged files during inference")
    temporary.replace(args.output)
    receipt = dict(
        task=TASK,
        num_rows=len(paths),
        complete=(
            [p.parent.name + "__" + p.name for p in paths]
            == expected_sample_ids("test")
        ),
        submission_tree_sha256=before,
        csv_sha256=sha256(args.output),
        data_receipt=receipts(args.data_root, "test"),
        device=device,
        wall_seconds=time.perf_counter() - start,
        runtime_scope="method import + load + all samples + CSV write",
    )
    write_json(args.output.with_name(args.output.stem + "_runtime.json"), receipt)
    return receipt


def validate_csv(path, root):
    """Reject partial coverage, invalid labels, malformed compression and row order."""
    expected = expected_sample_ids("test")
    if [p.parent.name + "__" + p.name for p in sample_paths(root, "test")] != expected:
        raise ValueError("Need all 10 Test houses / 150 samples")
    with Path(path).open(newline="") as f:
        reader = csv.reader(f)
        if next(reader, None) != ["Id", "Prediction"]:
            raise ValueError("CSV header must be Id,Prediction")
        rows = list(reader)
    if (
        len(rows) != 150
        or any(len(r) != 2 for r in rows)
        or [r[0] for r in rows] != expected
    ):
        raise ValueError("Need exactly 150 ordered unique sample IDs")
    for _, encoded in rows:
        decode_labels(encoded)
    return dict(num_rows=150, csv_sha256=sha256(path))


def probe(args):
    """Run the real fixed-grid interface twice, checking independent repeatability."""
    torch.set_num_threads(2)
    adapter = load_adapter(args.submission_dir)
    device = device_name(args.device)
    model = adapter.load_model(args.submission_dir / "weights", device)
    rgb = np.zeros((3, 3, 224, 224), dtype=np.float32)
    k = np.array([[112, 0, 111.5], [0, 112, 111.5], [0, 0, 1]], dtype=np.float32)
    inputs = dict(
        rgb=rgb,
        depth_m=np.ones((3, 1, 224, 224), dtype=np.float32),
        intrinsics=k,
        t_world_camera=np.tile(np.eye(4, dtype=np.float32), (3, 1, 1)),
        t_world_anchor=np.eye(4, dtype=np.float32),
    )
    # Use actual calibrated inputs whenever available. Do not impose an
    # arbitrary image size or grid on a valid replacement architecture.
    available = sorted(Path(args.data_root).glob("samples/*/sample_*/manifest.json"))
    if available:
        inputs = load_inputs(available[0].parent)
    else:
        flip = np.diag([1.0, -1.0, 1.0, 1.0]).astype(np.float32)
        poses = []
        for yaw in [-45, 0, 45]:
            angle = np.deg2rad(yaw)
            c, s = np.cos(angle), np.sin(angle)
            rotation = np.array(
                [[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]],
                dtype=np.float32,
            )
            poses.append(rotation @ flip)
        inputs["t_world_camera"] = np.stack(poses)
        inputs["t_world_anchor"] = poses[1]
    first = validate_prediction(adapter.predict_sample(model, inputs, device)).copy()
    other = (
        load_inputs(available[1].parent)
        if len(available) > 1
        else dict(inputs, rgb=1 - inputs["rgb"])
    )
    validate_prediction(adapter.predict_sample(model, other, device))
    second = validate_prediction(adapter.predict_sample(model, inputs, device))
    if not np.array_equal(first, second):
        raise ValueError("Repeated identical sample produces different output")
    print("Fixed-grid inputs-only interface probe passed")


def package(args):
    """Build one auditable LMS ZIP and its separate final SHA-256 manifest."""
    source = args.submission_dir.resolve()
    if not args.student_id.replace("_", "").isalnum():
        raise ValueError("Invalid NetID")
    for name in ["submission.py", "method.json", "requirements.txt"]:
        if not (source / name).is_file():
            raise FileNotFoundError(f"Missing final method file: {name}")
    csv_path = Path("outputs/kaggle/submission.csv")
    csv_info = validate_csv(csv_path, args.data_root)
    runtime = json.loads(csv_path.with_name("submission_runtime.json").read_text())
    val = json.loads(Path("outputs/evaluation/val/summary.json").read_text())
    digest = tree_hash(source)
    if (
        runtime["submission_tree_sha256"] != digest
        or val["submission_tree_sha256"] != digest
    ):
        raise ValueError("Stale validation/CSV: rerun after code or weight edits")
    if (
        runtime["csv_sha256"] != csv_info["csv_sha256"]
        or not runtime["complete"]
        or not val["complete"]
    ):
        raise ValueError("Need complete matching Val/Test evidence")
    for evidence, split in [(runtime, "test"), (val, "val")]:
        if (
            evidence["data_receipt"] != receipts(args.data_root, split)
            or not evidence["data_receipt"]
        ):
            raise ValueError("Missing/mismatched pinned download receipt")
        receipt = evidence["data_receipt"]
        if (
            len(receipt["shards"]) != 10
            or receipt["split"] != split
            or receipt["repo_id"] != "hongchi/cs498"
        ):
            raise ValueError("Need the complete official split receipt")
        if receipt["revision"] != DEFAULT_REVISION:
            raise ValueError("Use the released pinned data revision")
    method = json.loads((source / "method.json").read_text())
    if method.get("student_id") != args.student_id:
        raise ValueError("Set your NetID in method.json")
    if sum(p.stat().st_size for p in source.rglob("*") if p.is_file()) > 500 * 1024**2:
        raise ValueError("Code/model limit is 500 MiB")
    probe_args = copy.copy(args)
    probe_args.device = "cpu"
    probe(probe_args)
    if tree_hash(source) != digest:
        raise ValueError("Method changed its packaged files during the interface probe")
    validate_figures(source, Path("outputs/evaluation/val"), Path("outputs/report"))
    required = [
        Path("report.pdf"),
        csv_path,
        csv_path.with_name("submission_runtime.json"),
        Path("outputs/evaluation/val/summary.json"),
    ]
    required += [
        Path("outputs/report") / f for f in [*FIGURES, "evidence_manifest.json"]
    ]
    for p in required:
        if not p.is_file() or not p.stat().st_size:
            raise FileNotFoundError(f"Missing {p}")
    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output / f"cs498-track1-{args.student_id}.zip"
    files = []
    for p in sorted(source.rglob("*")):
        if not p.is_file() or "__pycache__" in p.parts or p.suffix == ".pyc":
            continue
        sensitive_name = p.name.lower() in {
            "kaggle.json",
            ".netrc",
        } or p.stem.lower() in {
            "credential",
            "credentials",
            "token",
            "hf_token",
            "api_token",
            "api_key",
            ".env",
        }
        if (
            p.is_symlink()
            or sensitive_name
            or any(
                part in {".git", ".env", ".venv", "credential", "credentials"}
                for part in p.relative_to(source).parts
            )
        ):
            raise ValueError(f"Unsafe package file {p}")
        files.append(p)
    temporary = target.with_suffix(".zip.partial")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
        for p in files:
            archive.write(p, Path("student_submission") / p.relative_to(source))
        for p in required:
            archive.write(p, p.as_posix())
    temporary.replace(target)
    manifest = dict(
        task=TASK,
        student_id=args.student_id,
        kaggle_username=args.kaggle_username,
        package=target.name,
        zip_sha256=sha256(target),
        csv_sha256=csv_info["csv_sha256"],
        submission_tree_sha256=digest,
    )
    write_json(args.output / "package_manifest.json", manifest)
    print(target)
