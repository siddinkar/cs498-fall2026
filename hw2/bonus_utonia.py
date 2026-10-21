"""Optional bonus: Utonia frozen-feature linear probe for chair part segmentation.

Run from the handout folder, inside an environment that has the official
Utonia package and its sparse-convolution dependencies installed (see
https://github.com/Pointcept/Utonia, "package mode"). CUDA is required by
spconv. Example:

    python bonus_utonia.py                      # full Utonia encoder
    python bonus_utonia.py --model utonia_tiny  # smaller encoder, smaller cache
    python bonus_utonia.py --max-train 32 --max-val 16 --epochs 2  # smoke test

Pipeline: download the pinned official checkpoint, freeze the encoder, run the
official transform and extract per-point features (cached as float16 memmaps),
then train only a point-wise nn.Linear head and save the bonus outputs under
outputs/bonus/.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from hw2_utils import PART_COLORS, PART_NAMES, load_split, normalize_points, plot_point_clouds, plot_training, write_json
from task1_data import compute_confusion_matrix, iou_from_confusion


UTONIA_REPO_ID = "Pointcept/Utonia"
UTONIA_HF_REVISION = "9457f025e36c7001ee962a542f0d27a5b600e1e7"  # HF main on 2026-10-21
UTONIA_SOURCE_COMMIT = "de1f9deef4ee65e508d74347f8d0eb848d878abb"  # GitHub main on 2026-10-16
UTONIA_MODELS = ("utonia", "utonia_small", "utonia_tiny")

EXAMPLE_INDICES = [0, 7]  # same fixed validation chairs as Tasks 2 and 3
NUM_POINTS = 2048


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=UTONIA_MODELS, default="utonia", help="official encoder name")
    parser.add_argument("--hf-revision", default=UTONIA_HF_REVISION, help="Hugging Face revision of Pointcept/Utonia")
    parser.add_argument("--download-root", default=os.path.expanduser("~/.cache/utonia/ckpt"))
    parser.add_argument("--checkpoint", default=None, help="local .pth path; skips the Hugging Face download")
    parser.add_argument("--no-flash", action="store_true", help="force the non-FlashAttention backend")
    parser.add_argument("--scale", type=float, default=1.0, help="RandomScale factor of the official transform")
    parser.add_argument("--grid-size", type=float, default=0.01, help="GridSample voxel size of the official transform")
    parser.add_argument("--upcast-levels", type=int, default=4,
                        help="number of coarse levels concatenated during feature upcasting (README: 4 for evaluation)")
    parser.add_argument("--no-color", action="store_true", help="feed zeros instead of RGB to the encoder")
    parser.add_argument("--no-normal", action="store_true", help="feed zeros instead of normals to the encoder")
    parser.add_argument("--extract-batch-size", type=int, default=8)
    parser.add_argument("--cache-dir", default=None, help="feature cache folder (default outputs/bonus/feature_cache)")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--in-memory", action="store_true", help="load cached features fully into RAM")
    parser.add_argument("--no-standardize", action="store_true",
                        help="disable the fixed (non-trainable) per-dimension feature standardization")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=498)
    parser.add_argument("--max-train", type=int, default=None, help="debug: use only the first K training chairs")
    parser.add_argument("--max-val", type=int, default=None, help="debug: use only the first K validation chairs")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


# ----------------------------------------------------------------------------
# Utonia encoder loading and frozen feature extraction
# ----------------------------------------------------------------------------
def load_utonia(args, device):
    import utonia
    from huggingface_hub import hf_hub_download

    try:
        import flash_attn  # noqa: F401
        have_flash = True
    except ImportError:
        have_flash = False

    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
    else:
        checkpoint_path = Path(hf_hub_download(
            repo_id=UTONIA_REPO_ID,
            filename=f"{args.model}.pth",
            repo_type="model",
            revision=args.hf_revision,
            local_dir=args.download_root,
        ))

    custom_config = None
    if args.no_flash or not have_flash:  # same fallback as the official demos
        custom_config = dict(enc_patch_size=[1024 for _ in range(5)], enable_flash=False)
    model = utonia.model.load(str(checkpoint_path), custom_config=custom_config).to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    sha256 = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = json.loads(json.dumps(checkpoint["config"], default=str))
    provenance = {
        "repo_id": UTONIA_REPO_ID,
        "model_name": args.model,
        "hf_revision": args.hf_revision,
        "checkpoint_source": "local path (pre-downloaded)" if args.checkpoint else "hf_hub_download",
        "checkpoint_file": str(checkpoint_path),
        "checkpoint_sha256": sha256,
        "source_repository": "https://github.com/Pointcept/Utonia",
        "source_commit": UTONIA_SOURCE_COMMIT,
        "flash_attention": bool(have_flash and not args.no_flash),
        "encoder_config": config,
        "encoder_parameter_count": int(sum(p.numel() for p in model.parameters())),
        "utonia_module_path": str(Path(utonia.__file__).resolve().parent),
    }
    return model, provenance


def build_transform(args):
    """Official default transform with NormalizeCoord enabled for single objects."""
    import utonia

    config = [
        dict(type="NormalizeCoord"),
        dict(type="RandomScale", scale=[args.scale, args.scale]),
        dict(type="CenterShift", apply_z=True),
        dict(
            type="GridSample",
            grid_size=args.grid_size,
            hash_type="fnv",
            mode="train",
            return_grid_coord=True,
            return_inverse=True,
        ),
        dict(type="NormalizeColor"),
        dict(type="ToTensor"),
        dict(
            type="Collect",
            keys=("coord", "grid_coord", "color", "inverse"),
            feat_keys=("coord", "color", "normal"),
        ),
    ]
    return utonia.transform.Compose(config)


def point_dict(split, index, use_color, use_normal):
    """One chair in the dictionary format expected by the Utonia transform."""
    coord = split["xyz"][index].astype(np.float32).copy()
    if use_color:
        color = (split["rgb"][index] * 255.0).astype(np.float32)
    else:
        color = np.zeros_like(coord)
    if use_normal and "normal" in split:
        normal = split["normal"][index].astype(np.float32).copy()
    else:
        normal = np.zeros_like(coord)
    return {"coord": coord, "color": color, "normal": normal}


def upcast_features(point, levels):
    """Concatenate `levels` coarse feature maps, then propagate to the finest level."""
    for _ in range(levels):
        if "pooling_parent" not in point.keys():
            break
        parent = point.pop("pooling_parent")
        inverse = point.pop("pooling_inverse")
        parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
        point = parent
    while "pooling_parent" in point.keys():
        parent = point.pop("pooling_parent")
        inverse = point.pop("pooling_inverse")
        parent.feat = point.feat[inverse]
        point = parent
    return point


@torch.inference_mode()
def extract_batch(model, transform, split, indices, args, device):
    """Return frozen features with shape (len(indices), 2048, D) for the given chairs."""
    import utonia

    samples = [transform(point_dict(split, index, not args.no_color, not args.no_normal)) for index in indices]
    point = utonia.data.collate_fn(samples)
    for key in point.keys():
        if isinstance(point[key], torch.Tensor):
            point[key] = point[key].to(device, non_blocking=True)
    point = model(point)
    point = upcast_features(point, args.upcast_levels)

    # GridSample changed the point count; `inverse` maps each original point back.
    offset = point.offset.tolist()  # cumulative grid-sampled point counts per chair
    inverse = point.inverse
    features = []
    start = 0
    for batch_index, end in enumerate(offset):
        chair_feat = point.feat[start:end]
        chair_inverse = inverse[batch_index * NUM_POINTS:(batch_index + 1) * NUM_POINTS]
        features.append(chair_feat[chair_inverse])
        start = end
    return torch.stack(features).float().cpu().numpy()


def cache_signature(args):
    return {
        "model": args.model,
        "hf_revision": args.hf_revision if not args.checkpoint else str(args.checkpoint),
        "scale": args.scale,
        "grid_size": args.grid_size,
        "upcast_levels": args.upcast_levels,
        "use_color": not args.no_color,
        "use_normal": not args.no_normal,
        "seed": args.seed,
    }


def cached_features(name, split, count, model, transform, args, device, cache_dir):
    """Extract (or reuse) frozen features for `count` chairs of one split as a float16 memmap."""
    feature_path = cache_dir / f"{name}_features.npy"
    meta_path = cache_dir / f"{name}_meta.json"
    signature = dict(cache_signature(args), count=count)
    if feature_path.exists() and meta_path.exists() and not args.rebuild_cache:
        meta = json.loads(meta_path.read_text())
        if meta.get("signature") == signature and meta.get("complete"):
            print(f"Reusing cached {name} features from {feature_path}")
            return np.load(feature_path, mmap_mode="r"), meta
        print(f"Cache signature changed; rebuilding {name} features")

    start = time.perf_counter()
    memmap = None
    for first in range(0, count, args.extract_batch_size):
        indices = list(range(first, min(first + args.extract_batch_size, count)))
        features = extract_batch(model, transform, split, indices, args, device)
        if memmap is None:
            feature_dim = features.shape[-1]
            size_gb = count * NUM_POINTS * feature_dim * 2 / 1e9
            print(f"{name}: feature dim {feature_dim}; float16 cache {size_gb:.1f} GB at {feature_path}")
            memmap = np.lib.format.open_memmap(
                feature_path, mode="w+", dtype=np.float16, shape=(count, NUM_POINTS, feature_dim)
            )
        memmap[indices[0]:indices[-1] + 1] = features.astype(np.float16)
        if (first // args.extract_batch_size) % 25 == 0:
            print(f"  extracted {indices[-1] + 1}/{count} {name} chairs")
    memmap.flush()
    seconds = time.perf_counter() - start
    meta = {"signature": signature, "feature_dim": int(memmap.shape[-1]),
            "extraction_seconds": seconds, "complete": True}
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    del memmap
    return np.load(feature_path, mmap_mode="r"), meta


# ----------------------------------------------------------------------------
# Linear probe
# ----------------------------------------------------------------------------
class FeatureDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return (
            torch.from_numpy(np.ascontiguousarray(self.features[index])),
            torch.from_numpy(self.labels[index]),
        )


class LinearProbe(nn.Module):
    """Point-wise linear layer on frozen features.

    `mean` and `std` are fixed training statistics stored as buffers; only
    `self.linear` has trainable parameters.
    """

    def __init__(self, feature_dim, num_classes, mean=None, std=None):
        super().__init__()
        if mean is None:
            mean = torch.zeros(feature_dim)
        if std is None:
            std = torch.ones(feature_dim)
        self.register_buffer("mean", mean.float())
        self.register_buffer("std", std.float())
        self.linear = nn.Linear(feature_dim, num_classes)

    def forward(self, features):
        return self.linear((features.float() - self.mean) / self.std)


def feature_statistics(features, chunk=16):
    """Per-dimension mean and std over all training points."""
    count = 0
    total = None
    total_sq = None
    for first in range(0, len(features), chunk):
        block = np.asarray(features[first:first + chunk], dtype=np.float64).reshape(-1, features.shape[-1])
        total = block.sum(0) if total is None else total + block.sum(0)
        total_sq = (block ** 2).sum(0) if total_sq is None else total_sq + (block ** 2).sum(0)
        count += len(block)
    mean = total / count
    variance = np.maximum(total_sq / count - mean ** 2, 0.0)
    std = np.sqrt(variance) + 1e-6
    return torch.from_numpy(mean), torch.from_numpy(std)


def probe_loss(logits, labels, class_weights):
    return F.cross_entropy(logits.flatten(0, 1), labels.flatten().long(), weight=class_weights, reduction="mean")


@torch.no_grad()
def evaluate_probe(head, loader, device, class_weights):
    head.eval()
    predictions, targets = [], []
    loss_sum = 0.0
    weight_sum = 0.0
    for features, labels in loader:
        features, labels = features.to(device), labels.to(device)
        logits = head(features)
        loss_sum += F.cross_entropy(
            logits.flatten(0, 1), labels.flatten().long(), weight=class_weights, reduction="sum"
        ).item()
        weight_sum += class_weights[labels].sum().item()
        predictions.append(logits.argmax(-1).cpu().numpy())
        targets.append(labels.cpu().numpy())
    prediction = np.concatenate(predictions)
    target = np.concatenate(targets)
    confusion = compute_confusion_matrix(prediction, target, len(PART_NAMES))
    per_class_iou = iou_from_confusion(confusion)
    return (
        float(np.nanmean(per_class_iou)),
        per_class_iou,
        prediction,
        loss_sum / weight_sum,
        float(np.mean(prediction == target)),
        confusion,
    )


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------
def pca_colors(features):
    """First three principal components of (P, D) features, rescaled to [0, 1]."""
    features = features.float()
    q = min(6, features.shape[-1])
    _, _, v = torch.pca_lowrank(features, q=q, center=True, niter=5)
    projection = (features - features.mean(0, keepdim=True)) @ v[:, :3]
    low = projection.min(0, keepdim=True).values
    high = projection.max(0, keepdim=True).values
    return ((projection - low) / (high - low).clamp_min(1e-6)).clamp(0.0, 1.0)


def plot_rgb_grid(xyz_rows, color_rows, title_rows, path):
    """Rows of point clouds with explicit RGB colors."""
    rows = len(xyz_rows)
    columns = len(xyz_rows[0])
    figure = plt.figure(figsize=(4.2 * columns, 4.0 * rows))
    for row in range(rows):
        for column in range(columns):
            xyz = xyz_rows[row][column]
            axis = figure.add_subplot(rows, columns, row * columns + column + 1, projection="3d")
            axis.scatter(*xyz.T, c=np.clip(color_rows[row][column], 0, 1), s=7, depthshade=False)
            extent = max(float(np.abs(xyz).max()), 1e-3)
            axis.set(xlim=(-extent, extent), ylim=(-extent, extent), zlim=(-extent, extent))
            axis.set_title(title_rows[row][column])
            axis.set_axis_off()
            axis.view_init(elev=18, azim=-58)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


# ----------------------------------------------------------------------------
def main():
    args = parse_args()
    root = Path(__file__).resolve().parent
    output = root / "outputs/bonus"
    output.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir) if args.cache_dir else output / "feature_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)  # GridSample uses NumPy's global RNG
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("Warning: Utonia's spconv/flash-attention backbone expects CUDA; extraction may fail on", device)

    train = load_split(root / "data/train.npz")
    val = load_split(root / "data/val.npz")
    train_count = min(len(train["xyz"]), args.max_train) if args.max_train else len(train["xyz"])
    val_count = min(len(val["xyz"]), args.max_val) if args.max_val else len(val["xyz"])
    val_count = max(val_count, max(EXAMPLE_INDICES) + 1)
    train_labels = train["labels"][:train_count]
    val_labels = val["labels"][:val_count]

    # Frozen encoder, official transform, cached per-point features.
    model, provenance = load_utonia(args, device)
    transform = build_transform(args)
    print(f"Loaded {args.model}: {provenance['encoder_parameter_count'] / 1e6:.2f}M frozen parameters")
    train_features, train_meta = cached_features("train", train, train_count, model, transform, args, device, cache_dir)
    val_features, val_meta = cached_features("val", val, val_count, model, transform, args, device, cache_dir)
    feature_dim = train_features.shape[-1]
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if args.in_memory:
        train_features = np.asarray(train_features)
        val_features = np.asarray(val_features)

    # Linear probe: same class weights and optimizer family as the required tasks.
    counts = np.bincount(train["labels"].reshape(-1), minlength=len(PART_NAMES))
    class_weights = torch.tensor(np.sqrt(counts.sum() / counts), dtype=torch.float32, device=device)
    class_weights /= class_weights.mean()

    if args.no_standardize:
        mean, std = None, None
    else:
        mean, std = feature_statistics(train_features)
    head = LinearProbe(feature_dim, len(PART_NAMES), mean, std).to(device)
    trainable = sum(p.numel() for p in head.parameters() if p.requires_grad)
    assert trainable == feature_dim * len(PART_NAMES) + len(PART_NAMES), "only the linear head may be trainable"
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_loader = DataLoader(FeatureDataset(train_features, train_labels), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(FeatureDataset(val_features, val_labels), batch_size=args.batch_size)

    history = {"train_loss": [], "val_loss": [], "val_miou": []}
    best_miou, best_epoch = -1.0, 0
    checkpoint_path = output / "linear_probe.pt"
    start = time.perf_counter()
    for epoch in range(args.epochs):
        head.train()
        batch_losses = []
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = probe_loss(head(features), labels, class_weights)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
        val_miou, _, _, val_loss, _, _ = evaluate_probe(head, val_loader, device, class_weights)
        train_loss = float(np.mean(batch_losses))
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_miou"].append(val_miou)
        if val_miou > best_miou:
            best_miou, best_epoch = val_miou, epoch + 1
            torch.save(head.state_dict(), checkpoint_path)
        print(f"epoch {epoch + 1:03d}/{args.epochs}: train loss={train_loss:.4f}, "
              f"val loss={val_loss:.4f}, val mIoU={val_miou:.3f}")
    training_time = time.perf_counter() - start
    plot_training(history, output / "training_curves.png")
    print("Bonus checkpoint A: training_curves.png")

    # Best-checkpoint metrics and figures on the two fixed validation chairs.
    head.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    val_miou, per_class_iou, prediction, val_loss, accuracy, confusion = evaluate_probe(
        head, val_loader, device, class_weights
    )
    example_xyz = normalize_points(torch.from_numpy(val["xyz"][EXAMPLE_INDICES])).numpy()
    plot_point_clouds(
        [example_xyz[0], example_xyz[0], example_xyz[1], example_xyz[1]],
        [val_labels[EXAMPLE_INDICES[0]], prediction[EXAMPLE_INDICES[0]],
         val_labels[EXAMPLE_INDICES[1]], prediction[EXAMPLE_INDICES[1]]],
        ["Sample 1: ground truth", "Sample 1: Utonia linear probe",
         "Sample 2: ground truth", "Sample 2: Utonia linear probe"],
        output / "predictions.png",
    )

    # Joint PCA over both chairs so their colors are comparable.
    example_features = torch.from_numpy(np.asarray(val_features[EXAMPLE_INDICES], dtype=np.float32))
    joint_colors = pca_colors(example_features.reshape(-1, feature_dim)).reshape(2, NUM_POINTS, 3).numpy()
    plot_rgb_grid(
        [[example_xyz[row]] * 4 for row in range(2)],
        [[val["rgb"][EXAMPLE_INDICES[row]], joint_colors[row],
          PART_COLORS[val_labels[EXAMPLE_INDICES[row]]], PART_COLORS[prediction[EXAMPLE_INDICES[row]]]]
         for row in range(2)],
        [[f"Sample {row + 1}: input RGB", f"Sample {row + 1}: frozen-feature PCA (3 comps)",
          f"Sample {row + 1}: ground truth", f"Sample {row + 1}: linear probe"] for row in range(2)],
        output / "feature_pca.png",
    )
    np.savetxt(output / "confusion_matrix.txt", confusion, fmt="%d")

    metrics = {
        "split": "validation",
        "validation_miou": val_miou,
        "validation_loss": val_loss,
        "overall_point_accuracy": accuracy,
        "per_class_iou": per_class_iou.tolist(),
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "trainable_parameter_count": int(trainable),
        "frozen_encoder_parameter_count": provenance["encoder_parameter_count"],
        "training_time_seconds": training_time,
        "feature_extraction_seconds": {"train": train_meta.get("extraction_seconds"),
                                       "val": val_meta.get("extraction_seconds")},
        "checkpoint_selection": "highest validation_miou",
        "linear_probe": {
            "head": f"nn.Linear({feature_dim}, {len(PART_NAMES)})",
            "feature_dim": int(feature_dim),
            "fixed_feature_standardization": not args.no_standardize,
            "optimizer": f"AdamW(lr={args.lr}, weight_decay={args.weight_decay})",
            "batch_size": args.batch_size,
            "train_chairs": train_count,
            "val_chairs": val_count,
            "augmentation": "none (features are extracted once from un-augmented chairs)",
        },
        "preprocessing": {
            "transform": ["NormalizeCoord", f"RandomScale[{args.scale},{args.scale}]", "CenterShift(apply_z=True)",
                          f"GridSample(grid_size={args.grid_size}, fnv, train, inverse)", "NormalizeColor",
                          "Collect(feat=coord+color+normal)"],
            "upcast_levels": args.upcast_levels,
            "encoder_inputs": {"xyz": True, "rgb": not args.no_color, "normal": not args.no_normal},
        },
        "utonia": provenance,
        "command": " ".join(sys.argv),
    }
    write_json(output / "metrics.json", metrics)
    print("Bonus checkpoint B: feature_pca.png, predictions.png, metrics.json, and linear_probe.pt")
    return metrics


if __name__ == "__main__":
    print("Bonus complete:", main())
