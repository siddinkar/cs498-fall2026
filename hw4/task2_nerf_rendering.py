"""Task 2: Tiny NeRF model and differentiable volume rendering."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn

from task1_rays_encoding import encoded_dim, positional_encoding


class TinyNeRF(nn.Module):
    """Compact NeRF MLP.

    The model returns raw RGB logits and raw density in the order
    ``[rgb_raw, sigma_raw]``. Activations are applied by raw_to_outputs.
    """

    def __init__(
        self,
        input_ch: int,
        input_ch_dir: int = 0,
        hidden_dim: int = 128,
        num_layers: int = 6,
        skip_layer: int = 3,
        use_viewdirs: bool = True,
    ):
        """Create the TinyNeRF network layers.

        Args:
            input_ch: Width ``C_xyz`` of each encoded 3D point feature. This
                should equal ``encoded_dim(3, num_freqs_xyz)`` from Task 1.
            input_ch_dir: Width ``C_dir`` of each encoded viewing-direction
                feature. This should equal ``encoded_dim(3, num_freqs_dir)``
                when view-dependent color is enabled. Set to ``0`` when not
                using viewing directions.
            hidden_dim: Width ``H`` of the hidden MLP activations and most
                hidden linear layers.
            num_layers: Number of point-processing linear layers in
                ``self.pts_layers``.
            skip_layer: Index of the point-processing layer that receives the
                skip connection. Immediately before this layer, concatenate the
                current hidden activation with the original encoded point
                feature, so that layer's input width is ``H + C_xyz``.
            use_viewdirs: Whether RGB should depend on encoded viewing
                directions. The effective value is ``False`` when
                ``input_ch_dir == 0``.
        """
        super().__init__()
        self.input_ch = input_ch
        self.input_ch_dir = input_ch_dir
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.skip_layer = skip_layer
        self.use_viewdirs = use_viewdirs and input_ch_dir > 0

        layers = []
        for layer_idx in range(num_layers):
            if layer_idx == 0:
                layer_input = input_ch
            elif layer_idx == skip_layer:
                layer_input = hidden_dim + input_ch
            else:
                layer_input = hidden_dim
            layers.append(nn.Linear(layer_input, hidden_dim))
        self.pts_layers = nn.ModuleList(layers)
        self.sigma_head = nn.Linear(hidden_dim, 1)
        if self.use_viewdirs:
            view_hidden_dim = max(1, hidden_dim // 2)
            self.feature_head = nn.Linear(hidden_dim, hidden_dim)
            self.view_layer = nn.Linear(hidden_dim + input_ch_dir, view_hidden_dim)
            self.rgb_head = nn.Linear(view_hidden_dim, 3)
        else:
            self.rgb_head = nn.Linear(hidden_dim, 3)

    def forward(self, x: torch.Tensor, viewdirs: torch.Tensor | None = None) -> torch.Tensor:
        """Return raw predictions for encoded sample points.

        Args:
            x: Encoded 3D sample points with shape ``(..., C_xyz)``, where
                ``C_xyz == self.input_ch``. The leading dimensions can be any
                batch shape, such as ``(N,)`` for flattened samples or
                ``(N, S)`` for ``N`` rays with ``S`` samples per ray.
            viewdirs: Encoded viewing directions with shape ``(..., C_dir)``,
                where ``C_dir == self.input_ch_dir``. Its leading dimensions
                must match ``x``. This argument is required when
                ``self.use_viewdirs`` is ``True`` and can be ``None`` otherwise.

        Returns:
            Tensor with shape ``(..., 4)`` on the same device as ``x``. The
            first three channels are raw RGB logits, and the final channel is
            raw density ``sigma``. Activation functions for RGB and density are
            applied later in ``raw_to_outputs``.
        """
        # ======= INSERT YOUR CODE BELOW =======
        # TODO(student): apply ReLU point MLP, skip connection, and output heads.
        # Predict sigma before view conditioning. If use_viewdirs is true,
        # combine a learned point feature with encoded viewdirs for RGB.
        # Runnable placeholder: preserve the public shape and a gradient path
        # so later scripts fail visibly on quality rather than dimensions.
        zero = self.sigma_head.weight.sum() * 0.0
        return x.new_zeros((*x.shape[:-1], 4)) + zero
        # ======= DO NOT MODIFY CODE OUTSIDE THIS BLOCK =======


def sample_along_rays(
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    near: float,
    far: float,
    num_samples: int,
    perturb: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample depths and 3D points along flattened rays.

    Args:
        rays_o: Ray origins with shape ``(N, 3)``, where ``N`` is the number of
            flattened rays. ``rays_o[n]`` is the world-space origin of ray
            ``n``.
        rays_d: Ray directions with shape ``(N, 3)``. ``rays_d[n]`` is the
            world-space direction paired with ``rays_o[n]``. Directions do not
            need to be unit length.
        near: Near depth bound. The first deterministic sample lies at this
            scalar depth along each ray.
        far: Far depth bound. The last deterministic sample lies at this scalar
            depth along each ray.
        num_samples: Number of samples ``S`` to draw on each ray.
        perturb: If ``False``, use deterministic samples uniformly spaced from
            ``near`` to ``far``. If ``True``, draw one stratified random sample
            inside each interval while keeping all depths inside ``[near, far]``.

    Returns:
        A tuple ``(pts, z_vals)``:
        ``pts`` has shape ``(N, S, 3)`` and contains world-space sample points
        computed as ``rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., None]``.
        ``z_vals`` has shape ``(N, S)`` and contains the sampled depth values.
        Both tensors should use the same dtype and device as ``rays_o``.
    """
    # ======= INSERT YOUR CODE BELOW =======
    # TODO(student): produce z_vals shape (N, S) and points shape (N, S, 3).
    # With perturb=False, samples are uniformly spaced from near to far,
    # inclusive. With perturb=True, jitter one sample inside each stratified
    # interval without leaving [near, far].
    num_rays = rays_o.shape[0]
    z_vals = torch.linspace(near, far, num_samples, device=rays_o.device, dtype=rays_o.dtype).expand(
        num_rays, num_samples
    )
    pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., None]
    return pts, z_vals
    # ======= DO NOT MODIFY CODE OUTSIDE THIS BLOCK =======


def raw_to_outputs(
    raw: torch.Tensor,
    z_vals: torch.Tensor,
    rays_d: torch.Tensor,
    white_background: bool = True,
) -> dict[str, torch.Tensor]:
    """Composite raw samples into one rendered value per ray.

    Args:
        raw: Network predictions with shape ``(N, S, 4)``. ``N`` is the number
            of flattened rays and ``S`` is the number of samples per ray. The
            first three channels ``raw[..., :3]`` are raw RGB logits, and the
            final channel ``raw[..., 3]`` is raw density before the nonnegative
            activation.
        z_vals: Sample depths with shape ``(N, S)``. These should be the depths
            returned by ``sample_along_rays`` for the same rays and samples.
        rays_d: Ray directions with shape ``(N, 3)``. Use their vector norms to
            convert adjacent depth differences into world-space distances.
        white_background: If ``True``, composite any remaining transparency
            over white by adding ``1 - acc`` to the RGB result.

    Hint:
        Keep compositing vectorized over the ``S`` samples; no Python ``for``
        loop is needed. ``torch.cumprod`` is useful for the transmittance prefix
        product. Remember that transmittance at sample ``i`` includes only
        samples before ``i``, so shift the cumulative product by prepending its
        multiplicative identity.

    Returns:
        Dictionary containing:
        ``rgb`` with shape ``(N, 3)``, the final rendered color per ray.
        ``depth`` with shape ``(N,)``, the expected depth using the unnormalized
        compositing weights.
        ``acc`` with shape ``(N,)``, the accumulated opacity per ray.
        ``weights`` with shape ``(N, S)``, the front-to-back alpha-compositing
        weight for each sample. Do not renormalize these weights to sum to one.
        ``sample_rgb`` with shape ``(N, S, 3)``, the sigmoid-activated RGB value
        at each sample before compositing.
    """
    # ======= INSERT YOUR CODE BELOW =======
    # TODO(student): implement front-to-back alpha compositing.
    # RGB uses sigmoid(raw[..., :3]); density uses a nonnegative activation.
    # Return rgb, depth, acc, weights, and sample_rgb with the shapes described
    # in the handout. Do not normalize weights to sum to one. Avoid a Python
    # loop over samples; torch.cumprod can form the exclusive prefix product.
    sample_rgb = torch.sigmoid(raw[..., :3])
    weights = torch.zeros_like(z_vals)
    rgb_map = sample_rgb.mean(dim=1)
    depth_map = torch.zeros(raw.shape[0], device=raw.device, dtype=raw.dtype)
    acc_map = torch.zeros(raw.shape[0], device=raw.device, dtype=raw.dtype)
    return {"rgb": rgb_map, "depth": depth_map, "acc": acc_map, "weights": weights, "sample_rgb": sample_rgb}
    # ======= DO NOT MODIFY CODE OUTSIDE THIS BLOCK =======


def render_rays(
    model: nn.Module,
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    cfg: dict[str, object],
) -> dict[str, torch.Tensor]:
    """Render flattened rays with the task implementations."""
    pts, z_vals = sample_along_rays(
        rays_o,
        rays_d,
        float(cfg["near"]),
        float(cfg["far"]),
        int(cfg["num_samples"]),
        bool(cfg.get("perturb", False)),
    )
    pts_flat = pts.reshape(-1, 3)
    embedded_pts = positional_encoding(pts_flat, int(cfg["num_freqs_xyz"]))

    viewdirs = rays_d / torch.linalg.norm(rays_d, dim=-1, keepdim=True).clamp_min(1.0e-8)
    viewdirs = viewdirs[:, None, :].expand_as(pts).reshape(-1, 3)
    embedded_dirs = positional_encoding(viewdirs, int(cfg["num_freqs_dir"]))

    raw = model(embedded_pts, embedded_dirs)
    raw = raw.reshape(rays_o.shape[0], int(cfg["num_samples"]), 4)
    return raw_to_outputs(raw, z_vals, rays_d, bool(cfg["white_background"]))


def save_task2_visual_checkpoint(path: str | Path) -> None:
    """Save the Task 2 visual checkpoint for sampling and compositing."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(498)

    num_rays = 32
    num_samples = 16
    rays_o = torch.zeros(num_rays, 3)
    rays_d = torch.tensor([[0.0, 0.0, -1.0]]).expand(num_rays, 3)
    _, z_det = sample_along_rays(rays_o, rays_d, 2.0, 6.0, num_samples, perturb=False)
    _, z_pert = sample_along_rays(rays_o, rays_d, 2.0, 6.0, num_samples, perturb=True)

    case_names = ["near red", "far blue", "transparent"]
    case_raw = torch.full((3, num_samples, 4), -8.0)
    case_raw[..., :3] = -6.0
    case_raw[0, 2, :3] = torch.tensor([6.0, -6.0, -6.0])
    case_raw[0, 2, 3] = 40.0
    case_raw[1, -3, :3] = torch.tensor([-6.0, -6.0, 6.0])
    case_raw[1, -3, 3] = 40.0
    case_raw[2, :, :3] = 0.0
    case_z = torch.linspace(1.0, 4.0, num_samples).expand(3, num_samples)
    case_rays_d = torch.tensor([[0.0, 0.0, -1.0]]).expand(3, 3)
    rendered = raw_to_outputs(case_raw, case_z, case_rays_d, white_background=True)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    im0 = axes[0, 0].imshow(z_det.detach().cpu(), aspect="auto", interpolation="nearest")
    axes[0, 0].set_title("Deterministic sample depths")
    axes[0, 0].set_xlabel("Sample index")
    axes[0, 0].set_ylabel("Ray index")
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    im1 = axes[0, 1].imshow(z_pert.detach().cpu(), aspect="auto", interpolation="nearest")
    axes[0, 1].set_title("Perturbed sample depths")
    axes[0, 1].set_xlabel("Sample index")
    axes[0, 1].set_ylabel("Ray index")
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    weights = rendered["weights"].detach().cpu()
    im2 = axes[1, 0].imshow(weights, aspect="auto", vmin=0.0, vmax=1.0, interpolation="nearest")
    axes[1, 0].set_title("Compositing weights")
    axes[1, 0].set_xlabel("Sample index")
    axes[1, 0].set_yticks(range(len(case_names)), labels=case_names)
    fig.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)

    rgb = rendered["rgb"].detach().cpu().clamp(0.0, 1.0)
    axes[1, 1].imshow(rgb.reshape(len(case_names), 1, 3), interpolation="nearest")
    axes[1, 1].set_title("Rendered RGB swatches")
    axes[1, 1].set_xticks([])
    axes[1, 1].set_yticks(range(len(case_names)), labels=case_names)
    for row, name in enumerate(case_names):
        depth = float(rendered["depth"][row].detach().cpu())
        acc = float(rendered["acc"][row].detach().cpu())
        print(f"{name}: rgb={rgb[row].tolist()} depth={depth:.3f} acc={acc:.3f}")

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"saved {path}")


def run_basic_checks() -> None:
    """Run lightweight interface checks that keep the starter executable."""
    input_ch = encoded_dim(3, 4)
    input_ch_dir = encoded_dim(3, 2)
    model = TinyNeRF(input_ch=input_ch, input_ch_dir=input_ch_dir, hidden_dim=32, num_layers=4, skip_layer=2)
    raw = model(torch.zeros(5, input_ch), torch.zeros(5, input_ch_dir))
    assert raw.shape == (5, 4)

    rays_o = torch.zeros(4, 3)
    rays_d = torch.tensor([[0.0, 0.0, -1.0]]).expand(4, 3)
    pts, z_vals = sample_along_rays(rays_o, rays_d, 2.0, 6.0, 8, perturb=False)
    assert pts.shape == (4, 8, 3)
    assert z_vals.shape == (4, 8)
    out = raw_to_outputs(torch.zeros(4, 8, 4), z_vals, rays_d)
    assert out["rgb"].shape == (4, 3)
    assert out["weights"].shape == (4, 8)
    print("Task 2 starter checks passed")


def main() -> None:
    """Run Task 2 checks and optionally write the visual checkpoint."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Run starter interface checks without saving the visual checkpoint.")
    parser.add_argument("--output", default="outputs/task2/task2_visual_checkpoint.png")
    args = parser.parse_args()
    run_basic_checks()
    if not args.check:
        save_task2_visual_checkpoint(args.output)


if __name__ == "__main__":
    main()
