from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


def project_voxel_centers(
    points, intrinsics, t_world_camera, t_world_anchor, image_height, image_width
):
    """Project local y-up centers into image-camera y-down coordinates.

    Return an align_corners=False sampling grid [B*3,N,1,2], the valid
    image projection mask [B,3,N], and optical-axis camera depths [B,3,N].
    Poses map camera to world. Intrinsics refer to the original image size.
    """
    # BEGIN YOUR CODE: project_voxel_centers
    b, v = t_world_camera.shape[:2]
    n = len(points)
    return (
        points.new_zeros((b * v, n, 1, 2)),
        torch.zeros((b, v, n), dtype=torch.bool, device=points.device),
        points.new_zeros((b, v, n)),
    )
    # END YOUR CODE


def factorized_logits(raw):
    """Convert occupancy + conditional semantic scores to ten class logits.

    p(occupied)=sigmoid(raw[:,0]); occupied semantics are conditional softmax.
    The returned free logit is zero; softmax over all ten logits is normalized.
    """
    # BEGIN YOUR CODE: factorized_logits
    return raw * 0
    # END YOUR CODE


def _group_count(channels: int) -> int:
    """Choose the largest small GroupNorm group count dividing the channel width."""
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


@dataclass(frozen=True)
class AnchorVolumeSpec:
    """Dense camera-relative prediction grid with tensor order ``(z, y, x)``."""

    x_range_m: tuple[float, float]
    y_range_m: tuple[float, float]
    z_range_m: tuple[float, float]
    voxel_size_m: float

    def __post_init__(self) -> None:
        """Reject invalid bounds before constructing a dense prediction volume."""
        if self.voxel_size_m <= 0:
            raise ValueError("voxel_size_m must be positive")
        for name, bounds in (
            ("x_range_m", self.x_range_m),
            ("y_range_m", self.y_range_m),
            ("z_range_m", self.z_range_m),
        ):
            if len(bounds) != 2 or bounds[1] <= bounds[0]:
                raise ValueError(f"{name} must contain increasing lower/upper bounds")
            cells = (bounds[1] - bounds[0]) / self.voxel_size_m
            if not math.isclose(cells, round(cells), abs_tol=1e-5):
                raise ValueError(f"{name} extent must be divisible by voxel_size_m")

    @property
    def shape_zyx(self) -> tuple[int, int, int]:
        """Return tensor dimensions in depth, height, width order."""
        return tuple(
            int(round((bounds[1] - bounds[0]) / self.voxel_size_m))
            for bounds in (self.z_range_m, self.y_range_m, self.x_range_m)
        )

    @property
    def num_voxels(self) -> int:
        """Count cells in the complete fixed prediction grid."""
        z, y, x = self.shape_zyx
        return z * y * x

    def centers(self, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Return local ``(x-right, y-up, z-forward)`` voxel centers."""
        x = torch.arange(self.shape_zyx[2], dtype=dtype)
        y = torch.arange(self.shape_zyx[1], dtype=dtype)
        z = torch.arange(self.shape_zyx[0], dtype=dtype)
        x = self.x_range_m[0] + (x + 0.5) * self.voxel_size_m
        y = self.y_range_m[0] + (y + 0.5) * self.voxel_size_m
        z = self.z_range_m[0] + (z + 0.5) * self.voxel_size_m
        zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
        return torch.stack([xx, yy, zz], dim=-1)

    def normalized_coordinates(self) -> torch.Tensor:
        """Return three local-coordinate channels [3,Z,Y,X], each scaled to [-1,1]."""
        centers = self.centers()
        coordinate_channels = []
        for values, bounds in zip(
            (centers[..., 0], centers[..., 1], centers[..., 2]),
            (self.x_range_m, self.y_range_m, self.z_range_m),
            strict=True,
        ):
            coordinate_channels.append(
                2.0 * (values - bounds[0]) / (bounds[1] - bounds[0]) - 1.0
            )
        return torch.stack(coordinate_channels, dim=0)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> AnchorVolumeSpec:
        """Read the portable volume bounds and resolution from the model settings."""
        model = config["triview_model"]
        volume = model["volume"]
        return cls(
            x_range_m=tuple(float(value) for value in volume["x_range_m"]),
            y_range_m=tuple(float(value) for value in volume["y_range_m"]),
            z_range_m=tuple(float(value) for value in volume["z_range_m"]),
            voxel_size_m=float(volume["voxel_size_m"]),
        )


class ResidualBlock2d(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, *, stride: int = 1):
        """Construct two image convolutions and a shape-matched residual branch."""
        super().__init__()
        self.conv1 = nn.Conv2d(
            input_channels,
            output_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.norm1 = nn.GroupNorm(_group_count(output_channels), output_channels)
        self.conv2 = nn.Conv2d(
            output_channels, output_channels, 3, padding=1, bias=False
        )
        self.norm2 = nn.GroupNorm(_group_count(output_channels), output_channels)
        self.skip = (
            nn.Identity()
            if input_channels == output_channels and stride == 1
            else nn.Conv2d(
                input_channels, output_channels, 1, stride=stride, bias=False
            )
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """Encode [B,C,H,W], optionally downsampling by the configured stride."""
        residual = self.skip(values)
        values = F.silu(self.norm1(self.conv1(values)), inplace=True)
        return F.silu(self.norm2(self.conv2(values)) + residual, inplace=True)


class SharedImageEncoder(nn.Module):
    """A small shared encoder; all three camera views use identical weights."""

    def __init__(self, input_channels: int, feature_channels: int):
        """Build the shared RGB(-D) encoder with an overall image stride of four."""
        super().__init__()
        stem_channels = max(8, feature_channels // 2)
        self.network = nn.Sequential(
            nn.Conv2d(
                input_channels, stem_channels, 5, stride=2, padding=2, bias=False
            ),
            nn.GroupNorm(_group_count(stem_channels), stem_channels),
            nn.SiLU(inplace=True),
            ResidualBlock2d(stem_channels, stem_channels),
            ResidualBlock2d(stem_channels, feature_channels, stride=2),
            ResidualBlock2d(feature_channels, feature_channels),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """Encode a flattened batch of camera images into shared feature maps."""
        return self.network(values)


class ResidualBlock3d(nn.Module):
    def __init__(self, input_channels: int, output_channels: int):
        """Construct a spatial residual block preserving all three grid dimensions."""
        super().__init__()
        self.conv1 = nn.Conv3d(
            input_channels, output_channels, 3, padding=1, bias=False
        )
        self.norm1 = nn.GroupNorm(_group_count(output_channels), output_channels)
        self.conv2 = nn.Conv3d(
            output_channels, output_channels, 3, padding=1, bias=False
        )
        self.norm2 = nn.GroupNorm(_group_count(output_channels), output_channels)
        self.skip = (
            nn.Identity()
            if input_channels == output_channels
            else nn.Conv3d(input_channels, output_channels, 1, bias=False)
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """Transform [B,C,Z,Y,X] features with a channel-matched skip connection."""
        residual = self.skip(values)
        values = F.silu(self.norm1(self.conv1(values)), inplace=True)
        return F.silu(self.norm2(self.conv2(values)) + residual, inplace=True)


class SpatialUNet3d(nn.Module):
    def __init__(self, input_channels: int, base_channels: int, output_channels: int):
        """Build two downsampling levels, skip decoders and a pointwise output head."""
        super().__init__()
        width = int(base_channels)
        self.stem = nn.Conv3d(input_channels, width, 1, bias=False)
        self.encoder1 = ResidualBlock3d(width, width)
        self.down1 = nn.Conv3d(width, width * 2, 3, stride=2, padding=1, bias=False)
        self.encoder2 = ResidualBlock3d(width * 2, width * 2)
        self.down2 = nn.Conv3d(width * 2, width * 4, 3, stride=2, padding=1, bias=False)
        self.bottleneck = nn.Sequential(
            ResidualBlock3d(width * 4, width * 4),
            ResidualBlock3d(width * 4, width * 4),
        )
        self.decoder2 = ResidualBlock3d(width * 4 + width * 2, width * 2)
        self.decoder1 = ResidualBlock3d(width * 2 + width, width)
        self.head = nn.Conv3d(width, output_channels, 1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """Decode spatial features, interpolating to exact skip sizes before concatenation.

        Input [B,C,Z,Y,X]; output [B,10,Z,Y,X]. The height 14 is not divisible
        by four, so use skip.shape[-3:] rather than a fixed scale factor.
        """
        # BEGIN YOUR CODE: SpatialUNet3d.forward
        return values[:, :1].expand(-1, self.head.out_channels, -1, -1, -1) * 0
        # END YOUR CODE


class TriViewLiftFuseUNet(nn.Module):
    """Lift three image feature maps into 3D, fuse them, then complete semantics.

    The model consumes only RGB, optional depth, intrinsics, and camera poses.
    Semantic/instance annotations and target masks are deliberately absent from
    the forward API.
    """

    def __init__(
        self,
        num_semantic_classes: int,
        volume: AnchorVolumeSpec,
        *,
        use_depth: bool = True,
        image_feature_channels: int = 24,
        volume_feature_channels: int = 24,
        base_channels: int = 24,
        depth_truncation_m: float = 0.5,
    ):
        """Build shared encoding, ordered fusion and completion heads for one sample."""
        super().__init__()
        if num_semantic_classes <= 0:
            raise ValueError("num_semantic_classes must be positive")
        if depth_truncation_m <= 0:
            raise ValueError("depth_truncation_m must be positive")
        self.num_semantic_classes = int(num_semantic_classes)
        self.volume = volume
        self.use_depth = bool(use_depth)
        self.depth_truncation_m = float(depth_truncation_m)
        self.image_encoder = SharedImageEncoder(
            4 if self.use_depth else 3,
            int(image_feature_channels),
        )
        per_view_channels = int(image_feature_channels) + 3
        lifted_channels = 3 * per_view_channels + 3
        self.view_fusion = nn.Sequential(
            nn.Conv3d(lifted_channels, volume_feature_channels, 1, bias=False),
            nn.GroupNorm(
                _group_count(volume_feature_channels), volume_feature_channels
            ),
            nn.SiLU(inplace=True),
            ResidualBlock3d(volume_feature_channels, volume_feature_channels),
        )
        # One occupancy logit and conditional logits for occupied semantics.
        self.completion_network = SpatialUNet3d(
            volume_feature_channels,
            base_channels,
            1 + self.num_semantic_classes,
        )
        self.register_buffer(
            "local_centers",
            volume.centers().reshape(-1, 3),
            persistent=False,
        )
        self.register_buffer(
            "coordinate_channels",
            volume.normalized_coordinates(),
            persistent=False,
        )

    def _project_and_sample(
        self,
        image_features: torch.Tensor,
        depth_m: torch.Tensor | None,
        intrinsics: torch.Tensor,
        t_world_camera: torch.Tensor,
        t_world_anchor: torch.Tensor,
        *,
        image_height: int,
        image_width: int,
    ) -> torch.Tensor:
        """Lift features and depth cues into the fixed volume without target masks.

        Project centers, bilinearly sample with align_corners=False, mask invalid
        projections, append depth-valid/residual/surface cues per view, then
        concatenate the three ordered view blocks and local xyz channels.
        Return [B,3*(C+3)+3,Z,Y,X], i.e. 84 channels for C=24.
        """
        # BEGIN YOUR CODE: TriViewLiftFuseUNet._project_and_sample
        return image_features.sum() * 0 + image_features.new_zeros(
            (
                len(image_features),
                self.view_fusion[0].in_channels,
                *self.volume.shape_zyx,
            )
        )
        # END YOUR CODE

    def fuse_views(self, lifted):
        """Apply the learned ordered-view fusion module to [B,84,Z,Y,X]."""
        # BEGIN YOUR CODE: TriViewLiftFuseUNet.fuse_views
        return lifted[:, : self.view_fusion[0].out_channels] * 0
        # END YOUR CODE

    def forward(
        self,
        rgb: torch.Tensor,
        depth_m: torch.Tensor | None,
        intrinsics: torch.Tensor,
        t_world_camera: torch.Tensor,
        t_world_anchor: torch.Tensor,
    ) -> torch.Tensor:
        """Predict ten class scores from three images, optional depth and calibration."""
        if rgb.ndim != 5 or rgb.shape[1] != 3 or rgb.shape[2] != 3:
            raise ValueError(
                "rgb must have shape [batch, 3 views, 3 channels, height, width]"
            )
        batch_size, view_count, _, image_height, image_width = rgb.shape
        if intrinsics.shape != (batch_size, 3, 3):
            raise ValueError("intrinsics must have shape [batch, 3, 3]")
        if t_world_camera.shape != (batch_size, view_count, 4, 4):
            raise ValueError("t_world_camera must have shape [batch, 3, 4, 4]")
        if t_world_anchor.shape != (batch_size, 4, 4):
            raise ValueError("t_world_anchor must have shape [batch, 4, 4]")
        if depth_m is not None and depth_m.shape != (
            batch_size,
            view_count,
            1,
            image_height,
            image_width,
        ):
            raise ValueError("depth_m must have shape [batch, 3, 1, height, width]")

        encoder_input = rgb
        if self.use_depth:
            if depth_m is None:
                raise ValueError("depth_m is required when use_depth=True")
            normalized_depth = (depth_m / self.volume.z_range_m[1]).clamp(0.0, 1.0)
            encoder_input = torch.cat([rgb, normalized_depth], dim=2)
        image_features = self.image_encoder(
            encoder_input.reshape(
                batch_size * view_count,
                encoder_input.shape[2],
                image_height,
                image_width,
            )
        )
        image_features = image_features.reshape(
            batch_size,
            view_count,
            image_features.shape[1],
            image_features.shape[2],
            image_features.shape[3],
        )
        lifted = self._project_and_sample(
            image_features,
            depth_m,
            intrinsics,
            t_world_camera,
            t_world_anchor,
            image_height=image_height,
            image_width=image_width,
        )
        raw = self.completion_network(self.fuse_views(lifted))
        return factorized_logits(raw)


def build_triview_model(
    config: dict[str, Any], num_semantic_classes: int
) -> TriViewLiftFuseUNet:
    """Instantiate the provided baseline; nine occupied classes plus free space."""
    settings = config["triview_model"]
    name = str(settings.get("name", "lift_fuse_3d_unet"))
    if name != "lift_fuse_3d_unet":
        raise ValueError(f"Unknown three-view model: {name}")
    return TriViewLiftFuseUNet(
        num_semantic_classes,
        AnchorVolumeSpec.from_config(config),
        use_depth=bool(settings.get("use_depth", True)),
        image_feature_channels=int(settings.get("image_feature_channels", 24)),
        volume_feature_channels=int(settings.get("volume_feature_channels", 24)),
        base_channels=int(settings.get("base_channels", 24)),
        depth_truncation_m=float(settings.get("depth_truncation_m", 0.5)),
    )
