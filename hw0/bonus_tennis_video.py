"""HW0 optional bonus: foreground-preserving virtual logo insertion for tennis clips.

Run from the hw0 directory after completing Task 1:

    python bonus_tennis_video.py                 # all five clips, MP4 for match005
    python bonus_tennis_video.py --clips match005 --no-video

Pipeline, all implemented here (no provided helper is used for the bonus):
  1. video loading with OpenCV;
  2. clean-background estimation as the temporal median of frames sampled
     across the clip (the broadcast camera is fixed, so moving players and the
     ball cancel out and the bare court remains);
  3. soft foreground mask per frame from the colour difference to that
     background, with a smooth threshold ramp, morphological clean-up, and a
     Gaussian feather so player edges blend instead of cutting hard;
  4. logo insertion by warping the RGBA logo through the Task 1 court
     homography into a 6 m x 3 m rectangle on the near half of the court,
     with the logo dimmed where the live frame is darker than the background so
     shadows fall on it naturally;
  5. composition where the logo alpha is multiplied by (1 - foreground), so
     the player pixels of the current frame remain in front of the logo;
  6. output writing: one representative PNG per clip (the frame with the most
     player/logo overlap) and one full MP4 for match005.

Geometry and alpha blending reuse the completed Task 1 functions.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from task1_homography import (
    TENNIS_XY,
    alpha_blend,
    estimate_homography,
    logo_to_image_homography,
)

ROOT = Path(__file__).resolve().parent
CLIPS = ("match001", "match005", "match028", "match091", "match125")
LOGO_LOWER_LEFT_XY = (-3.0, 0.5)  # metres, same rectangle as Task 1C
LOGO_SIZE_XY = (6.0, 3.0)


# ----------------------------------------------------------------------------
# Video loading
# ----------------------------------------------------------------------------

def open_video(path: Path) -> tuple[cv2.VideoCapture, float, int, int, int]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(f"cannot open video {path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return capture, fps, count, width, height


def read_frame(capture: cv2.VideoCapture, index: int) -> np.ndarray | None:
    capture.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = capture.read()
    return frame if ok else None


def iterate_frames(path: Path, start: int, stop: int):
    """Yield (index, BGR uint8 frame) for indices in [start, stop)."""
    capture, *_ = open_video(path)
    capture.set(cv2.CAP_PROP_POS_FRAMES, start)
    index = start
    while index < stop:
        ok, frame = capture.read()
        if not ok:
            break
        yield index, frame
        index += 1
    capture.release()


# ----------------------------------------------------------------------------
# Clean background: temporal median
# ----------------------------------------------------------------------------

def estimate_background(path: Path, start: int, stop: int, samples: int) -> np.ndarray:
    """Median over `samples` frames spread evenly across [start, stop)."""
    capture, *_ = open_video(path)
    indices = np.unique(np.linspace(start, stop - 1, samples).round().astype(int))
    stack = []
    for index in indices:
        frame = read_frame(capture, int(index))
        if frame is not None:
            stack.append(frame)
    capture.release()
    if not stack:
        raise RuntimeError(f"no frames could be read from {path}")
    return np.median(np.stack(stack), axis=0).astype(np.float32) / 255.0


# ----------------------------------------------------------------------------
# Soft foreground mask
# ----------------------------------------------------------------------------

def smoothstep(x: np.ndarray, low: float, high: float) -> np.ndarray:
    t = np.clip((x - low) / (high - low), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def foreground_mask(
    frame: np.ndarray,
    background: np.ndarray,
    low: float = 0.10,
    high: float = 0.28,
) -> np.ndarray:
    """Soft mask in [0, 1]; 1 = confidently foreground (player, racket, ball).

    frame and background are float32 BGR in [0, 1] with identical shape.
    """
    difference = np.abs(frame - background).max(axis=2)
    soft = smoothstep(difference, low, high)
    mask8 = (soft * 255.0).astype(np.uint8)
    small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask8 = cv2.morphologyEx(mask8, cv2.MORPH_OPEN, small)   # drop speckle noise
    mask8 = cv2.morphologyEx(mask8, cv2.MORPH_CLOSE, large)  # fill holes inside players
    mask8 = cv2.dilate(mask8, small)                          # protect thin edges
    mask8 = cv2.GaussianBlur(mask8, (9, 9), 0)                # feather
    return mask8.astype(np.float32) / 255.0


# ----------------------------------------------------------------------------
# Logo insertion
# ----------------------------------------------------------------------------

def load_logo_bgra(path: Path) -> np.ndarray:
    logo = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if logo is None:
        raise FileNotFoundError(path)
    if logo.ndim == 2:
        logo = cv2.cvtColor(logo, cv2.COLOR_GRAY2BGR)
    if logo.shape[2] == 3:
        logo = np.dstack([logo, np.full(logo.shape[:2], 255, np.uint8)])
    return logo.astype(np.float32) / 255.0


def warp_logo(logo_bgra: np.ndarray, court_to_image: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    """Warp the RGBA logo into the metric court rectangle, then into the image."""
    logo_to_image = logo_to_image_homography(
        court_to_image, logo_bgra.shape, LOGO_LOWER_LEFT_XY, LOGO_SIZE_XY
    )
    return cv2.warpPerspective(
        logo_bgra, logo_to_image, size_wh,
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )


def footprint_box(alpha: np.ndarray, margin: int = 24) -> tuple[slice, slice]:
    ys, xs = np.nonzero(alpha > 1e-3)
    y0, y1 = max(ys.min() - margin, 0), min(ys.max() + margin + 1, alpha.shape[0])
    x0, x1 = max(xs.min() - margin, 0), min(xs.max() + margin + 1, alpha.shape[1])
    return slice(y0, y1), slice(x0, x1)


# ----------------------------------------------------------------------------
# Composition
# ----------------------------------------------------------------------------

def composite(
    frame: np.ndarray,
    background: np.ndarray,
    logo_warped: np.ndarray,
    box: tuple[slice, slice],
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (composited uint8 BGR frame, foreground mask crop, overlap score)."""
    crop = frame[box].astype(np.float32) / 255.0
    background_crop = background[box]
    logo_crop = logo_warped[box]

    fg = foreground_mask(crop, background_crop)

    # Dim the logo where the live frame is darker than the clean background so
    # player shadows and the ball's shadow fall onto the painted logo as well.
    luminance = lambda img: 0.114 * img[..., 0] + 0.587 * img[..., 1] + 0.299 * img[..., 2]
    shading = np.clip(luminance(crop) / np.maximum(luminance(background_crop), 1e-3), 0.35, 1.15)
    shaded_logo_rgb = np.clip(logo_crop[..., :3] * shading[..., None], 0.0, 1.0)

    effective_alpha = logo_crop[..., 3] * (1.0 - fg)
    foreground_rgba = np.dstack([shaded_logo_rgb, effective_alpha])
    blended = alpha_blend(foreground_rgba, crop)

    out = frame.copy()
    out[box] = np.clip(blended * 255.0 + 0.5, 0, 255).astype(np.uint8)
    overlap = float((fg * logo_crop[..., 3]).sum())
    return out, fg, overlap


# ----------------------------------------------------------------------------
# Per-clip driver
# ----------------------------------------------------------------------------

def process_clip(name: str, logo_bgra: np.ndarray, out_dir: Path, samples: int, write_video: bool) -> dict:
    started = time.time()
    video_path = ROOT / "data/tennis" / f"{name}.mp4"
    meta = np.load(ROOT / "data/tennis" / f"{name}.npz", allow_pickle=False)
    anchors_uv = meta["court_anchor_image_xy"].astype(float)
    start, stop = (int(v) for v in meta["valid_frame_range"])

    _, fps, count, width, height = open_video(video_path)
    stop = min(stop, count) if count > 0 else stop

    # Geometry from Task 1: metric anchors -> this clip's image anchors.
    court_to_image = estimate_homography(TENNIS_XY, anchors_uv)
    projected = (court_to_image @ np.column_stack([TENNIS_XY, np.ones(len(TENNIS_XY))]).T).T
    projected = projected[:, :2] / projected[:, 2:]
    anchor_rmse = float(np.sqrt(((projected - anchors_uv) ** 2).sum(axis=1).mean()))

    logo_warped = warp_logo(logo_bgra, court_to_image, (width, height))
    box = footprint_box(logo_warped[..., 3])

    background = estimate_background(video_path, start, stop, samples)
    cv2.imwrite(str(out_dir / f"{name}_background.png"), (background * 255.0 + 0.5).astype(np.uint8))

    writer = None
    if write_video:
        video_out = out_dir / f"{name}_occlusion_aware.mp4"
        for fourcc in ("avc1", "mp4v"):
            writer = cv2.VideoWriter(str(video_out), cv2.VideoWriter_fourcc(*fourcc), fps, (width, height))
            if writer.isOpened():
                break
            writer.release()
            writer = None
        if writer is None:
            print(f"  warning: no MP4 encoder available, skipping {video_out.name}")

    best = {"overlap": -1.0, "index": start, "frame": None, "mask": None}
    processed = 0
    for index, frame in iterate_frames(video_path, start, stop):
        out, fg, overlap = composite(frame, background, logo_warped, box)
        if writer is not None:
            writer.write(out)
        if overlap > best["overlap"]:
            best.update(overlap=overlap, index=index, frame=out, mask=fg)
        processed += 1
    if writer is not None:
        writer.release()

    cv2.imwrite(str(out_dir / f"{name}_representative.png"), best["frame"])
    mask_full = np.zeros((height, width), np.uint8)
    mask_full[box] = (best["mask"] * 255.0 + 0.5).astype(np.uint8)
    cv2.imwrite(str(out_dir / f"{name}_representative_mask.png"), mask_full)

    summary = {
        "clip": name,
        "frames_processed": processed,
        "fps": round(float(fps), 3),
        "anchor_rmse_px": round(anchor_rmse, 3),
        "representative_frame": int(best["index"]),
        "player_logo_overlap_px": round(best["overlap"], 1),
        "seconds": round(time.time() - started, 1),
    }
    print(f"  {name}: {processed} frames, anchor RMSE {anchor_rmse:.2f} px, "
          f"representative frame {best['index']} (overlap {best['overlap']:.0f} px), "
          f"{summary['seconds']} s")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clips", nargs="+", default=list(CLIPS), choices=CLIPS)
    parser.add_argument("--video-for", default="match005", help="clip that also gets a full MP4")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--samples", type=int, default=45, help="frames sampled for the median background")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/bonus")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    logo_bgra = load_logo_bgra(ROOT / "data/court/logo.png")
    print(f"Bonus: foreground-preserving insertion -> {args.out}")
    summaries = [
        process_clip(name, logo_bgra, args.out, args.samples,
                     write_video=(name == args.video_for and not args.no_video))
        for name in args.clips
    ]
    # Merge with any earlier run so partial invocations (--clips ...) keep the other clips' rows.
    summary_path = args.out / "bonus_summary.json"
    merged = {row["clip"]: row for row in (json.loads(summary_path.read_text()) if summary_path.exists() else [])}
    merged.update({row["clip"]: row for row in summaries})
    summary_path.write_text(json.dumps([merged[name] for name in CLIPS if name in merged], indent=2))
    print("Bonus complete.")


if __name__ == "__main__":
    main()
