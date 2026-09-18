"""Verify release data; optionally smoke-test graphics in an isolated process."""
from pathlib import Path
import argparse
import hashlib
import subprocess
import sys
import tempfile

ARCHIVES = {
    "registration_demo.npz": (147328, "d62a9583afab47f0538fa08893fa2b9a07d1e5e38d96039f221a6d58f56111aa"),
    "sequence00.npz": (60661905, "418ad17aefba6df3ebedf875df9b44caa62409eea9df6aae76ad4c0038d6e58b"),
}


def check_data(root):
    for name, (size, expected) in ARCHIVES.items():
        path = Path(root) / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}; see README.md download commands")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        if path.stat().st_size != size or digest.hexdigest() != expected:
            raise ValueError(f"{path}: size or SHA-256 mismatch; download again")
        print(f"{name}: size and SHA-256 verified")


def render_probe():
    import hw1_utils as utils
    import imageio.v2 as imageio
    import numpy as np
    import open3d as o3d
    renderer = o3d.visualization.rendering.OffscreenRenderer(64, 64)
    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultUnlit"
    renderer.scene.add_geometry("box", o3d.geometry.TriangleMesh.create_box(), material)
    renderer.setup_camera(60, np.array([.5, .5, .5], np.float32),
                          np.array([2, 2, 2], np.float32),
                          np.array([0, 0, 1], np.float32))
    frame = np.asarray(renderer.render_to_image())
    assert frame.shape == (64, 64, 3)
    with tempfile.TemporaryDirectory(prefix="hw1-render-") as tmp:
        path = Path(tmp) / "probe.mp4"
        with imageio.get_writer(path, fps=24, codec="libx264") as writer:
            writer.append_data(frame)
        decoded = imageio.get_reader(path)
        assert decoded.get_data(0).shape == frame.shape
        decoded.close()
    print("Open3D rendering and MP4 encoding/decoding passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--render-child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.render_child:
        render_probe()
    else:
        check_data(Path(__file__).resolve().parent / "data")
        if args.render:
            try:
                result = subprocess.run([sys.executable, __file__, "--render-child"],
                                        timeout=90, check=False)
                if result.returncode:
                    raise RuntimeError(f"renderer process exited {result.returncode}")
            except (RuntimeError, subprocess.TimeoutExpired) as error:
                parser.exit(1, f"Rendering preflight failed: {error}. Use --skip-rendering "
                            "for numerical work and contact staff with this output.\n")
