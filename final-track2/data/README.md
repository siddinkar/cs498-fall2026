# Provided data

There are 30 Dev, 10 Val, and 20 Test scenes, each containing 192 posed RGB
frames (512 x 512, 24 fps). You do not generate any scenes. Input inference may
read only RGB frames and the supplied calibration/poses. There is no GT depth
input. Category names are not required.

```
data/scenes/<dev|val|test>/scene_XXX/
    rgb/frame_0000.png ... frame_0191.png
    cameras.npz
    video.mp4                 # optional preview of the same RGB inputs
    scene.blend               # Dev/Val evaluation data ONLY; never in Test
    physics_eval.json         # legacy Dev/Val metadata; unused by current metric
```

`K` is shared or per-frame intrinsics. Camera CV axes are x-right, y-down,
z-forward. `camera_to_world_cv` maps CV camera coordinates into the dataset
world (right-handed, Z-up, meters). `camera_to_world_blender` uses Blender
camera axes instead. Use `frame_indices` for temporal alignment; do not guess
matrix conventions. `track2_io.load_scene` is the common input loader.

Run `python data/download_data.py --split dev` from `handout/`; repeat with
`--split val` and `--split test` when needed. The included manifest pins the
published archive URLs and SHA-256 hashes. Alternatively use a supplied
archive directory: `python data/download_data.py --from-local-dir /path/to/archives`.
Both routes verify every archive SHA-256 and reject path traversal, links,
duplicate entries, excessive uncompressed sizes, and Test GT artifacts.

Dev is for algorithm development and training. Val is for local evaluation and
controlled model selection. Do not train on Val GT or build per-scene lookup
tables. Test inputs are for the final frozen automatic method, not human
editing. Test GT and its scene-generation assets are never distributed.
