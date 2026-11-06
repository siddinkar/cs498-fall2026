# HW4: Tiny NeRF

Release and due dates will be announced on the course site.

In this homework you will complete six focused implementation blocks in a
compact Neural Radiance Field (NeRF) pipeline, train it on the Lego scene, and
summarize the supplied visual checks in a short report. Camera-pose creation,
network construction, marching cubes, evaluation, texturing, and sphere
intersection are already provided so that the work stays focused and
manageable.

## 1. Setup

Run all commands from this directory:

```bash
conda create -n cs498-hw4 python=3.11
conda activate cs498-hw4
pip install -r requirements.txt
python data/download_data.py
```

The downloader fetches a 2.7 MB public archive, verifies every file with
SHA-256, extracts 50 training, 5 validation, and 12 test views, and removes
the archive. The resulting scene is `data/nerf_synthetic/lego`.

Then run the quick preflight:

```bash
python -m unittest discover -s tests
python task3_train_mesh.py --mode visualize
```

The untouched starter is expected to fail tests for the six unfinished
blocks. The tests use synthetic inputs and do not train a NeRF.

## 2. What you implement

Edit only between `INSERT YOUR CODE BELOW` and
`DO NOT MODIFY CODE OUTSIDE THIS BLOCK`.

| File | Required block | Points |
|---|---|---:|
| `task1_rays_encoding.py` | `get_rays` | 1.0 |
| `task1_rays_encoding.py` | `positional_encoding` | 1.0 |
| `task2_nerf_rendering.py` | `TinyNeRF.forward` | 2.0 |
| `task2_nerf_rendering.py` | `sample_along_rays` | 1.0 |
| `task2_nerf_rendering.py` | `raw_to_outputs` | 1.5 |
| `task3_train_mesh.py` | `build_density_grid` | 2.0 |
| Generated training and held-out evidence | provided pipeline | 3.0 |
| Report completeness and reproducibility | `report.pdf` | 1.5 |
| **Required total** | | **13.0** |
| Optional ball insertion | `composite_opaque_sphere` | **+1.0** |
| **Maximum with bonus** | | **14.0** |

The detailed equations and expected tensor shapes are in `hw4_handout.pdf`.
Each script also documents its input and output contract beside the marked
block.

## 3. Fast validation workflow

Complete the homework in this order:

```bash
# Dataset-free checks and figures
python task1_rays_encoding.py
python task2_nerf_rendering.py
python task3_train_mesh.py --mode check
python -m unittest discover -s tests

# Data-backed pipeline
python task3_train_mesh.py --mode smoke
python task3_train_mesh.py --mode all
```

The full defaults use 10,000 training iterations, five validation views, a
12-frame held-out turntable, a 96-cubed density grid, and a 256-pixel texture
atlas. The separate 100-iteration smoke mode is intended for quick debugging;
it is not a substitute for the full run. On a recent desktop GPU, budget
several minutes for the complete workflow; runtime varies by hardware. CUDA is
used when available and otherwise falls back to CPU.

The scripts create additional local intermediates while they run, including
smoke outputs and model checkpoints. Keep them locally when needed for
evaluation or the bonus, but submit only the following canonical grading
evidence with these names and directories:

```text
outputs/
├── task1/
│   ├── ray_directions.png
│   └── task1_visual_checkpoint.png
├── task2/
│   └── task2_visual_checkpoint.png
└── task3/
    └── lego_train/
        ├── train_metrics.csv
        ├── train_psnr_curve.png
        ├── val_010000_view_00.png
        ├── val_010000_view_01.png
        ├── validation_metrics.csv
        ├── validation/view_00_comparison.png ... view_04_comparison.png
        ├── turntable.mp4
        ├── turntable_contact_sheet.png
        └── mesh/
            ├── comparison.csv
            ├── view_00_comparison.png
            ├── view_01_comparison.png
            ├── lego_textured.obj
            ├── lego_textured.mtl
            ├── lego_texture.png
            └── lego_vertex_color_mesh.ply
```

If you attempt the bonus, the canonical directory additionally contains
`ball_insertion.mp4`, `ball_insertion_contact_sheet.png`, and
`ball_visibility.png` directly under `outputs/task3/lego_train/`.

Validation and mesh metrics are PSNR and SSIM; both are higher-is-better.
They are diagnostics, not fixed score cutoffs. Grading uses the visual evidence
and whether the outputs are coherent with one another.

## 4. Optional bonus [+1.0]

Sphere intersection is supplied. Implement only
`composite_opaque_sphere`, then run:

```bash
python bonus_ball_insertion.py --mode check
python -m unittest tests.test_bonus_ball_insertion
python bonus_ball_insertion.py --mode render \
  --checkpoint outputs/task3/lego_train/model.pt
```

The result should use NeRF transmittance before the sphere hit, so the sphere
can be partially or fully hidden by foreground density. Include the contact
sheet and visibility image in the optional report section.

## 5. Report and submission

Use `report_template.tex` as the outline for a concise technical report
(recommended length: 4--6 pages). Include the requested results and analysis;
you may choose the layout and page breaks.

Submit one `netid_hw4.zip` containing:

- `task1_rays_encoding.py`
- `task2_nerf_rendering.py`
- `task3_train_mesh.py`
- `report.pdf`
- the lightweight canonical grading evidence shown above
- `bonus_ball_insertion.py` only if you attempt the bonus

Do not submit model checkpoints (`*.pt`), smoke/debug outputs, `data/`,
environments, caches, or other generated intermediates. Do not rename,
flatten, or reorganize the listed `outputs/` paths: submission checks use them
directly. There is no student-side grading bundle or grading tool.
