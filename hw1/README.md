# CS 498 HW1: Registration, Odometry, and Semantic Mapping

- Release: Friday, September 18, 2026
- Due: Wednesday, October 7, 2026 at 11:59 p.m. America/Chicago
- Work mode: individual
- Required credit: 15 points; optional loop closure: +1 point

Download both files from the [HW1 data folder](https://drive.google.com/drive/folders/13azu-aNzMn3jrZ_soxHsX5SNlRcGdr64?usp=sharing).
From this directory, use Python 3.12 in a virtual environment:

```bash
python -m pip install -r requirements.txt
python -m pip install gdown
gdown --continue 18ud_EZhidMx2CU5AXgd4fMvRCUK8mY9c -O data/registration_demo.npz
gdown --continue 1NFJb771W_R6aP5dBxFk7iiW0rnHNhJ3p -O data/sequence00.npz
python preflight.py
```

Direct file downloads avoid the extra `hw1_data/` directory created by a folder
download. If Drive stalls, retry the same command or download through a browser.
Preflight verifies both sizes and SHA-256 hashes; a downloader reporting success
alone is insufficient. Then run each pipeline:

```bash
python task1_alignment.py
python task2_ransac.py
python task3_odometry.py
python task4_mapping.py
```

Run `python preflight.py --render` before the first full run to test Open3D and
MP4 encoding in a separate process. On Linux, working EGL/Mesa or GPU drivers
are required; see [Open3D software rendering](https://www.open3d.org/docs/release/tutorial/visualization/cpu_rendering.html).
If rendering fails, all four scripts accept `--skip-rendering` so numerical work
and 2D diagnostics can continue without Open3D. This skips Open3D images/videos,
the online odometry video, and the Task 4 PLY; contact staff with the preflight
output to arrange rendering on a supported machine before final submission.
These files remain required for the final report. Do not edit the provided helpers.

Complete Task 1 before Tasks 2 and 3. Task 4 is independent and uses GT poses.
Full odometry processes 1,023 pairs; progress is printed every 50 pairs.
Budget several minutes per full run and several GB of RAM. The previous staff
run measured about 4 GB peak RAM for mapping; rendering also needs graphics
memory. Save numerical checkpoints before attempting expensive rendering.

The completed publication trial used Linux, Python 3.12.4, NumPy 2.1.3,
SciPy 1.14.1, PyTorch 2.13.0, Matplotlib 3.9.2, Open3D 0.19.0,
imageio 2.35.1, and imageio-ffmpeg 0.5.1. Other platforms have not been
validated by this trial; run the preflight before beginning.

Run `python -m unittest discover -s tests` for small numerical examples. Failures
are expected for untouched student functions; these tests do not replace the
full-data run or prove that an implementation is correct.

Geometry code uses homogeneous point columns with shape `[4, N]`, so transforms
act directly as `points_B = T_A_to_B @ points_A`.

The frozen data consists of one real Open3D registration pair and a 1,024-frame
SemanticKITTI sequence-00 clip covering original frames 2336--3359. Task 1 uses
1,000 real paired points. Task 3 uses the supplied stronger ICP initialization
on the Open3D pair before running adjacent-frame odometry across the complete
clip. Task 4 uses every sequence frame while retaining at most 2,500
range-filtered points from each frame for memory control.

Each task file has one clearly marked edit block. Modify only that block. Every
pipeline has a visible `main()` and saves named checkpoints in `outputs/task1/`
through `outputs/task4/`. Untouched placeholders run to completion and label
their metrics `starter_placeholder`; they are not correct solutions.

Implement exactly these eight functions:

- `fit_rigid_svd` and `nearest_correspondences`;
- `ransac_rigid`;
- `icp`, `chain_relative_poses`, and `relative_trajectory_error`; and
- `points_to_voxels` and `fuse_voxel_evidence`.

Task 3 additionally saves `icp_iterations.mp4`, `relative_transforms.npy`,
`icp_poses.npy`, and `icp_odometry_online.mp4`. The online video keeps the
LiDAR centered in a heading-up $\pm$50 m BEV: its left panel uses every point
in the current stored sweep, while its right panel accumulates intensity only
through the current frame and overlays the inferred trajectory up to that
frame in cyan. Task 4 renders equal-metric Open3D voxel cubes as a turntable
and creates a side-by-side LiDAR/map follow video. Both follow-video panels use
the same moving camera, located 3 m above the LiDAR. The map uses 0.25 m voxels,
and the left panel renders every stored sweep point with inferno range colors.
Matplotlib is reserved for
2D BEV, trajectory, residual, and convergence diagnostics; it is not used to
fake 3D views.

Do not modify `hw1_utils.py`; it only contains provided rendering and output
machinery, and you do not need to read its internals. Submit the four revised
task scripts and one PDF report built from `report_template.tex`. If you attempt the +1 bonus, submit
your own directly runnable `bonus_loop_closure.py`. The supplied
`pose_graph.py` provides the optimizer and documents edge direction, units, and
a small runnable example (`python pose_graph.py`). Use the same
`sequence00.npz` and Task 3 poses; no bonus data archive, starter, retrieval
method, or registration helper is provided. A candidate pair must be separated
by at least 400 original frame IDs before you verify it geometrically. See the
PDF handout for equations, checkpoints, required analysis, submission details,
and the generative-AI policy.
