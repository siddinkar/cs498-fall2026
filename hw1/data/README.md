# HW1 data

Download both required course files from
<https://drive.google.com/drive/folders/13azu-aNzMn3jrZ_soxHsX5SNlRcGdr64?usp=sharing>
and place them directly in this directory. The parent README provides resumable
`gdown` commands. Run `python preflight.py` from the parent directory containing
the task scripts (`hw1/` in the student repository) to verify the downloads.

- `registration_demo.npz`: the real two-fragment Open3D
  `DemoICPPointClouds` scene. It includes 1,000 real Task 1 point pairs, the
  reference transform used for evaluation, and the deliberately perturbed transform
  used to debug Task 3 ICP;
- `sequence00.npz`: a compact 1,024-frame SemanticKITTI sequence-00 clip used
  for adjacent-frame odometry, semantic voxel mapping, and the optional loop
  closure.

## Release audit

The frozen demo's rounded rotation blocks have a small scale/shear error.
The provided loader projects both demo initialization and evaluation rotations
to the nearest proper rotation before use. This correction does not alter the
archive bytes or hashes; students should not implement or repeat it.

| File | Contents | Bytes | SHA-256 |
| --- | --- | ---: | --- |
| `registration_demo.npz` | Open3D pair; 1,000 real Task 1 pairs; perturbed Task 3 initialization | 147,328 | `d62a9583afab47f0538fa08893fa2b9a07d1e5e38d96039f221a6d58f56111aa` |
| `sequence00.npz` | frames 2336--3359; 1,024 frames; 8,192,000 points | 60,661,905 | `418ad17aefba6df3ebedf875df9b44caa62409eea9df6aae76ad4c0038d6e58b` |

The vehicle travels 869.749 m. Its reference trajectory spans 189.235 m in
world x and 262.317 m in world y. Each original LiDAR scan is spatially sampled
with 0.25 m voxels and capped at 8,000 points. Spatial sampling preserves scene
coverage better than taking the first or uniformly spaced raw points.

## Sequence archive schema

All arrays load with `np.load(..., allow_pickle=False)`.

| Key | Dtype and shape | Meaning |
| --- | --- | --- |
| `format_version` | `uint16 []` | Schema version 2. |
| `sequence` | string scalar | Always `"00"`. |
| `frame_ids` | `int32 [1024]` | Original IDs 2336--3359. |
| `frame_offsets` | `int64 [1025]` | Packed point boundaries. |
| `points` | `float16 [8192000,4]` | Local-frame `(x,y,z,intensity)`. |
| `semantic_gt` | `uint8 [8192000]` | Class IDs 0--18; 255 means ignore. |
| `semantic_logits_gt` | `int8 [8192000,19]` | GT-derived per-point semantic logits. |
| `semantic_class_names` | string `[19]` | Class order for labels and logits. |
| `timestamps` | `float64 [1024]` | Seconds from sequence start. |
| `poses_lidar_to_world` | `float64 [1024,4,4]` | KITTI reference poses. |
| `T_cam0_velo` | `float64 [4,4]` | Velodyne-to-camera-0 calibration. |
| `sample_voxel_size_m` | `float32 []` | Per-frame sampling voxel size, 0.25 m. |
| `max_points_per_frame` | `int32 []` | Per-frame cap, 8,000. |

Frame `i` occupies rows `frame_offsets[i]:frame_offsets[i + 1]`. Convert a
selected XYZ slice to `float32` before geometry. Labels 0--18 follow
`semantic_class_names`; 255 means ignore. `semantic_logits_gt` contains +4 for
the GT class and -4 for other classes; ignored points contain all zeros. These
scores test coordinate conversion and repeated-observation fusion, not semantic
model quality.

The course convention is `T_A_to_B`: it maps coordinates from frame A into
frame B. Thus `poses_lidar_to_world[i]` is `T_Li_to_W`, while each saved Task 3
relative transform is `T_L(i+1)_to_Li`. With homogeneous column vectors,
`p_B_h = T_A_to_B @ p_A_h`. Code stacks point columns as `[4, N]`, so
the full point set uses `points_B = T_A_to_B @ points_A`. Raw `[N, 3]` arrays
are converted only when loaded or passed to SciPy/Open3D. Velodyne coordinates
use x forward, y left, and z up. Task 3 normalizes the first supplied LiDAR pose to
identity and runs ICP on every adjacent pair. It saves relative transforms and
chained poses for later inspection and for the bonus.

Task 4 uses all 1,024 frames. For each frame it keeps labeled points within 40
m in the LiDAR xy plane and with `-2.8 < z < 3.5` m, then retains at most 2,500
evenly spaced scan-order samples. This cap bounds memory without dropping a frame from
the 0.25 m map. The final 3D products are equal-metric Open3D voxel cubes:
a turntable and a side-by-side LiDAR/map follow video. The two follow views use
the same camera 3 m above the moving LiDAR. The left panel uses every point in
the stored 8,000-point sweep, without the map's range/height filter or 2,500-point
cap, and colors points by range with the inferno colormap.

There is no optional loop-closure archive. The bonus uses `sequence00.npz` and
the poses produced by Task 3. Candidate pairs must differ by at least 400 in
their original `frame_ids`; ground-truth poses may not be used to retrieve a
candidate.

The linked Drive folder distributes the frozen archives. LiDAR, poses, and
calibration come from KITTI odometry sequence 00; point labels come from
SemanticKITTI. Cite both datasets when reusing the supplied data:

- <https://semantic-kitti.org/dataset.html>
- <https://www.cvlibs.net/datasets/kitti/>
- <https://www.open3d.org/docs/release/python_api/open3d.data.DemoICPPointClouds.html>
