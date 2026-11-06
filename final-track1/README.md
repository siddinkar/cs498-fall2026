# Final Project - Track 1: Multi-View 3D Semantic Occupancy Prediction

Individual, non-AI-assisted project. Base credit: 20 points.
Release: Wednesday, November 4, 2026. Due December 9,
2026 at 11:59 p.m. America/Chicago. Read the handout PDF for the complete rubric.
The verified reference score is **0.65135**, at Kaggle's five-decimal display
precision. This fixed benchmark does not need a leaderboard participant entry.

## Kaggle competition

- Competition: https://www.kaggle.com/competitions/cs498-fall-2026-multiview-occupancy
- **Join first** with the course invitation:
  https://www.kaggle.com/t/f04e3dce46f047c5b52deec9fbe73701
  Sign in to a Kaggle account whose verified email ends in `@illinois.edu`.
  The competition is private, so until you join through the invitation the
  competition URL shows a 404 page; that is expected.
- Submission steps: [KAGGLE_SUBMISSION.md](KAGGLE_SUBMISSION.md). Teams are limited
  to one member (individual project).

Predict a semantic occupancy grid from three synchronized views at one position:
front-left, front, front-right. RGB is required, depth is optional. There is no
video or temporal accumulation. The canonical output is an integer `(40,14,80)`
grid in **(z,y,x)** order, with labels 0 (free) through 9. Only eligible hidden
completion cells are ranked, macro-averaged by house.

## Setup and first smoke test

Start with [SETUP.md](SETUP.md) for the complete Ubuntu installation and
Colab workflow. Ubuntu and Colab are the supported paths; Mac/Windows users
should use Colab. The fallback notebook is `notebooks/track1_colab.ipynb`.

Use Python 3.11-3.12 (the Kaggle SDK requires 3.11+). A CUDA GPU is recommended;
LaTeX is needed for the report.
Run all commands from this directory.
Use `python run.py --help` to list commands, or `python run.py train --help`
for one command's options.

```text
run.py                        # one entry point for the provided tools
student_submission/           # your code, method.json, dependencies and weights
alternative_method_template/  # optional replacement-method interface
cs498_track1/                 # provided data, training, scoring and packaging tools
data/                         # download instructions and downloaded data
tests/                        # public function and infrastructure checks
report_template.tex           # write your report here
outputs/                      # created by the workflow; not starter source code
```

```bash
python -m pip install -r requirements.txt
python run.py doctor
python -m pytest -q tests
python run.py download --split train --houses train_house_000
python run.py download --split val --houses val_house_000
python run.py train --smoke
```

The starter is runnable, but its eight implementation blocks deliberately return
noncompetitive placeholders. The public suite initially reports eight `XFAIL`
checks, with infrastructure checks passing. Editing a block activates its test.
A smoke run checks plumbing, not correctness or useful performance.
Smoke checkpoints and logs are kept under `outputs/train/smoke/`; they never
replace your final `student_submission/weights/checkpoint_best.pt`.

## Eight baseline functions

All are inside `student_submission/semocc/`:

| File | Function |
|---|---|
| `model.py` | `project_voxel_centers` |
| `model.py` | `TriViewLiftFuseUNet._project_and_sample` |
| `model.py` | `TriViewLiftFuseUNet.fuse_views` |
| `model.py` | `SpatialUNet3d.forward` |
| `model.py` | `factorized_logits` |
| `learning.py` | `completion_loss` |
| `learning.py` | `class_weights_from_histogram` |
| `learning.py` | `metrics_from_confusion` |

The handout explains each function's math, dimensions and common pitfalls.
Provided modules own data loading, optimization, evaluation, prediction transport
and packaging. `cs498_track1/protocol.py` is the immutable benchmark contract,
not a TODO. Leave `run.py` and `cs498_track1/` unchanged.

## Complete workflow

```bash
python run.py download --split train
python run.py download --split val
python run.py train
```

After choosing the final weights, fill `student_submission/method.json` before
generating the final evidence:

```bash
python run.py evaluate
python run.py figures
```

Train: 200 houses x 150 samples. Val: 10 houses x 15 samples. Each sample has
three images per modality. Prepared Train/Val targets need no data generation.
The provided trainer is capped at 15 epochs and selects weights by Val house-macro completion
mIoU. `--batch-size 8` and `--workers 0` can reduce resource requirements;
`--rgb-only` runs without depth. `run.py evaluate` is the canonical evaluator,
not a proxy metric. Full evidence needs all 150 Val samples.

Training supports epoch-boundary recovery with `--resume outputs/train/checkpoint_last.pt`.
Keep the original code, data, Torch version, device, batch size and total epoch
target unchanged. An interrupted epoch is repeated. See SETUP.md before resuming.

Then produce and submit the Test predictions:

```bash
python run.py download --split test
python run.py predict --max-samples 3 --output outputs/kaggle/smoke.csv
python run.py predict
python run.py submit outputs/kaggle/submission.csv
python run.py submit outputs/kaggle/submission.csv --submit
python run.py report
python run.py package --student-id YOUR_NETID --kaggle-username YOUR_KAGGLE_NAME
```

The partial smoke CSV is not accepted by Kaggle. The full CSV contains 150 rows.
Without `--submit`, the upload tool only validates locally. CSV generation is
atomic: a failed run leaves the previous complete CSV intact. See
`KAGGLE_SUBMISSION.md` for registration, encoding, rules and troubleshooting.

Upload the generated `outputs/submission_package/cs498-track1-YOUR_NETID.zip`
**and** `outputs/submission_package/package_manifest.json` to the LMS. Upload
only `outputs/kaggle/submission.csv` to Kaggle. The ZIP contains your final
`student_submission/` code/model tree, `report.pdf`, canonical Val summary,
four report figures and their evidence manifest, selected CSV and runtime receipt.
No datasets or credentials.
Finalize code, weights and `method.json` before full Val/Test inference. The
packager rejects stale source hashes, CSV hashes, data receipts and figure hashes.
If you rerun evaluation, regenerate the figures and rebuild the report before
packaging. These checks prevent accidental stale files; they do not replace
your responsibility to report truthfully or the final reproduction audit.

## Your own algorithm

The runnable `alternative_method_template/` illustrates the interface without
prescribing a method. Copy its adapter and declaration into `student_submission/`
and add your algorithm. You do not have to implement unused baseline functions
when replacing the method. Keep the provided tools unchanged. The required API is:

```python
load_model(model_dir, device)
predict_sample(model, inputs, device)  # integer [40,14,80], labels 0..9
```

`load_model` receives the `student_submission/weights/` directory (not the
method root) and the selected device string. It runs once. The CPU interface
probe must pass even if you use CUDA for training and full inference.

The input dictionary contains NumPy float32 arrays: `rgb` `[3,3,H,W]` in `[0,1]`,
`depth_m` `[3,1,H,W]` in metres (zero means invalid) or `None`,
`intrinsics` `[3,3]`, `t_world_camera` `[3,4,4]`, and `t_world_anchor` `[4,4]`.
It contains no house ID or target. Every prediction is independent: never pool
other samples or use sample-specific lookup rules. Different internal grids
are allowed, but the returned grid is fixed. Keep the final code/model tree
within 500 MiB and declare dependencies and permitted external sources.
Place helper modules under `student_submission/`, preferably in your own
uniquely named package such as `my_method/`. The adapter can import
`from my_method.network import Model`; package-relative imports such as
`from .my_method.network import Model` also work. Do not name a package
`cs498_track1`, which is reserved for the provided infrastructure.

```bash
python run.py validate
python run.py evaluate --max-samples 3
python run.py evaluate
python run.py figures
```

The provided figure command works with any canonical-grid method. If you draw
your own figures, use the same four required filenames, then run
`python run.py figures --record-existing` to record their hashes against the
current method and Val summary. Only do this for figures you have verified
represent that evaluation. Rebuild `report.pdf` after changing images.

Baseline-specific architecture tests are not requirements for a deliberately
replaced architecture. Run `python -m pytest -q tests/test_protocol.py
tests/test_workflow.py` for the common infrastructure, plus your own method tests.

## Data and coordinates

Camera coordinates: x-right, y-down, z-forward. Poses map camera to world.
Depth is optical-axis z in metres. Output coordinates: x-right, y-up,
z-forward in the front-camera frame. Bounds: x `[-10,10)`, y `[-1.75,1.75)`,
z `[0,10)` metres; cell width 0.25 m. Output arrays are `(z,y,x)`.

Labels: 0 free; 1 floor; 2 wall/ceiling; 3 opening; 4 furniture; 5 appliance;
6 plumbing fixture; 7 electronics; 8 small object; 9 other occupied.
Ignore label -1 is supervision only, never a prediction.

Targets describe same-room, first-hit-instance occlusion completion within
the three views and fixed bounds. No unseen-room hallucination is required.
Canonical targets use occupied-priority aggregation from 10 cm source labels;
hidden target labels are selected only from hidden source cells. The exact rule
is in the handout. Some samples have no hidden cells after coarsening; they
contribute no counts, but all sample rows remain mandatory. Houses, not samples
or voxel counts, have equal ranking weight. Test labels stay private.

## Common problems

- Missing data: run the downloader for the needed split or exact house.
- Missing checkpoint: run training first; keep weights in `student_submission/weights/`.
- CPU memory or worker issue: try `--workers 0 --batch-size 8`.
- Low smoke score: placeholders are not implementations; check the eight focused tests.
- Packaging says stale evidence: rerun full Val, Test prediction, figures and report after the final edit.
- Report images missing: run full evaluation, then `python run.py figures`.
- Dataset size: the pinned manifest reports archive bytes; budget for HF cache plus extraction.
- Re-downloading a subset preserves receipts for other already verified houses
  at the same pinned revision. Changing revisions starts a new split receipt.

Do not use generative AI or coding agents to complete the assignment. Follow
`NON_AI_POLICY.md`. Credentials never belong in code, reports, Git, HF, or ZIPs.
