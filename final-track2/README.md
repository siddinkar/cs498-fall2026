# Final Project - Track 2

## Reconstruct complete, textured 3D instances

Turn a posed RGB video into a complete, textured, instance-separated 3D scene.
Recognizing visible surfaces is not enough: reconstruct occluded parts, preserve
object placement, and avoid interpenetration. Static collision-free checks are scored;
no gravity simulation is required. Collision-free does not mean dynamically stable. **Category names are
not required.** This is an individual, AI-assisted project. You receive the full
baseline, not a set of required TODO functions. Improve it or replace it while
preserving the input, output, and evaluation contracts.

Read `final_track2_handout.pdf` for the complete assignment. All commands below
run from this directory. Do not generate scenes.

## Kaggle competition

- Competition: https://www.kaggle.com/competitions/cs498-fall-2026-track2-reconstruction
- **Join first** with the course invitation:
  https://www.kaggle.com/t/27cd04d4d854493f9f8f45a814bb1cd5
  Sign in to an individual Kaggle account whose verified email ends in
  `illinois.edu`. The competition is private, so until you join through the
  invitation the competition URL shows a 404 page; that is expected.
- You upload only the generated Test CSV; the submission steps are in
  section 5 below ("Kaggle, HF results and one small final course ZIP").
  One submission per UTC day; teams are limited to one member.

Deadline: **December 9, 2026, 11:59 p.m. America/Chicago**.

## Folder guide

- `scripts/student/`: supported run, evaluation, preview, validation and packaging commands.
- `methods/`: your algorithm; `interface_example.py` demonstrates the API only.
- `src/`: shared input/output, baseline and evaluation implementation.
- `scripts/setup/` and `env/`: setup scripts and environment specifications.
- `configs/`: reusable settings; see `configs/README.md` for path conventions.
- `data/`: downloader and data manifest. `examples/`: editable submission/AI/model examples.
- `tests/`: student-only checks; no instructor files are needed.

`scripts/baseline/` and `scripts/evaluation/` are implementation entry points
used by the supported tools, not extra assignment steps.

## 1. Lightweight setup and data

```bash
bash scripts/setup/tools.sh
source env/tools/.venv/bin/activate
python -m pytest tests -q
python data/download_data.py --split dev
```

Start with Linux and Python 3 with `venv` support (Ubuntu package
`python3-venv`). The installer pins uv and provisions Python 3.11.16 in a
separate tools environment; it does not replace system Python or install any
models. See the [uv installation](https://docs.astral.sh/uv/getting-started/installation/)
and [Python provisioning](https://docs.astral.sh/uv/guides/install-python/)
documentation. A working network connection is needed for first installation.
Use Ubuntu for the CUDA baseline; native macOS/Windows baseline execution is
not supported. The reference was measured on RTX 4080. This track does not
require or provide a Colab fallback; use a suitable Ubuntu/NVIDIA machine for
the supplied baseline. Cloud GPU compatibility has not been validated.

The data installer also accepts `--from-local-dir /path/to/provided_archives`.
See `data/README.md` for calibration conventions and split boundaries. Dev/Val
include evaluation GT; Test contains RGB and calibration only. The loader used
by methods opens only RGB and calibration even for Dev/Val.

## 2. Set up the provided baseline

The complete model baseline needs Linux, an NVIDIA CUDA GPU, `git`, `uv`, Conda,
the system `ffmpeg` package (including `ffprobe`),
and space for model weights and native extensions. It uses separate perception
(Python 3.11/Torch 2.5.1) and reconstruction (Python 3.10/Torch 2.7.1) runtimes.
Their versions are intentionally separate. A 16 GiB RTX 4080 was used for the
reference. Budget at least 30 GiB scratch for an active case, in addition to
model/environment/data storage. The fresh reconstruction weights occupy about
45 GiB; allow roughly 120 GiB or more for both runtimes, models, build files,
data and an active case. This is a storage budget, not a download-time promise.
Other GPUs may require a different
`TORCH_CUDA_ARCH_LIST` when building native extensions.

```bash
bash scripts/setup/perception.sh
bash scripts/setup/reconstruction.sh
python scripts/student/run_baseline.py --scene dev_scene_000 --check
python scripts/student/run_baseline.py --scene dev_scene_000
```

The pipeline is Pi3X predicted RGB-D -> Florence-2/SAM2 proposals -> multi-view
3D instance fusion -> Fire3D reconstruction and PBR appearance -> GLB export.
Fire3D is not used for perception. All 192 frames are used by the reference
baseline. Its minimum track support is 25% of frames, followed by class-agnostic
3D bbox NMS at IoU 0.65. These are baseline choices, not constraints on your own
algorithm. Detected label strings can help that baseline associate proposals;
they are not scored and your method may use `unknown` throughout.

The handout cites the papers behind these components at their point of use.
Reusable BibTeX entries are in `references.bib`, including DINOv3/AnyUp image
conditioning, TRELLIS.2 shape/material decoding, and the SSIM metric. Pi3X is
identified separately as an official engineering release of the Pi3 project.
Cite the components you use and add sources for your own algorithm in the report.

Use a new `--algorithm-name` for each changed configuration. Fixed reference
profiles protect the provided baseline from accidental changes; they are not
an assignment restriction. For parameter changes inside the Fire3D wrapper,
remove/replace its frozen profile and corresponding protocol assertions in
your experimental copy, document the change, and use a new run name. Alternatively
use the unrestricted method interface below.

The default `infer` stage exports results without opening `scene.blend` or
running GT comparisons. Use the supported evaluation and preview tools below.
Approximate measured baseline runtimes were
7-11 minutes per case on the reference GPU; your machine and method may differ.
Native remesh/UV operations can change on reruns: save the exact GLBs evaluated
in each experiment rather than assuming a seed reconstructs identical files.

## 3. Evaluate Dev/Val with the Kaggle metric

Use the tools environment and the fixed local Blender renderer:

    bash scripts/setup/evaluation.sh
    export BLENDER_EXECUTABLE="$PWD/env/blender/blender-4.2.0-linux-x64/blender"
    python scripts/student/evaluate.py \
      --manifest outputs/baseline/dev_scene_000/evaluation/mesh_manifest.json

CPU rendering is the default; configured NVIDIA OptiX hardware can use
--device gpu. Both use identical camera/lighting settings, 512² RGBA PNGs,
32 Cycles samples and Standard view transform. Do not change canonical settings.
Open evaluation/reports/evaluation/default/index.html for paired evidence.

The numerical scorer is self-contained CPU NumPy/SciPy/image processing, using
the same source online. All rendering, surface sampling and voxelization run
locally. Kaggle compares cached point arrays, images and sparse voxel intervals;
it never processes triangles, runs Blender/physics/student code, or uses the
network. GT preprocessing is performed once before competition setup.
No Test GT is needed to export predictions.

| Group | Definition | Benchmark weight |
|---|---|---:|
| Perception | Class-agnostic instance F1; AABB IoU ≥ .25; missing/extras penalized | 20% |
| Scene geometry | Area-uniform complete-surface CD/F1 in world placement | 20% |
| Object geometry | Matched-object geometry × foreground recall | 20% |
| Scene appearance | 12 shared fixed world views, derived only from public calibration | 15% |
| Object appearance | Six world-axis views, each side's own AABB center/uniform scale; × recall | 15% |
| Collision-free | Matched collision-free foreground GT recall; table match required | 10% |

Geometry combines symmetric **unsquared** CD in meters and surface F1 at 1 cm:
G = (exp(-CD/(.05 × GT_box_diagonal)) + F1)/2.
The local exporter caches 100,000 scene samples and 20,000 per object, using
public fixed seeds independent of GT IDs. No ICP, per-object
rescaling or GT pose correction is applied to these world-space metrics.

Appearance balances full-image and foreground-union normalized PSNR (40 dB
ceiling) and SSIM. Object views use each side's AABB framing, never independent
rotations/nonuniform stretching. Size/placement remain measured by geometry.
Both object groups equal foreground matching recall times the mean over all
matched instances, including a matched support table; no matches earn zero.
Foreground omissions reduce coverage credit. Support omissions are penalized
by scene metrics and make collision credit zero; the object groups alone are
not support-completeness checks.

Collision uses a **5 mm world-anchored sparse voxel grid**, not triangle
intersection. Local preprocessing preserves closed concave components; open
3D components use a convex fallback. Occupied cells contain an interior grid
center or a quantized sampled surface point. A component's core is its interior
eroded by one cell in all 26 neighboring directions. Two instances collide
when either one's core intersects the other's occupied cells. Planar pieces
contribute sampled surface cells but no fictitious volume. This conservative
contact rule can miss sub-grid/thin penetration; it is not exact collision or
a support/stability proof. Kaggle intersects run-length encoded columns without
expanding a dense grid. Floor penetration uses the original minimum Z and a
2 mm tolerance. Invalid/over-fragmented collision solids earn no collision credit.
Extra objects participate; missing GT objects earn no credit.
A local 600-second per-scene voxelization deadline or collision budget failure
zeroes **only that scene's collision**, not other groups.

Evaluate all Val cases, then summarize:

    python scripts/student/summarize_scores.py \
      --predictions outputs/my_method --split val

All ten Val cases contribute; missing/failed cases count as zero. Dev has 30
scenes, Test 20. Contextual means are in configs/baseline_reference.json.
Re-evaluate the exact exported GLBs after texture packing; never copy reference
numbers into your measurements.

## 4. Further improve with your own algorithm

You may replace depth/perception, temporal association, completion, surface
reconstruction, appearance or automatic collision avoidance. Neural and
non-neural methods are welcome. Use Dev for development/training and Val for
measurement/model selection. No GT inference input, manual masks, mesh editing,
object placement or Test-specific lookup tables.

Create methods/my_method.py:

    def predict(scene):
        # scene.rgb(i): uint8 RGB; scene.calibration: known numeric camera arrays.
        # One fully automatic algorithm, using only these inputs.
        # Separate trimesh meshes in dataset world, Z-up, meters.
        return {1: table_mesh, 2: object_mesh, 3: another_object_mesh}

    python scripts/student/run_method.py \
      --method methods.my_method:predict --scene dev_scene_000
    python scripts/student/evaluate.py \
      --manifest outputs/my_method/dev_scene_000/evaluation/mesh_manifest.json

methods/interface_example.py is a runnable one-box **format demonstration**,
not a competitive algorithm. The adapter preserves UV textures or converts
vertex/face colors to embedded atlases. Constant roughness/metallic maps fill
missing channels; this does not invent unseen appearance or ensure quality.
Other implementations may directly generate the same validated GLB format.
Disclose model sources, licenses, training data and dependencies.

Try substantive ideas and controlled comparisons. Submit **one final version**,
selected by the highest mean benchmark over all ten Val scenes. Final Val
results, code/config/model, Test GLBs and Kaggle CSV must describe that same
method. Do not cherry-pick cases/metrics from different variants.

## 5. Kaggle, HF results and one small final course ZIP

### Rehearse the submission locally on Val first

Do not use a scarce Test upload to debug CSV formatting. After producing one
Val reconstruction, export exactly the same representation used for Test:

    python scripts/student/export_kaggle.py \
      --predictions outputs/my_method --split val --scenes val_scene_000 \
      --output outputs/val_one.csv --device cpu
    python scripts/student/check_kaggle_submission.py \
      --csv outputs/val_one.csv --scenes val_scene_000 \
      --predictions outputs/my_method --output outputs/val_one_check.json

This diagnostic has 2,048 rows and uses public Val GT locally. The checker
validates transport, images/arrays/voxel bounds, and (when `--predictions` is
provided) the exact GLB source hashes and strict 20 MB budget. It runs the **same
numerical scorer as Kaggle** and writes each group plus the total to JSON.
The JSON also has a readable `submission_format` inventory: actual IDs,
point/bounds shapes, image counts, sparse-run counts and preprocessing status.
Use it to understand the generated representation without editing Base64.
It never downloads or opens Test GT, and it does not make an online submission.
This rehearsal receipt does not replace the required Dev/Val side-by-side
reports: continue to use `evaluate.py` and `summarize_scores.py` for those.

Then omit `--scenes` in both commands and use new output filenames to rehearse
all ten Val scenes (20,480 rows). For the final Test export change only
`--split val` to `--split test`, keep all 20 scenes (40,960 rows), and upload the
generated Test CSV. **Never rename Val IDs into Test IDs** or upload Val CSV to
the Test competition. View/sampling/voxel caches are reused when their source
and settings have not changed; final CSVs/check receipts are not overwritten.
The shared local cache is `outputs/.evaluation_cache/`; each CSV also gets a
self-contained `<CSV stem>_cache/` snapshot beside it. Report snapshots remain
self-contained. The initial rendering pass is heavy; repeated numeric checks
reuse the cache. A format pass is not proof of algorithm provenance.

Run the frozen method on all 20 Test inputs. Validate and export:

    python scripts/student/validate_submission.py \
      --predictions outputs/my_method --split test
    python scripts/student/export_kaggle.py \
      --predictions outputs/my_method --split test \
      --output outputs/submission.csv --device cpu

First open
[the course invitation](https://www.kaggle.com/t/27cd04d4d854493f9f8f45a814bb1cd5)
with your individual Kaggle account. The restricted-access check uses an
`illinois.edu` email address; follow Kaggle's email verification prompts.
Before you join through the invitation, a private-page/404 response is expected,
not a data-download failure. After joining, upload the generated **plain CSV** to
[the Track 2 competition](https://www.kaggle.com/competitions/cs498-fall-2026-track2-reconstruction),
accept the applicable rules, and check the final scoring status. Upload success
alone is not a score. No course API token is needed; never submit credentials.

The columns are Id,Prediction. Every scene has 2,048 fixed rows of at most
65,536 characters, padded with a hyphen. All 40,960 rows must remain in sample
order. The exporter handles this automatically. Do not hand-edit chunks or
submit .partial/.gz files. Kaggle reassembles whole scenes and computes one
equal-scene mean over all 20 Test cases; there is no second hidden subset.
Encoding/compression is transport, **not encryption or an e-signature**.

Images must come from the exact GLBs and unchanged canonical renderer. Hashes
are consistency receipts, not proof of image/sample/voxel production. Final
code/model/artifact review can require local reproduction using the provided
tools. No production Kaggle job re-renders or parses GLBs.
Fake renders, manually repaired Test outputs, evaluator exploitation and hidden
label access are prohibited.

Kaggle permits **at most one submission per day (UTC)**. You may submit on
later days, but choose **one final complete version** using the highest mean
benchmark across all ten Val scenes, not by Test feedback. Select the matching
final entry on Kaggle and include that exact submission ID/CSV in the HF results
archive referenced by your course ZIP. Local Dev/Val/format checks are unlimited. Do not combine different methods
per case or per metric. Infrastructure recovery resumes the same artifacts.

Build the minimal mesh ZIP and record the actual Kaggle ID:

    python scripts/student/package_submission.py \
      --predictions outputs/my_method --output outputs/test_results.zip
    python scripts/student/kaggle_receipt.py \
      --csv outputs/submission.csv --submission-id YOUR_SUBMISSION_ID \
      --output submission/kaggle_receipt.json

Only the CSV goes to Kaggle. Large result files go to a dataset repository in
**your own Hugging Face account**, not the course upload. Report and actual
source code must still be submitted through the course system. Prepare this
complete local staging tree first; it is split automatically in the next step:

    submission/
      report.pdf
      README.md
      code/README.md
      code/final_config.json
      code/requirements.txt
      code/...                          # actual final source snapshot
      models/manifest.json              # weight paths/HTTPS URLs + file SHA-256
      models/...                        # optional local weights/licenses, moved to HF
      ai_collaboration.jsonl
      test_results.zip                  # all 20 minimal GLB results
      kaggle_submission.csv             # exact uploaded CSV, renamed only
      kaggle_receipt.json                # submission ID + CSV SHA-256
      evaluation/val_summary.json
      evaluation/evidence_selection.json
      evaluation/dev/<selected-scene>/evaluation/reports/evaluation/default/...
      evaluation/val/<scene>/evaluation/reports/evaluation/default/...
      evaluation/test/<selected-scene>/prediction_preview/...

The inner ZIP contains only evaluator inputs, not code/models/report/videos/GT.
Copy the exact uploaded CSV bytes as kaggle_submission.csv. Include all available
Val metrics and complete selected Dev/Val report folders: metrics.json,
matching.json, visualizations.json, index.html and linked images. Keep failures
in the ten-case summary. Extra analyses go under evaluation/analysis/.

Test evidence is prediction-only. The official Kaggle total may be quoted
with its submission ID; do not invent private GT comparisons/per-object scores.
Start selection/model declarations from examples/. List every required weight
using an included file or a stable HTTPS URL and its file SHA-256. Use
required:false and an empty list only if no weights are needed. Exclude data,
environments, caches, credentials, expiring links and Git history.

### Upload the results to your own HF dataset

    python scripts/student/project_submission.py --submission-dir submission \
      --netid YOUR_NETID --prepare-hf-results --output outputs/results.zip

Create a **dataset**, not a model or Space, at
https://huggingface.co/new-dataset using your own account. Upload only the generated
archive under the exact name `results.zip` using Files and versions → Add file.
Do **not** upload your full working folder, source code, report, AI record,
credentials, course data, environments or caches. `results.zip` contains
`evaluation/`, the Test ZIP, exact Kaggle CSV/receipt, optional locally supplied
weights, and a generated hash inventory. Keep required source-license notices.
Upload weights only when their license permits redistribution; otherwise use
the original provider's stable pinned URL and file SHA-256 in the model manifest.

Keep the repository **private during development**, then make it **public and
ungated at final course hand-in**. The exact submitted revision must be
downloadable without login. A private repo link is not a completed hand-in;
never send an access token in your report or ZIP. Verify anonymous access before
the deadline and retain the repo/files until grading and appeals finish.
Public predictions can be copied by other competitors; do not publish unfinished
results during the competition. Downloading another participant's outputs and
claiming them as your own is not allowed.

From the upload commit's page, copy the **full 40-character commit hash**.
Do not use `main`, a tag, a short hash or an expiring download URL. HF supports
[commit-pinned downloads](https://huggingface.co/docs/huggingface_hub/en/guides/download).
Record the dataset ID and full commit in the report, then make the small course ZIP:

    python scripts/student/project_submission.py --submission-dir submission \
      --netid YOUR_NETID --output outputs/YOUR_NETID_track2.zip \
      --hf-results-zip outputs/results.zip --hf-repo YOUR_HF_USER/YOUR_REPO \
      --hf-revision YOUR_FULL_40_CHARACTER_COMMIT

The **course-platform ZIP** contains only:

    report.pdf
    README.md
    code/...                     # actual final source, configs, dependencies
    models/manifest.json         # stable URLs or paths restored from HF results
    ai_collaboration.jsonl
    results_location.json        # generated repo ID, full revision, bytes, SHA-256
    submission_manifest.json     # generated course file inventory

Only this small ZIP is uploaded to the course system. The report must repeat the
HF dataset ID, full revision, `results.zip` SHA-256 and actual Kaggle submission ID,
matching the generated `results_location.json`. A GitHub/HF code link is **not** a
replacement for the submitted `code/` snapshot. The packer verifies the local
results archive matches your staging tree; it does not upload to HF or prove that
the remote file is accessible. After uploading, download the **pinned revision's**
`results.zip` to a fresh directory and compare its SHA-256 with the packer output.
If outputs change, generate/upload a new archive and regenerate the course ZIP
and report pointer; never silently replace the submitted version.

The packer checks layout, Val means, all Test cases/calibration, mesh budgets
and the CSV receipt; it adds a file/hash inventory. Missing visual evidence
produces explicit partial-credit warnings; core errors stop packaging.
The result ZIP safety bound is 8 GB, not a mesh allowance. Keep the result and
course ZIP hashes. The grader retrieves only the pinned file, checks its size and
SHA-256, and restores the same canonical evaluation tree without executing code.
Download/access failures require recovery of the exact submitted artifact, not a
new method or automatic numerical zero. Unresolved missing/inaccessible required
results cannot be treated as a complete submission.

All GLBs of a scene together must be **strictly below 20,000,000 bytes**,
including table/textures. At most 64 meshes, IDs 1–32767, combined scene totals
of two million vertices/two million triangles, and an 80 MB uncompressed scene
record are supported. Automatic UV/texture packing preserves geometry/IDs/
placement; re-evaluate the exact resulting files. No GLB external resources.

If omitting Test entirely, omit its inner ZIP and Kaggle files, select no Test
IDs and use --without-test. The whole base is halved once; neither bonus is
available. A 19-scene ZIP is not accepted as a complete submission.

## 6. Grading and AI collaboration

| Component | Points |
|---|---:|
| Results and visual evidence | 8 |
| Report | 8 |
| Perception | 1.5 |
| Reconstruction | 1.5 |
| Collision-free recall | 1.0 |
| **Base total** | **20** |
| Eligible top-five Test ranking bonus | **+2** |
| Implemented/evaluated novel-idea bonus | **up to +2** |

For numeric credit, macro-average each entire Val/Test split first. Each split
receives half the credit. Full credit needs absolute normalized .75 in each
split, not .75 of baseline or a per-case threshold; below .75, credit is linear.
Per split: perception .75 points, collision-free .5, and each of four
reconstruction metrics .1875. Sum before rounding.

Show three distinct scenes from each of Dev/Val/Test, from the same final
method. Four evidence groups are worth 2 points each:

- Instances/matching: Dev/Val matches/misses/extras and labeled views; Test predicted IDs.
- Scene geometry/appearance: Dev/Val errors and paired views; Test whole-scene predictions.
- Object geometry/appearance: Dev/Val complete matched-object dumps, representative report views and unmatched objects; Test isolated predicted objects.
- Collision-free: Dev/Val status, proxies and marked witnesses (or explicit clean result); Test prediction-only placement/contact views, with official checks verified later.

Each group earns 2 × verified selected scenes / 9. No baseline-score threshold.
Honest failures/timeouts are evidence; missing artifacts or fabricated successes
are not. No post-evaluation report revision is required.

Create Test qualitative montages without GT:

    python scripts/student/preview_outputs.py \
      --manifest outputs/my_method/test_scene_000/evaluation/mesh_manifest.json

Copy its prediction_preview/ folder for each selected Test scene.
No simulation video is required. Canonical metric renders come from the exporter.

The eight report points cover four two-point criteria: ideas/method/rationale
and related work; controlled experiments/analysis/limitations; reproducibility;
AI-human collaboration and verification. Value clear hypotheses and informative
failures, not only improvements. Use report.tex and record real AI tasks,
suggestions, decisions, checks and fixes; remove sample log entries.
Do not send secrets/private data to external tools.

### Build the report PDF

Edit `report.tex`; keep `references.bib` and your figure files beside it.
For a browser-based route, upload them to
[Overleaf](https://www.overleaf.com/learn/how-to/Creating_a_document_in_Overleaf),
set `report.tex` as the main document, select pdfLaTeX, recompile, and download
the result as `report.pdf`. No project data or credentials belong on Overleaf.

On Ubuntu, install the report tools separately from the Python environments:

```bash
sudo apt-get install texlive-latex-extra texlive-fonts-recommended
pdflatex -interaction=nonstopmode -halt-on-error report.tex
# If you enable the bibliography commands and use citations:
bibtex report
pdflatex -interaction=nonstopmode -halt-on-error report.tex
pdflatex -interaction=nonstopmode -halt-on-error report.tex
```

With no bibliography, only the first `pdflatex` invocation is needed. The
template includes commented bibliography commands; uncomment them and add
your citations before the BibTeX sequence. Inspect the final PDF, remove the
template instructions, and copy it to `submission/report.pdf`.

Novel-idea bonus: 0 without a substantive idea implemented in the final method,
full-Val experiments and official Test results; 1 with limited controls/analysis;
2 with reproducible controlled evaluation and clear interpretation.
A proposal, run-name, presentation or compute-budget change alone is insufficient;
a gain on every metric is not required. The independent top-five bonus requires
valid, reproducible results and policy compliance. Both use the same final
deliverables. Maximum **20 +2 +2 =24**.

Without Test, assess Val numeric work and six Dev/Val evidence scenes, then
halve the entire base once: maximum 10/20, neither bonus eligible.
