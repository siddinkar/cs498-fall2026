# Kaggle submission guide

Competition: https://www.kaggle.com/competitions/cs498-fall-2026-multiview-occupancy

Course invitation (join here first): https://www.kaggle.com/t/f04e3dce46f047c5b52deec9fbe73701

The competition is private. Open the invitation while signed in to a Kaggle
account whose verified email address ends in `@illinois.edu`. Until you have
joined this way, the plain competition URL shows a 404 page; that is expected
and is not a data-download problem. After joining, accept the rules and compete
as an individual (one-person team, e.g. `cs498-YOUR_NETID`; teams are limited to
one member).
Install API credentials using Kaggle's normal account setup, outside this folder.

## Self-service scoring

Public Test inputs -> your final method -> 150-row CSV -> Kaggle private GT ->
one live ranking over all ten Test houses. Kaggle scores predictions; it does
not execute your code or accept your model as the leaderboard submission.

```bash
python run.py download --split test
python run.py predict
python run.py submit outputs/kaggle/submission.csv
python run.py submit outputs/kaggle/submission.csv --submit
```

The third command only validates. The last uploads. The web Submit Prediction
button accepts the same CSV. Record its full SHA-256 and the Kaggle entry you
select. Do not edit rows or encoded values manually.

The CSV header is `Id,Prediction`. IDs are ordered, unique, and have the form
`test_house_000__sample_00000`. There are 10 houses x 15 samples = 150 rows.
Each `Prediction` is an int8 `(40,14,80)` grid in **(z,y,x)** order, flattened
in C order, zlib-compressed and Base64-encoded. Labels must be 0-9.
The local validator bounds decoding and checks coverage, order, shape and labels.
A `--max-samples` smoke run is deliberately incomplete and cannot be submitted.

## Metric

Only eligible completion cells are scored. Pool confusion across 15 samples
within each house; compute class IoU for target-present classes 0-9, including
free; mean those IoUs; then average all ten house scores equally.
Some samples have no hidden canonical cells and contribute no counts. All rows
are still required. All ten houses contribute to one ranking partition.
Kaggle's name `Public` for that partition does **not** mean labels are public.

House-macro completion mIoU is the only ranking metric. Local end-to-end runtime
(import, loading, inference, CSV writing) breaks an exact tie only after an audit
on the same hardware/software, not by comparing uncontrolled local timings.
The fixed reference score is **0.65135**, verified online and compared at
Kaggle's displayed precision, not unpublished extra decimal places. It is a
course benchmark, not a competing student team or a required leaderboard entry.

## Integrity and final package

Test inputs are public; Test labels, source houses and annotations are not.
Encoding is not encryption. Student signatures cannot prove that a model
generated predictions, so provenance is checked by reproducing your final
code/model package and comparing recorded source/data/CSV hashes.

Do not annotate Test images, use simulator assets to recover targets, combine
other samples, use sample-specific rules, share predictions, or probe the metric.
Use Val for development and checkpoint selection. The project is individual
and non-AI-assisted. Declare public pretrained sources and external unlabeled
data; external labeled semantic/occupancy data are prohibited.

Finalize code, weights and `method.json` before full Val/Test runs. Submit the
course ZIP plus `package_manifest.json` to the LMS, not Kaggle. This includes
the exact selected CSV and `submission_runtime.json`, along with report and Val
evidence. Every credited entry is subject to reproduction audit.

The top three final finishers may earn 1-10 percentage points of bonus
(+0.2 to +2.0 project points), based on performance and successful audit.
After closing, eligible winners publish the exact code in a public tagged
GitHub release and learned weights in a public HF model repo. Include environment,
commands, license, citations and matching checkpoint hashes. Training-free methods
declare that no weights exist. Do not publish before the deadline.

## Troubleshooting

- 401: reinstall your Kaggle API credential; never paste it into the report.
- 403: join and accept the rules in your browser.
- Validation failure: regenerate the CSV using all Test samples; do not patch it.
- Pending score: check the submission status and any displayed scoring error.
- Stale evidence: finalize code/model first, rerun full Val and Test prediction,
  regenerate figures (or record your verified custom figures), and rebuild the report.
