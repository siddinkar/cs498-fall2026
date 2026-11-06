# Environment and getting started

Supported paths: **Ubuntu + NVIDIA GPU**, or **Google Colab**. Colab is the
fallback for students without a suitable GPU and is recommended for **macOS and
Windows**. Native Mac/MPS, Windows and WSL are not required or validated paths.
This project is individual and non-AI-assisted, including in Colab: do not use
Gemini or AI code completion to implement your work.

## 1. Obtain the student release

On or after November 4, use the course student repository, not the staff repo:

```bash
git clone https://github.com/shenlongwang/cs498-fall2026.git
cd cs498-fall2026/final-track1
```

If the course distributes `final-track1.zip`, extract it and enter its
`final-track1/` directory instead. It must contain `run.py`, `student_submission/`
and the handout PDF. A missing Track 1 directory before release is not a Python
error. Never copy a different homework or an older streaming project to replace it.

## 2. Ubuntu installation

The reproducibility target is Ubuntu 24.04, Python 3.12, PyTorch 2.11.0 with
CUDA 12.8 wheels. Python 3.11 is also within the supported API range. Use a
working NVIDIA driver and a CUDA-capable GPU for full training. The measured
reference machine has 16 GiB GPU memory; this is not a minimum-memory promise.
The provided code does not require compiling a CUDA extension or installing a
separate CUDA toolkit. Use the official [PyTorch selector](https://pytorch.org/get-started/locally/)
if a different driver needs another supported wheel; run all checks afterward.

```bash
nvidia-smi
sudo apt-get update
sudo apt-get install -y python3.12-venv git latexmk texlive-latex-base texlive-latex-recommended texlive-latex-extra
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt -c environment/ubuntu-py312.txt
python run.py doctor --require-gpu --require-latex --min-free-gib 15
```

Run every subsequent command with this environment active. Check
`python -c "import sys; print(sys.executable)"` if an import works in one terminal
but not another. Do not install into your system Python. If you only want a CPU
smoke, install Torch from `https://download.pytorch.org/whl/cpu`, omit
`--require-gpu`, and use `--device cpu`; CPU is not the full-training fallback.

Budget at least 15 GiB of free local space for the 4.23 GiB archive cache,
roughly 5.2 GiB extracted corpus, checkpoints and outputs, in addition to the
Python environment. More space is needed for multiple experiments. The many
small input files should be on local disk, not a network/Drive-mounted dataset.

## 3. First run and editing

Public data downloads require **no HF account or token**:

```bash
python run.py download --split train --houses train_house_000
python run.py download --split val --houses val_house_000
python -m pytest -q tests
python run.py train --smoke --device cpu --batch-size 8 --workers 0
python run.py train --smoke --device cuda --batch-size 8 --workers 0
```

Eight `XFAIL`s are expected for untouched TODOs. A smoke with placeholders
checks plumbing, not useful accuracy or the completed model's memory use.
Edit `student_submission/semocc/model.py` and `learning.py` yourself using a text
editor. Save the actual `.py` file before running its focused tests, for example
`python -m pytest -q tests/test_functions.py`. The README and PDF list all eight
functions. Your own architecture may use the alternative adapter instead.
For that path, install `student_submission/requirements.txt`, run your own
training entry point under that directory, and rejoin the common validation,
figure, prediction and packaging commands. The provided `train` command is for
the supplied baseline, not arbitrary algorithms.

## 4. Full training, interruption and resumption

```bash
python run.py download --split train
python run.py download --split val
python run.py doctor --require-gpu --splits train val
python run.py train --device cuda --epochs 15 --batch-size 8 --workers 0
```

This is the full dataset, not the two-house smoke. Batch 8/workers 0 is a
resource-conscious starting setting; if it runs out of memory, begin a new run
with a smaller batch. Default batch 32 reproduces the measured 4080 setting.
Do not interpret that machine's timing as a Colab/T4 guarantee.

After interruption, keep the code, data revision, Torch version, device, batch
size and **original total epoch target** unchanged:

```bash
python run.py train --device cuda --epochs 15 --batch-size 8 --workers 0 \
  --resume outputs/train/checkpoint_last.pt
```

Recovery occurs at completed epoch boundaries; an unfinished epoch is repeated.
Different allocated GPUs and CUDA operations may produce small numerical
differences; recovery is not a promise of bitwise equality across hardware.
The recovery file restores model, optimizer, scheduler, scaler, sampler, RNG,
history and historical best weights. It is saved separately from the compact
inference weights under `student_submission/weights/`. Only resume your own
checkpoints. Legacy inference-only files cannot resume training. If no epoch
has finished, start normally; a `.partial` file is not a complete checkpoint.
Changing code or training settings is a new experiment, not an exact resume:
use a fresh output directory and do not mix its evidence with the old method.

## 5. Colab fallback (recommended for Mac/Windows)

Open `notebooks/track1_colab.ipynb` at [Colab](https://colab.research.google.com/)
using **File > Upload notebook**, then save your own notebook copy in Drive.
After publication, the [direct Colab link](https://colab.research.google.com/github/shenlongwang/cs498-fall2026/blob/main/final-track1/notebooks/track1_colab.ipynb)
opens the same notebook. Select **Runtime > Change runtime type > GPU** and
run the numbered cells. Use a Python 3.11/3.12 runtime. The notebook checks this
instead of silently using an unsupported runtime.

The notebook mounts Drive with your consent. It keeps editable code, model
weights, recovery checkpoint, outputs and report under your persistent project
directory. Data and HF cache live on local `/content/` storage and must be
downloaded again after a runtime reset. Do not extract all 333,150 data files
onto mounted Drive. Saving a notebook alone does **not** save `.py` edits or
files in `/content/`; Git alone also does not back up ignored weights/outputs.

Edit the saved `.py` files using Colab's Files panel, or edit locally and use
the notebook's explicit single-file upload cell. Do not overwrite your method
with a fresh starter when reconnecting. To continue: remount Drive, reuse the
existing project, reinstall runtime dependencies, redownload the same splits
locally, run the data check, then use the resume command. The notebook has
separate smoke, full-data, train/resume and final-submission cells.

Colab supplies Torch; the notebook preserves an already compatible installation
rather than blindly replacing it and breaking related packages. It installs
the other requirements and report tools. Free GPU availability, runtime length
and GPU type are [not guaranteed](https://research.google.com/colaboratory/faq.html).
Do not use keep-alive or quota-bypass tricks. Checkpoints, not uninterrupted
sessions, are the recovery mechanism. Stop before full training if no GPU is
allocated; retry later or use the supported Ubuntu GPU path.

## 6. Report, Kaggle and final downloads

After finalizing your method and `method.json`, follow README for full Val,
figures and Test inference. `python run.py report` compiles the filled template
to `report.pdf`. Alternatively, use a LaTeX editor such as Overleaf with the
template and required images, then place the compiled file at `report.pdf`.
The notebook installs LaTeX so local desktop LaTeX is unnecessary.

Kaggle browser upload is the simplest route: sign in, join the competition,
accept its rules, create your individual team, and upload only the generated
`outputs/kaggle/submission.csv`. No Kaggle token is needed for that route.
Optional CLI upload is described in KAGGLE_SUBMISSION.md; never put a token in
the notebook, Drive project, code, report, Git or final ZIP.

Run the final packager and download **both** the ZIP and the separate manifest
for LMS submission. Also keep the exact selected CSV and a backup of your work.
The notebook includes download cells. If a browser blocks multiple downloads,
download the files individually from the Files panel. Inspect them before
leaving the runtime; a printed path is not a copy on your computer.

## Troubleshooting

- `No module named ...`: check the interpreter and rerun requirements installation.
- CUDA unavailable: run `doctor --require-gpu`; select a Colab GPU or fix the driver/wheel.
- LaTeX missing or `xurl.sty` missing: install the listed `texlive-latex-extra` package.
- Missing data: redownload the required split into the same `--data-root` used by all commands.
- Resume rejected: use the original code/config/total epochs and recorded Torch version; do not bypass the check.
- Out of memory: start a new lower-batch run, close unused models, or use the Ubuntu GPU path.
- Reset Colab session: notebook/Drive files may survive, local data/runtime packages do not; rerun setup and download.
- HTTP 403 on Kaggle: join and accept the rules in your own browser after launch.
