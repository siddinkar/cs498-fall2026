"""Read-only student environment checks; no credentials or target data are read."""

import importlib
import importlib.metadata
import json
import platform
import shutil
import sys
from pathlib import Path


def diagnose(args):
    """Print actionable checks and optionally save a non-secret environment report."""
    checks = []

    def check(name, passed, detail, fix):
        checks.append(
            dict(
                name=name, passed=bool(passed), detail=detail, fix="" if passed else fix
            )
        )

    check(
        "Python",
        (3, 11) <= sys.version_info[:2] <= (3, 12),
        platform.python_version(),
        "Use Python 3.11 or 3.12; see SETUP.md.",
    )
    requirements = Path(__file__).resolve().parents[1] / "requirements.txt"
    for line in requirements.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        try:
            from packaging.requirements import Requirement

            requirement = Requirement(line)
            version = importlib.metadata.version(requirement.name)
            check(
                requirement.name,
                version in requirement.specifier,
                version,
                "Install the documented requirements with this Python interpreter.",
            )
            if requirement.name != "kaggle":
                module = {"Pillow": "PIL", "huggingface-hub": "huggingface_hub"}.get(
                    requirement.name, requirement.name
                )
                try:
                    importlib.import_module(module)
                    check(f"{module} import", True, "Import succeeded", "")
                except Exception as error:
                    check(
                        f"{module} import",
                        False,
                        f"{type(error).__name__}: {error}",
                        "Reinstall the documented requirements in a clean environment.",
                    )
        except (importlib.metadata.PackageNotFoundError, ImportError) as error:
            check(
                line,
                False,
                type(error).__name__,
                "Run python -m pip install -r requirements.txt.",
            )
    try:
        import torch

        check("Torch import", True, str(torch.__version__), "")
        cuda = torch.cuda.is_available()
        if cuda:
            value = torch.ones(8, device="cuda").sum().item()
            detail = f"{torch.cuda.get_device_name(0)}; allocation/sum={value}; CUDA {torch.version.cuda}"
        else:
            detail = "CUDA unavailable; CPU smoke is available, not a full-training recommendation."
        check(
            "GPU",
            cuda or not args.require_gpu,
            detail,
            "Select a Colab GPU runtime or fix the Ubuntu NVIDIA/PyTorch installation.",
        )
    except Exception as error:
        check(
            "Torch/CUDA",
            False,
            f"{type(error).__name__}: {error}",
            "Reinstall the supported Torch build, then restart the runtime.",
        )
    latex = bool(shutil.which("latexmk") or shutil.which("pdflatex"))
    check(
        "Report tools",
        latex or not args.require_latex,
        f"LaTeX available: {latex}",
        "Install the LaTeX packages in SETUP.md, or compile the report externally.",
    )
    location = storage_path(args.data_root)
    free = shutil.disk_usage(location).free / 1024**3
    check(
        "Free disk",
        free >= args.min_free_gib,
        f"{free:.1f} GiB free in data filesystem ({location})",
        f"Free at least {args.min_free_gib:g} GiB; use local Colab storage for data.",
    )
    for split in args.splits:
        try:
            from .download import DEFAULT_REVISION
            from .protocol import expected_sample_ids
        except ImportError as error:
            check(
                f"{split} coverage",
                False,
                str(error),
                "Install requirements before checking data.",
            )
            continue
        ids = sorted(
            p.parent.parent.name + "__" + p.parent.name
            for p in args.data_root.glob(
                f"samples/{split}_house_*/sample_*/manifest.json"
            )
        )
        expected = expected_sample_ids(split)
        receipt_path = args.data_root / "download_receipts" / f"{split}.json"
        try:
            receipt = json.loads(receipt_path.read_text())
        except (OSError, ValueError):
            receipt = {}
        check(
            f"{split} coverage",
            ids == expected and receipt.get("revision") == DEFAULT_REVISION,
            f"{len(ids)}/{len(expected)} manifests; pinned receipt={receipt.get('revision') == DEFAULT_REVISION}",
            f"Run python run.py download --split {split} --output {args.data_root}.",
        )
    result = dict(
        passed=all(c["passed"] for c in checks),
        python=platform.python_version(),
        platform=platform.platform(),
        checks=checks,
        scope="Local environment/manifest preflight, not model correctness or hosted-Colab certification",
    )
    for item in checks:
        print(
            f"{'PASS' if item['passed'] else 'FAIL'} {item['name']}: {item['detail']}"
        )
        if item["fix"]:
            print("  Fix:", item["fix"])
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    if not result["passed"]:
        raise SystemExit(1)
    return result


def storage_path(data_root):
    """Find the actual data filesystem even before the directory is created."""
    path = Path(data_root).resolve()
    while not path.exists() and path != path.parent:
        path = path.parent
    return path
