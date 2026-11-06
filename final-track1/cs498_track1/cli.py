"""Command-line interface for provided tools, separate from student algorithms."""

import argparse
from pathlib import Path


def common(parser, device="auto"):
    """Add shared method/data options; paths are relative to the handout root."""
    parser.add_argument("--data-root", type=Path, default=Path("data/triview"))
    parser.add_argument(
        "--submission-dir", type=Path, default=Path("student_submission")
    )
    choices = ["cpu"] if device == "cpu" else ["auto", "cpu", "cuda"]
    parser.add_argument("--device", choices=choices, default=device)


def main(argv=None):
    """Dispatch exactly one command; uploads require an explicit --submit flag."""
    parser = argparse.ArgumentParser(
        description="Track 1 provided tools. Run from the handout root."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("download", help="Download verified public data")
    p.add_argument("--split", choices=["train", "val", "test"], required=True)
    p.add_argument("--output", type=Path, default=Path("data/triview"))
    p.add_argument(
        "--revision",
        help="Override the official pinned revision (not for final graded evidence)",
    )
    p.add_argument(
        "--houses", nargs="+", help="Exact house IDs; omit for the full split"
    )
    p.add_argument("--keep-archives", action="store_true")

    p = commands.add_parser("train", help="Train the baseline, at most 15 epochs")
    common(p)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--rgb-only", action="store_true")
    p.add_argument(
        "--resume",
        type=Path,
        help="Your own checkpoint_last.pt; epochs remains the original total",
    )
    p.add_argument("--output", type=Path, default=Path("outputs/train"))

    for name, help_text, output in [
        ("evaluate", "Canonical Val evaluation", "outputs/evaluation/val"),
        ("predict", "Generate the 150-row Test CSV", "outputs/kaggle/submission.csv"),
    ]:
        p = commands.add_parser(name, help=help_text)
        common(p)
        p.add_argument("--max-samples", type=int)
        p.add_argument("--output", type=Path, default=Path(output))

    p = commands.add_parser(
        "figures", help="Render report figures from final Val predictions"
    )
    common(p)
    p.add_argument("--evaluation", type=Path, default=Path("outputs/evaluation/val"))
    p.add_argument("--output", type=Path, default=Path("outputs/report"))
    p.add_argument(
        "--record-existing",
        action="store_true",
        help="Record your own four figures, instead of rendering provided plots",
    )

    p = commands.add_parser(
        "validate", help="Probe the inputs-only method interface on CPU"
    )
    common(p, device="cpu")
    p = commands.add_parser(
        "package", help="Build the final LMS ZIP and SHA-256 manifest"
    )
    common(p, device="cpu")
    p.add_argument("--student-id", required=True)
    p.add_argument("--kaggle-username", required=True)
    p.add_argument("--output", type=Path, default=Path("outputs/submission_package"))

    p = commands.add_parser(
        "submit", help="Validate CSV locally; upload only with --submit"
    )
    p.add_argument("csv", type=Path)
    p.add_argument("--data-root", type=Path, default=Path("data/triview"))
    p.add_argument("--submit", action="store_true")
    commands.add_parser("report", help="Compile report_template.tex to report.pdf")
    p = commands.add_parser(
        "doctor", help="Check the environment before downloading/training"
    )
    p.add_argument("--data-root", type=Path, default=Path("data/triview"))
    p.add_argument("--require-gpu", action="store_true")
    p.add_argument("--require-latex", action="store_true")
    p.add_argument("--min-free-gib", type=float, default=0)
    p.add_argument("--splits", nargs="*", choices=["train", "val", "test"], default=[])
    p.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    if args.command == "download":
        from .download import DEFAULT_REVISION, download

        args.revision = args.revision or DEFAULT_REVISION
        return download(args)
    if args.command == "doctor":
        from .diagnostics import diagnose

        return diagnose(args)
    if args.command == "report":
        from .report import main as build_report

        return build_report()
    if args.command == "figures":
        if args.record_existing:
            from .evidence import record_figures

            return record_figures(args.submission_dir, args.evaluation, args.output)
        from .figures import generate

        return generate(
            args.data_root, args.evaluation, args.output, args.submission_dir
        )

    from . import workflow

    if args.command == "submit":
        info = workflow.validate_csv(args.csv, args.data_root)
        print(info)
        if args.submit:
            from kaggle.api.kaggle_api_extended import KaggleApi
            from .protocol import COMPETITION

            api = KaggleApi()
            api.authenticate()
            print(
                api.competition_submit(
                    str(args.csv), "CSV SHA256 " + info["csv_sha256"][:16], COMPETITION
                )
            )
        return info
    handler = {
        "train": workflow.train,
        "evaluate": workflow.evaluate,
        "predict": workflow.generate_csv,
        "validate": workflow.probe,
        "package": workflow.package,
    }[args.command]
    return handler(args)
