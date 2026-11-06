"""Summarize every required Dev/Val case, including failures as zero."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap
from course_metrics import summarize
from track2_io import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--split', choices=['dev', 'val'], default='val')
    parser.add_argument('--run-name', default='default')
    parser.add_argument('--output', type=Path, default=Path('outputs/val_summary.json'))
    args = parser.parse_args()
    result = summarize(args.predictions, args.split, args.run_name)
    write_json(args.output, result)
    print(json.dumps({k: result[k] for k in ['split', 'expected', 'completed', 'missing', 'means']}, indent=2))


if __name__ == '__main__':
    main()
