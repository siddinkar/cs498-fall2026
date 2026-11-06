"""Small atomic JSON helpers for evaluation logs and artifact receipts."""

import json
import os
from pathlib import Path


def read(path):
    """Load numeric/log metadata; no pickle or executable object deserialization."""
    return json.loads(Path(path).read_text())


def write(path, value):
    """Atomically replace a generated JSON artifact and reject NaN/Infinity."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
