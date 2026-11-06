"""Optional cross-process GPU admission for overlapping inference/evaluation.

CPU simulation and diagnostics keep running while GPU sections take turns.
The gate changes scheduling only; model/render configuration stays untouched.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import gc
import json
import os
from pathlib import Path
import sys
import threading
import time

_local = threading.local()


def _write(path, data):
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.{threading.get_ident()}.tmp')
    temporary.write_text(json.dumps(data, indent=2)+'\n')
    temporary.replace(path)


def _event(kind, label, **extra):
    folder = os.environ.get('TABLETOP_GPU_EVENTS')
    if not folder:
        return
    path = Path(folder);path.mkdir(parents=True, exist_ok=True)
    data = {'event': kind, 'label': label, 'pid': os.getpid(), 'thread': threading.get_ident(),
        'utc': datetime.now(timezone.utc).isoformat(), 'monotonic_seconds': time.monotonic(), **extra}
    with (path/f'{os.getpid()}.jsonl').open('a') as output:
        output.write(json.dumps(data)+'\n')


def _release_cuda():
    # Avoid importing/initializing Torch in the evaluator or parent processes.
    torch = sys.modules.get('torch')
    if torch is not None and torch.cuda.is_initialized():
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()


@contextmanager
def gpu_lease(label):
    name = os.environ.get('TABLETOP_GPU_LOCK')
    if not name or getattr(_local, 'active', False):
        yield
        return
    path = Path(name);path.parent.mkdir(parents=True, exist_ok=True)
    owner = path.with_name(path.name+'.owner.json')
    requested = time.monotonic()
    _event('waiting', label)
    with path.open('a+') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        acquired = time.monotonic();_local.active = True
        details = {'active': True, 'pid': os.getpid(), 'label': label,
            'acquired_utc': datetime.now(timezone.utc).isoformat(), 'wait_seconds': acquired-requested}
        try:
            _write(owner, details);_event('acquired', label, wait_seconds=acquired-requested)
            yield
        finally:
            try:
                _release_cuda()
            finally:
                _local.active = False
                _write(owner, {**details, 'active': False, 'released_utc': datetime.now(timezone.utc).isoformat()})
                _event('released', label, held_seconds=time.monotonic()-acquired)
                fcntl.flock(handle, fcntl.LOCK_UN)
