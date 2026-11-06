"""External wall-clock protection, including cleanup of descendant processes."""
import math
import os
from pathlib import Path
import signal
import subprocess
import time


class EvaluationTimeoutError(TimeoutError):
    """An incomplete evaluation, never an automatic zero or a partial score."""


def _stop_group(process, grace):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        pass
    # Kill remaining descendants even if the leader already exited on SIGTERM.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def run_bounded(command, *, output, timeout, grace=3.0, status_path=None,
                cwd=None, env=None, context=None):
    """Run a process in its own POSIX session and atomically record its outcome.

    The timer starts after spawn, including imports and native rendering setup.
    No in-process timer is trusted to interrupt a blocked native library.
    Timeout and cancellation always terminate the whole child process group.
    """
    from .common import write
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('Evaluation timeout must be finite and positive')
    if not math.isfinite(grace) or grace <= 0:
        raise ValueError('Termination grace must be finite and positive')
    started = time.perf_counter()
    process = subprocess.Popen(command, cwd=cwd, env=env, stdout=output,
                               stderr=subprocess.STDOUT, start_new_session=True)

    def record(status, **extra):
        if status_path is not None:
            write(Path(status_path), dict(status=status, pid=process.pid,
                elapsed_seconds=time.perf_counter() - started,
                timeout_seconds=timeout, context=context or {}, **extra))

    try:
        record('running')
        returncode = process.wait(timeout=timeout)
        if returncode:
            record('failed', returncode=returncode)
            raise subprocess.CalledProcessError(returncode, command)
        record('complete', returncode=returncode)
    except subprocess.TimeoutExpired as error:
        _stop_group(process, grace)
        record('timed_out', returncode=process.returncode,
               score_available=False, reason='External wall-clock deadline exceeded')
        raise EvaluationTimeoutError(
            f'Evaluation timed out after {timeout:g}s; no score produced. '
            f'Process status: {status_path}') from error
    except subprocess.CalledProcessError:
        _stop_group(process, grace)
        raise
    except BaseException:
        _stop_group(process, grace)
        record('cancelled', score_available=False, returncode=process.returncode)
        raise
