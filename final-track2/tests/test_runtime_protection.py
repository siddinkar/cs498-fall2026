"""Hard budgets, honest incomplete results, and real process-group timeouts."""

import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from evaluation.common import read
from evaluation.subprocess_utils import EvaluationTimeoutError, run_bounded

ROOT = Path(__file__).resolve().parents[1]


def test_bounded_process_success_and_failure(tmp_path):
    status = tmp_path / "process.json"
    with (tmp_path / "run.log").open("w") as output:
        run_bounded(
            [sys.executable, "-c", 'print("ok")'],
            output=output,
            timeout=5,
            status_path=status,
        )
        assert read(status)["status"] == "complete"
        with pytest.raises(subprocess.CalledProcessError):
            run_bounded(
                [sys.executable, "-c", "raise SystemExit(7)"],
                output=output,
                timeout=5,
                status_path=status,
            )
    assert read(status)["status"] == "failed"
    assert read(status)["returncode"] == 7


@pytest.mark.skipif(os.name != "posix", reason="The evaluator runs on Linux/POSIX")
@pytest.mark.parametrize("parent_ignores_sigterm", [False, True])
def test_real_timeout_kills_sigterm_resistant_child_even_if_parent_exits(
    tmp_path, parent_ignores_sigterm
):
    child_pid = tmp_path / "child.pid"
    child_code = (
        "import os,signal,time,pathlib; "
        "signal.signal(signal.SIGTERM,signal.SIG_IGN); "
        f"pathlib.Path({str(child_pid)!r}).write_text(str(os.getpid())); "
        "time.sleep(60)"
    )
    parent_signal = (
        "signal.signal(signal.SIGTERM,signal.SIG_IGN); "
        if parent_ignores_sigterm
        else ""
    )
    parent_code = (
        "import subprocess,sys,signal,time; "
        + parent_signal
        + f'subprocess.Popen([sys.executable,"-c",{child_code!r}]); '
        "time.sleep(60)"
    )
    status = tmp_path / "process.json"
    started = time.monotonic()
    with (tmp_path / "timeout.log").open("w") as output:
        with pytest.raises(EvaluationTimeoutError, match="no score produced"):
            run_bounded(
                [sys.executable, "-c", parent_code],
                output=output,
                timeout=1,
                grace=0.1,
                status_path=status,
            )
    assert time.monotonic() - started < 5
    result = read(status)
    assert result["status"] == "timed_out"
    assert result["score_available"] is False
    assert child_pid.is_file(), "Fixture must spawn the child before timing out"
    pid = int(child_pid.read_text())
    # A killed descendant may briefly remain a zombie pending init's reap.
    for _ in range(100):
        path = Path(f"/proc/{pid}/stat")
        try:
            state = path.read_text().split()[2]
        except (FileNotFoundError, ProcessLookupError):
            break
        if state == "Z":
            break
        time.sleep(0.01)
    else:
        pytest.fail("Timeout left a running child process")


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
def test_invalid_timeout_never_spawns_a_process(tmp_path, timeout):
    with (tmp_path / "log").open("w") as output:
        with pytest.raises(ValueError, match="finite and positive"):
            run_bounded(
                [sys.executable, "-c", "raise SystemExit(99)"],
                output=output,
                timeout=timeout,
            )
