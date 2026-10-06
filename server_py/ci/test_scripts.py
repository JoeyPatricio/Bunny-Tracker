"""pytest as a runner over the check-scripts, not a rewrite of them.

Each script in tests/ and training/ stays independently runnable with plain
`python`, exits 1 on any FAIL, and prints its own PASS/FAIL/SKIP lines. This
file runs each one as a subprocess under the venv it belongs to and asserts
exit 0, so there is one green/red signal without collecting the scripts
themselves. That matters: tests/ defines module-level `async def test_*`
functions, and pytest collecting those without an async plugin would report
them as skips that read like passes. pyproject.toml's `testpaths = ["ci"]`
keeps pytest out of tests/ for the same reason.

Usage (from server_py/; pytest is not in either venv, so run it with uvx):
    uvx --from pytest==9.1.1 pytest                       # everything
    uvx --from pytest==9.1.1 pytest -m "not local_data"   # what CI runs

Interpreters default to the two venvs beside this directory. Override with
BT_RUNTIME_PYTHON / BT_TRAIN_PYTHON to point at others.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

SERVER_PY = Path(__file__).resolve().parent.parent


def _venv_python(venv: str) -> Path:
    bindir = "Scripts" if sys.platform == "win32" else "bin"
    exe = "python.exe" if sys.platform == "win32" else "python"
    return SERVER_PY / venv / bindir / exe


PYTHONS = {
    "runtime": Path(os.environ.get("BT_RUNTIME_PYTHON") or _venv_python(".venv")),
    "train": Path(os.environ.get("BT_TRAIN_PYTHON") or _venv_python(".venv-train")),
}

# (script, venv). All hermetic today: each one seeds its own temp dirs or uses
# tests/fixtures/. test_security_regressions.py does need client/dist/, which a
# clean clone does not have, so build the client first (CI does). A script that
# genuinely needs real recordings or a trained model gets
# marks=pytest.mark.local_data so CI's `-m "not local_data"` leaves it out.
SCRIPTS = [
    ("tests/test_capture_loop.py", "runtime"),
    ("tests/test_clipping_mode.py", "runtime"),
    ("tests/test_motion_parity.py", "runtime"),
    ("tests/test_security_regressions.py", "runtime"),
    ("tests/test_server_robustness.py", "runtime"),
    ("training/test_export_guards.py", "train"),
    ("training/test_train_guards.py", "train"),
]


@pytest.mark.parametrize(
    ("script", "venv"),
    [pytest.param(s, v, id=Path(s).stem) for s, v in SCRIPTS],
)
def test_script(script: str, venv: str) -> None:
    python = PYTHONS[venv]
    # A missing interpreter fails rather than skips. A skip here would turn
    # "the training venv was never built" into a green run.
    assert python.is_file(), (
        f"{venv} interpreter not found at {python}. Build it per docs/pipeline.md, "
        f"or set BT_{venv.upper()}_PYTHON."
    )
    result = subprocess.run(
        [str(python), script],
        cwd=SERVER_PY,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, (
        f"{script} exited {result.returncode}\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )
