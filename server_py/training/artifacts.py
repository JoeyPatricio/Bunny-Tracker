"""What counts as a deployable model artifact, and how it is hashed.

Shared by export_onnx.py (training venv, torch) and promote.py (runtime venv,
no torch), so it imports nothing heavier than the standard library.
"""
import hashlib
from pathlib import Path

MANIFEST_NAME = "manifest.json"
GATE_RESULT_NAME = "gate_result.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def is_artifact(path: Path) -> bool:
    """The .onnx graphs and their external-data sidecars. Anything else in a
    model directory (gate_result.json, manifests, head_state.pt) is
    provenance or training state, not something the agent loads."""
    return path.is_file() and (path.suffix == ".onnx" or path.name.endswith(".onnx.data"))


def artifacts_in(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.iterdir() if is_artifact(p))
