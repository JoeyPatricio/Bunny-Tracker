"""Promoted window and metadata must drive runtime, independent of later training."""
import asyncio
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent import capture
from app.routers import model
from inference.deployment import read_promoted
from shared.labels import LABELS
from tests.checks import check, finish


def exercise_model_bundle(root):
    source = root / "bundle-source"
    source.mkdir()
    artifacts = {"backbone_int8.onnx": b"backbone", "head.onnx": b"head graph",
                 "head.onnx.data": b"external head weights"}
    for name, data in artifacts.items():
        (source / name).write_bytes(data)
    manifest = {"labels": LABELS, "window": 12, "view": "full", "artifacts": {
        name: hashlib.sha256(data).hexdigest() for name, data in artifacts.items()
    }}
    (source / "promoted.json").write_text(json.dumps({"manifest": manifest}))
    tool = Path(__file__).resolve().parents[2] / "deploy" / "pi" / "model_bundle.py"
    archive = root / "model.tar.gz"

    def run(*args):
        return subprocess.run([sys.executable, str(tool), *map(str, args)],
                              capture_output=True, text=True, timeout=10)

    result = run("pack", source, archive)
    check("model bundle packs a promotion and its external weights", result.returncode == 0,
          result.stderr)
    if result.returncode != 0:
        return
    destination = root / "bundle-destination"
    destination.mkdir()
    with tarfile.open(archive) as bundle:
        check("model bundle includes every manifest artifact and the promotion record",
              set(bundle.getnames()) == {*artifacts, "promoted.json"})
        bundle.extractall(destination, filter="data")
    result = run("verify", destination)
    check("transferred bundle verifies independently of training files", result.returncode == 0,
          result.stderr)
    (destination / "head.onnx.data").unlink()
    result = run("verify", destination)
    check("model bundle refuses missing external weights", result.returncode != 0
          and "missing head.onnx.data" in result.stderr, result.stderr)
    (destination / "head.onnx.data").write_bytes(b"wrong weights")
    result = run("verify", destination)
    check("model bundle refuses changed external weights", result.returncode != 0
          and "sha256 mismatch for head.onnx.data" in result.stderr, result.stderr)


async def exercise(root):
    check("absent promotion has no deployment record", read_promoted(root) is None)
    record = {"manifest": {"labels": LABELS, "window": 12, "view": "full", "cv_arm": "gru@albu_v1"},
              "gate_result": {"cv": {"gru_macro_f1": 0.61}}}
    (root / "promoted.json").write_text(json.dumps(record))
    (root / "head.onnx").touch()
    (root / "backbone_int8.onnx").touch()
    # A newer candidate must not change what the deployed panel describes.
    (root / "train_result.json").write_text(json.dumps({"window": 4, "val_acc": 1.0}))
    with patch.object(model, "MODELS_ONNX_DIR", root):
        result = model._read_deployed_model()
    check("panel reads promoted labels and window", result["labels"] == LABELS and result["window"] == 12)
    check("panel shows recorded OOF score instead of a later selection score",
          result["cvMacroF1"] == 0.61 and "valAcc" not in result)
    check("panel records the deployed arm", result["arm"] == "gru@albu_v1")
    with patch.object(capture, "MODELS_DIR", root), \
            patch.object(capture, "Backbone", return_value=SimpleNamespace()), \
            patch.object(capture, "TemporalHead", return_value=SimpleNamespace(num_classes=len(LABELS))), \
            patch.object(capture, "AgentClient", return_value=SimpleNamespace(close=AsyncMock())), \
            patch.dict(os.environ):
        os.environ.pop("AGENT_EMBED_WINDOW", None)
        agent = capture.Agent()
        try:
            await agent.load_models()
            check("agent uses the promoted twelve-frame window", agent.embed_window == 12
                  and agent.decision_loop.predictor.window == 12)
        finally:
            await agent.client.close()
        os.environ["AGENT_EMBED_WINDOW"] = "8"
        agent = capture.Agent()
        try:
            await agent.load_models()
            check("conflicting window override disables incompatible inference",
                  agent.decision_loop.predictor is None)
        finally:
            await agent.client.close()
        (root / "promoted.json").unlink()
        agent = capture.Agent()
        try:
            await agent.load_models()
            check("unpromoted graphs cannot enable classification", agent.decision_loop.predictor is None)
            with patch.object(model, "MODELS_ONNX_DIR", root):
                check("panel also reports no promoted model", model._read_deployed_model() == {"exists": False})
        finally:
            await agent.client.close()
    for name, changed in (("crop", {"view": "crop"}), ("stale vocabulary", {"labels": ["normal"]}),
                          ("zero window", {"window": 0})):
        bad = {**record, "manifest": {**record["manifest"], **changed}}
        (root / "promoted.json").write_text(json.dumps(bad))
        try:
            read_promoted(root)
            check(f"runtime refuses {name}", False)
        except ValueError:
            check(f"runtime refuses {name}", True)


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="bunny-deployment-") as directory:
        asyncio.run(exercise(Path(directory)))
        exercise_model_bundle(Path(directory))
    finish("deployment")
