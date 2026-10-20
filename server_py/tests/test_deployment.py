"""Promoted window and metadata must drive runtime, independent of later training."""
import asyncio
import json
import os
import sys
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
    finish("deployment")
