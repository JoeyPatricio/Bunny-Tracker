"""Model metadata for the read-only Training panel.

Originally a port of server/routes/model.js, which described the tfjs model in
server/model/. That model is dead: the agent loads the ONNX artifacts in
server_py/models/ and there is no code path to run the tfjs one from Python at
all. Reporting it meant the panel showed the mtime of a file the system does
not use and a backups list that retraining would never change (fixes.md 4.2).

This now describes what actually runs: exists/savedAt/labels as before, plus
valAcc and window, because otherwise the panel has almost nothing true to say.
`backups` is gone with the tfjs model that had them; TrainingStudio.jsx is the
only consumer of this route.
"""
import asyncio
import json

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.auth import is_authed
from app.config import MODELS_ONNX_DIR
from app.lib.iso_time import to_iso_millis

router = APIRouter()

BACKBONE_NAME = "backbone_int8.onnx"
HEAD_NAME = "head.onnx"


def _read_deployed_model() -> dict:
    backbone = MODELS_ONNX_DIR / BACKBONE_NAME
    head = MODELS_ONNX_DIR / HEAD_NAME
    if not backbone.is_file() or not head.is_file():
        return {"exists": False}

    # train_result.json carries no timestamp, so "last trained at" is the mtime
    # of the newest artifact the export actually wrote. Both files come out of
    # the same export_onnx.py run.
    saved_at = max(backbone.stat().st_mtime, head.stat().st_mtime)

    labels: list[str] = []
    val_acc = None
    window = None
    try:
        result = json.loads((MODELS_ONNX_DIR / "train_result.json").read_text(encoding="utf-8"))
        # per_class is keyed by shared/labels.py's LABELS, in the order the
        # classifier emits. Read it from here rather than importing shared into
        # the web tier, which app deliberately does not do anywhere else.
        labels = list(result.get("per_class") or {})
        val_acc = result.get("val_acc")
        window = result.get("window")
    except (OSError, json.JSONDecodeError):
        pass

    return {
        "exists": True,
        "savedAt": to_iso_millis(saved_at),
        "labels": labels,
        "valAcc": val_acc,
        "window": window,
    }


@router.get("")
@router.get("/")
async def get_model(request: Request):
    """PRIVATE: exposes the deployed model's label map and held-out accuracy."""
    if not is_authed(request):
        return JSONResponse(status_code=401, content={"error": "Login required"})
    try:
        return await asyncio.to_thread(_read_deployed_model)
    except OSError:
        return {"exists": False}
