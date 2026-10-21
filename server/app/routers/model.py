"""Model metadata for the read-only Training panel.

Originally a port of server/routes/model.js, which described the tfjs model in
data/model/. That model is dead: the agent loads the ONNX artifacts in
server/models/ and there is no code path to run the tfjs one from Python at
all. Reporting it meant the panel showed the mtime of a file the system does
not use and a backups list that retraining would never change.

This describes the promoted artifacts and their recorded OOF headline.
`backups` is gone with the tfjs model that had them; TrainingStudio.jsx is the
only consumer of this route.
"""
import asyncio

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.auth import is_authed
from app.config import MODELS_ONNX_DIR
from app.lib.iso_time import to_iso_millis
from inference.deployment import read_promoted

router = APIRouter()

BACKBONE_NAME = "backbone_int8.onnx"
HEAD_NAME = "head.onnx"


def _read_deployed_model() -> dict:
    backbone = MODELS_ONNX_DIR / BACKBONE_NAME
    head = MODELS_ONNX_DIR / HEAD_NAME
    if not backbone.is_file() or not head.is_file():
        return {"exists": False}

    promoted = read_promoted(MODELS_ONNX_DIR)
    if promoted is None:
        return {"exists": False}
    manifest = promoted["manifest"]
    cv = (promoted.get("gate_result") or {}).get("cv") or {}
    saved_at = max(backbone.stat().st_mtime, head.stat().st_mtime)

    return {
        "exists": True,
        "savedAt": to_iso_millis(saved_at),
        "labels": manifest["labels"],
        "cvMacroF1": cv.get("gru_macro_f1"),
        "window": manifest["window"],
        "arm": manifest.get("cv_arm"),
    }


@router.get("")
@router.get("/")
async def get_model(request: Request):
    if not is_authed(request):
        return JSONResponse(status_code=401, content={"error": "Login required"})
    try:
        return await asyncio.to_thread(_read_deployed_model)
    except (OSError, ValueError, KeyError) as err:
        return JSONResponse(status_code=500, content={"error": f"Cannot read deployed model metadata: {err}"})
