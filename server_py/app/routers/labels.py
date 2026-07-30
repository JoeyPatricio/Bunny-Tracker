"""Port of server/routes/labels.js."""
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.auth import is_authed
from app.lib.label_store import read_auto_labels, read_labels, update_auto_labels, update_labels
from app.lib.recording_name import is_recording_filename
from app.lib.valid_labels import VALID_LABELS

router = APIRouter()


class LabelBody(BaseModel):
    label: str | None = None


@router.get("/valid")
async def get_valid_labels():
    return {"labels": VALID_LABELS}


@router.get("")
@router.get("/")
async def get_labels(request: Request):
    """PRIVATE: full map requires login. `labels` holds human-confirmed labels
    only (the training set); `suggestions` holds agent-predicted labels
    awaiting review in Label Studio."""
    if not is_authed(request):
        return JSONResponse(status_code=401, content={"error": "Login required"})
    try:
        labels = await read_labels()
        suggestions = await read_auto_labels()
        return {"labels": labels, "suggestions": suggestions}
    except Exception as err:
        return JSONResponse(status_code=500, content={"error": "Failed to read labels", "detail": str(err)})


@router.post("/{filename:path}/suggest")
async def suggest_label(filename: str, body: LabelBody):
    """Record the agent's predicted label as a review suggestion. Deliberately
    NOT written to labels.json: the model's own predictions must never feed
    back into its training data unreviewed."""
    if not is_recording_filename(filename):
        return JSONResponse(status_code=400, content={"error": "Invalid filename"})
    if body.label not in VALID_LABELS:
        return JSONResponse(status_code=400, content={"error": f"Invalid label: {body.label}"})
    try:
        await update_auto_labels(lambda auto: {**auto, filename: body.label})
        return {"filename": filename, "suggested": body.label}
    except Exception as err:
        return JSONResponse(status_code=500, content={"error": "Failed to save suggestion", "detail": str(err)})


@router.post("/{filename:path}")
async def set_label(filename: str, body: LabelBody):
    """Set label for a clip."""
    if not is_recording_filename(filename):
        return JSONResponse(status_code=400, content={"error": "Invalid filename"})
    if body.label not in VALID_LABELS:
        return JSONResponse(status_code=400, content={"error": f"Invalid label: {body.label}"})
    try:
        await update_labels(lambda labels: {**labels, filename: body.label})
        # The human has reviewed this clip — the agent's suggestion is resolved.
        try:
            await update_auto_labels(lambda auto: {k: v for k, v in auto.items() if k != filename})
        except Exception:
            pass
        return {"filename": filename, "label": body.label}
    except Exception as err:
        return JSONResponse(status_code=500, content={"error": "Failed to save label", "detail": str(err)})


@router.delete("/{filename:path}")
async def delete_label(filename: str):
    """Remove label (and any suggestion) from a clip."""
    if not is_recording_filename(filename):
        return JSONResponse(status_code=400, content={"error": "Invalid filename"})
    try:
        await update_labels(lambda labels: {k: v for k, v in labels.items() if k != filename})
        try:
            await update_auto_labels(lambda auto: {k: v for k, v in auto.items() if k != filename})
        except Exception:
            pass
        return {"deleted": filename}
    except Exception as err:
        return JSONResponse(status_code=500, content={"error": "Failed to delete label", "detail": str(err)})
