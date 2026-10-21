"""Port of server/routes/highlights.js."""
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.auth import is_authed
from app.lib.hidden_store import set_hidden
from app.lib.list_highlights import list_highlights
from app.lib.recording_name import is_recording_filename

router = APIRouter()


class HighlightBody(BaseModel):
    hidden: bool = False


@router.get("")
@router.get("/")
async def get_highlights(request: Request):
    """PRIVATE owner view: every auto-highlight candidate (non-normal labeled
    clip) with its current homepage visibility."""
    if not is_authed(request):
        return JSONResponse(status_code=401, content={"error": "Login required"})
    try:
        return {"highlights": await list_highlights(include_hidden=True)}
    except Exception as err:
        return JSONResponse(status_code=500, content={"error": "Failed to load highlights", "detail": str(err)})


@router.post("/{filename:path}")
async def set_highlight_visibility(filename: str, body: HighlightBody):
    """{ hidden: boolean } — show/hide on the demo."""
    if not is_recording_filename(filename):
        return JSONResponse(status_code=400, content={"error": "Invalid filename"})
    try:
        await set_hidden(filename, body.hidden)
        return {"filename": filename, "hidden": body.hidden}
    except Exception as err:
        return JSONResponse(status_code=500, content={"error": "Failed to update highlight", "detail": str(err)})
