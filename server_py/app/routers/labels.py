"""Port of server/routes/labels.js, plus the Phase 2 coding sidecars.

Route order matters here. The bare `/{filename:path}` routes match anything,
so every fixed-prefix route (`/meta/...`, `/reliability/...`) is registered
above them. A recording filename always starts with `recording-`, so the two
can never collide in the other direction.
"""
import asyncio
import logging
from datetime import UTC, datetime

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.auth import is_authed
from app.config import RECORDINGS_DIR
from app.lib.label_store import (
    read_auto_labels,
    read_label_meta,
    read_labels,
    read_reliability,
    read_reliability_sample,
    update_auto_labels,
    update_label_meta,
    update_labels,
    update_reliability,
    update_reliability_sample,
)
from app.lib.recording_name import is_recording_filename
from app.lib.reliability_sample import SAMPLE_FRACTION, SAMPLE_SEED, draw_sample
from app.lib.valid_labels import VALID_LABELS
from shared.labels import ETHOGRAM_VERSION

log = logging.getLogger(__name__)

router = APIRouter()


class LabelBody(BaseModel):
    label: str | None = None
    # Optional so older clients and scripts that send only {label} keep
    # working; the studio always sends it.
    coder: str | None = None


class MetaBody(BaseModel):
    uncertain: bool
    coder: str | None = None


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _clean_coder(coder: str | None) -> str | None:
    if coder is None:
        return None
    coder = coder.strip()[:64]
    return coder or None


def _list_recordings() -> list[str]:
    if not RECORDINGS_DIR.exists():
        return []
    return [p.name for p in RECORDINGS_DIR.iterdir() if is_recording_filename(p.name)]


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
        meta = await read_label_meta()
        return {"labels": labels, "suggestions": suggestions, "meta": meta}
    except Exception as err:
        return JSONResponse(status_code=500, content={"error": "Failed to read labels", "detail": str(err)})


@router.post("/meta/{filename:path}")
async def set_label_meta(filename: str, body: MetaBody):
    """Set the uncertainty flag. A toggle, not a label: an uncertain clip keeps
    its label and stays in the dataset, so the paper can report "n flagged
    uncertain" and the clips stay findable for a second look."""
    if not is_recording_filename(filename):
        return JSONResponse(status_code=400, content={"error": "Invalid filename"})
    coder = _clean_coder(body.coder)

    def mutate(meta):
        entry = dict(meta.get(filename) or {})
        entry["uncertain"] = body.uncertain
        entry.setdefault("ethogram_version", ETHOGRAM_VERSION)
        if coder and not entry.get("coder"):
            entry["coder"] = coder
        entry.setdefault("coded_at", None)
        return {**meta, filename: entry}

    try:
        meta = await update_label_meta(mutate)
        return {"filename": filename, "meta": meta[filename]}
    except Exception as err:
        return JSONResponse(status_code=500, content={"error": "Failed to save metadata", "detail": str(err)})


@router.get("/reliability")
async def get_reliability():
    """The re-code sample (or null if not drawn), the re-codes so far, and
    whether a draw is currently allowed. Never returns first-pass labels: the
    studio's blind mode must not need them, so it cannot leak them."""
    try:
        sample = await read_reliability_sample()
        recodes = await read_reliability()
        labels = await read_labels()
        recordings = await asyncio.to_thread(_list_recordings)
    except Exception as err:
        return JSONResponse(status_code=500, content={"error": "Failed to read reliability state", "detail": str(err)})
    unlabeled = sum(1 for fn in recordings if fn not in labels)
    return {
        "sample": sample or None,
        "recodes": recodes,
        "unlabeled": unlabeled,
        "canDraw": not sample and unlabeled == 0 and len(recordings) > 0,
    }


@router.post("/reliability/sample")
async def draw_reliability_sample(body: LabelBody | None = None):
    """Draw the re-code sample. Once only.

    Refused while any clip lacks a first-pass label. That is the stricter of
    the two options and it is deliberate: a sample drawn from a partial pool
    silently excludes every clip coded afterwards, and re-coding interleaved
    with first-pass coding lets recent decisions leak into the "blind" pass.
    """
    try:
        existing = await read_reliability_sample()
        if existing:
            return JSONResponse(status_code=409, content={"error": "Sample already drawn; it is never redrawn", "sample": existing})
        labels = await read_labels()
        recordings = await asyncio.to_thread(_list_recordings)
    except Exception as err:
        return JSONResponse(status_code=500, content={"error": "Failed to read labels", "detail": str(err)})
    unlabeled = [fn for fn in recordings if fn not in labels]
    if unlabeled:
        return JSONResponse(
            status_code=409,
            content={"error": f"Finish the first pass first: {len(unlabeled)} clip(s) still unlabeled"},
        )
    pool = [fn for fn in recordings if fn in labels]
    if not pool:
        return JSONResponse(status_code=409, content={"error": "No labeled clips to sample from"})
    sample = {
        "seed": SAMPLE_SEED,
        "fraction": SAMPLE_FRACTION,
        "pool_size": len(pool),
        "ethogram_version": ETHOGRAM_VERSION,
        "drawn_at": _now_iso(),
        "drawn_by": _clean_coder(body.coder) if body else None,
        "clips": draw_sample(pool),
    }

    def mutate(current):
        # Re-checked under the store's lock so two concurrent draws cannot
        # both write.
        if current:
            raise ValueError("Sample already drawn")
        return sample

    try:
        await update_reliability_sample(mutate)
    except ValueError as err:
        return JSONResponse(status_code=409, content={"error": str(err)})
    except Exception as err:
        return JSONResponse(status_code=500, content={"error": "Failed to save sample", "detail": str(err)})
    return {"sample": sample}


@router.post("/reliability/{filename:path}")
async def set_reliability_label(filename: str, body: LabelBody):
    """Record a blind re-code. Writes labels.reliability.json only, never
    labels.json, and only for clips in the frozen sample."""
    if not is_recording_filename(filename):
        return JSONResponse(status_code=400, content={"error": "Invalid filename"})
    if body.label not in VALID_LABELS:
        return JSONResponse(status_code=400, content={"error": f"Invalid label: {body.label}"})
    try:
        sample = await read_reliability_sample()
    except Exception as err:
        return JSONResponse(status_code=500, content={"error": "Failed to read sample", "detail": str(err)})
    if not sample or filename not in sample.get("clips", []):
        return JSONResponse(status_code=400, content={"error": "Clip is not in the reliability sample"})
    entry = {
        "label": body.label,
        "coder": _clean_coder(body.coder),
        "coded_at": _now_iso(),
        "ethogram_version": ETHOGRAM_VERSION,
    }
    try:
        await update_reliability(lambda recodes: {**recodes, filename: entry})
        return {"filename": filename, "recode": entry}
    except Exception as err:
        return JSONResponse(status_code=500, content={"error": "Failed to save re-code", "detail": str(err)})


@router.delete("/reliability/{filename:path}")
async def delete_reliability_label(filename: str):
    """Undo a re-code, for a mis-click. The sample itself is untouched."""
    if not is_recording_filename(filename):
        return JSONResponse(status_code=400, content={"error": "Invalid filename"})
    try:
        await update_reliability(lambda recodes: {k: v for k, v in recodes.items() if k != filename})
        return {"deleted": filename}
    except Exception as err:
        return JSONResponse(status_code=500, content={"error": "Failed to delete re-code", "detail": str(err)})


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
        # Provenance after the label, and best-effort: the label is the
        # training data and has already saved, so a sidecar failure must not
        # read as a failed save. It is reported in the response instead, so
        # the studio can say so rather than drop it silently.
        coder = _clean_coder(body.coder)

        def stamp(meta):
            entry = dict(meta.get(filename) or {})
            entry["ethogram_version"] = ETHOGRAM_VERSION
            entry["coder"] = coder
            entry["coded_at"] = _now_iso()
            entry.setdefault("uncertain", False)
            return {**meta, filename: entry}

        try:
            await update_label_meta(stamp)
            meta_saved = True
        except Exception as err:
            log.warning("labels.meta.json not updated for %s: %s", filename, err)
            meta_saved = False
        return {"filename": filename, "label": body.label, "metaSaved": meta_saved}
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
        # Provenance describes a label that no longer exists.
        try:
            await update_label_meta(lambda meta: {k: v for k, v in meta.items() if k != filename})
        except Exception as err:
            log.warning("labels.meta.json not cleared for %s: %s", filename, err)
        return {"deleted": filename}
    except Exception as err:
        return JSONResponse(status_code=500, content={"error": "Failed to delete label", "detail": str(err)})
