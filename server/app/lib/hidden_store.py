"""Explicit owner approvals for public highlights, stored atomically.

Legacy hidden.json is deliberately ignored: absence from a denylist is not
approval to publish an existing clip.
"""
from app.config import DATA_DIR
from app.lib.json_store import create_store
from app.lib.recording_name import is_recording_filename

_store = create_store(DATA_DIR, "approved_highlights.json", [])


def _approved_set(value) -> set[str]:
    if not isinstance(value, list) or not all(is_recording_filename(name) for name in value):
        raise ValueError("approved_highlights.json must contain an array of recording filenames")
    return set(value)


async def read_approved() -> set[str]:
    return _approved_set(await _store.read())


async def set_hidden(filename: str, hidden: bool) -> set[str]:
    def mutate(value):
        approved = _approved_set(value)
        if hidden:
            approved.discard(filename)
        else:
            approved.add(filename)
        return sorted(approved)

    return set(await _store.update(mutate))
