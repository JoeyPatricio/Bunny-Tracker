"""Port of server/lib/hiddenStore.js.

Set of recording filenames the owner has hidden from the public demo page.
Stored as a JSON array through the shared atomic/serialized store, so
concurrent hide/unhide toggles can't lose an update or corrupt the file.
"""
from app.config import SERVER_DIR
from app.lib.json_store import create_store

_store = create_store(SERVER_DIR, "hidden.json", [])


async def read_hidden() -> set:
    return set(await _store.read())


async def set_hidden(filename: str, hidden: bool) -> set:
    def mutate(lst):
        s = set(lst)
        if hidden:
            s.add(filename)
        else:
            s.discard(filename)
        return list(s)

    next_value = await _store.update(mutate)
    return set(next_value)
