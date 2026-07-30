"""Atomic, serialized JSON store for one file. Port of server/lib/jsonStore.js.

Works for either an object map or an array (pass the matching default_value).
All read-modify-write operations are serialized per file via an asyncio.Lock,
so two requests can never interleave and clobber each other, and every write
goes through a temp file plus rename so a crash mid-write can't leave a
half-written file.
"""
import asyncio
import copy
import json
import os
from pathlib import Path
from typing import Any, Callable


def _size(v: Any) -> int:
    return len(v)


def _is_empty(v: Any) -> bool:
    return _size(v) == 0


class JsonStore:
    def __init__(self, path: Path, default_value: Any, guard_wipe: bool = False):
        self.path = path
        self.tmp_path = path.with_name(path.name + ".tmp")
        self.bak_path = path.with_name(path.name + ".bak")
        self.default_value = default_value
        self.guard_wipe = guard_wipe
        self._lock = asyncio.Lock()

    async def read(self) -> Any:
        """Returns a copy of default_value ONLY when the file legitimately
        doesn't exist or is blank. On a parse error (a mid-write partial read
        or corruption) this RAISES, so callers abort instead of overwriting
        good data with an empty value."""
        try:
            raw = await asyncio.to_thread(self.path.read_text, encoding="utf-8")
        except FileNotFoundError:
            return copy.deepcopy(self.default_value)
        text = raw.lstrip("﻿").strip()  # strip BOM/whitespace
        if text == "":
            return copy.deepcopy(self.default_value)
        try:
            return json.loads(text)
        except json.JSONDecodeError as err:
            raise ValueError(f"{self.path.name} is corrupt, refusing to overwrite: {err}") from err

    async def _write_atomic(self, value: Any) -> None:
        text = json.dumps(value, indent=2)
        await asyncio.to_thread(self.tmp_path.write_text, text, encoding="utf-8")
        # os.replace(), not os.rename(): os.rename() raises FileExistsError on
        # Windows when the destination exists, unlike Node's fs.rename.
        await asyncio.to_thread(os.replace, self.tmp_path, self.path)

    async def update(self, mutate: Callable[[Any], Any]) -> Any:
        """Run a read-modify-write transaction safely and serially.
        `mutate(copy)` receives a copy of the current value and returns the new one.
        Keeps a .bak of the previous good state before each write."""
        async with self._lock:
            current = await self.read()
            next_value = mutate(copy.deepcopy(current))
            if self.guard_wipe and _size(current) > 5 and _is_empty(next_value):
                raise ValueError(f"Refusing to clear all of {self.path.name} in one operation")
            try:
                await asyncio.to_thread(_copy_file, self.path, self.bak_path)
            except FileNotFoundError:
                pass  # best-effort backup, matches the JS .catch(() => {})
            await self._write_atomic(next_value)
            return next_value


def _copy_file(src: Path, dst: Path) -> None:
    dst.write_bytes(src.read_bytes())


def create_store(server_dir: Path, filename: str, default_value: Any, guard_wipe: bool = False) -> JsonStore:
    return JsonStore(server_dir / filename, default_value, guard_wipe)
