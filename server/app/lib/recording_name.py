"""Canonical recording-filename validator. Port of server/lib/recordingName.js.

Use this everywhere a filename arrives from a request; do not hand-roll a
prefix/suffix check. The character class excludes '/' and '\\' — the whole
point, since a loose startswith/endswith check would accept a path-traversal
filename like 'recording-x%2F..%2F..%2Fseg.webm'.
"""
import re

RECORDING_RE = re.compile(r"^recording-[A-Za-z0-9._-]+\.webm$")


def is_recording_filename(name) -> bool:
    return isinstance(name, str) and bool(RECORDING_RE.match(name))
