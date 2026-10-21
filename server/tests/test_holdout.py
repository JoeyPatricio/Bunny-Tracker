"""Regression gate for the prospective temporal holdout
(docs/ethogram.md "Prospective temporal holdout").

The property that matters: once a cut date is set, a clip captured on or after
it can be coded but its label never lands in labels.json, which is the only
file any training, selection or calibration code reads. With the cut unset,
labeling behaves exactly as before.

Hermetic. Every label store and the router's RECORDINGS_DIR are repointed at a
temp dir seeded with synthetic clips, and holdout.HOLDOUT_CUTOFF is patched
per check, so nothing under data/ or data/.env is read or written.

Usage (from server/, runtime venv):
    .venv/bin/python tests/test_holdout.py
"""
import asyncio
import json
import os
import shutil
import sys
import tempfile
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient

from app import auth
from app.lib import backup_labels, clip_budget, holdout, label_store
from app.main import app
from app.routers import labels as labels_router
from tests.checks import check, finish

CUT = datetime(2026, 9, 15, tzinfo=UTC)
PRE = "recording-clip-2026-09-10T10-00-00-000Z.webm"
POST = "recording-clip-2026-09-20T10-00-00-000Z.webm"
AT_CUT = "recording-clip-2026-09-15T00-00-00-000Z.webm"
JUST_BEFORE = "recording-clip-2026-09-14T23-59-59-999Z.webm"
UNSTAMPED_OLD = "recording-import-old.webm"
UNSTAMPED_NEW = "recording-import-new.webm"
CLIPS = [PRE, POST, AT_CUT, JUST_BEFORE, UNSTAMPED_OLD, UNSTAMPED_NEW]


@contextmanager
def sandbox(cutoff):
    """Temp data + recordings dir, every label store repointed there, and
    the cut date set to `cutoff` (None = the placeholder state)."""
    tmp = Path(tempfile.mkdtemp(prefix="bunnyholdout-"))
    rec_dir = tmp / "recordings"
    rec_dir.mkdir()
    for fn in CLIPS:
        (rec_dir / fn).touch()
    # Unstamped names fall back to mtime: one well before the cut, one after.
    os.utime(rec_dir / UNSTAMPED_OLD, (CUT.timestamp() - 86400,) * 2)
    os.utime(rec_dir / UNSTAMPED_NEW, (CUT.timestamp() + 86400,) * 2)
    saved = [(s, s.path, s.tmp_path, s.bak_path) for s in label_store.ALL_STORES]
    saved_rec = labels_router.RECORDINGS_DIR
    try:
        for s in label_store.ALL_STORES:
            s.path = tmp / s.path.name
            s.tmp_path = s.path.with_name(s.path.name + ".tmp")
            s.bak_path = s.path.with_name(s.path.name + ".bak")
        labels_router.RECORDINGS_DIR = rec_dir
        with patch.object(holdout, "HOLDOUT_CUTOFF", cutoff), \
                patch.object(auth, "is_authed", lambda _req: True), \
                patch.object(labels_router, "is_authed", lambda _req: True):
            yield tmp
    finally:
        for s, p, t, b in saved:
            s.path, s.tmp_path, s.bak_path = p, t, b
        labels_router.RECORDINGS_DIR = saved_rec
        shutil.rmtree(tmp, ignore_errors=True)


def read(tmp: Path, name: str) -> dict:
    path = tmp / name
    return json.loads(path.read_text()) if path.exists() else {}


def post(client: TestClient, fn: str, label: str = "grooming"):
    return client.post(f"/api/labels/{fn}", json={"label": label, "coder": "joe"})


def test_parse_cutoff() -> None:
    """HOLDOUT_CUTOFF parsing: unset is the placeholder, a typo is loud but not fatal."""
    check("unset is no cut", holdout.parse_cutoff(None) is None)
    check("blank is no cut", holdout.parse_cutoff("  ") is None)
    check("a date means midnight UTC", holdout.parse_cutoff("2026-09-15") == CUT)
    check("a full ISO timestamp is accepted",
          holdout.parse_cutoff("2026-09-15T00:00:00Z") == CUT)
    check("an offset is normalised to UTC",
          holdout.parse_cutoff("2026-09-14T17:00:00-07:00") == CUT)
    check("garbage is no cut, not a crash", holdout.parse_cutoff("next tuesday") is None)


def test_unset_changes_nothing(client: TestClient) -> None:
    """With the cut unset, every label goes to labels.json as before."""
    with sandbox(None) as tmp:
        for fn in CLIPS:
            res = post(client, fn)
            check(f"{fn} saved", res.status_code == 200, f"got {res.status_code}")
        check("every clip is in labels.json", set(read(tmp, "labels.json")) == set(CLIPS))
        check("no holdout file is written", not (tmp / "labels.holdout.json").exists())
        body = client.get("/api/labels").json()
        check("GET reports no cut", body.get("holdoutCutoff") is None, str(body.get("holdoutCutoff")))
        check("GET holdout map is empty", body.get("holdout") == {}, str(body.get("holdout")))


def test_cut_routes_labels(client: TestClient) -> None:
    """With a cut set, post-cut labels land only in labels.holdout.json."""
    with sandbox(CUT) as tmp:
        for fn in CLIPS:
            post(client, fn)
        labels = read(tmp, "labels.json")
        held = read(tmp, "labels.holdout.json")
        check("pre-cut clip is in labels.json", PRE in labels and PRE not in held)
        check("post-cut clip is in the holdout file", POST in held)
        check("labels.json never contains a post-cut clip",
              not ({POST, AT_CUT, UNSTAMPED_NEW} & set(labels)), str(sorted(labels)))
        check("the two files never share a clip", not (set(labels) & set(held)))
        check("holdout file keeps the flat schema", held.get(POST) == "grooming", str(held.get(POST)))
        meta = read(tmp, "labels.meta.json")
        check("meta marks the clip as holdout", meta.get(POST, {}).get("holdout") is True)
        check("meta marks a pool clip as not holdout", meta.get(PRE, {}).get("holdout") is False)
        body = client.get("/api/labels").json()
        check("GET reports the cut", body.get("holdoutCutoff") == CUT.isoformat(), str(body.get("holdoutCutoff")))
        check("GET marks holdout labels separately", POST in body.get("holdout", {}) and POST not in body["labels"])
        check("POST response says holdout", post(client, POST).json().get("holdout") is True)


def test_boundary_is_inclusive(client: TestClient) -> None:
    """A clip captured exactly at the cut is holdout; one millisecond earlier is pool."""
    with sandbox(CUT) as tmp:
        post(client, AT_CUT)
        post(client, JUST_BEFORE)
        check("exactly at the cut -> holdout", AT_CUT in read(tmp, "labels.holdout.json"))
        check("1 ms before the cut -> labels.json", JUST_BEFORE in read(tmp, "labels.json"))


def test_unstamped_names_use_mtime(client: TestClient) -> None:
    """No timestamp in the name: fall back to mtime; no file either: holdout."""
    with sandbox(CUT) as tmp:
        post(client, UNSTAMPED_OLD)
        post(client, UNSTAMPED_NEW)
        check("old mtime -> labels.json", UNSTAMPED_OLD in read(tmp, "labels.json"))
        check("new mtime -> holdout", UNSTAMPED_NEW in read(tmp, "labels.holdout.json"))
        check("unknown capture time is treated as holdout",
              holdout.is_holdout("recording-import-gone.webm", CUT, tmp / "recordings"))


def test_no_bulk_move(client: TestClient) -> None:
    """Setting a cut moves nothing; only an explicit re-label moves that one clip."""
    with sandbox(None) as tmp:
        post(client, PRE)
        post(client, POST)  # coded before any cut existed
        before = read(tmp, "labels.json")
        with patch.object(holdout, "HOLDOUT_CUTOFF", CUT):
            client.get("/api/labels")
            check("setting a cut rewrites nothing", read(tmp, "labels.json") == before)
            check("and creates no holdout file", not (tmp / "labels.holdout.json").exists())
            post(client, POST, "rearing")
            labels = read(tmp, "labels.json")
            check("re-labeling a post-cut clip removes it from labels.json", POST not in labels)
            check("and puts it in the holdout file", read(tmp, "labels.holdout.json").get(POST) == "rearing")
            check("other labels.json entries are untouched", labels.get(PRE) == before[PRE])
            client.delete(f"/api/labels/{POST}")
            check("delete clears the holdout file too", POST not in read(tmp, "labels.holdout.json"))


def test_holdout_does_not_block_pool_work(client: TestClient) -> None:
    """Post-cut clips neither block the reliability draw nor fill the backlog cap."""
    with sandbox(CUT) as tmp:
        for fn in (PRE, JUST_BEFORE, UNSTAMPED_OLD):
            post(client, fn)
        rel = client.get("/api/labels/reliability").json()
        check("uncoded post-cut clips do not count as pool backlog", rel.get("unlabeled") == 0, str(rel))
        check("the re-code sample can be drawn", rel.get("canDraw") is True, str(rel))
        drawn = client.post("/api/labels/reliability/sample", json={"coder": "joe"}).json()
        clips = set(drawn.get("sample", {}).get("clips", []))
        check("the sample never includes a holdout clip",
              not (clips & {POST, AT_CUT, UNSTAMPED_NEW}), str(sorted(clips)))
        post(client, POST)
        usage = asyncio.run(clip_budget.recordings_usage(
            recordings_dir=tmp / "recordings", max_unlabeled=0, min_free_gb=0))
        check("a coded holdout clip is not backlog",
              usage["unlabeled"] == 2, f"{usage['unlabeled']} unlabeled, expected AT_CUT and UNSTAMPED_NEW")


def test_holdout_is_backed_up() -> None:
    """labels.holdout.json is in the daily backup set."""
    check("in BACKED_UP", "labels.holdout.json" in backup_labels.BACKED_UP)
    tmp = Path(tempfile.mkdtemp(prefix="bunnyholdoutbak-"))
    try:
        data_dir, backups = tmp / "data", tmp / "backups"
        data_dir.mkdir()
        (data_dir / "labels.holdout.json").write_text(json.dumps({POST: "grooming"}))
        with patch.object(backup_labels, "DATA_DIR", data_dir), \
                patch.object(backup_labels, "LABELS_BACKUP_DIR", backups):
            asyncio.run(backup_labels._make_backup())
        names = [f.name for f in backups.iterdir()] if backups.exists() else []
        check("a dated holdout backup is written",
              any(n.startswith("labels.holdout-") for n in names), str(names))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> None:
    # No `with` block: entering the lifespan would start the label-backup
    # task, which writes to data/backups/.
    client = TestClient(app)
    for fn, needs_client in (
        (test_parse_cutoff, False),
        (test_unset_changes_nothing, True),
        (test_cut_routes_labels, True),
        (test_boundary_is_inclusive, True),
        (test_unstamped_names_use_mtime, True),
        (test_no_bulk_move, True),
        (test_holdout_does_not_block_pool_work, True),
        (test_holdout_is_backed_up, False),
    ):
        print(f"\n{fn.__doc__.splitlines()[0]}")
        fn(client) if needs_client else fn()
    finish("holdout")


if __name__ == "__main__":
    main()
