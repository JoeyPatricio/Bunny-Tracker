"""Regression gate for the coding sidecars.

Covers labels.meta.json (coder, ethogram version, coded_at, uncertainty flag),
the frozen reliability sample, and labels.reliability.json. The property that
matters most is the last one: a blind re-code must never be able to modify the
first-pass labels it is measured against.

Hermetic. Every store in app/lib/label_store.py and the router's
RECORDINGS_DIR are repointed at a temp dir seeded with synthetic clips, so
nothing under server/ is read or written. Auth is patched open for the
functional checks and left alone for the auth checks, so this needs no
ADMIN_PASSWORD or AGENT_TOKEN.

Usage (from server_py/, runtime venv):
    .venv/bin/python tests/test_labeling_sidecars.py
"""
import asyncio
import json
import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient

from app import auth
from app.lib import backup_labels, label_store
from app.lib.reliability_sample import SAMPLE_FRACTION, draw_sample, sample_size
from app.main import app
from app.routers import labels as labels_router
from shared.labels import ETHOGRAM_VERSION
from tests.checks import check, finish

CLIPS = [f"recording-clip-2026-09-{d:02d}T10-00-00-000Z.webm" for d in range(1, 13)]


@contextmanager
def sandbox(clips=CLIPS):
    """Temp server dir + recordings dir; every label store repointed there."""
    tmp = Path(tempfile.mkdtemp(prefix="bunnylabels-"))
    rec_dir = tmp / "recordings"
    rec_dir.mkdir()
    for fn in clips:
        (rec_dir / fn).touch()
    saved = [(s, s.path, s.tmp_path, s.bak_path) for s in label_store.ALL_STORES]
    saved_rec = labels_router.RECORDINGS_DIR
    try:
        for s in label_store.ALL_STORES:
            s.path = tmp / s.path.name
            s.tmp_path = s.path.with_name(s.path.name + ".tmp")
            s.bak_path = s.path.with_name(s.path.name + ".bak")
        labels_router.RECORDINGS_DIR = rec_dir
        yield tmp
    finally:
        for s, p, t, b in saved:
            s.path, s.tmp_path, s.bak_path = p, t, b
        labels_router.RECORDINGS_DIR = saved_rec
        shutil.rmtree(tmp, ignore_errors=True)


@contextmanager
def logged_in():
    # The guard middleware and the GET handler each call is_authed by name.
    with patch.object(auth, "is_authed", lambda _req: True), \
            patch.object(labels_router, "is_authed", lambda _req: True):
        yield


def read(tmp: Path, name: str):
    path = tmp / name
    return json.loads(path.read_text()) if path.exists() else None


def label_all(client: TestClient, clips=CLIPS) -> None:
    for i, fn in enumerate(clips):
        client.post(f"/api/labels/{fn}", json={"label": "grooming" if i % 2 else "rearing", "coder": "joe"})


def test_meta_round_trip(client: TestClient) -> None:
    """Labeling writes provenance to labels.meta.json; labels.json stays flat."""
    with sandbox() as tmp, logged_in():
        fn = CLIPS[0]
        res = client.post(f"/api/labels/{fn}", json={"label": "rearing", "coder": "joe"})
        check("label save ok", res.status_code == 200, f"got {res.status_code} {res.text}")
        check("meta reported saved", res.json().get("metaSaved") is True, res.text)
        labels = read(tmp, "labels.json")
        check("labels.json stays a flat string map", labels == {fn: "rearing"}, str(labels))
        meta = read(tmp, "labels.meta.json") or {}
        entry = meta.get(fn, {})
        check("meta has the coder", entry.get("coder") == "joe", str(entry))
        check("meta has the ethogram version", entry.get("ethogram_version") == ETHOGRAM_VERSION, str(entry))
        check("meta has coded_at", isinstance(entry.get("coded_at"), str) and entry["coded_at"].endswith("Z"), str(entry))
        check("uncertain defaults to false", entry.get("uncertain") is False, str(entry))

        got = client.get("/api/labels").json()
        check("GET /api/labels exposes meta", got.get("meta", {}).get(fn, {}).get("coder") == "joe", str(got)[:200])
        check("GET /api/labels keeps the flat labels key", got.get("labels") == {fn: "rearing"}, str(got)[:200])

        client.delete(f"/api/labels/{fn}")
        check("deleting a label clears its meta", fn not in (read(tmp, "labels.meta.json") or {}))


def test_uncertain_flag(client: TestClient) -> None:
    """The uncertainty toggle round-trips and survives a relabel."""
    with sandbox() as tmp, logged_in():
        fn = CLIPS[1]
        client.post(f"/api/labels/{fn}", json={"label": "feeding", "coder": "joe"})
        res = client.post(f"/api/labels/meta/{fn}", json={"uncertain": True, "coder": "joe"})
        check("flag save ok", res.status_code == 200, f"got {res.status_code} {res.text}")
        check("flag set", (read(tmp, "labels.meta.json") or {}).get(fn, {}).get("uncertain") is True)
        check("flag did not touch the label", read(tmp, "labels.json") == {fn: "feeding"})

        client.post(f"/api/labels/{fn}", json={"label": "grooming", "coder": "joe"})
        check("flag survives a relabel", (read(tmp, "labels.meta.json") or {}).get(fn, {}).get("uncertain") is True)

        client.post(f"/api/labels/meta/{fn}", json={"uncertain": False})
        check("flag clears", (read(tmp, "labels.meta.json") or {}).get(fn, {}).get("uncertain") is False)

        res = client.post("/api/labels/meta/notes.txt", json={"uncertain": True})
        check("flag rejects a non-recording filename", res.status_code == 400, f"got {res.status_code}")


def test_guard_wipe() -> None:
    """The sidecars refuse to be emptied in one write, like labels.json."""
    with sandbox():
        for name, store in (("meta", label_store._meta_store), ("reliability", label_store._reliability_store)):
            check(f"{name} store has guard_wipe", store.guard_wipe is True)
            asyncio.run(store.update(lambda _m: {fn: {"x": 1} for fn in CLIPS[:6]}))
            try:
                asyncio.run(store.update(lambda _m: {}))
                check(f"{name} store refuses a full wipe", False, "wipe was allowed")
            except ValueError:
                check(f"{name} store refuses a full wipe", True)
            check(f"{name} store kept its data", len(asyncio.run(store.read())) == 6)


def test_sample_is_deterministic() -> None:
    """Seed 0, sorted pool: same clips every call and in a fresh process."""
    pool = [f"recording-{i:03d}.webm" for i in range(112)]
    a = draw_sample(pool)
    b = draw_sample(list(reversed(pool)))
    check("same sample regardless of pool order", a == b)
    check("112 clips gives 28", len(a) == 28, f"got {len(a)}")
    check("sample has no duplicates", len(set(a)) == len(a))
    check("sample is drawn from the pool", set(a) <= set(pool))

    code = (
        "import json,sys; sys.path.insert(0,'.');"
        "from app.lib.reliability_sample import draw_sample;"
        f"print(json.dumps(draw_sample({json.dumps(pool)})))"
    )
    out = subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).resolve().parent.parent,
                         capture_output=True, text=True)
    fresh = json.loads(out.stdout) if out.returncode == 0 else None
    check("same sample in a fresh process", fresh == a, out.stderr[-300:])

    check("fraction is 25%", SAMPLE_FRACTION == 0.25)
    sizes = {n: sample_size(n) for n in (1, 4, 10, 13, 14, 112)}
    check("rounding is ceil, never under 25%", sizes == {1: 1, 4: 1, 10: 3, 13: 4, 14: 4, 112: 28}, str(sizes))


def test_sample_route(client: TestClient) -> None:
    """The sample is refused until the first pass is done, then frozen."""
    with sandbox() as tmp, logged_in():
        label_all(client, CLIPS[:-1])
        res = client.post("/api/labels/reliability/sample", json={"coder": "joe"})
        check("draw refused with a clip unlabeled", res.status_code == 409, f"got {res.status_code}")
        state = client.get("/api/labels/reliability").json()
        check("state reports the unlabeled count", state.get("unlabeled") == 1 and state.get("canDraw") is False, str(state))

        label_all(client)
        res = client.post("/api/labels/reliability/sample", json={"coder": "joe"})
        check("draw allowed once complete", res.status_code == 200, f"got {res.status_code} {res.text}")
        sample = res.json().get("sample", {})
        check("sample matches draw_sample", sample.get("clips") == draw_sample(CLIPS), str(sample))
        check("sample is 25% of 12", len(sample.get("clips", [])) == 3)
        check("sample records seed and fraction", sample.get("seed") == 0 and sample.get("fraction") == 0.25)

        res = client.post("/api/labels/reliability/sample", json={"coder": "joe"})
        check("second draw refused", res.status_code == 409, f"got {res.status_code}")
        check("stored sample unchanged", read(tmp, "labels.reliability_sample.json") == sample)

        state = client.get("/api/labels/reliability").json()
        check("state returns the frozen sample", state.get("sample") == sample)
        check("state never exposes first-pass labels", "labels" not in state, str(sorted(state)))


def test_recode_never_touches_labels(client: TestClient) -> None:
    """Re-codes go to labels.reliability.json and nowhere else."""
    with sandbox() as tmp, logged_in():
        label_all(client)
        client.post("/api/labels/reliability/sample", json={"coder": "joe"})
        sample = read(tmp, "labels.reliability_sample.json")["clips"]
        before = (tmp / "labels.json").read_bytes()
        meta_before = (tmp / "labels.meta.json").read_bytes()

        target = sample[0]
        res = client.post(f"/api/labels/reliability/{target}", json={"label": "feeding", "coder": "joe"})
        check("re-code saves", res.status_code == 200, f"got {res.status_code} {res.text}")
        recodes = read(tmp, "labels.reliability.json") or {}
        check("re-code written", recodes.get(target, {}).get("label") == "feeding", str(recodes))
        check("re-code records coder and version",
              recodes.get(target, {}).get("coder") == "joe"
              and recodes.get(target, {}).get("ethogram_version") == ETHOGRAM_VERSION)
        check("labels.json byte-identical after re-code", (tmp / "labels.json").read_bytes() == before)
        check("labels.meta.json byte-identical after re-code", (tmp / "labels.meta.json").read_bytes() == meta_before)

        outside = next(fn for fn in CLIPS if fn not in sample)
        res = client.post(f"/api/labels/reliability/{outside}", json={"label": "feeding"})
        check("re-code refused outside the sample", res.status_code == 400, f"got {res.status_code}")
        res = client.post(f"/api/labels/reliability/{target}", json={"label": "not_a_label"})
        check("re-code refuses an invalid label", res.status_code == 400, f"got {res.status_code}")

        client.delete(f"/api/labels/reliability/{target}")
        check("re-code can be undone", target not in (read(tmp, "labels.reliability.json") or {}))
        check("labels.json still byte-identical", (tmp / "labels.json").read_bytes() == before)


def test_auth_required(client: TestClient) -> None:
    """Every new route is behind the admin guard."""
    with sandbox():
        fn = CLIPS[0]
        for method, path, body in (
            ("POST", f"/api/labels/meta/{fn}", {"uncertain": True}),
            ("GET", "/api/labels/reliability", None),
            ("POST", "/api/labels/reliability/sample", {}),
            ("POST", f"/api/labels/reliability/{fn}", {"label": "feeding"}),
            ("DELETE", f"/api/labels/reliability/{fn}", None),
        ):
            res = client.request(method, path, json=body)
            check(f"{method} {path.replace(fn, '<clip>')} is 401 without login", res.status_code == 401, f"got {res.status_code}")


def test_sidecars_are_backed_up() -> None:
    """The daily backup covers every sidecar, each with its own pruning."""
    tmp = Path(tempfile.mkdtemp(prefix="bunnybackup-"))
    try:
        server, backups = tmp / "server", tmp / "backups"
        server.mkdir()
        for name in backup_labels.BACKED_UP:
            (server / name).write_text(json.dumps({"x.webm": name}), encoding="utf-8")
        # Ten stale labels.json backups, plus one stale meta backup that a
        # loose `labels-` pattern would also have matched and pruned.
        backups.mkdir()
        for d in range(1, 11):
            (backups / f"labels-2026-01-{d:02d}.json").write_text("{}")
        (backups / "labels.meta-2026-01-01.json").write_text("{}")
        with patch.object(backup_labels, "SERVER_DIR", server), \
                patch.object(backup_labels, "LABELS_BACKUP_DIR", backups):
            asyncio.run(backup_labels._make_backup())
        names = sorted(f.name for f in backups.iterdir())
        for name in backup_labels.BACKED_UP:
            stem = name.removesuffix(".json")
            check(f"{name} backed up today", any(n.startswith(f"{stem}-2") and "2026-01" not in n for n in names), str(names))
        check("labels.json series pruned to KEEP",
              sum(1 for n in names if n.startswith("labels-")) == backup_labels.KEEP, str(names))
        check("pruning one series never touches another", "labels.meta-2026-01-01.json" in names, str(names))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> None:
    # No `with` block: entering the lifespan would start the label-backup
    # task, which writes to server/backups/.
    client = TestClient(app)
    for fn, needs_client in (
        (test_meta_round_trip, True),
        (test_uncertain_flag, True),
        (test_guard_wipe, False),
        (test_sample_is_deterministic, False),
        (test_sample_route, True),
        (test_recode_never_touches_labels, True),
        (test_auth_required, True),
        (test_sidecars_are_backed_up, False),
    ):
        print(f"\n{fn.__doc__.splitlines()[0]}")
        fn(client) if needs_client else fn()
    finish("labeling sidecar")


if __name__ == "__main__":
    main()
