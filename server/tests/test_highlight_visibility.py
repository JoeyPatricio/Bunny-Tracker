"""Public highlights require persisted, explicit owner approval."""
import json
import sys
import tempfile
import time
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient

from app import auth, main
from app.lib import hidden_store, list_highlights
from app.lib.json_store import create_store
from tests.checks import check, finish

CLIP = "recording-highlight.webm"
OTHER = "recording-other.webm"
RESTING = "recording-resting.webm"
UNLABELED = "recording-unlabeled.webm"


def main_test():
    with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
        root = Path(tmp)
        for name in (CLIP, OTHER, RESTING, UNLABELED):
            (root / name).write_bytes(b"test video")
        legacy = root / "hidden.json"
        legacy.write_text(json.dumps([OTHER]))
        store = create_store(root, "approved_highlights.json", [])
        labels = {CLIP: "locomotion", OTHER: "locomotion", RESTING: "resting_lying"}
        stack.enter_context(patch.object(hidden_store, "_store", store))
        for module in (main, list_highlights):
            stack.enter_context(patch.object(module, "RECORDINGS_DIR", root))
            stack.enter_context(patch.object(module, "read_labels", AsyncMock(return_value=labels)))
        stack.enter_context(patch.dict(auth._sessions, {"visibility-test": time.time() + 3600}))
        public = TestClient(main.app)
        owner = TestClient(main.app, cookies={auth.COOKIE_NAME: "visibility-test"})
        stack.callback(public.close)
        stack.callback(owner.close)

        def listing():
            response = public.get("/api/recordings/highlights")
            check("public list succeeds", response.status_code == 200)
            return response.json()["recordings"]

        def mutate(hidden):
            return owner.post(f"/api/highlights/{CLIP}", json={"hidden": hidden})

        def video_status():
            return public.get(f"/recordings/{CLIP}").status_code

        items = owner.get("/api/highlights").json()["highlights"]
        check("existing candidates default hidden despite legacy denylist",
              {i["filename"] for i in items} == {CLIP, OTHER} and all(i["hidden"] for i in items))
        check("default public list empty", listing() == [])
        for name in (CLIP, OTHER):
            check(f"unapproved direct video blocked: {name}", public.get(f"/recordings/{name}").status_code == 403)
        check("owner can preview hidden clip", owner.get(f"/recordings/{CLIP}").content == b"test video")
        check("anonymous mutation rejected", public.post(f"/api/highlights/{CLIP}", json={"hidden": False}).status_code == 401)
        for body in ({}, {"hidden": None}, {"hidden": "false"}, {"hidden": 0}):
            check(f"explicit boolean required: {body}", owner.post(f"/api/highlights/{CLIP}", json=body).status_code == 422)
        check("invalid requests do not create approvals", not store.path.exists())

        response = mutate(False)
        check("Show preserves response contract", response.status_code == 200 and response.json() == {"filename": CLIP, "hidden": False})
        check("Show persists approval", json.loads(store.path.read_text()) == [CLIP])
        check("Show publishes only approved clip", [i["filename"] for i in listing()] == [CLIP])
        check("Show authorizes direct video", public.get(f"/recordings/{CLIP}").content == b"test video")
        newcomer = "recording-new.webm"
        (root / newcomer).write_bytes(b"new clip")
        labels[newcomer] = "grooming"
        check("new candidate remains hidden after another is approved",
              [i["filename"] for i in listing()] == [CLIP]
              and public.get(f"/recordings/{newcomer}").status_code == 403)
        check("owner sees hidden false", next(i for i in owner.get("/api/highlights").json()["highlights"] if i["filename"] == CLIP)["hidden"] is False)
        with patch.object(hidden_store, "_store", create_store(root, "approved_highlights.json", [])):
            check("approval survives store recreation", [i["filename"] for i in listing()] == [CLIP])
            response = mutate(True)
            check("Hide preserves response contract", response.status_code == 200 and response.json() == {"filename": CLIP, "hidden": True})
        check("Hide persists removal", json.loads(store.path.read_text()) == [])
        check("Hide removes public listing", listing() == [])
        check("Hide revokes direct video", video_status() == 403)
        check("legacy hidden file unchanged", json.loads(legacy.read_text()) == [OTHER])

        for name in (RESTING, UNLABELED):
            check("non-highlight approval saved", owner.post(f"/api/highlights/{name}", json={"hidden": False}).status_code == 200)
            check(f"approval cannot bypass label gate: {name}", public.get(f"/recordings/{name}").status_code == 403)
        check("non-highlight approvals stay out of list", listing() == [])
        for raw in ('{broken', json.dumps({CLIP: False}), json.dumps(CLIP), '[null]'):
            store.path.write_text(raw)
            check(f"invalid approval data blocks public list: {raw}", public.get("/api/recordings/highlights").status_code == 500)
            check("invalid approval data blocks video", video_status() == 503)
            check("invalid approval data blocks mutation", mutate(False).status_code == 500)
            check("invalid data preserved", store.path.read_text() == raw)
        store.path.write_text('[]')
        with patch.object(store, "read", AsyncMock(side_effect=PermissionError("unreadable"))):
            check("read failure blocks public list", public.get("/api/recordings/highlights").status_code == 500)
            check("read failure blocks direct video", video_status() == 503)
            check("read failure reported to owner", owner.get("/api/highlights").status_code == 500)
            check("read failure blocks mutation", mutate(False).status_code == 500)
        with patch("app.lib.json_store.os.replace", side_effect=OSError("disk error")):
            check("failed Show reports error", mutate(False).status_code == 500)
        check("failed Show leaves persisted state unchanged", store.path.read_text() == '[]')
        check("failed Show cannot publish", listing() == [] and video_status() == 403)
        check("Show succeeds after storage recovers", mutate(False).status_code == 200)
        with patch("app.lib.json_store.os.replace", side_effect=OSError("disk error")):
            check("failed Hide reports error", mutate(True).status_code == 500)
        check("failed Hide retains previous persisted approval", json.loads(store.path.read_text()) == [CLIP])
        for raw in ('', '[]'):
            store.path.write_text(raw)
            check("empty approvals fail closed", listing() == [] and video_status() == 403)
        store.path.unlink()
        check("missing approvals fail closed", listing() == [] and video_status() == 403)
    finish("highlight visibility")


if __name__ == "__main__":
    main_test()
