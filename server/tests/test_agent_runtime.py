"""Shared transient paths, promoted benchmark inputs, and agent upload contracts.

Uses temporary dotenv and deployment files without loading server auth stores.
Run with .venv/bin/python -m tests.test_agent_runtime from server/.
"""
import asyncio
import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent import capture
from scripts import bench_pi
from shared.labels import LABELS
from tests.checks import check, finish


def test_shared_paths(root):
    # Import the resolver before dotenv to catch accidentally cached env reads.
    probe = """
import sys
from pathlib import Path
from unittest.mock import patch
from shared import paths
from dotenv import load_dotenv
dotenv_path, mode, root = sys.argv[1:]
root = Path(root)
with patch('dotenv.load_dotenv', side_effect=lambda *args, **kwargs: load_dotenv(dotenv_path)):
    from app import config
    from agent import capture
    with patch.object(capture, 'AgentClient'):
        agent = capture.Agent()
if mode == 'default':
    segments = paths.DATA_DIR / 'agent' / 'segments'
    live = paths.SERVER_DIR / 'agent' / 'live_frame.jpg'
else:
    segments = root / mode / 'segments'
    live = root / mode / 'live' / 'frame.jpg'
assert config.SEGMENTS_DIR == agent.seg_dir == segments
assert agent.live_frame_path == live
"""
    for mode in ("default", "dotenv", "env"):
        dotenv = root / f"{mode}.env"
        dotenv.write_text("" if mode == "default" else
                          f"AGENT_SEGMENT_DIR={root / 'dotenv' / 'segments'}\n"
                          f"AGENT_LIVE_FRAME_PATH={root / 'dotenv' / 'live' / 'frame.jpg'}\n")
        env = dict(os.environ)
        for key in ("AGENT_SEGMENT_DIR", "AGENT_LIVE_FRAME_PATH"):
            env.pop(key, None)
        if mode == "env":
            env.update(AGENT_SEGMENT_DIR=str(root / mode / "segments"),
                       AGENT_LIVE_FRAME_PATH=str(root / mode / "live" / "frame.jpg"))
        result = subprocess.run([sys.executable, "-c", probe, str(dotenv), mode, str(root)],
                                cwd=Path(__file__).resolve().parent.parent,
                                env=env, capture_output=True, text=True, timeout=10)
        check(f"agent and server share {mode} paths after dotenv load", result.returncode == 0,
              result.stderr)


def test_promoted_benchmark(root):
    models = root / "models"
    models.mkdir()
    cache = models / "cache"
    cache.mkdir()
    (cache / "manifest.json").write_text(json.dumps({"window": 8}))
    record = {"manifest": {"labels": LABELS, "window": 12, "view": "full"}}
    promotion = models / "promoted.json"
    promotion.write_text(json.dumps(record))
    shapes = {}

    def bench(name, path, make_input):
        shapes[name] = make_input().shape
        return None

    with patch.object(bench_pi, "MODELS_DIR", models), \
            patch.object(bench_pi, "embedding_dim", return_value=1280), \
            patch.object(bench_pi, "bench_session", side_effect=bench), \
            contextlib.redirect_stdout(io.StringIO()) as output:
        bench_pi.main()
        check_window = shapes["head"] == (1, 12, 1280)
        (cache / "manifest.json").unlink()
        bench_pi.main()
        check_no_cache = shapes["head"] == (1, 12, 1280)
        rejected = []
        for manifest in (None, {**record["manifest"], "labels": ["normal"]},
                         {**record["manifest"], "view": "crop"},
                         {**record["manifest"], "window": 0}):
            if manifest is None:
                promotion.unlink()
            else:
                promotion.write_text(json.dumps({"manifest": manifest}))
            try:
                bench_pi.main()
                rejected.append(False)
            except ValueError:
                rejected.append(True)
    check("benchmark uses the promoted twelve-frame window over cache eight", check_window)
    check("benchmark needs no training cache", check_no_cache)
    check("benchmark identifies the current CPU host", "CPU inference benchmark on " in output.getvalue())
    check("benchmark validates promotion presence, labels, view, and window", all(rejected))
    check("benchmark imports no torch or training modules", "torch" not in sys.modules
          and not any(name.startswith("training.") for name in sys.modules))


async def test_upload_source(root):
    client = SimpleNamespace(post=AsyncMock(return_value=SimpleNamespace(
        status_code=200, json=lambda: {"filename": "clip.webm", "sent": False})))
    with patch.object(capture, "AgentClient", return_value=client):
        agent = capture.Agent()
    agent.seg_dir = root
    (root / "segment.webm").write_bytes(b"test segment")
    agent._latest_finished_segment = AsyncMock(return_value="segment.webm")
    agent.alert_webhook_url = None
    await agent._upload_motion_clip(5)
    await agent._upload_and_alert({"label": "rearing", "confidence": 95})
    uploads = [call.kwargs for call in client.post.call_args_list if call.args[0] == "/api/recordings"]
    check("motion clips send the disk-budget source", uploads[0]["data"] == {"source": "motion"})
    check("alert clips preserve their existing untagged source", uploads[1]["data"] is None)
    client.post.return_value = SimpleNamespace(status_code=507, json=lambda: {"reason": "disk floor"})
    with patch.object(capture, "log") as log:
        await agent._upload_motion_clip(5)
        check("motion refusal does not claim clipping was stopped",
              "Motion capture refused" in log.call_args.args[0] and "Clipping stopped" not in log.call_args.args[0])
        await agent._upload_motion_clip(5)
        check("consecutive motion refusals log once", log.call_count == 1)
        client.post.return_value = SimpleNamespace(status_code=200, json=lambda: {"filename": "clip.webm"})
        await agent._upload_motion_clip(5)
        check("successful upload clears the budget logging flag", not agent.clip_budget_blocked)
        client.post.return_value = SimpleNamespace(status_code=507, json=lambda: {"reason": "disk floor"})
        await agent._upload_motion_clip(5)
        refusals = [call for call in log.call_args_list if "Motion capture refused" in call.args[0]]
        check("a later low-disk episode logs its motion refusal", len(refusals) == 2)


async def test_startup_settings():
    state = {"enabled": True, "clipping": True, "clippingMotionThreshold": 7,
             "clippingCooldownSec": 19, "clippingDryRun": True}
    client = SimpleNamespace(verify=AsyncMock(), get=AsyncMock(return_value=SimpleNamespace(json=lambda: state)))
    with patch.object(capture, "AgentClient", return_value=client):
        agent = capture.Agent()
    with tempfile.TemporaryDirectory(prefix="bunny-startup-") as tmp:
        agent.seg_dir = Path(tmp) / "segments"
        agent.live_frame_path = Path(tmp) / "live" / "frame.jpg"

        async def load_models():
            agent.decision_loop = capture.AgentDecisionLoop()

        async def start():
            loop = agent.decision_loop
            check("persisted clipping settings apply before first capture", loop.clipping
                  and loop.clip_threshold == 7 and loop.clip_cooldown_sec == 19 and loop.dry_run)
            check("startup creates both transient parents", agent.seg_dir.is_dir()
                  and agent.live_frame_path.parent.is_dir())
            raise asyncio.CancelledError

        agent.load_models = load_models
        agent.start_capture = start
        try:
            await agent.run()
        except asyncio.CancelledError:
            pass


def main():
    with tempfile.TemporaryDirectory(prefix="bunny-runtime-") as tmp:
        root = Path(tmp)
        test_shared_paths(root)
        test_promoted_benchmark(root)
        asyncio.run(test_upload_source(root))
        asyncio.run(test_startup_settings())
    finish("agent runtime")


if __name__ == "__main__":
    main()
