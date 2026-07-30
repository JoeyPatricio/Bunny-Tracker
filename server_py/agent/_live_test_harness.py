"""One-off supervised test driver for the Phase 5 live-camera verification.
Not part of the permanent agent — runs the real Agent for a bounded window,
then explicitly stops ffmpeg and AWAITS its exit before the process ends, so
no external kill (and no orphaned ffmpeg child) is ever needed for the normal
path. Prints a clear DONE marker at the end so the caller knows it exited
clean.

Usage: .venv/Scripts/python.exe -m agent._live_test_harness <duration_sec>
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.capture import Agent, log


async def main():
    duration = float(sys.argv[1]) if len(sys.argv) > 1 else 60.0
    agent = Agent()

    run_task = asyncio.ensure_future(agent.run())
    try:
        await asyncio.wait_for(asyncio.shield(run_task), timeout=duration)
    except asyncio.TimeoutError:
        log(f"Test window ({duration:.0f}s) elapsed — stopping cleanly")

    if agent.ffmpeg:
        agent.ffmpeg.stop()
    if agent.live_poller:
        agent.live_poller.stop()
    run_task.cancel()
    try:
        await run_task
    except (asyncio.CancelledError, Exception):
        pass

    if agent.ffmpeg and agent.ffmpeg.proc:
        try:
            await asyncio.wait_for(agent.ffmpeg.proc.wait(), timeout=5.0)
            log("ffmpeg child exited cleanly")
        except asyncio.TimeoutError:
            log("ffmpeg child did not exit within 5s — killing it directly")
            agent.ffmpeg.proc.kill()
            await agent.ffmpeg.proc.wait()

    log("HARNESS_DONE — clean shutdown, no orphaned ffmpeg")


if __name__ == "__main__":
    asyncio.run(main())
