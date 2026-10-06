"""Move a gated candidate into models/, the only way a model gets deployed.

export_onnx.py writes to models/candidate/. The agent and the server read
models/*.onnx directly, so before this existed an export WAS the deploy, and
models/gate_result.json could read `"gate_passed": false` while that model was
live. This script refuses unless:

1. Every artifact in the candidate matches the sha256 recorded in
   candidate/manifest.json, and nothing unrecorded sits beside them. Never
   bypassable: a hash mismatch means the files being installed are not the
   files that were evaluated.
2. candidate/gate_result.json has `"gate_passed": true`. Bypassable only with
   --force "<reason>", and the reason is written into models/promoted.json so
   an override stays visible next to the model it let through.

gate_result.json minimal schema (written by the evaluate.py rewrite):

    {
      "gate_passed": bool,
      "gates": {"<id>": {"passed": bool, ...}, ...},
      "artifacts": {"<name>": "<sha256>", ...}   # optional
    }

If `artifacts` is present it must equal the manifest's, which binds the gate
result to the exact candidate it evaluated rather than to whatever happens to
be in the directory now.

On success the currently deployed artifacts are copied to
models/archive/<UTC timestamp>/ (with the previous promoted.json), the
candidate files are installed one by one via a temp name and os.replace, any
deployed artifact the new candidate does not have is removed (it is already in
the archive), and models/promoted.json records the candidate manifest, the
gate result, when, and the force reason or null.

Runtime venv or training venv: standard library only.

Usage (from server_py/):
    .venv/bin/python -m training.promote [--force "reason"] [--models-dir DIR]
"""
import argparse
import json
import os
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.artifacts import GATE_RESULT_NAME, MANIFEST_NAME, artifacts_in, sha256_file

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
PROMOTED_NAME = "promoted.json"


class PromoteRefused(Exception):
    pass


def _read_json(path: Path, what: str) -> dict:
    if not path.is_file():
        raise PromoteRefused(f"No {what} at {path}.")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as err:
        raise PromoteRefused(f"{what} at {path} is not valid JSON: {err}") from err


def verify_hashes(candidate_dir: Path, manifest: dict) -> dict[str, str]:
    recorded = manifest.get("artifacts") or {}
    if not recorded:
        raise PromoteRefused(f"{candidate_dir / MANIFEST_NAME} lists no artifacts.")
    present = {p.name for p in artifacts_in(candidate_dir)}
    problems = []
    for name in sorted(set(recorded) - present):
        problems.append(f"{name}: in the manifest but missing from the candidate")
    for name in sorted(present - set(recorded)):
        problems.append(f"{name}: in the candidate but not in the manifest")
    for name in sorted(set(recorded) & present):
        actual = sha256_file(candidate_dir / name)
        if actual != recorded[name]:
            problems.append(f"{name}: sha256 {actual[:12]}... but manifest says {recorded[name][:12]}...")
    if problems:
        raise PromoteRefused(
            "Candidate does not match its manifest, so it is not the model that "
            "was evaluated. Re-export and re-evaluate. --force does not bypass this.\n  "
            + "\n  ".join(problems)
        )
    return dict(recorded)


def check_gate(gate: dict, artifacts: dict[str, str]) -> list[str]:
    """Reasons the gate blocks this candidate; empty if it passes."""
    reasons = []
    if gate.get("gate_passed") is not True:
        failed = sorted(k for k, v in (gate.get("gates") or {}).items()
                        if isinstance(v, dict) and v.get("passed") is False)
        reasons.append("gate_passed is not true" + (f" (failed: {', '.join(failed)})" if failed else ""))
    if "artifacts" in gate and gate["artifacts"] != artifacts:
        reasons.append("gate_result.json was computed on different artifacts than this candidate")
    return reasons


def promote(models_dir: Path = MODELS_DIR, candidate_dir: Path | None = None,
            force_reason: str | None = None, now: datetime | None = None) -> dict:
    models_dir = Path(models_dir)
    candidate_dir = Path(candidate_dir) if candidate_dir else models_dir / "candidate"
    now = now or datetime.now(UTC)
    if force_reason is not None and not force_reason.strip():
        raise PromoteRefused("--force needs a non-empty reason; it is recorded in promoted.json.")

    manifest = _read_json(candidate_dir / MANIFEST_NAME, "candidate manifest")
    artifacts = verify_hashes(candidate_dir, manifest)

    gate_path = candidate_dir / GATE_RESULT_NAME
    gate = _read_json(gate_path, "gate result") if gate_path.is_file() else None
    blockers = ["no gate_result.json; run training.evaluate first"] if gate is None else check_gate(gate, artifacts)
    if blockers and force_reason is None:
        raise PromoteRefused(
            "Gate does not pass: " + "; ".join(blockers)
            + '. Fix the candidate, or override with --force "<reason>".'
        )

    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    archive_dir = models_dir / "archive" / stamp
    deployed = artifacts_in(models_dir)
    previous = models_dir / PROMOTED_NAME
    if deployed or previous.is_file():
        archive_dir.mkdir(parents=True, exist_ok=False)
        for path in deployed:
            shutil.copy2(path, archive_dir / path.name)
        if previous.is_file():
            shutil.copy2(previous, archive_dir / PROMOTED_NAME)

    # Copy-then-replace per file, so models/ never holds a half-written graph.
    # The set as a whole is not atomic; a reader between two replaces could
    # see a new backbone with the old head. The agent loads on start, so
    # restart it after a promote.
    for name in sorted(artifacts):
        tmp = models_dir / f".{name}.promote-tmp"
        shutil.copy2(candidate_dir / name, tmp)
        os.replace(tmp, models_dir / name)
    for path in deployed:
        if path.name not in artifacts:
            path.unlink()  # already archived

    mismatched = [n for n, h in artifacts.items() if sha256_file(models_dir / n) != h]
    if mismatched:
        raise PromoteRefused(
            f"Installed files do not match the manifest after copy: {mismatched}. "
            f"The previous model is in {archive_dir}."
        )

    record = {
        "promoted_at": now.isoformat(timespec="seconds").replace("+00:00", "Z"),
        "forced": force_reason,
        "gate_blockers_overridden": blockers if force_reason is not None else [],
        "archived_previous_to": str(archive_dir.relative_to(models_dir)) if archive_dir.exists() else None,
        "manifest": manifest,
        "gate_result": gate,
    }
    (models_dir / PROMOTED_NAME).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--candidate-dir", type=Path, default=None,
                        help="default <models-dir>/candidate")
    parser.add_argument("--force", metavar="REASON", default=None,
                        help="promote despite a failing gate; REASON is recorded")
    args = parser.parse_args()
    try:
        record = promote(args.models_dir, args.candidate_dir, args.force)
    except PromoteRefused as err:
        raise SystemExit(f"promote refused: {err}") from None
    print(f"Promoted {len(record['manifest']['artifacts'])} artifacts into {args.models_dir}.")
    if record["archived_previous_to"]:
        print(f"Previous model archived to {args.models_dir / record['archived_previous_to']}.")
    if record["forced"] is not None:
        print(f"FORCED past: {'; '.join(record['gate_blockers_overridden'])}. Reason: {record['forced']}")
    print("Restart the agent so it loads the new model.")


if __name__ == "__main__":
    main()
