"""Regression gate for the Phase 3 deploy gate and the two methods reports.

promote.py is the only path from models/candidate/ into models/, so its
refusals are the checks that matter: a failing gate must block (unless forced,
with the reason recorded), and a hash mismatch must block even when forced.
recoding_report.py and reliability.py are checked on small synthetic inputs
with hand-computed answers.

Hermetic: every models/ and data/ path is a temp dir. Standard library and
numpy only, so it runs in the runtime venv (no torch).

Usage (from server/, runtime venv):
    .venv/bin/python training/test_promote_reports.py
"""
import json
import shutil
import sys
import tempfile
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.checks import check, finish
from training import promote as P
from training import recoding_report as R
from training import reliability as K
from training.artifacts import GATE_RESULT_NAME, MANIFEST_NAME, sha256_file

NOW = datetime(2026, 10, 6, 3, 4, 5, tzinfo=UTC)


@contextmanager
def models_sandbox(gate_passed: bool | None = True):
    """models/ holding an 'old' deployed pair and a candidate with a 'new' pair
    plus a matching manifest. gate_passed=None writes no gate_result.json."""
    tmp = Path(tempfile.mkdtemp(prefix="bunnypromote-"))
    try:
        models = tmp / "models"
        cand = models / "candidate"
        cand.mkdir(parents=True)
        (models / "backbone_int8.onnx").write_bytes(b"old backbone")
        (models / "head.onnx").write_bytes(b"old head")
        (models / "head.onnx.data").write_bytes(b"old head weights")
        (models / "head_state.pt").write_bytes(b"training state, not an artifact")
        (cand / "backbone_int8.onnx").write_bytes(b"new backbone")
        (cand / "head.onnx").write_bytes(b"new head")
        manifest = {"artifacts": {p.name: sha256_file(p) for p in sorted(cand.iterdir())}, "labels": ["a"]}
        (cand / MANIFEST_NAME).write_text(json.dumps(manifest))
        if gate_passed is not None:
            gate = {"gate_passed": gate_passed, "gates": {"G4": {"passed": gate_passed}}}
            (cand / GATE_RESULT_NAME).write_text(json.dumps(gate))
        yield models
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def refused(fn) -> tuple[bool, str]:
    try:
        fn()
    except P.PromoteRefused as err:
        return True, str(err)
    return False, "did not refuse"


def test_failing_gate_refuses() -> None:
    """promote refuses a candidate whose gate did not pass, and changes nothing."""
    with models_sandbox(gate_passed=False) as models:
        ok, msg = refused(lambda: P.promote(models, now=NOW))
        check("gate_passed false is refused", ok, msg)
        check("the refusal names the failed gate", "G4" in msg, msg)
        check("deployed model untouched", (models / "head.onnx").read_bytes() == b"old head")
        check("nothing archived", not (models / "archive").exists())
    with models_sandbox(gate_passed=None) as models:
        ok, msg = refused(lambda: P.promote(models, now=NOW))
        check("a missing gate_result.json is refused", ok, msg)


def test_hash_mismatch_refuses_even_forced() -> None:
    """A candidate that differs from its manifest is never installed."""
    for desc, tamper in (
        ("modified artifact", lambda c: (c / "head.onnx").write_bytes(b"tampered")),
        ("unlisted artifact", lambda c: (c / "extra.onnx").write_bytes(b"x")),
        ("missing artifact", lambda c: (c / "head.onnx").unlink()),
    ):
        with models_sandbox() as models:
            tamper(models / "candidate")
            ok, msg = refused(lambda m=models: P.promote(m, now=NOW))
            check(f"{desc} is refused", ok, msg)
            ok, msg = refused(lambda m=models: P.promote(m, force_reason="trust me", now=NOW))
            check(f"{desc} is refused even with --force", ok, msg)
            check(f"{desc}: deployed model untouched", (models / "head.onnx").read_bytes() == b"old head")


def test_gate_bound_to_other_artifacts_refuses() -> None:
    """A passing gate computed on different artifacts does not count."""
    with models_sandbox() as models:
        gate_path = models / "candidate" / GATE_RESULT_NAME
        gate = json.loads(gate_path.read_text())
        gate["artifacts"] = {"head.onnx": "0" * 64}
        gate_path.write_text(json.dumps(gate))
        ok, msg = refused(lambda: P.promote(models, now=NOW))
        check("gate result for other artifacts is refused", ok, msg)


def test_force_records_the_reason() -> None:
    """--force gets past a failing gate and the reason lands in promoted.json."""
    with models_sandbox(gate_passed=False) as models:
        ok, msg = refused(lambda: P.promote(models, force_reason="   ", now=NOW))
        check("an empty force reason is refused", ok, msg)
        record = P.promote(models, force_reason="demo for the lab meeting", now=NOW)
        written = json.loads((models / P.PROMOTED_NAME).read_text())
        check("forced promote installs the candidate", (models / "head.onnx").read_bytes() == b"new head")
        check("reason recorded", written["forced"] == "demo for the lab meeting", str(written["forced"]))
        check("overridden blockers recorded", bool(written["gate_blockers_overridden"]), str(written))
        check("returned record matches file", record == written)


def test_successful_promote() -> None:
    """A passing, hash-clean candidate is archived-then-installed."""
    with models_sandbox() as models:
        record = P.promote(models, now=NOW)
        archive = models / "archive" / "20261006T030405Z"
        check("old artifacts archived", sorted(p.name for p in archive.iterdir())
              == ["backbone_int8.onnx", "head.onnx", "head.onnx.data"], str(sorted(archive.iterdir())))
        check("archived bytes are the old model", (archive / "head.onnx").read_bytes() == b"old head")
        for name in ("backbone_int8.onnx", "head.onnx"):
            check(f"{name} installed byte-identical",
                  (models / name).read_bytes() == (models / "candidate" / name).read_bytes())
        check("stale sidecar the candidate lacks is removed", not (models / "head.onnx.data").exists())
        check("non-artifact files left alone", (models / "head_state.pt").is_file())
        check("no temp files left behind", not list(models.glob(".*promote-tmp")))
        written = json.loads((models / P.PROMOTED_NAME).read_text())
        check("promoted.json carries the manifest", written["manifest"]["artifacts"] == record["manifest"]["artifacts"])
        check("promoted.json says not forced", written["forced"] is None)
        check("promoted.json points at the archive", written["archived_previous_to"] == "archive/20261006T030405Z",
              str(written["archived_previous_to"]))

        # A second promote archives the first promotion's provenance too.
        later = datetime(2026, 10, 7, tzinfo=UTC)
        P.promote(models, now=later)
        check("previous promoted.json archived on the next promote",
              (models / "archive" / "20261007T000000Z" / P.PROMOTED_NAME).is_file())


def test_recoding_report() -> None:
    """recoding_report cross-tabulates v1 rows against v2 columns."""
    v1 = {"a.webm": "normal", "b.webm": "normal", "c.webm": "normal", "d.webm": "zoomies", "e.webm": "yawn"}
    v2 = {"a.webm": "resting_lying", "b.webm": "resting_sitting", "d.webm": "locomotion_rapid",
          "e.webm": "out_of_view", "new.webm": "feeding"}
    rep = R.crosstab(v1, v2)
    t = rep["table"]
    check("rows are the v1 labels", rep["rows"] == ["normal", "yawn", "zoomies"], str(rep["rows"]))
    check("normal split across resting codes",
          t["normal"]["resting_lying"] == 1 and t["normal"]["resting_sitting"] == 1)
    check("unrecoded clip counted as unlabeled", t["normal"][R.UNLABELED] == 1)
    check("zoomies -> locomotion_rapid", t["zoomies"]["locomotion_rapid"] == 1)
    check("coding-only codes have a column", t["yawn"]["out_of_view"] == 1)
    check("row totals equal v1 counts", sum(t["normal"].values()) == 3)
    check("recoded and new counts", rep["n_recoded"] == 4 and rep["new_in_v2"] == 1, str(rep))


def test_reliability() -> None:
    """reliability.py kappa matches a hand-computed value and refuses partial re-codes."""
    # 10 clips: (g,g) x5, (g,f) x1, (f,f) x3, (f,g) x1. Observed 0.8, both
    # passes have marginals g=6 f=4, expected 0.36 + 0.16 = 0.52, so
    # kappa = (0.8 - 0.52) / 0.48 = 7/12.
    clips = [f"c{i}.webm" for i in range(10)]
    first = ["grooming"] * 6 + ["feeding"] * 4
    second = ["grooming"] * 5 + ["feeding"] + ["feeding"] * 3 + ["grooming"]
    labels = dict(zip(clips, first, strict=True))
    recode = {fn: {"label": lab, "coder": "joe"} for fn, lab in zip(clips, second, strict=True)}
    sample = {"clips": clips, "seed": 0, "fraction": 0.25}
    meta = {"c5.webm": {"uncertain": True}, "c9.webm": {"uncertain": True}, "zz.webm": {"uncertain": True}}

    res = K.compute(labels, recode, sample, meta)
    check("overall kappa is 7/12", abs(res["kappa"] - 7 / 12) < 1e-12, f"{res['kappa']!r}")
    check("percent agreement is 0.8", abs(res["percent_agreement"] - 0.8) < 1e-12)
    check("one-vs-rest kappa for a two-code sample equals overall",
          abs(res["per_class"]["grooming"]["kappa_one_vs_rest"] - 7 / 12) < 1e-12)
    check("unused code reports no kappa", res["per_class"]["rearing"]["kappa_one_vs_rest"] is None)
    check("uncertain counted in sample and overall",
          res["n_uncertain_in_sample"] == 2 and res["n_uncertain_overall"] == 3, str(res))
    check("dropping the two disagreeing uncertain clips gives kappa 1",
          res["kappa_excluding_uncertain"] == 1.0, f"{res['kappa_excluding_uncertain']!r}")
    check("coder recorded", res["coders"] == ["joe"])

    partial = {k: v for k, v in recode.items() if k != "c3.webm"}
    try:
        K.compute(labels, partial, sample, meta)
        check("incomplete re-code refused", False, "did not raise")
    except K.ReliabilityError as err:
        check("incomplete re-code refused with counts", "9 of 10" in str(err), str(err))
    try:
        K.compute(labels, recode, {}, meta)
        check("no sample refused", False, "did not raise")
    except K.ReliabilityError:
        check("no sample refused", True)


def main() -> None:
    for fn in (
        test_failing_gate_refuses,
        test_hash_mismatch_refuses_even_forced,
        test_gate_bound_to_other_artifacts_refuses,
        test_force_records_the_reason,
        test_successful_promote,
        test_recoding_report,
        test_reliability,
    ):
        print(f"\n{fn.__doc__.splitlines()[0]}")
        fn()
    finish("promote and report")


if __name__ == "__main__":
    main()
