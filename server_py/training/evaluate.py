"""Deployment gate for models/candidate/: G1-G7, written to gate_result.json.

The science lives in crossval.py, whose pooled out-of-fold numbers are the
only ones this project reports. This file answers a different question: is
THIS exported candidate safe to deploy? promote.py refuses to install a
candidate unless the gate_result.json written here says `gate_passed: true`
and its `artifacts` map equals the candidate manifest's, so a result can only
ever apply to the exact files it evaluated.

The gates:

G1  head.onnx output width equals len(LABELS). agent/capture.py keeps its own
    runtime guard as defence in depth; this one fires at build time, before a
    stale head can reach the agent at all, rather than at 3am when the agent
    restarts and quietly falls back to motion-only.
G2  The batch axis actually batches: head.onnx at batch 3 and each backbone at
    batch 2 return that many rows, and every row equals the same input run
    alone (within 1e-5). A statically batched export either raises here or
    fails the row comparison.
G3  The ONNX fp32 head reproduces the torch head (head_state.pt) within 1e-4
    max absolute logit difference on every cached eval embedding. Comparing
    logits, not accuracies: two different models can score the same accuracy,
    which is how the old check passed while proving nothing.
G4  int8 backbone macro-F1 within 0.03 of the fp32 backbone. See "G4's eval
    set" below.
G5  Not degenerate: the deployed (int8) pipeline must not predict a single
    class for every clip (metrics.check_degenerate).
G6  The candidate's arm beats static_logreg in cv_result.json: gru pooled OOF
    macro-F1 (mean over repeats) strictly above static_logreg's on the same
    folds, and cv_result.json was computed on the labels this candidate's cache
    holds. See "G6 and significance" below.
G7  Not worse than the deployed model by more than one fold-sd: the
    candidate's gru pooled OOF macro-F1 must be at least the deployed one's
    (recorded in models/promoted.json at its promotion) minus the sd of the
    candidate's per-outer-fold gru macro-F1. With no promoted.json, or one
    whose gate result has no CV headline (a first run, or a --force with no
    gate), it passes with detail "no deployed baseline". If the deployed
    model was scored under a different ethogram version the numbers measure
    different tasks, so it passes with that stated in the detail rather than
    comparing them.

G4's eval set. The candidate is the --all-clips model, trained on every
labeled clip, so no held-out clips exist for it. G4 does not need them: it
asks whether int8 quantization changed what the same weights predict, which
is a property of the two numeric implementations, not of generalization. It
runs every cached clip's eval sample (frames/<clip>.eval<W>.npy, the identical
uint8 frames crossval.py's eval embeddings came from) through both backbones
into the same head, and compares macro-F1 against the human labels. Those
macro-F1s are in-sample and are NOT accuracy results; only their difference
is the gate. The fp32/int8 argmax agreement rate is reported alongside, since
it is the more sensitive view of the same damage.

G6 and significance. G6 requires gru to beat static_logreg on the point
estimate and reports the Nadeau-Bengio corrected paired p-value, but does not
require significance. At n=112 with 5 outer folds the corrected test has
little power: a real 3-5 point gain will usually not reach p < 0.05, and a
gate that demanded it would block most genuine improvements. The gate's job is
to stop shipping a temporal head that loses to its own static baseline; the
significance claim belongs in the paper, where the p-value is reported as is.

Also writes candidate/quant_result.json, the G4 numbers on their own.

Usage (from server_py/, training venv, after export_onnx.py):
    .venv-train/bin/python -m training.evaluate [--candidate-dir DIR] [--models-dir DIR]
        [--cache-dir DIR] [--cv-result PATH] [--head-state PATH] [--train-result PATH]
Exits 1 if the gate fails, after writing gate_result.json either way.
"""
import argparse
import contextlib
import io
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.frames import normalize
from shared.labels import ETHOGRAM_VERSION, LABELS
from training.artifacts import GATE_RESULT_NAME, MANIFEST_NAME, artifacts_in, sha256_file
from training.cache import DEFAULT_CACHE_DIR, LEGACY_POLICY, load_manifest
from training.metrics import check_degenerate, confusion_matrix, macro_f1
from training.model import TemporalHead

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

LOGIT_TOLERANCE = 1e-4   # G3
BATCH_TOLERANCE = 1e-5   # G2
INT8_TOLERANCE = 0.03    # G4: macro-F1 points int8 may lose to fp32

BACKBONES = ("backbone_fp32.onnx", "backbone_int8.onnx")
HEAD = "head.onnx"


def _session(path: Path) -> ort.InferenceSession:
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def _run_head(sess, windows: np.ndarray) -> np.ndarray:
    return sess.run(None, {"embeddings": windows.astype(np.float32)})[0]


def _run_backbone(sess, frames_uint8: np.ndarray) -> np.ndarray:
    batch = np.stack([normalize(f) for f in frames_uint8]).astype(np.float32)
    return sess.run(None, {"pixel_values": batch})[0]


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class Inputs:
    """Everything the gates read, loaded once. Paths are explicit so the test
    can point every one of them at a temp dir."""

    def __init__(self, *, candidate_dir: Path, models_dir: Path, cache_dir: Path, cv_result: Path,
                 head_state: Path, train_result: Path):
        self.candidate_dir = Path(candidate_dir)
        self.models_dir = Path(models_dir)
        self.cache_dir = Path(cache_dir)
        self.cv_result_path = Path(cv_result)
        self.head_state = Path(head_state)

        manifest_path = self.candidate_dir / MANIFEST_NAME
        if not manifest_path.is_file():
            raise SystemExit(f"No candidate manifest at {manifest_path}. Run training.export_onnx first.")
        self.manifest = json.loads(manifest_path.read_text())
        self._check_artifacts_match_manifest()

        tr = Path(train_result)
        self.train_result = json.loads(tr.read_text()) if tr.is_file() else {}
        self.window = self.manifest.get("window") or self.train_result.get("window")
        if not self.window:
            raise SystemExit("Neither the candidate manifest nor train_result.json records a window.")
        self.window = int(self.window)

        self.arm = self.train_result.get("cv_arm", "gru")
        if (self.train_result.get("view", "full") != "full"
                or self.manifest.get("view", "full") != "full"):
            raise SystemExit("Crop is an offline CV arm. The agent has no matching ROI preprocessing.")
        self.cache = load_manifest(self.cache_dir, self.train_result.get("aug_policy", LEGACY_POLICY))
        key = str(self.window)
        if key not in {str(w) for w in self.cache["eval_windows"]}:
            raise SystemExit(
                f"The cache has no eval sample for window={self.window} (has "
                f"{self.cache['eval_windows']}). Regenerate with extract_frames.py --eval-windows."
            )
        clips = [c for c in self.cache["clips"] if c["label"] in LABELS]
        self.filenames = [c["filename"] for c in clips]
        self.y = np.array([LABELS.index(c["label"]) for c in clips])
        emb_dir, frames_dir = self.cache_dir / "emb", self.cache_dir / "frames"
        self.embeddings = np.stack([np.load(emb_dir / c["eval"][key]) for c in clips]).astype(np.float32)
        self.frames_paths = [frames_dir / c["eval"][key] for c in clips]

        self.cv = json.loads(self.cv_result_path.read_text()) if self.cv_result_path.is_file() else None

    def _check_artifacts_match_manifest(self) -> None:
        # Evaluating files that are not the ones the manifest hashed would bind
        # the gate result to the wrong model. promote.py re-checks at install.
        recorded = self.manifest.get("artifacts") or {}
        present = {p.name: sha256_file(p) for p in artifacts_in(self.candidate_dir)}
        if present != recorded:
            raise SystemExit(
                f"{self.candidate_dir} does not match its manifest (differing: "
                f"{sorted(set(present.items()) ^ set(recorded.items()))[:4]}). Re-export."
            )

    def path(self, name: str) -> Path:
        return self.candidate_dir / name


def gate_g1(inp: Inputs) -> dict:
    sess = _session(inp.path(HEAD))
    declared = sess.get_outputs()[0].shape
    out = _run_head(sess, inp.embeddings[:1])
    width = int(out.shape[1])
    ok = width == len(LABELS)
    return {"passed": ok, "width": width, "declared_shape": [str(d) for d in declared],
            "detail": f"head predicts {width} classes, ethogram {ETHOGRAM_VERSION} declares {len(LABELS)}"}


def _batch_check(run, batch: np.ndarray) -> tuple[bool, str]:
    try:
        together = run(batch)
    except Exception as err:  # a static batch axis raises here
        return False, f"batch {len(batch)} raised: {str(err).splitlines()[0][:200]}"
    if together.shape[0] != len(batch):
        return False, f"batch {len(batch)} returned {together.shape[0]} rows"
    alone = np.concatenate([run(batch[i:i + 1]) for i in range(len(batch))])
    diff = float(np.abs(together - alone).max())
    return diff <= BATCH_TOLERANCE, f"batch {len(batch)}, max row diff {diff:.2e}"


def gate_g2(inp: Inputs) -> dict:
    head = _session(inp.path(HEAD))
    rows = inp.embeddings[np.arange(3) % len(inp.embeddings)]
    results = {"head": _batch_check(lambda b: _run_head(head, b), rows)}
    frames = np.load(inp.frames_paths[0])[:2]
    if len(frames) < 2:
        frames = np.concatenate([frames, frames])
    for name in BACKBONES:
        sess = _session(inp.path(name))
        results[name] = _batch_check(lambda b, s=sess: _run_backbone(s, b), frames)
    ok = all(r[0] for r in results.values())
    return {"passed": ok, "detail": "; ".join(f"{k}: {v[1]}" for k, v in results.items())}


def gate_g3(inp: Inputs) -> dict:
    head = TemporalHead()
    head.load_state_dict(torch.load(inp.head_state, weights_only=True))  # strict: a stale width raises
    head.eval()
    with torch.no_grad():
        torch_logits = head(torch.from_numpy(inp.embeddings)).numpy()
    onnx_logits = _run_head(_session(inp.path(HEAD)), inp.embeddings)
    diff = float(np.abs(torch_logits - onnx_logits).max())
    return {"passed": diff <= LOGIT_TOLERANCE, "max_abs_logit_diff": diff,
            "detail": f"max |torch - onnx| logit {diff:.2e} over {len(inp.y)} clips (tolerance {LOGIT_TOLERANCE:g})"}


def pipeline_probs(inp: Inputs, backbone_name: str) -> np.ndarray:
    backbone = _session(inp.path(backbone_name))
    head = _session(inp.path(HEAD))
    windows = np.stack([_run_backbone(backbone, np.load(p)) for p in inp.frames_paths])
    return _softmax(_run_head(head, windows))


def gate_g4(inp: Inputs, probs: dict) -> dict:
    p32, p8 = probs["backbone_fp32.onnx"], probs["backbone_int8.onnx"]
    f1_32 = macro_f1(confusion_matrix(inp.y.tolist(), p32.argmax(1).tolist()))
    f1_8 = macro_f1(confusion_matrix(inp.y.tolist(), p8.argmax(1).tolist()))
    agreement = float((p32.argmax(1) == p8.argmax(1)).mean())
    delta = f1_32 - f1_8
    return {"passed": delta <= INT8_TOLERANCE, "fp32_macro_f1": f1_32, "int8_macro_f1": f1_8,
            "delta": delta, "argmax_agreement": agreement, "n": len(inp.y),
            "detail": f"int8 macro-F1 {f1_8:.3f} vs fp32 {f1_32:.3f} (delta {delta:+.3f}, tolerance "
                      f"{INT8_TOLERANCE}); argmax agreement {agreement:.1%}. In-sample: only the delta is the gate."}


def gate_g5(inp: Inputs, probs: dict) -> dict:
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        degenerate = check_degenerate(probs["backbone_int8.onnx"], inp.y)
    predicted = sorted({LABELS[c] for c in probs["backbone_int8.onnx"].argmax(1)})
    return {"passed": not degenerate, "predicted_classes": predicted,
            "detail": f"int8 pipeline predicts {len(predicted)} distinct classes: {predicted}"}


def _gru_headline(cv: dict, arm: str = "gru") -> dict:
    gru = cv["arms"][arm]
    folds = np.array([f["macro_f1"] for f in gru["per_fold"]])
    return {
        "arm": arm,
        "run_id": cv.get("run_id"),
        "labels_sha256": cv.get("labels_sha256"),
        "ethogram_version": cv.get("ethogram_version"),
        "gru_macro_f1": gru["pooled_oof"]["macro_f1"]["mean"],
        "gru_macro_f1_sd_repeats": gru["pooled_oof"]["macro_f1"]["sd"],
        "gru_fold_sd": float(folds.std(ddof=1)) if len(folds) > 1 else 0.0,
    }


def gate_g6(inp: Inputs) -> dict:
    cv = inp.cv
    if cv is None:
        return {"passed": False, "detail": f"no cv_result.json at {inp.cv_result_path}; run training.crossval"}
    problems = []
    arm = inp.arm
    for name, data in (("candidate", inp.manifest), ("cache", inp.cache),
                       ("CV result", cv), ("training result", inp.train_result)):
        if data.get("ethogram_version") != ETHOGRAM_VERSION:
            problems.append(f"{name} uses a different ethogram version")
        if name != "training result" and data.get("labels") != LABELS:
            problems.append(f"{name} uses a different ordered label vocabulary")
    if cv.get("config", {}).get("quick"):
        problems.append("cv_result.json is a --quick smoke run, not deployment evidence")
    for key in ("cv_arm", "cv_run_id", "labels_sha256", "aug_policy", "view"):
        if key in inp.manifest and inp.manifest[key] != inp.train_result.get(key):
            problems.append(f"candidate manifest and train_result.json disagree on {key}")
    if cv.get("labels_sha256") != inp.cache.get("labels_sha256"):
        problems.append("cv_result.json was computed on different labels than the cache holds")
    tr_sha = inp.train_result.get("labels_sha256")
    if tr_sha is not None and tr_sha != cv.get("labels_sha256"):
        problems.append("the candidate's head was trained on different labels than cv_result.json")
    tr_run = inp.train_result.get("cv_run_id")
    if tr_run is not None and tr_run != cv.get("run_id"):
        problems.append(f"train_result.json took its config from CV run {tr_run}, not {cv.get('run_id')}")
    arms = cv.get("arms", {})
    if arm not in arms or "static_logreg" not in arms:
        return {"passed": False, "detail": f"cv_result.json lacks the {arm} or static_logreg arm"}
    cache = cv.get("arm_cache", {}).get(arm)
    if cache is not None and cache != {"aug_policy": inp.train_result.get("aug_policy", LEGACY_POLICY), "view": "full"}:
        problems.append("the head's cache differs from its selected CV arm")
    gru = arms[arm]["pooled_oof"]["macro_f1"]["mean"]
    logreg = arms["static_logreg"]["pooled_oof"]["macro_f1"]["mean"]
    p = ((cv.get("paired") or {}).get("corrected_resampled_t") or {}).get("p_two_sided")
    if arm != "gru":
        from training.crossval import paired_comparison

        paired = paired_comparison({a: arms[a]["per_fold"] for a in (arm, "static_logreg")},
                                   arm, "static_logreg", cv["n_clips"], cv["config"]["n_splits"])
        p = paired["corrected_resampled_t"]["p_two_sided"]
    if not gru > logreg:
        problems.append(f"{arm} does not beat static_logreg ({gru:.3f} vs {logreg:.3f})")
    p_text = f"{p:.3f}" if p is not None else "n/a"
    return {"passed": not problems, "arm": arm, "gru_macro_f1": gru, "static_logreg_macro_f1": logreg,
            "corrected_paired_p": p, "significance_required": False,
            "detail": "; ".join(problems) if problems else
            f"{arm} {gru:.3f} > static_logreg {logreg:.3f} (corrected paired p = {p_text}, reported, not required)"}


def gate_g7(inp: Inputs) -> dict:
    if inp.cv is None or inp.arm not in inp.cv.get("arms", {}):
        return {"passed": False, "detail": "no candidate CV headline to compare"}
    cand = _gru_headline(inp.cv, inp.arm)
    promoted_path = inp.models_dir / "promoted.json"
    baseline = None
    if promoted_path.is_file():
        promoted = json.loads(promoted_path.read_text())
        baseline = ((promoted.get("gate_result") or {}).get("cv")) or None
    if baseline is None or baseline.get("gru_macro_f1") is None:
        return {"passed": True, "detail": "no deployed baseline", "candidate": cand["gru_macro_f1"]}
    if baseline.get("ethogram_version") != cand["ethogram_version"]:
        return {"passed": True, "candidate": cand["gru_macro_f1"], "deployed": baseline["gru_macro_f1"],
                "detail": f"deployed model was scored under ethogram {baseline.get('ethogram_version')}, "
                          f"candidate under {cand['ethogram_version']}: not comparable, not compared"}
    floor = baseline["gru_macro_f1"] - cand["gru_fold_sd"]
    ok = cand["gru_macro_f1"] >= floor
    return {"passed": ok, "candidate": cand["gru_macro_f1"], "deployed": baseline["gru_macro_f1"],
            "fold_sd": cand["gru_fold_sd"],
            "detail": f"candidate {cand['gru_macro_f1']:.3f} vs deployed {baseline['gru_macro_f1']:.3f} "
                      f"minus one fold-sd {cand['gru_fold_sd']:.3f} = floor {floor:.3f}"}


def _guarded(fn, *args) -> dict:
    """A gate that raises has failed, with the error as its detail, and the
    remaining gates still run so the report shows everything wrong at once."""
    try:
        return fn(*args)
    except Exception as err:
        return {"passed": False, "detail": f"{type(err).__name__}: {str(err).splitlines()[0][:300]}"}


def _with_probs(gate, inp: Inputs, probs: dict) -> dict:
    """G4 and G5 share one pass of every clip through each backbone; if that
    pass failed for either backbone, both gates fail with its error."""
    for value in probs.values():
        if isinstance(value, Exception):
            raise value
    return gate(inp, probs)


def evaluate(*, candidate_dir: Path, models_dir: Path = MODELS_DIR, cache_dir: Path = DEFAULT_CACHE_DIR,
             cv_result: Path | None = None, head_state: Path | None = None,
             train_result: Path | None = None) -> dict:
    models_dir = Path(models_dir)
    inp = Inputs(
        candidate_dir=candidate_dir, models_dir=models_dir, cache_dir=cache_dir,
        cv_result=cv_result or models_dir / "cv_result.json",
        head_state=head_state or models_dir / "head_state.pt",
        train_result=train_result or models_dir / "train_result.json",
    )
    probs = {}
    for name in BACKBONES:
        try:
            probs[name] = pipeline_probs(inp, name)
        except Exception as err:
            probs[name] = err

    gates = {
        "G1": _guarded(gate_g1, inp),
        "G2": _guarded(gate_g2, inp),
        "G3": _guarded(gate_g3, inp),
        "G4": _guarded(_with_probs, gate_g4, inp, probs),
        "G5": _guarded(_with_probs, gate_g5, inp, probs),
        "G6": _guarded(gate_g6, inp),
        "G7": _guarded(gate_g7, inp),
    }
    result = {
        "gate_passed": all(g["passed"] for g in gates.values()),
        "gates": gates,
        "artifacts": dict(inp.manifest["artifacts"]),
        "window": inp.window,
        "n_clips": len(inp.y),
        "ethogram_version": ETHOGRAM_VERSION,
        "cv": _guarded(_gru_headline, inp.cv, inp.arm) if inp.cv else None,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
    }
    inp.candidate_dir.joinpath(GATE_RESULT_NAME).write_text(json.dumps(result, indent=2) + "\n")
    if "fp32_macro_f1" in gates["G4"]:
        quant = {k: gates["G4"][k] for k in ("fp32_macro_f1", "int8_macro_f1", "delta", "argmax_agreement", "n")}
        quant["note"] = "In-sample macro-F1 of the all-clips candidate; only the fp32-int8 delta is meaningful."
        inp.candidate_dir.joinpath("quant_result.json").write_text(json.dumps(quant, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--candidate-dir", type=Path, default=MODELS_DIR / "candidate")
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR,
                        help="where promoted.json, cv_result.json, head_state.pt and train_result.json live")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--cv-result", type=Path, default=None)
    parser.add_argument("--head-state", type=Path, default=None)
    parser.add_argument("--train-result", type=Path, default=None)
    args = parser.parse_args()
    result = evaluate(candidate_dir=args.candidate_dir, models_dir=args.models_dir, cache_dir=args.cache_dir,
                      cv_result=args.cv_result, head_state=args.head_state, train_result=args.train_result)
    print(f"\n{'=' * 60}")
    for gid, g in result["gates"].items():
        print(f"  {'PASS' if g['passed'] else 'FAIL'}  {gid}  {g.get('detail', '')}")
    print(f"{'=' * 60}")
    print(f"gate_passed: {result['gate_passed']}  ->  {args.candidate_dir / GATE_RESULT_NAME}")
    if not result["gate_passed"]:
        sys.exit(1)
    print("Next: .venv/bin/python -m training.promote")


if __name__ == "__main__":
    main()
