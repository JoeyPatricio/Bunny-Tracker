"""Regression gate for training/crossval.py (docs/upgrade-plan.md, Phase 3).

Hermetic: builds a synthetic schema-2 cache in a temp dir, with embeddings
that carry a learnable per-class signal, and runs crossval in --quick mode on
it. Nothing under server/ or models/ is read or written.

The checks that matter most are the leakage ones. The GRU's emb_mean/emb_std
buffers and class weights must be computed from that fit's training clips
alone, which is checked bit-for-bit against an independent computation from
the cache files, and no clip may ever be trained on (in any augmented variant)
in a fold that evaluates it.

calibrate_gate.py runs on the runtime venv, so its checks run it as a
subprocess under BT_RUNTIME_PYTHON (default server_py/.venv/bin/python).

Usage (from server_py/, training venv):
    .venv-train/bin/python training/test_crossval_guards.py
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.labels import LABELS
from tests.checks import check, finish
from training import crossval as CV
from training import train as T
from training.cache import split_entries
from training.splits import group_key, make_folds

SERVER_PY = Path(__file__).resolve().parent.parent
RUNTIME_PYTHON = Path(os.environ.get("BT_RUNTIME_PYTHON") or SERVER_PY / ".venv" / "bin" / "python")

FRAMES, WINDOW, DIM, VARIANTS = 8, 4, 1280, 3
CFG = {"repeats": 2, "n_splits": 3, "n_inner": 2, "seed": 0}


def build_cache(cache: Path, *, per_class: dict[str, int] | None = None, dominant: int = 6,
                seed: int = 0) -> dict:
    """One clip per source group (a distinct date-hour, see splits.group_key).
    Class 0 gets `dominant` extra groups so the majority class is the same in
    every outer-train set, which makes pooled majority balanced accuracy
    exactly 1/len(LABELS) and not just per fold."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, 1, (len(LABELS), DIM)).astype(np.float32)
    emb = cache / "emb"
    emb.mkdir(parents=True)
    per_class = per_class or {label: 4 + (dominant if c == 0 else 0) for c, label in enumerate(LABELS)}
    clips, n = [], 0
    for c, label in enumerate(LABELS):
        for _ in range(per_class.get(label, 0)):
            n += 1
            fn = f"recording-clip-2026-{1 + n // 28:02d}-{1 + n % 28:02d}T10-00-00-000Z.webm"
            variants = []
            for k in range(VARIANTS):
                x = centers[c] * 0.6 + rng.normal(0, 1, (FRAMES, DIM))
                np.save(emb / f"{fn}.v{k}.npy", x.astype(np.float32))
                variants.append(f"{fn}.v{k}.npy")
            ev = centers[c] * 0.6 + rng.normal(0, 1, (WINDOW, DIM))
            np.save(emb / f"{fn}.eval{WINDOW}.npy", ev.astype(np.float32))
            clips.append({"filename": fn, "label": label, "group": group_key(fn),
                          "variants": variants, "eval": {str(WINDOW): f"{fn}.eval{WINDOW}.npy"}})
    manifest = {"schema": 2, "clips": clips, "window": WINDOW, "eval_windows": [WINDOW],
                "train_frames": FRAMES, "backbone": "synthetic", "labels": list(LABELS),
                "labels_sha256": "ab" * 32, "ethogram_version": "test", "aug_policy": "synthetic"}
    (cache / "manifest.json").write_text(json.dumps(manifest))
    return manifest


def independent_stats(cache: Path, manifest: dict, idx) -> tuple[torch.Tensor, torch.Tensor]:
    """emb_mean and class weights for a set of clips, straight from the cache
    files, without going through cache.py, train.py or dataset.py."""
    frames = []
    counts = [0] * len(LABELS)
    for i in sorted(int(j) for j in idx):
        clip = manifest["clips"][i]
        for name in clip["variants"]:
            frames.append(np.load(cache / "emb" / name))
            counts[LABELS.index(clip["label"])] += 1
    mean = torch.from_numpy(np.concatenate(frames, axis=0).mean(axis=0)).float()
    total = sum(counts)
    weights = torch.tensor([total / (len(LABELS) * max(1, c)) for c in counts], dtype=torch.float32)
    return mean, weights


def test_folds_partition_and_never_leak(manifest: dict, cache: Path) -> None:
    """Every clip is evaluated exactly once per repeat and never trained on there."""
    folds = make_folds([c["filename"] for c in manifest["clips"]], [c["label"] for c in manifest["clips"]],
                       n_splits=CFG["n_splits"], n_inner=CFG["n_inner"], repeats=CFG["repeats"],
                       seed=CFG["seed"])
    n = len(manifest["clips"])
    once, leak_free, groups_ok = True, True, True
    for r in range(CFG["repeats"]):
        seen = np.zeros(n, dtype=int)
        for f in (f for f in folds if f.repeat == r):
            seen[f.eval] += 1
            train, ev = split_entries(manifest, f.train, f.eval, cache_dir=cache)
            # Filenames carry every variant: an augmented copy of an eval clip
            # in training would show up here under the same filename.
            leak_free &= not ({e["filename"] for e in train} & {e["filename"] for e in ev})
            groups = [group_key(c["filename"]) for c in manifest["clips"]]
            groups_ok &= not ({groups[i] for i in f.train} & {groups[i] for i in f.eval})
            for itr, ival in f.inner:
                groups_ok &= not ({groups[i] for i in itr} & {groups[i] for i in ival})
                leak_free &= not (set(itr.tolist()) & set(f.eval.tolist()))
        once &= bool((seen == 1).all())
    check("every clip in exactly one outer eval fold per repeat", once)
    check("no eval clip (in any variant) is trained on in its fold", leak_free)
    check("no group straddles an outer or inner split", groups_ok)


def test_thin_class_fails_loudly(tmp: Path) -> None:
    """A 3-clip class stops crossval with the class name and the remedy."""
    counts = {label: 4 for label in LABELS}
    counts[LABELS[3]] = 2
    cache = tmp / "thin"
    build_cache(cache, per_class=counts)
    try:
        CV.crossval(cache_dir=cache, out_dir=tmp / "thin-out", repeats=1, n_splits=3, n_inner=2, quick=True)
        check("thin class raises", False, "crossval ran")
    except SystemExit as err:
        msg = str(err)
        check("thin class raises naming the class and the remedy",
              LABELS[3] in msg and "Clipping Mode" in msg, msg[:200])
    check("nothing written for a refused run", not (tmp / "thin-out" / "cv_result.json").exists())


def run_crossval(manifest: dict, cache: Path, out: Path) -> tuple[dict, list]:
    fits = []

    def on_fit(arm, fold, phase, train_idx, res):
        if arm == "gru":
            fits.append((fold, phase, np.asarray(train_idx), res["emb_mean"], res["class_weights"]))

    result = CV.crossval(cache_dir=cache, out_dir=out, quick=True, on_fit=on_fit,
                         repeats=CFG["repeats"], n_splits=CFG["n_splits"], n_inner=CFG["n_inner"],
                         seed=CFG["seed"])
    return result, fits


def test_standardization_is_per_fold(manifest: dict, cache: Path, fits: list) -> None:
    """emb_mean and class weights come from each fit's own training clips."""
    all_mean, _ = independent_stats(cache, manifest, range(len(manifest["clips"])))
    mean_ok, weights_ok, not_global = True, True, True
    outer = inner = 0
    for _fold, phase, train_idx, emb_mean, weights in fits:
        exp_mean, exp_weights = independent_stats(cache, manifest, train_idx)
        mean_ok &= torch.equal(emb_mean, exp_mean)
        weights_ok &= torch.equal(weights, exp_weights)
        not_global &= not torch.equal(emb_mean, all_mean)
        outer += phase == "outer"
        inner += phase == "inner"
    expected_outer = CFG["repeats"] * CFG["n_splits"]
    check("one outer gru fit per fold", outer == expected_outer, f"{outer} vs {expected_outer}")
    check("inner gru fits ran", inner == expected_outer * CFG["n_inner"], f"{inner}")
    check("emb_mean per fit is bit-identical to its training clips' mean", mean_ok)
    check("class weights per fit match its training clips' counts", weights_ok)
    check("no fit used the all-clips mean", not_global)


def test_results(result: dict, out: Path, manifest: dict) -> None:
    """cv_result.json, cv_oof.npz and runs/index.jsonl carry what Phase 3 needs."""
    on_disk = json.loads((out / "cv_result.json").read_text())
    check("cv_result.json matches the returned result", on_disk["run_id"] == result["run_id"])
    for key in ("run_id", "git_sha", "labels_sha256", "ethogram_version", "config", "arms", "paired"):
        check(f"cv_result has {key}", key in on_disk)
    arms = on_disk["arms"]
    check("all four arms evaluated", set(arms) == set(CV.ARMS), str(sorted(arms)))

    one_over_k = 1 / len(LABELS)
    per_fold_bal = [f["balanced_accuracy"] for f in arms["majority"]["per_fold"]]
    check("majority balanced accuracy is exactly 1/7 in every fold",
          all(b == one_over_k for b in per_fold_bal), str(per_fold_bal))
    pooled = arms["majority"]["pooled_oof"]["balanced_accuracy"]["mean"]
    check("majority pooled balanced accuracy is exactly 1/7", pooled == one_over_k, repr(pooled))
    floor = arms["majority"]["pooled_oof"]["macro_f1"]["mean"]
    for arm in ("gru", "static_logreg", "static_meanstd_mlp"):
        got = arms[arm]["pooled_oof"]["macro_f1"]["mean"]
        check(f"{arm} beats majority on learnable data", got > floor + 0.3, f"{got:.3f} vs {floor:.3f}")

    sel = arms["gru"]["selections"]
    check("gru selection recorded per outer fold", len(sel) == CFG["repeats"] * CFG["n_splits"])
    check("selections carry window, lr and epochs",
          all({"window", "lr", "epochs", "inner_val_loss"} <= set(s) for s in sel))
    rec = arms["gru"]["per_class"][LABELS[0]]
    check("per-class recall has a Wilson interval", len(rec["recall_ci95"]) == 2 and rec["support"] > 0)
    paired = on_disk["paired"]
    check("paired comparison is gru minus static_logreg", (paired["a"], paired["b"]) == ("gru", "static_logreg"))
    check("paired comparison uses the corrected resampled t-test",
          "p_two_sided" in paired["corrected_resampled_t"])
    const = CV.corrected_resampled_t(np.full(6, 0.1), n_train=80, n_eval=20)
    check("constant nonzero fold differences give no p-value, with a note",
          const["p_two_sided"] is None and "not estimable" in const.get("note", ""), str(const))
    check("all-zero fold differences give p=1",
          CV.corrected_resampled_t(np.zeros(6), n_train=80, n_eval=20)["p_two_sided"] == 1.0)

    oof = np.load(out / "cv_oof.npz")
    shape = oof["gru_P"].shape
    check("cv_oof.npz holds [repeats, clips, classes] per arm",
          shape == (CFG["repeats"], len(manifest["clips"]), len(LABELS)), str(shape))
    check("every OOF row is a probability distribution", np.allclose(oof["gru_P"].sum(axis=2), 1.0, atol=1e-5))

    lines = [json.loads(line) for line in (out / "runs" / "index.jsonl").read_text().splitlines()]
    check("index.jsonl has one line per arm", len(lines) == len(CV.ARMS), str(len(lines)))
    check("index.jsonl lines carry provenance",
          all({"run_id", "git_sha", "seed", "labels_sha256", "ethogram_version", "arm", "metrics", "created_at"} <= set(x)
              for x in lines))


def test_fits_are_deterministic(manifest: dict, cache: Path) -> None:
    """The same fit twice in one process gives identical probabilities."""
    idx = list(range(len(manifest["clips"])))
    train, ev = split_entries(manifest, idx[::2], idx[1::2], cache_dir=cache)
    a = CV.fit_gru(train, ev, lr=1e-3, seed=7, epochs=3, patience=5, early_stop=False)["probs"]
    CV.fit_gru(train, ev, lr=3e-4, seed=8, epochs=2, patience=5, early_stop=False)  # disturb global RNG
    b = CV.fit_gru(train, ev, lr=1e-3, seed=7, epochs=3, patience=5, early_stop=False)["probs"]
    check("a gru fit is reproducible from its own seed", np.array_equal(a, b))


def test_all_clips_training(cache: Path, out: Path) -> None:
    """train.py --all-clips uses CV's selections and reports no accuracy."""
    result = T.train_all_clips(cache, out, seed=0)
    tr = json.loads((out / "train_result.json").read_text())
    check("train_result.json written with the window export_onnx.py needs", tr.get("window") == WINDOW)
    check("train_result.json points at cv_result.json", tr.get("cv_result") == "cv_result.json")
    quotable = [k for k in tr if "acc" in k or "f1" in k or "kappa" in k or k == "confusion_matrix"]
    check("train_result.json carries no quotable accuracy", not quotable, str(quotable))
    check("head_state.pt written", (out / "head_state.pt").is_file())
    all_mean, _ = independent_stats(cache, json.loads((cache / "manifest.json").read_text()),
                                    range(result["n_clips"]))
    check("all-clips head standardizes on all clips", torch.equal(result["emb_mean"], all_mean))

    sels = [{"window": 4, "lr": 1e-3, "epochs": 5}, {"window": 8, "lr": 1e-3, "epochs": 9},
            {"window": 8, "lr": 3e-4, "epochs": 30}, {"window": 4, "lr": 1e-3, "epochs": 7}]
    agg = T.aggregate_selection(sels, default_window=8)
    check("aggregation: modal lr, tie on window goes to the default, median epochs",
          agg == {"window": 8, "lr": 1e-3, "epochs": 8}, str(agg))

    stale = json.loads((out / "cv_result.json").read_text())
    stale["labels_sha256"] = "cd" * 32
    (out / "cv_result.json").write_text(json.dumps(stale))
    try:
        T.train_all_clips(cache, out, seed=0)
        check("all-clips refuses a cv_result from other labels", False, "did not raise")
    except SystemExit as err:
        check("all-clips refuses a cv_result from other labels", "Re-run crossval.py" in str(err), str(err)[:200])


def test_calibrate_gate_uses_oof(out: Path) -> None:
    """calibrate_gate.py reads OOF probabilities and never touches an ONNX model."""
    source = (SERVER_PY / "training" / "calibrate_gate.py").read_text()
    check("calibrate_gate loads no ONNX model",
          not any(s in source for s in ("backbone_int8", "head.onnx", "inference.backbone",
                                        "inference.temporal_head", "InferenceSession")))
    if not RUNTIME_PYTHON.is_file():
        check("runtime venv available for calibrate_gate", False, f"{RUNTIME_PYTHON} missing; set BT_RUNTIME_PYTHON")
        return
    run = subprocess.run([str(RUNTIME_PYTHON), "-m", "training.calibrate_gate", "--oof", str(out / "cv_oof.npz")],
                         cwd=SERVER_PY, capture_output=True, text=True, timeout=120)
    check("calibrate_gate runs on the OOF file", run.returncode == 0, run.stdout[-400:] + run.stderr[-400:])
    check("it reports out-of-fold probabilities", "Out-of-fold probabilities for" in run.stdout)
    missing = subprocess.run([str(RUNTIME_PYTHON), "-m", "training.calibrate_gate", "--oof", str(out / "nope.npz")],
                             cwd=SERVER_PY, capture_output=True, text=True, timeout=120)
    check("calibrate_gate refuses without crossval output",
          missing.returncode != 0 and "Run training/crossval.py" in missing.stderr, missing.stderr[-300:])


def test_parallel_matches_serial(cache: Path, out: Path, tmp: Path) -> None:
    """--jobs does not change the result: parallel folds reproduce the serial run."""
    par = tmp / "parallel"
    CV.crossval(cache_dir=cache, out_dir=par, quick=True, arms=("static_logreg", "gru"), repeats=1,
                jobs=2, n_splits=CFG["n_splits"], n_inner=CFG["n_inner"], seed=CFG["seed"])
    for arm in ("static_logreg", "gru"):
        serial = np.load(out / "cv_oof.npz")[f"{arm}_P"][0]
        parallel = np.load(par / "cv_oof.npz")[f"{arm}_P"][0]
        check(f"repeat 0 {arm} OOF probabilities identical at jobs=2", np.array_equal(serial, parallel),
              f"max abs diff {np.abs(serial - parallel).max():g}")


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="bunnycv-"))
    try:
        cache, out = tmp / "cache", tmp / "out"
        manifest = build_cache(cache)
        print(f"synthetic cache: {len(manifest['clips'])} clips")

        print(f"\n{test_folds_partition_and_never_leak.__doc__.splitlines()[0]}")
        test_folds_partition_and_never_leak(manifest, cache)
        print(f"\n{test_thin_class_fails_loudly.__doc__.splitlines()[0]}")
        test_thin_class_fails_loudly(tmp)

        print("\nrunning crossval --quick on the synthetic cache")
        result, fits = run_crossval(manifest, cache, out)
        for fn, args in (
            (test_standardization_is_per_fold, (manifest, cache, fits)),
            (test_results, (result, out, manifest)),
            (test_fits_are_deterministic, (manifest, cache)),
            (test_parallel_matches_serial, (cache, out, tmp)),
            (test_calibrate_gate_uses_oof, (out,)),
            (test_all_clips_training, (cache, out)),
        ):
            print(f"\n{fn.__doc__.splitlines()[0]}")
            fn(*args)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    finish("crossval guard")


if __name__ == "__main__":
    main()
