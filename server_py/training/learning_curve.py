"""Measured training-size experiments on the reference grouped outer folds.

CPU-only, sequential, using crossval.run_fold for the same inner selection and
outer refit as the primary experiment. Fractions refer to whole source groups.
Small subsets without class support fail rather than changing the split rules.
"""
import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.labels import LABELS
from training.pose_features import fold_membership, reference_folds, sha256
from training.splits import Fold, check_support, group_key

DEFAULT_FRACTIONS = (0.25, 0.5, 0.75, 1.0)


def training_subsets(folds, names, labels, *, fractions, seed, n_inner):
    fractions = tuple(sorted(set(fractions)))
    if not fractions or any(not np.isfinite(f) or not 0 < f <= 1 for f in fractions):
        raise ValueError("fractions must be finite in (0, 1]")
    y = np.array([LABELS.index(label) for label in labels])
    groups = np.array([group_key(name) for name in names])
    plans = {fraction: [] for fraction in fractions}
    for fold in folds:
        ordered = sorted(set(groups[fold.train]))
        rng_seed = int(np.random.SeedSequence([seed, fold.repeat, fold.outer, 713]).generate_state(1)[0])
        np.random.default_rng(rng_seed).shuffle(ordered)
        sizes = [int(np.ceil(len(ordered) * fraction)) for fraction in fractions]
        if len(set(sizes)) != len(sizes):
            raise ValueError("requested fractions select identical numbers of groups. Use fewer fractions or collect more groups")
        for fraction, size in zip(fractions, sizes, strict=True):
            selected = set(ordered[:size])
            train = np.array([i for i in fold.train if groups[i] in selected], dtype=int)
            where = f"learning curve fraction={fraction}, repeat={fold.repeat}, outer={fold.outer}"
            check_support([labels[i] for i in train], groups[train].tolist(), LABELS, n_inner, where=where)
            inner_seed = int(np.random.SeedSequence([seed, fold.repeat, fold.outer]).generate_state(1)[0])
            relative = list(StratifiedGroupKFold(n_splits=n_inner, shuffle=True, random_state=inner_seed)
                            .split(np.zeros((len(train), 1)), y[train], groups[train]))
            inner = tuple((train[a], train[b]) for a, b in relative)
            for k, (a, b) in enumerate(inner):
                for part, indices in (("train", a), ("val", b)):
                    check_support([labels[i] for i in indices], groups[indices].tolist(), LABELS, 1,
                                  where=f"{where}, inner={k} {part}")
            plans[fraction].append(Fold(fold.repeat, fold.outer, train, fold.eval, inner))
    return plans


def run(*, cache_dir, manifest_path, reference_path, oof_path, out_dir,
        fractions=DEFAULT_FRACTIONS, arms=("static_logreg", "gru")):
    # Import only here so subset tests do not import torch or touch its thread state.
    import torch

    from training.cache import load_manifest
    from training.crossval import ARMS, git_sha, run_fold, summarise

    reference = json.loads(reference_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    if not arms or set(arms) - set(ARMS):
        raise ValueError(f"learning curve arms must be drawn from {ARMS}")
    config = reference["config"]
    loaded = load_manifest(cache_dir, config["aug_policy"])
    if loaded != manifest:
        raise ValueError("--manifest must match the primary policy manifest loaded from --cache-dir")
    for key in ("backbone", "aug_policy"):
        if manifest.get(key) != config.get(key):
            raise ValueError(f"reference CV and learning cache disagree on {key}")
    with np.load(oof_path, allow_pickle=False) as archive:
        arrays = dict(archive)
    comparator = "static_logreg" if "static_logreg" in reference["arms"] else next(iter(reference["arms"]))
    folds, y, _ = reference_folds(manifest, reference, arrays, comparator)
    names = arrays["filenames"].tolist()
    plans = training_subsets(folds, names, [c["label"] for c in manifest["clips"]],
                             fractions=fractions, seed=config["seed"], n_inner=config["n_inner"])
    paths = {"manifest": manifest_path, "reference_result": reference_path, "reference_oof": oof_path}
    provenance = {key: {"path": str(path.resolve()), "sha256": sha256(path)} for key, path in paths.items()}
    out_dir.mkdir(parents=True, exist_ok=True)
    points = []
    saved_threads = torch.get_num_threads()
    try:
        for number, (fraction, subset_folds) in enumerate(plans.items()):
            results, predictions = {}, {}
            for arm in arms:
                runs = [run_fold(arm, manifest, fold, config["windows"], seed=config["seed"],
                                 quick=config["quick"], epochs=config["epochs_cap"],
                                 patience=config["patience"], cache_dir=cache_dir)
                        for fold in subset_folds]
                summary, p = summarise(runs, y, config["repeats"],
                                       groups=[group_key(name) for name in names], seed=config["seed"])
                results[arm], predictions[f"{arm}_P"] = summary, p
            membership = fold_membership(subset_folds, names)
            created = datetime.now(UTC)
            result = {
                "run_id": f"learning-{created.strftime('%Y%m%dT%H%M%S%fZ')}-{number}",
                "created_at": created.isoformat(), "git_sha": git_sha(),
                "labels": list(LABELS), "labels_sha256": manifest["labels_sha256"],
                "ethogram_version": manifest["ethogram_version"], "n_clips": len(names),
                "config": {**config, "jobs": 1, "learning_curve": {
                    "group_fraction": fraction,
                    "n_train_per_fold": [len(f.train) for f in subset_folds],
                    "n_groups_per_fold": [len({group_key(names[i]) for i in f.train}) for f in subset_folds],
                    "subset_rule": "seeded nested prefixes of complete outer-training groups"}},
                "reference_run_id": reference["run_id"], "inputs": provenance,
                "implementation_sha256": {"learning_curve": sha256(Path(__file__)),
                                          "crossval": sha256(Path(__file__).with_name("crossval.py"))},
                "folds": membership,
                "folds_sha256": hashlib.sha256(json.dumps(membership, sort_keys=True).encode()).hexdigest(),
                "arms": results,
            }
            directory = out_dir / f"point-{number}"
            directory.mkdir(exist_ok=True)
            np.savez(directory / "cv_oof.npz", filenames=np.array(names), y=y, **predictions)
            result["oof_sha256"] = sha256(directory / "cv_oof.npz")
            (directory / "cv_result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
            points.append({"result": f"point-{number}/cv_result.json", "oof": f"point-{number}/cv_oof.npz"})
            print(f"fraction {fraction:g}: outer train clip counts {result['config']['learning_curve']['n_train_per_fold']}", flush=True)
    finally:
        torch.set_num_threads(saved_threads)
    spec = {"schema": 1, "reference_run_id": reference["run_id"], "points": points, "inputs": provenance}
    (out_dir / "learning_curve.json").write_text(json.dumps(spec, indent=2) + "\n")
    return spec


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache-dir", type=Path, default=Path("models/cache"))
    parser.add_argument("--manifest", type=Path, default=Path("models/cache/manifest.json"))
    parser.add_argument("--reference", type=Path, default=Path("models/cv_result.json"))
    parser.add_argument("--oof", type=Path, default=Path("models/cv_oof.npz"))
    parser.add_argument("--out-dir", type=Path, default=Path("models/learning"))
    parser.add_argument("--fractions", type=float, nargs="+", default=list(DEFAULT_FRACTIONS))
    parser.add_argument("--arms", nargs="+", default=["static_logreg", "gru"],
                        choices=("majority", "static_logreg", "static_meanstd_mlp", "gru"))
    args = parser.parse_args()
    try:
        run(cache_dir=args.cache_dir, manifest_path=args.manifest, reference_path=args.reference,
            oof_path=args.oof, out_dir=args.out_dir, fractions=args.fractions, arms=args.arms)
    except (OSError, ValueError, KeyError) as err:
        parser.exit(2, f"learning_curve: {err}\n")
    print(f"Wrote {args.out_dir / 'learning_curve.json'}")


if __name__ == "__main__":
    main()
