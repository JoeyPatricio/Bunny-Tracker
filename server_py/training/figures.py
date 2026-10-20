"""Export paper figures from recorded model reports and OOF predictions.

No training, annotation or placeholder results. Missing requested inputs are
errors. Each SVG/PDF has a JSON sidecar with plotted data and source hashes.
See paper/README.md for inputs, including the learning-curve run contract.
"""
import argparse
import csv
import io
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.metrics import confusion_matrix
from training.pose_features import sha256, validate_oof

KINDS = ("head_ablation", "oof_confusion", "recoding", "kappa", "quantization", "learning_curve")
ROOT = Path(__file__).resolve().parents[2]


class Inputs:
    def __init__(self, models_dir):
        self.root = Path(models_dir).resolve()
        self.sources = {}

    def path(self, name):
        path = (self.root / name).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError(f"figure inputs must be under {self.root}: {name}")
        if not path.is_file():
            raise ValueError(f"Missing {path}. Produce this report first, or select available figures with --only")
        self.sources[str(path)] = {"path": str(path), "sha256": sha256(path)}
        return path

    def json(self, name):
        path = self.path(name)
        data = json.loads(path.read_text())
        for key in ("run_id", "git_sha", "labels_sha256", "ethogram_version"):
            if key in data:
                self.sources[str(path)][key] = data[key]
        return data

    def cv(self, result="cv_result.json", oof="cv_oof.npz"):
        data = self.json(result)
        for key in ("run_id", "labels_sha256", "ethogram_version"):
            if not data.get(key):
                raise ValueError(f"{result} lacks {key}. Use a versioned CV report from training.crossval")
        if not data.get("arms"):
            raise ValueError(f"{result} contains no measured arms")
        path = self.path(oof)
        if data.get("oof_sha256") and data["oof_sha256"] != sha256(path):
            raise ValueError(f"{result} OOF hash does not match {oof}")
        with np.load(path, allow_pickle=False) as archive:
            arrays = dict(archive)
        for arm in data["arms"]:
            validate_oof(data, arrays, arm=arm)
        return data, arrays


def _finite(value, name, low=None, high=None):
    if not isinstance(value, (int, float)) or not np.isfinite(value):
        raise ValueError(f"{name} must be a measured finite number")
    if (low is not None and value < low) or (high is not None and value > high):
        raise ValueError(f"{name} is outside [{low}, {high}]")
    return value


def prepare(inputs, kind, *, arm="gru", repeat=0):
    if kind in ("head_ablation", "oof_confusion", "kappa"):
        cv, arrays = inputs.cv()
        arms = list(cv["arms"])
        if kind == "head_ablation":
            return {"arms": arms, "metric": "macro_f1", "uncertainty": "standard deviation across repeats, not a confidence interval",
                    "mean": [cv["arms"][a]["pooled_oof"]["macro_f1"]["mean"] for a in arms],
                    "sd": [cv["arms"][a]["pooled_oof"]["macro_f1"]["sd"] for a in arms]}
        if kind == "oof_confusion":
            if arm not in arms or not 0 <= repeat < cv["config"]["repeats"]:
                raise ValueError(f"choose an available --arm {arms} and --repeat in [0,{cv['config']['repeats'] - 1}]")
            cm = confusion_matrix(arrays["y"], arrays[f"{arm}_P"][repeat].argmax(1), len(cv["labels"]))
            return {"arm": arm, "repeat": repeat, "labels": cv["labels"], "matrix": cm.tolist(),
                    "n": int(cm.sum()), "normalization": "counts, one OOF prediction per clip"}
        reliability = inputs.json("reliability.json")
        if reliability.get("ethogram_version") != cv["ethogram_version"]:
            raise ValueError("reliability.json and cv_result.json use different ethograms. Recompute on the frozen recode")
        human = _finite(reliability["kappa"], "reliability kappa", -1, 1)
        n = _finite(reliability["n"], "reliability n", 1)
        return {"arms": arms + ["Human blind re-code"],
                "mean": [cv["arms"][a]["pooled_oof"]["kappa"]["mean"] for a in arms] + [human],
                "sd": [cv["arms"][a]["pooled_oof"]["kappa"]["sd"] for a in arms] + [None],
                "human_n": n, "model_n": cv["n_clips"],
                "note": "model: pooled OOF, mean +/- repeat SD. Human: blind intra-observer subset, no estimated interval. Different populations and coding vocabularies, not an upper bound"}
    if kind == "recoding":
        report = inputs.json("recoding_report.json")
        matrix = np.array([[report["table"][r][c] for c in report["columns"]] for r in report["rows"]])
        if matrix.size == 0 or matrix.dtype.kind not in "iu" or (matrix < 0).any() or matrix.sum() != report["n_v1"]:
            raise ValueError("recoding_report.json needs nonnegative integer counts summing to n_v1")
        return {"rows": report["rows"], "columns": report["columns"], "matrix": matrix.tolist(),
                "n_recoded": report["n_recoded"], "n_v1": report["n_v1"],
                "note": "unlabeled clips stay visible. Counts describe recoding progress, not model accuracy"}
    if kind == "quantization":
        q = inputs.json("candidate/quant_result.json")
        n = _finite(q["n"], "quantization n", 1)
        fp = _finite(q["fp32_macro_f1"], "fp32 macro-F1", 0, 1)
        int8 = _finite(q["int8_macro_f1"], "int8 macro-F1", 0, 1)
        if not np.isclose(fp - int8, q["delta"], atol=1e-8):
            raise ValueError("quantization delta disagrees with the recorded macro-F1 scores")
        return {"columns": ["Model", "Macro-F1", "Clips", "Evaluation"],
                "rows": [["fp32", fp, n, "in-sample deployment check"],
                         ["int8 candidate", int8, n, "in-sample deployment check"]],
                "note": "G4 uses all cached clips for the final all-clips model. These are deployment scores. Per-tensor and latency measurements are absent and are not inferred"}
    if kind == "learning_curve":
        spec_path = Path("learning/learning_curve.json")
        spec = inputs.json(spec_path)
        if spec.get("schema") != 1 or len(spec.get("points", [])) < 2:
            raise ValueError("learning_curve.json needs schema=1 and at least two points with result/oof paths. Run measured training-size experiments first")
        points, base, base_arrays = [], None, None
        for point in spec["points"]:
            cv, arrays = inputs.cv(spec_path.parent / point["result"], spec_path.parent / point["oof"])
            metadata = cv["config"].get("learning_curve", {})
            counts = np.asarray(metadata.get("n_train_per_fold", []))
            if counts.shape != (cv["config"]["repeats"] * cv["config"]["n_splits"],) or counts.dtype.kind not in "iu" or (counts <= 0).any():
                raise ValueError(f"{point['result']} must record config.learning_curve.n_train_per_fold, actual clip counts for every outer fit")
            if (counts >= cv["n_clips"]).any():
                raise ValueError("learning-curve training counts must exclude outer evaluation clips")
            folds = cv.get("folds", [])
            if len(folds) != len(counts) or [len(f["train"]) for f in folds] != counts.tolist():
                raise ValueError("learning-curve counts must match recorded outer-training membership")
            if cv.get("reference_run_id") != spec.get("reference_run_id"):
                raise ValueError("learning-curve point belongs to a different reference run")
            if base is not None:
                keys = ("labels_sha256", "ethogram_version", "labels", "n_clips")
                config_keys = ("seed", "repeats", "n_splits", "n_inner", "backbone", "aug_policy")
                if (any(cv.get(k) != base.get(k) for k in keys)
                        or any(cv["config"].get(k) != base["config"].get(k) for k in config_keys)
                        or set(cv["arms"]) != set(base["arms"])
                        or not np.array_equal(arrays["filenames"], base_arrays["filenames"])
                        or not np.array_equal(arrays["y"], base_arrays["y"])
                        or [(f["repeat"], f["outer"], f["eval"]) for f in folds]
                           != [(f["repeat"], f["outer"], f["eval"]) for f in base["folds"]]):
                    raise ValueError("learning-curve runs must share clips, labels, seeds, grouped splits, backbone, augmentation and arms")
            base, base_arrays = cv, arrays
            points.append({"n_train_mean": float(counts.mean()), "n_train_per_fold": counts.tolist(),
                           "arms": {a: cv["arms"][a]["pooled_oof"]["macro_f1"] for a in cv["arms"]}})
        points.sort(key=lambda p: p["n_train_mean"])
        if len({p["n_train_mean"] for p in points}) != len(points):
            raise ValueError("learning-curve points must represent different training sizes")
        return {"points": points, "metric": "macro_f1", "uncertainty": "standard deviation across repeats, not a confidence interval"}
    raise ValueError(f"unknown figure {kind}")


def render(kind, data):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    if kind in ("head_ablation", "kappa"):
        if kind == "kappa":
            ax.bar(np.arange(len(data["arms"]) - 1), data["mean"][:-1], yerr=data["sd"][:-1], capsize=3)
            ax.bar(len(data["arms"]) - 1, data["mean"][-1])
        else:
            ax.bar(np.arange(len(data["arms"])), data["mean"], yerr=data["sd"], capsize=3)
        ax.set_xticks(np.arange(len(data["arms"])), data["arms"], rotation=25, ha="right")
        ax.set_ylabel("Pooled OOF macro-F1" if kind == "head_ablation" else "Cohen's kappa")
        max_sd = max((sd for sd in data["sd"] if sd is not None), default=0)
        ax.set_ylim(min(-0.1, min(data["mean"]) - max_sd - .05), 1.05)
        if kind == "kappa":
            ax.set_title(f"Model CV pool n={data['model_n']}, human blind subset n={data['human_n']}\n"
                         "Human point has no estimated interval and is not an upper bound", fontsize=10)
        else:
            ax.set_title("Arms on identical grouped folds, mean +/- repeat SD")
    elif kind in ("oof_confusion", "recoding"):
        matrix = np.asarray(data["matrix"])
        rows = data.get("rows", data.get("labels"))
        cols = data.get("columns", data.get("labels"))
        fig.set_size_inches(max(8, len(cols) * 1.1), max(5, len(rows) * .65))
        ax.imshow(matrix, cmap="Blues", aspect="auto")
        ax.set_xticks(np.arange(len(cols)), cols, rotation=40, ha="right")
        ax.set_yticks(np.arange(len(rows)), rows)
        for (i, j), value in np.ndenumerate(matrix):
            ax.text(j, i, str(value), ha="center", va="center",
                    color="white" if value > matrix.max() / 2 else "black")
        ax.set_xlabel("Predicted label" if kind == "oof_confusion" else "v2 code")
        ax.set_ylabel("True label" if kind == "oof_confusion" else "v1 label")
        ax.set_title(f"{data['arm']}, OOF repeat {data['repeat']}, n={data['n']}" if kind == "oof_confusion"
                     else f"Recoding progress: {data['n_recoded']}/{data['n_v1']}")
    elif kind == "quantization":
        ax.axis("off")
        shown = [[row[0], f"{row[1]:.4f}", str(row[2]), row[3]] for row in data["rows"]]
        table = ax.table(cellText=shown, colLabels=data["columns"], loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        ax.set_title("Quantization deployment check, in-sample")
    else:
        points = data["points"]
        for arm in points[0]["arms"]:
            ax.errorbar([p["n_train_mean"] for p in points],
                        [p["arms"][arm]["mean"] for p in points],
                        yerr=[p["arms"][arm]["sd"] for p in points], marker="o", capsize=3, label=arm)
        ax.set_xlabel("Mean training clips per outer fold")
        ax.set_ylabel("Pooled OOF macro-F1, mean +/- repeat SD")
        ax.legend()
    return fig


def generate(*, models_dir, out_dir, kinds=KINDS, arm="gru", repeat=0):
    # Validate every requested input before writing any figures.
    prepared = []
    for kind in kinds:
        inputs = Inputs(models_dir)
        data = prepare(inputs, kind, arm=arm, repeat=repeat)
        prepared.append((kind, data, list(inputs.sources.values())))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for kind, data, sources in prepared:
        fig = render(kind, data)
        provenance = {"schema": 1, "figure": kind, "created_at": datetime.now(UTC).isoformat(),
                      "sources": sources, "data": data, "generator_sha256": sha256(Path(__file__)),
                      "versions": {"numpy": np.__version__, "matplotlib": matplotlib.__version__}}
        description = json.dumps({"figure": kind, "sources": sources})
        outputs = {}
        try:
            for suffix, metadata in (("svg", {"Description": description}), ("pdf", {"Subject": description})):
                path = out_dir / f"{kind}.{suffix}"
                fig.savefig(path, metadata=metadata)
                outputs[path.name] = sha256(path)
        finally:
            plt.close(fig)
        if kind == "quantization":
            text = io.StringIO(newline="")
            writer = csv.writer(text)
            writer.writerow(data["columns"])
            writer.writerows(data["rows"])
            path = out_dir / "quantization.csv"
            path.write_text(text.getvalue())
            outputs[path.name] = sha256(path)
        provenance["outputs"] = outputs
        (out_dir / f"{kind}.json").write_text(json.dumps(provenance, indent=2, allow_nan=False) + "\n")
    return [kind for kind, _, _ in prepared]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models-dir", type=Path, default=ROOT / "server_py" / "models")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "paper" / "figures")
    parser.add_argument("--only", nargs="+", choices=KINDS, default=list(KINDS))
    parser.add_argument("--arm", default="gru")
    parser.add_argument("--repeat", type=int, default=0)
    args = parser.parse_args()
    try:
        made = generate(models_dir=args.models_dir, out_dir=args.out_dir,
                        kinds=args.only, arm=args.arm, repeat=args.repeat)
    except (OSError, ValueError, KeyError, ImportError) as err:
        parser.exit(2, f"figures: {err}. See paper/README.md for input producers and schemas.\n")
    print(f"Wrote {', '.join(made)} to {args.out_dir}")


if __name__ == "__main__":
    main()
