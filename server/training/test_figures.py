"""Hermetic figure exports, numerical traceability and missing-input failures."""
import copy
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.labels import ETHOGRAM_VERSION
from tests.checks import check, finish
from training.figures import KINDS, Inputs, generate, prepare
from training.pose_features import fold_membership, sha256
from training.recoding_report import crosstab
from training.test_pose_features import reference, rejects


def build_reports(tmp):
    _, cv, arrays, folds = reference(tmp)
    names = arrays["filenames"].tolist()
    table = crosstab({name: "normal" for name in names}, {name: cv["labels"][i] for name, i in zip(names, arrays["y"], strict=True)})
    (tmp / "recoding_report.json").write_text(json.dumps(table))
    (tmp / "reliability.json").write_text(json.dumps({"kappa": .5, "n": 12, "ethogram_version": ETHOGRAM_VERSION}))
    (tmp / "candidate").mkdir()
    (tmp / "candidate" / "quant_result.json").write_text(json.dumps({"fp32_macro_f1": .6, "int8_macro_f1": .59, "delta": .01, "n": len(names)}))
    (tmp / "learning").mkdir()
    points = []
    for number, count in enumerate([21, 42]):
        point = tmp / "learning" / f"point-{number}"
        point.mkdir()
        data = copy.deepcopy(cv)
        data["reference_run_id"] = cv["run_id"]
        data["config"]["learning_curve"] = {"n_train_per_fold": [count] * len(folds)}
        data["folds"] = fold_membership(folds, names)
        for f in data["folds"]:
            f["train"] = f["train"][:count]
        (point / "cv_result.json").write_text(json.dumps(data))
        np.savez(point / "cv_oof.npz", **arrays)
        points.append({"result": f"point-{number}/cv_result.json", "oof": f"point-{number}/cv_oof.npz"})
    (tmp / "learning" / "learning_curve.json").write_text(json.dumps({"schema": 1, "reference_run_id": cv["run_id"], "points": points}))
    return cv, arrays


def test_exports(tmp):
    cv, arrays = build_reports(tmp)
    output = tmp / "figures"
    made = generate(models_dir=tmp, out_dir=output, arm="static_logreg")
    check("all six figure kinds export standalone SVG and PDF", made == list(KINDS)
          and all((output / f"{kind}.{suffix}").stat().st_size > 1000 for kind in KINDS for suffix in ("svg", "pdf")))
    checked = True
    for kind in made:
        provenance = json.loads((output / f"{kind}.json").read_text())
        checked &= all(sha256(Path(source["path"])) == source["sha256"] for source in provenance["sources"])
        checked &= all(sha256(output / path) == digest for path, digest in provenance["outputs"].items())
        checked &= "sources" in (output / f"{kind}.svg").read_text()
    check("every chart records source hashes, plotted data and artifact hashes", checked)
    ablation = json.loads((output / "head_ablation.json").read_text())["data"]
    check("ablation uses recorded pooled OOF values", ablation["mean"] == [cv["arms"]["static_logreg"]["pooled_oof"]["macro_f1"]["mean"]])
    confusion = json.loads((output / "oof_confusion.json").read_text())["data"]
    check("confusion matrix counts one repeat without inflating sample size", sum(map(sum, confusion["matrix"])) == len(arrays["y"])
          and confusion["matrix"] == cv["arms"]["static_logreg"]["per_repeat_confusion"][0])
    quant = (output / "quantization.csv").read_text()
    check("quantization is labeled in-sample and lacks invented latency", "in-sample deployment check" in quant and "latency" not in quant and "0.59" in quant)
    learning = json.loads((output / "learning_curve.json").read_text())["data"]
    check("learning curve uses actual recorded per-fold clip counts", [p["n_train_mean"] for p in learning["points"]] == [21, 42])
    human = json.loads((output / "kappa.json").read_text())["data"]
    check("human subset population and missing interval stay explicit", human["human_n"] == 12
          and human["sd"][-1] is None and "not an upper bound" in human["note"])


def test_refusals(tmp):
    out = tmp / "missing-output"
    empty = tmp / "empty"
    empty.mkdir()
    rejects("missing requested report fails actionably", lambda: generate(models_dir=empty, out_dir=out), "Produce this report first")
    check("missing inputs produce no partial figure directory", not out.exists())
    rejects("input paths cannot escape models tree", lambda: Inputs(tmp).json("../outside.json"), "must be under")
    bad = copy.deepcopy(json.loads((tmp / "cv_result.json").read_text()))
    bad["arms"]["static_logreg"]["pooled_oof"]["macro_f1"]["mean"] = .123
    (tmp / "bad.json").write_text(json.dumps(bad))
    rejects("mismatched result JSON and OOF fail", lambda: Inputs(tmp).cv("bad.json"), "disagrees")
    rejects("unavailable arm fails", lambda: prepare(Inputs(tmp), "oof_confusion", arm="missing"), "available --arm")
    point = tmp / "learning" / "point-0" / "cv_result.json"
    original = point.read_text()
    bad = json.loads(original)
    del bad["config"]["learning_curve"]
    point.write_text(json.dumps(bad))
    rejects("learning curve refuses missing measured training counts", lambda: prepare(Inputs(tmp), "learning_curve"), "actual clip counts")
    point.write_text(original)
    bad = json.loads(original)
    bad["folds"][0]["eval"] = bad["folds"][1]["eval"]
    point.write_text(json.dumps(bad))
    rejects("learning points with different evaluation folds fail", lambda: prepare(Inputs(tmp), "learning_curve"), "must share")
    point.write_text(original)
    report = tmp / "reliability.json"
    original = report.read_text()
    bad = json.loads(original)
    bad["ethogram_version"] = "old"
    report.write_text(json.dumps(bad))
    rejects("different reliability vocabulary fails", lambda: prepare(Inputs(tmp), "kappa"), "different ethograms")
    report.write_text(original)


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="bunny-figures-test-") as directory:
        tmp = Path(directory)
        test_exports(tmp)
        test_refusals(tmp)
    finish("paper figures")
