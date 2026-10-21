"""Regression gate for training/evaluate.py's deployment gates G1-G7.

Every gate must pass on a good candidate and fail on one crafted to break
exactly it, and the result must flow through promote.py: a passing candidate
installs, a failing one is refused.

Hermetic. The real MobileNet backbone is replaced by a tiny conv net exported
to ONNX and int8-quantized with the same onnxruntime quantize_static the real
export uses. Clips are synthetic frames whose colour encodes the class, so a
small TemporalHead fitted on their embeddings separates them and the
in-sample numbers G4 and G5 look at are meaningful. Cache, cv_result.json,
train_result.json, candidate and models dirs all live in a temp dir.

Usage (from server/, training venv):
    .venv-train/bin/python training/test_evaluate_gates.py
"""
import contextlib
import io
import json
import shutil
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
from onnxruntime.quantization import CalibrationDataReader, QuantType, quantize_static

from shared.frames import normalize
from shared.labels import ETHOGRAM_VERSION, LABELS
from tests.checks import check, finish
from training import evaluate as E
from training import export_onnx, promote
from training.model import EMBEDDING_DIM, TemporalHead

WINDOW = 4
CLIPS_PER_CLASS = 2
LABELS_SHA = "a" * 64


class StubBackbone(nn.Module):
    """[B,3,224,224] -> [B,1280], standing in for MobileNet."""

    def __init__(self, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.conv = nn.Conv2d(3, 8, kernel_size=3, stride=8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(8, EMBEDDING_DIM)

    def forward(self, x):
        return self.fc(self.pool(torch.relu(self.conv(x))).flatten(1))


def synth_frames(k: int, clip: int) -> np.ndarray:
    """W uint8 frames whose mean colour encodes class k."""
    rng = np.random.default_rng(100 * k + clip)
    hue = np.array([(k * 37) % 256, (k * 91 + 40) % 256, (k * 173 + 80) % 256], dtype=np.float32)
    frames = hue + rng.normal(0, 6, size=(WINDOW, 224, 224, 3))
    return np.clip(frames, 0, 255).astype(np.uint8)


def build_cache(root: Path, backbone: StubBackbone) -> Path:
    cache = root / "cache"
    (cache / "frames").mkdir(parents=True)
    (cache / "emb").mkdir()
    clips = []
    for k, label in enumerate(LABELS):
        for c in range(CLIPS_PER_CLASS):
            fn = f"recording-clip-2026-09-{k + 1:02d}T{c:02d}-00-00-000Z.webm"
            name = f"{fn}.eval{WINDOW}.npy"
            frames = synth_frames(k, c)
            np.save(cache / "frames" / name, frames)
            with torch.no_grad():
                emb = backbone(torch.from_numpy(np.stack([normalize(f) for f in frames]))).numpy()
            np.save(cache / "emb" / name, emb.astype(np.float32))
            clips.append({"filename": fn, "label": label, "group": fn, "variants": [], "eval": {str(WINDOW): name}})
    manifest = {"schema": 2, "clips": clips, "window": WINDOW, "eval_windows": [WINDOW], "train_frames": WINDOW,
                "backbone": "stub", "labels": list(LABELS), "labels_sha256": LABELS_SHA,
                "ethogram_version": ETHOGRAM_VERSION, "aug_policy": "none"}
    (cache / "manifest.json").write_text(json.dumps(manifest))
    return cache


def fit_head(cache: Path, num_classes: int = len(LABELS)) -> TemporalHead:
    manifest = json.loads((cache / "manifest.json").read_text())
    x = torch.from_numpy(np.stack([np.load(cache / "emb" / c["eval"][str(WINDOW)]) for c in manifest["clips"]]))
    y = torch.tensor([LABELS.index(c["label"]) % num_classes for c in manifest["clips"]])
    flat = x.reshape(-1, EMBEDDING_DIM)
    head = TemporalHead(num_classes=num_classes, emb_mean=flat.mean(0), emb_std=flat.std(0) + 1e-6)
    torch.manual_seed(0)
    opt = torch.optim.Adam(head.parameters(), lr=1e-2)
    head.train()
    for _ in range(150):
        opt.zero_grad()
        nn.functional.cross_entropy(head(x), y).backward()
        opt.step()
    head.eval()
    return head


class FramesReader(CalibrationDataReader):
    def __init__(self, cache: Path):
        names = sorted((cache / "frames").iterdir())
        self._it = iter({"pixel_values": normalize(f)[np.newaxis].astype(np.float32)}
                        for p in names for f in np.load(p)[:1])

    def get_next(self):
        return next(self._it, None)


def export_backbones(out: Path, cache: Path, backbone: StubBackbone, int8: str = "quantized") -> None:
    dummy = torch.randn(1, 3, 224, 224)
    fp32 = out / "backbone_fp32.onnx"
    torch.onnx.export(backbone, dummy, str(fp32), input_names=["pixel_values"], output_names=["embedding"],
                      dynamic_axes={"pixel_values": {0: "batch"}, "embedding": {0: "batch"}},
                      opset_version=17, dynamo=False)
    int8_path = out / "backbone_int8.onnx"
    if int8 == "quantized":
        quantize_static(str(fp32), str(int8_path), FramesReader(cache), per_channel=True,
                        weight_type=QuantType.QInt8)
    else:  # "perturbed": a different network entirely, standing in for a quantization collapse
        torch.onnx.export(StubBackbone(seed=99), dummy, str(int8_path), input_names=["pixel_values"],
                          output_names=["embedding"],
                          dynamic_axes={"pixel_values": {0: "batch"}, "embedding": {0: "batch"}},
                          opset_version=17, dynamo=False)


def export_head(head: TemporalHead, out: Path, static_batch: bool = False) -> None:
    axes = {"embeddings": {1: "window"}} if static_batch else \
        {"embeddings": {0: "batch", 1: "window"}, "logits": {0: "batch"}}
    torch.onnx.export(head, torch.randn(1, WINDOW, EMBEDDING_DIM), str(out / "head.onnx"),
                      input_names=["embeddings"], output_names=["logits"], dynamic_axes=axes,
                      opset_version=17, dynamo=False)


def cv_result(gru: float = 0.60, logreg: float = 0.50, labels_sha: str = LABELS_SHA,
              fold_f1=(0.55, 0.60, 0.65, 0.58, 0.62)) -> dict:
    def arm(mean, folds):
        return {"pooled_oof": {"macro_f1": {"mean": mean, "sd": 0.02}},
                "per_fold": [{"macro_f1": f} for f in folds]}
    return {"run_id": "run-1", "labels_sha256": labels_sha, "ethogram_version": ETHOGRAM_VERSION,
            "labels": list(LABELS),
            "arms": {"gru": arm(gru, fold_f1), "static_logreg": arm(logreg, [logreg] * len(fold_f1))},
            "paired": {"corrected_resampled_t": {"p_two_sided": 0.21}}}


class World:
    """One temp dir holding a shared cache and backbone, plus per-scenario
    candidate and models dirs."""

    def __init__(self):
        self.root = Path(tempfile.mkdtemp(prefix="bunnygate-"))
        self.backbone = StubBackbone().eval()
        self.cache = build_cache(self.root, self.backbone)
        self.good_head = fit_head(self.cache)
        self.n = 0

    def scenario(self, *, head: TemporalHead | None = None, state: dict | None = None, static_batch=False,
                 int8="quantized", cv: dict | None = None, train: dict | None = None,
                 promoted: dict | None = None) -> tuple[Path, Path]:
        self.n += 1
        models = self.root / f"models{self.n}"
        cand = models / "candidate"
        cand.mkdir(parents=True)
        head = head or self.good_head
        export_backbones(cand, self.cache, self.backbone, int8)
        export_head(head, cand, static_batch)
        torch.save(state if state is not None else head.state_dict(), models / "head_state.pt")
        (models / "cv_result.json").write_text(json.dumps(cv if cv is not None else cv_result()))
        trained = {"mode": "all-clips", "window": WINDOW, "labels_sha256": LABELS_SHA,
                   "cv_run_id": "run-1", "cv_arm": "gru", "aug_policy": "legacy", "view": "full",
                   "ethogram_version": ETHOGRAM_VERSION, **(train or {})}
        (models / "train_result.json").write_text(json.dumps(trained))
        if promoted is not None:
            (models / "promoted.json").write_text(json.dumps(promoted))
        with contextlib.redirect_stdout(io.StringIO()):
            export_onnx.write_manifest(cand, WINDOW, trained)
        return models, cand

    def evaluate(self, models: Path, cand: Path) -> dict:
        return E.evaluate(candidate_dir=cand, models_dir=models, cache_dir=self.cache)

    def close(self):
        shutil.rmtree(self.root, ignore_errors=True)


def gates_failed(result: dict) -> list[str]:
    return sorted(k for k, g in result["gates"].items() if not g["passed"])


def test_good_candidate(w: World) -> dict:
    """A well-formed candidate passes every gate and promotes."""
    models, cand = w.scenario()
    res = w.evaluate(models, cand)
    for gid in ("G1", "G2", "G3", "G4", "G5", "G6", "G7"):
        check(f"{gid} passes on a good candidate", res["gates"][gid]["passed"], res["gates"][gid].get("detail", ""))
    check("gate_passed is true", res["gate_passed"] is True)
    manifest = json.loads((cand / "manifest.json").read_text())
    written = json.loads((cand / "gate_result.json").read_text())
    check("gate_result.json is written to the candidate", written["gate_passed"] is True)
    check("gate_result artifacts map equals the manifest's", written["artifacts"] == manifest["artifacts"])
    check("G7 with no promoted.json says so", res["gates"]["G7"]["detail"] == "no deployed baseline")
    check("G6 reports the paired p without requiring it",
          res["gates"]["G6"]["corrected_paired_p"] == 0.21 and res["gates"]["G6"]["significance_required"] is False)
    check("quant_result.json carries the G4 numbers", (cand / "quant_result.json").is_file())

    record = promote.promote(models_dir=models)
    check("promote installs a passing candidate",
          all((models / name).is_file() for name in manifest["artifacts"]) and record["forced"] is None)
    check("promoted.json carries the CV headline G7 reads next time",
          (record["gate_result"] or {}).get("cv", {}).get("gru_macro_f1") == 0.60)
    parity = subprocess.run(
        [sys.executable, "-m", "training.verify_predictor_parity", "--models-dir", str(models),
         "--cache-dir", str(w.cache)], capture_output=True, text=True,
        cwd=Path(__file__).resolve().parent.parent, timeout=60,
    )
    check("promoted ONNX graphs match the runtime ring buffer", parity.returncode == 0,
          parity.stdout + parity.stderr)
    return record


def test_g1_wrong_width(w: World) -> None:
    """G1: a head trained for the wrong number of classes fails."""
    narrow = fit_head(w.cache, num_classes=len(LABELS) - 2)
    res = w.evaluate(*w.scenario(head=narrow))
    check("G1 fails on a head with the wrong width", not res["gates"]["G1"]["passed"], res["gates"]["G1"].get("detail", ""))
    check("and the whole gate fails", res["gate_passed"] is False)


def test_g2_static_batch(w: World) -> None:
    """G2: a head exported with a static batch axis fails."""
    res = w.evaluate(*w.scenario(static_batch=True))
    check("G2 fails on a statically batched head", not res["gates"]["G2"]["passed"], res["gates"]["G2"].get("detail", ""))


def test_g3_torch_onnx_mismatch(w: World) -> None:
    """G3: head_state.pt that is not the exported head fails."""
    state = {k: v.clone() for k, v in w.good_head.state_dict().items()}
    state["fc.weight"] += 0.05
    res = w.evaluate(*w.scenario(state=state))
    check("G3 fails when torch and ONNX heads differ", not res["gates"]["G3"]["passed"], res["gates"]["G3"].get("detail", ""))
    check("G3 records the logit difference", res["gates"]["G3"].get("max_abs_logit_diff", 0) > E.LOGIT_TOLERANCE)


def test_g4_int8_collapse(w: World) -> None:
    """G4: an int8 backbone that no longer computes the fp32 function fails."""
    res = w.evaluate(*w.scenario(int8="perturbed"))
    check("G4 fails on a collapsed int8 backbone", not res["gates"]["G4"]["passed"], res["gates"]["G4"].get("detail", ""))


def test_g5_constant_head(w: World) -> None:
    """G5: a head that predicts one class for everything fails."""
    head = fit_head(w.cache)
    with torch.no_grad():
        head.fc.weight.zero_()
        head.fc.bias.zero_()
        head.fc.bias[2] = 5.0
    res = w.evaluate(*w.scenario(head=head))
    check("G5 fails on a constant classifier", not res["gates"]["G5"]["passed"], res["gates"]["G5"].get("detail", ""))
    check("G3 still passes on it (the gates are independent)", res["gates"]["G3"]["passed"])


def test_g6_baseline_and_labels(w: World) -> None:
    """G6: losing to static_logreg, or CV on different labels, fails."""
    res = w.evaluate(*w.scenario(cv=cv_result(gru=0.45, logreg=0.50)))
    check("G6 fails when gru loses to static_logreg", not res["gates"]["G6"]["passed"], res["gates"]["G6"].get("detail", ""))
    res = w.evaluate(*w.scenario(cv=cv_result(gru=0.50, logreg=0.50)))
    check("G6 fails on a tie (strictly better is required)", not res["gates"]["G6"]["passed"])
    res = w.evaluate(*w.scenario(cv=cv_result(labels_sha="b" * 64)))
    check("G6 fails when cv_result was computed on other labels", not res["gates"]["G6"]["passed"],
          res["gates"]["G6"].get("detail", ""))
    res = w.evaluate(*w.scenario(train={"window": WINDOW, "labels_sha256": LABELS_SHA, "cv_run_id": "run-0"}))
    check("G6 fails when the head took its config from another CV run", not res["gates"]["G6"]["passed"])

    models, cand = w.scenario(cv=cv_result(gru=0.45, logreg=0.50))
    w.evaluate(models, cand)
    try:
        promote.promote(models_dir=models)
        check("promote refuses a failing candidate", False, "it promoted")
    except promote.PromoteRefused as err:
        check("promote refuses a failing candidate", "G6" in str(err), str(err))


def test_g7_against_deployed(w: World) -> None:
    """G7: worse than the deployed model by more than one fold-sd fails."""
    def promoted(f1, version=ETHOGRAM_VERSION):
        return {"gate_result": {"cv": {"gru_macro_f1": f1, "ethogram_version": version}}}
    # Candidate 0.60, fold sd of (0.55, 0.60, 0.65, 0.58, 0.62) is about 0.038.
    res = w.evaluate(*w.scenario(promoted=promoted(0.70)))
    check("G7 fails 0.10 below the deployed model", not res["gates"]["G7"]["passed"], res["gates"]["G7"].get("detail", ""))
    res = w.evaluate(*w.scenario(promoted=promoted(0.62)))
    check("G7 passes within one fold-sd of the deployed model", res["gates"]["G7"]["passed"],
          res["gates"]["G7"].get("detail", ""))
    res = w.evaluate(*w.scenario(promoted=promoted(0.95, version="1.0")))
    check("G7 does not compare across ethogram versions", res["gates"]["G7"]["passed"]
          and "not comparable" in res["gates"]["G7"]["detail"])
    res = w.evaluate(*w.scenario(promoted={"gate_result": None, "forced": "first deploy"}))
    check("G7 treats a forced promote with no gate as no baseline",
          res["gates"]["G7"]["detail"] == "no deployed baseline")


def test_selected_arm_and_provenance(w: World) -> None:
    """G6 follows the trained arm, even when the primary GRU scores better."""
    cv = cv_result(gru=0.90)
    cv["arms"]["gru@albu_v1"] = cv_result(gru=0.40)["arms"]["gru"]
    cv["n_clips"] = len(LABELS) * CLIPS_PER_CLASS
    cv["config"] = {"n_splits": 5}
    cv["arm_cache"] = {"gru@albu_v1": {"aug_policy": "albu_v1", "view": "full"}}
    manifest = json.loads((w.cache / "manifest.json").read_text())
    manifest["aug_policy"] = "albu_v1"
    (w.cache / "manifest.albu_v1.json").write_text(json.dumps(manifest))
    train = {"window": WINDOW, "labels_sha256": LABELS_SHA, "cv_run_id": "run-1",
             "cv_arm": "gru@albu_v1", "aug_policy": "albu_v1", "view": "full"}
    models, cand = w.scenario(cv=cv, train=train)
    inp = E.Inputs(candidate_dir=cand, models_dir=models, cache_dir=w.cache,
                   cv_result=models / "cv_result.json", head_state=models / "head_state.pt",
                   train_result=models / "train_result.json")
    g6 = E.gate_g6(inp)
    check("selected augmentation arm cannot borrow primary GRU's better score",
          not g6["passed"] and g6["gru_macro_f1"] == 0.40, str(g6))
    headline = E._gru_headline(cv, inp.arm)
    check("deployed headline belongs to the selected arm",
          headline["arm"] == inp.arm and headline["gru_macro_f1"] == 0.40)
    inp.manifest["cv_arm"] = "gru"
    check("candidate and training arm mismatch blocks the gate",
          "disagree on cv_arm" in E.gate_g6(inp)["detail"])
    del inp.manifest["cv_arm"]
    inp.cv["config"]["quick"] = True
    check("a quick smoke run is not deployment evidence", "smoke run" in E.gate_g6(inp)["detail"])
    inp.cv["config"]["quick"] = False
    inp.cv["ethogram_version"] = "1.0"
    check("CV from an obsolete ethogram blocks deployment", "CV result uses a different ethogram" in E.gate_g6(inp)["detail"])
    inp.cv["ethogram_version"] = ETHOGRAM_VERSION
    inp.cv["labels"] = list(reversed(LABELS))
    check("a reordered CV vocabulary blocks deployment", "different ordered label" in E.gate_g6(inp)["detail"])


def test_tampered_candidate(w: World) -> None:
    """A candidate whose files no longer match its manifest is not evaluated."""
    models, cand = w.scenario()
    with (cand / "head.onnx").open("ab") as f:
        f.write(b"\0")
    try:
        w.evaluate(models, cand)
        check("evaluate refuses a tampered candidate", False, "it evaluated")
    except SystemExit as err:
        check("evaluate refuses a tampered candidate", "manifest" in str(err), str(err))
    check("and writes no gate result for it", not (cand / "gate_result.json").exists())


def main() -> None:
    w = World()
    try:
        for fn in (
            test_good_candidate,
            test_g1_wrong_width,
            test_g2_static_batch,
            test_g3_torch_onnx_mismatch,
            test_g4_int8_collapse,
            test_g5_constant_head,
            test_g6_baseline_and_labels,
            test_g7_against_deployed,
            test_selected_arm_and_provenance,
            test_tampered_candidate,
        ):
            print(f"\n{fn.__doc__.splitlines()[0]}")
            with contextlib.redirect_stderr(io.StringIO()):
                fn(w)
    finally:
        w.close()
    finish("evaluate gate")


if __name__ == "__main__":
    main()
