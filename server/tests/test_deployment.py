"""Promoted window and metadata must drive runtime, independent of later training."""
import contextlib
import hashlib
import io
import json
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.routers import model
from inference.deployment import read_promoted
from scripts import bench_pi
from shared.labels import LABELS
from tests.checks import check, finish


def exercise_model_bundle(root):
    source = root / "bundle-source"
    source.mkdir()
    artifacts = {"backbone_int8.onnx": b"backbone", "head.onnx": b"head graph",
                 "head.onnx.data": b"external head weights"}
    for name, data in artifacts.items():
        (source / name).write_bytes(data)
    manifest = {"labels": LABELS, "window": 12, "view": "full", "artifacts": {
        name: hashlib.sha256(data).hexdigest() for name, data in artifacts.items()
    }}
    (source / "promoted.json").write_text(json.dumps({"manifest": manifest}))
    tool = Path(__file__).resolve().parents[2] / "deploy" / "pi" / "model_bundle.py"
    archive = root / "model.tar.gz"

    def run(*args):
        return subprocess.run([sys.executable, str(tool), *map(str, args)],
                              capture_output=True, text=True, timeout=10)

    result = run("pack", source, archive)
    check("model bundle packs a promotion and its external weights", result.returncode == 0,
          result.stderr)
    if result.returncode != 0:
        return
    destination = root / "bundle-destination"
    destination.mkdir()
    with tarfile.open(archive) as bundle:
        check("model bundle includes every manifest artifact and the promotion record",
              set(bundle.getnames()) == {*artifacts, "promoted.json"})
        bundle.extractall(destination, filter="data")
    result = run("verify", destination)
    check("transferred bundle verifies independently of training files", result.returncode == 0,
          result.stderr)
    (destination / "head.onnx.data").unlink()
    result = run("verify", destination)
    check("model bundle refuses missing external weights", result.returncode != 0
          and "missing head.onnx.data" in result.stderr, result.stderr)
    (destination / "head.onnx.data").write_bytes(b"wrong weights")
    result = run("verify", destination)
    check("model bundle refuses changed external weights", result.returncode != 0
          and "sha256 mismatch for head.onnx.data" in result.stderr, result.stderr)


def exercise(root):
    check("absent promotion has no deployment record", read_promoted(root) is None)
    record = {"manifest": {"labels": LABELS, "window": 12, "view": "full", "cv_arm": "gru@albu_v1"},
              "gate_result": {"cv": {"gru_macro_f1": 0.61}}}
    (root / "promoted.json").write_text(json.dumps(record))
    (root / "head.onnx").touch()
    (root / "backbone_int8.onnx").touch()
    # A newer candidate must not change what the deployed panel describes.
    (root / "train_result.json").write_text(json.dumps({"window": 4, "val_acc": 1.0}))
    with patch.object(model, "MODELS_ONNX_DIR", root):
        result = model._read_deployed_model()
    check("panel reads promoted labels and window", result["labels"] == LABELS and result["window"] == 12)
    check("panel shows recorded OOF score instead of a later selection score",
          result["cvMacroF1"] == 0.61 and "valAcc" not in result)
    check("panel records the deployed arm", result["arm"] == "gru@albu_v1")
    (root / "promoted.json").unlink()
    with patch.object(model, "MODELS_ONNX_DIR", root):
        check("panel reports no promoted model", model._read_deployed_model() == {"exists": False})
    for name, changed in (("crop", {"view": "crop"}), ("stale vocabulary", {"labels": ["normal"]}),
                          ("zero window", {"window": 0})):
        bad = {**record, "manifest": {**record["manifest"], **changed}}
        (root / "promoted.json").write_text(json.dumps(bad))
        try:
            read_promoted(root)
            check(f"runtime refuses {name}", False)
        except ValueError:
            check(f"runtime refuses {name}", True)


def exercise_benchmark(root):
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


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="bunny-deployment-") as directory:
        exercise(Path(directory))
        exercise_model_bundle(Path(directory))
        exercise_benchmark(Path(directory))
    finish("deployment")
