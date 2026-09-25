"""Regression gate for the Phase 0 export fixes (docs/upgrade-plan.md).

Four things that were silently wrong and would each ship a broken model:
  0.1 head.onnx exported with a STATIC batch axis, so the head could not be
      evaluated in batches. torch 2.14 defaults to the dynamo exporter, which
      specializes a batch-1 example input to a literal 1 and rejects the
      dynamic window axis outright.
  0.2 export_head used strict load_state_dict; that must STAY strict, because
      strict=False would "succeed" against a stale checkpoint by leaving the
      final layer at its random init and export a model that predicts noise.
  0.3 int8 calibration frames were drawn without filtering on LABELS, so
      `out_of_view` clips (a valid label, not a model class) could set the
      quantized backbone's dynamic range.
  0.4 training/model.py hardcoded NUM_CLASSES = 5, which survived the ethogram
      v2 migration.

Nothing here writes into models/ or touches server/labels.json; the
calibration check runs against a temp labels file with module globals
repointed. No trained checkpoint is required.

Usage (from server_py/, training venv):
    .venv-train/bin/python -m training.test_train_guards
    .venv-train/bin/python -m training.test_export_guards
"""
import json
import sys
import tempfile
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

warnings.filterwarnings("ignore")

import onnx
import torch

from shared.labels import ALL_CODES, CODING_ONLY, LABELS
from training import export_onnx
from training.model import EMBEDDING_DIM, NUM_CLASSES, TemporalHead

failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    if ok:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}{': ' + detail if detail else ''}")
        failures.append(name)


def _onnx_shape(value_info) -> list:
    return [d.dim_param or d.dim_value for d in value_info.type.tensor_type.shape.dim]


def test_head_exports_with_dynamic_batch_axis() -> None:
    """0.1: head.onnx must keep batch and window dynamic."""
    head = TemporalHead()
    head.eval()
    out = Path(tempfile.mkdtemp()) / "head.onnx"
    export_onnx.HEAD_ONNX, real = out, export_onnx.HEAD_ONNX
    try:
        # Call the real exporter, not a copy of its arguments, so this test
        # fails if someone drops dynamo=False or edits the dynamic_axes.
        torch.save(head.state_dict(), state := Path(tempfile.mkdtemp()) / "head_state.pt")
        export_onnx.HEAD_STATE_PATH, real_state = state, export_onnx.HEAD_STATE_PATH
        try:
            export_onnx.export_head(window=8)
        finally:
            export_onnx.HEAD_STATE_PATH = real_state
    finally:
        export_onnx.HEAD_ONNX = real

    model = onnx.load(str(out))
    got_in = _onnx_shape(model.graph.input[0])
    got_out = _onnx_shape(model.graph.output[0])
    check("0.1 embeddings input keeps batch and window dynamic",
          got_in == ["batch", "window", EMBEDDING_DIM], f"got {got_in}")
    check("0.1 logits output has a dynamic batch axis",
          got_out[0] == "batch", f"got {got_out}")
    check("0.1 logits width matches the ethogram",
          got_out[-1] == len(LABELS), f"got {got_out[-1]}, ethogram has {len(LABELS)}")


def test_head_export_rejects_a_stale_checkpoint() -> None:
    """0.2: a checkpoint with the wrong class count must raise, not load."""
    stale = TemporalHead(num_classes=len(LABELS) - 1)
    state = Path(tempfile.mkdtemp()) / "stale.pt"
    torch.save(stale.state_dict(), state)

    export_onnx.HEAD_STATE_PATH, real = state, export_onnx.HEAD_STATE_PATH
    try:
        export_onnx.export_head(window=8)
        check("0.2 a stale checkpoint raises instead of exporting", False,
              "export_head accepted a head with the wrong class count")
    except RuntimeError as err:
        check("0.2 a stale checkpoint raises instead of exporting",
              "size mismatch" in str(err).lower() or "shape" in str(err).lower(),
              f"raised but not about shape: {err}")
    except Exception as err:  # noqa: BLE001 - any other failure is still a failure
        check("0.2 a stale checkpoint raises instead of exporting", False,
              f"raised {type(err).__name__}: {err}")
    finally:
        export_onnx.HEAD_STATE_PATH = real


def test_calibration_skips_non_model_classes() -> None:
    """0.3: int8 calibration must draw only from model classes."""
    tmp = Path(tempfile.mkdtemp())
    labels_path = tmp / "labels.json"
    # Three real clips, three out_of_view. Filenames need not exist on disk:
    # the filter runs before any decode, and a missing file is skipped anyway.
    labels_path.write_text(json.dumps({
        "recording-clip-a.webm": LABELS[0],
        "recording-clip-b.webm": LABELS[1],
        "recording-clip-c.webm": LABELS[2],
        "recording-clip-d.webm": CODING_ONLY[0],
        "recording-clip-e.webm": CODING_ONLY[0],
        "recording-clip-f.webm": CODING_ONLY[0],
    }))

    seen: list[str] = []

    def fake_extract(path, n):  # noqa: ANN001
        seen.append(Path(path).name)
        raise ValueError("no such clip")  # skipped by the existing handler

    export_onnx.LABELS_JSON, real_labels = labels_path, export_onnx.LABELS_JSON
    export_onnx.extract_frames, real_extract = fake_extract, export_onnx.extract_frames
    try:
        export_onnx.pick_calibration_frames()
    finally:
        export_onnx.LABELS_JSON = real_labels
        export_onnx.extract_frames = real_extract

    out_of_view_seen = [f for f in seen if f in {"recording-clip-d.webm",
                                                 "recording-clip-e.webm",
                                                 "recording-clip-f.webm"}]
    check("0.3 calibration considered the model-class clips", len(seen) == 3, f"saw {seen}")
    check("0.3 calibration skipped every out_of_view clip",
          not out_of_view_seen, f"sampled {out_of_view_seen}")


def test_calibration_fails_loudly_with_nothing_to_calibrate_on() -> None:
    """0.3: an all-out_of_view labels file must not silently calibrate on nothing."""
    tmp = Path(tempfile.mkdtemp())
    labels_path = tmp / "labels.json"
    labels_path.write_text(json.dumps({"recording-clip-a.webm": CODING_ONLY[0]}))

    export_onnx.LABELS_JSON, real = labels_path, export_onnx.LABELS_JSON
    try:
        export_onnx.pick_calibration_frames()
        check("0.3 an empty calibration set exits with a reason", False,
              "returned instead of exiting")
    except SystemExit as err:
        check("0.3 an empty calibration set exits with a reason",
              "calibration" in str(err).lower(), f"exited with: {err}")
    finally:
        export_onnx.LABELS_JSON = real


def test_class_count_is_derived_from_the_ethogram() -> None:
    """0.4: no hardcoded class count anywhere in the model definition."""
    check("0.4 model.NUM_CLASSES tracks the ethogram",
          NUM_CLASSES == len(LABELS), f"{NUM_CLASSES} vs {len(LABELS)}")
    check("0.4 a default-constructed head is the ethogram's width",
          TemporalHead().fc.out_features == len(LABELS),
          f"{TemporalHead().fc.out_features} vs {len(LABELS)}")
    check("0.4 out_of_view is a valid code but not a model class",
          set(ALL_CODES) - set(LABELS) == set(CODING_ONLY),
          f"ALL_CODES={ALL_CODES} LABELS={LABELS}")


def main() -> None:
    tests = (
        test_head_exports_with_dynamic_batch_axis,
        test_head_export_rejects_a_stale_checkpoint,
        test_calibration_skips_non_model_classes,
        test_calibration_fails_loudly_with_nothing_to_calibrate_on,
        test_class_count_is_derived_from_the_ethogram,
    )
    for fn in tests:
        print(f"\n{fn.__doc__.splitlines()[0]}")
        fn()

    if failures:
        print(f"\nFAIL: {len(failures)} check(s) failed")
        sys.exit(1)
    print("\nPASS: all export guard checks passed")


if __name__ == "__main__":
    main()
