"""Regression gate for the schema-2 embedding cache and the single-process decode.

Hermetic: synthesises short VP8 .webm clips with the bundled ffmpeg into a temp
dir, writes a temp labels.json, and builds the cache into a temp cache dir with
a stub backbone (no timm weights, CPU, fast). Nothing under server/ or models/
is read or written.

The invariant this exists for: an augmented variant of a clip must never
appear in a fold where that clip is being evaluated.

Usage (from server_py/, training venv):
    .venv-train/bin/python training/test_cache_v2.py
"""
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared import frames as F
from shared.labels import LABELS
from tests.checks import check, finish
from training import cache as C
from training import extract_frames as X
from training import train as T
from training.dataset import AUG_COPIES, TRAIN_FRAMES, WINDOW
from training.splits import make_folds

# Two model classes, four source groups each (distinct date-hours in the
# filename, see splits.group_key), so make_folds can run 2 outer x 2 inner.
CLASSES = [LABELS[0], LABELS[1]]
CLIPS = [
    (f"recording-clip-2026-08-{day:02d}T1{h}-00-00-000Z.webm", CLASSES[c])
    for c, day in enumerate((1, 2)) for h in range(4)
]


class StubBackbone(torch.nn.Module):
    """[T,3,224,224] -> [T,1280], deterministic and sensitive to the pixels,
    so a mirrored or jittered variant embeds differently from the original."""

    def forward(self, x):
        return x.flatten(1)[:, :: (3 * 224 * 224) // 1280][:, :1280].contiguous()


def make_clip(path: Path, hue: int) -> None:
    subprocess.run(
        [F.FFMPEG, "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=15:duration=2",
         "-vf", f"hue=h={hue}", "-c:v", "libvpx", "-b:v", "400k",
         "-loglevel", "error", "-y", str(path)],
        check=True,
    )


def setup(tmp: Path) -> tuple[Path, Path, Path]:
    rec = tmp / "recordings"
    rec.mkdir()
    for i, (fn, _) in enumerate(CLIPS):
        make_clip(rec / fn, hue=i * 40)
    make_clip(rec / "recording-clip-2026-08-03T10-00-00-000Z.webm", hue=300)
    labels = {fn: label for fn, label in CLIPS}
    labels["recording-clip-2026-08-03T10-00-00-000Z.webm"] = "out_of_view"
    labels_json = tmp / "labels.json"
    labels_json.write_text(json.dumps(labels, indent=2))
    return rec, labels_json, tmp / "cache"


def build(rec: Path, labels_json: Path, cache_dir: Path, counter: list) -> dict:
    real = X.ffmpeg_extract_frames

    def counting(path, n):
        counter.append((Path(path).name, n))
        return real(path, n=n)

    with patch.object(X, "ffmpeg_extract_frames", counting):
        return X.build_cache(cache_dir=cache_dir, labels_json=labels_json, recordings_dir=rec,
                             backbone=StubBackbone().eval(), backbone_name="stub")


def test_decode_parity(rec: Path) -> None:
    """The single-process decode must be byte-identical to the per-frame one."""
    for fn, _ in CLIPS[:3]:
        for n in (WINDOW, TRAIN_FRAMES):
            old = F.extract_frames_single(str(rec / fn), n)
            new = F.extract_frames(str(rec / fn), n)
            same = len(old) == len(new) and all(np.array_equal(a, b) for a, b in zip(old, new, strict=True))
            check(f"byte parity, {fn[-24:]} n={n}", same)
    path = str(rec / CLIPS[0][0])
    dup = F.read_frames(path, [0.5, 0.5, 0.0])
    check("a repeated timestamp yields the same frame twice",
          np.array_equal(dup[0], dup[1]) and np.array_equal(dup[0], F.read_frame_single(path, 0.5)))
    check("output order follows the requested order, not time order",
          np.array_equal(dup[2], F.read_frame_single(path, 0.0)))


def test_manifest(manifest: dict, labels_json: Path, cache_dir: Path) -> None:
    """Schema 2 fields, no split baked in, every file the right shape."""
    required = {"schema", "clips", "window", "eval_windows", "train_frames", "backbone",
                "labels", "labels_sha256", "ethogram_version", "aug_policy"}
    check("schema is 2", manifest.get("schema") == 2)
    check("every required field present", required <= set(manifest), f"missing {sorted(required - set(manifest))}")
    check("no train/val keys", not ({"train", "val"} & set(manifest)))
    check("labels_sha256 is the sha256 of labels.json",
          manifest["labels_sha256"] == hashlib.sha256(labels_json.read_bytes()).hexdigest())
    check("labels is the model vocabulary", manifest["labels"] == LABELS)
    names = [c["filename"] for c in manifest["clips"]]
    check("every model-class clip cached, out_of_view skipped",
          sorted(names) == sorted(fn for fn, _ in CLIPS), f"{len(names)} clips")
    check("clip entries carry filename, label, group, variants, eval",
          all({"filename", "label", "group", "variants", "eval"} <= set(c) for c in manifest["clips"]))

    clip = manifest["clips"][0]
    fr = np.load(cache_dir / "frames" / f"{clip['filename']}.npy")
    check("frames tier is uint8 [TRAIN_FRAMES,224,224,3]",
          fr.dtype == np.uint8 and fr.shape == (TRAIN_FRAMES, 224, 224, 3), f"{fr.dtype} {fr.shape}")
    check(f"{AUG_COPIES + 1} variants per clip", all(len(c["variants"]) == AUG_COPIES + 1 for c in manifest["clips"]))
    v = [np.load(cache_dir / "emb" / n) for n in clip["variants"]]
    check("variant embeddings are [TRAIN_FRAMES,1280]", all(e.shape == (TRAIN_FRAMES, 1280) for e in v))
    check("variants differ from each other", not np.array_equal(v[0], v[1]))
    ev = np.load(cache_dir / "emb" / clip["eval"][str(WINDOW)])
    check("eval embedding is [WINDOW,1280]", ev.shape == (WINDOW, 1280), str(ev.shape))


def test_frames_tier_reuse(rec, labels_json, cache_dir, first: dict) -> None:
    """A second run must not decode anything, and must reproduce the embeddings."""
    calls: list = []
    second = build(rec, labels_json, cache_dir, calls)
    check("second run decodes nothing", not calls, f"{len(calls)} decode call(s)")
    same = all(
        np.array_equal(np.load(cache_dir / "emb" / a), np.load(cache_dir / "emb" / b))
        for c1, c2 in zip(first["clips"], second["clips"], strict=True)
        for a, b in zip(c1["variants"], c2["variants"], strict=True)
    )
    check("variants are deterministic across runs", same)


def test_fold_invariant(manifest: dict, cache_dir: Path) -> None:
    """No variant of a clip ever trains in the fold that evaluates it."""
    filenames = [c["filename"] for c in manifest["clips"]]
    labels = [c["label"] for c in manifest["clips"]]
    folds = make_folds(filenames, labels, n_splits=2, n_inner=2, repeats=2, classes=CLASSES)
    leak, bad_eval, bad_train, wrong_tier = [], [], [], []
    evaluated: dict[int, list[str]] = {}
    for fold in folds:
        train, ev = C.fold_entries(manifest, fold, cache_dir=cache_dir)
        overlap = {e["filename"] for e in train} & {e["filename"] for e in ev}
        if overlap:
            leak.append((fold.repeat, fold.outer, sorted(overlap)))
        evaluated.setdefault(fold.repeat, []).extend(e["filename"] for e in ev)
        for e in ev:
            clip = manifest["clips"][filenames.index(e["filename"])]
            if e["variant"] != 0 or e["windows"] != [(0, WINDOW)] or e["embeddings"].shape[0] != WINDOW:
                bad_eval.append(e["filename"])
            if not np.array_equal(e["embeddings"], np.load(cache_dir / "emb" / clip["eval"][str(WINDOW)])):
                wrong_tier.append(e["filename"])
        per_clip: dict[str, set] = {}
        for e in train:
            per_clip.setdefault(e["filename"], set()).add(e["variant"])
            if e["windows"] != [(0, 8), (4, 12), (8, 16)]:
                bad_train.append(e["filename"])
        if any(v != set(range(AUG_COPIES + 1)) for v in per_clip.values()) or \
                len(per_clip) != len(fold.train):
            bad_train.append(f"fold {fold.outer}: variants or clips missing")
    check("no clip on both sides of any fold", not leak, str(leak[:3]))
    check("every clip evaluated exactly once per repeat",
          all(sorted(v) == sorted(filenames) for v in evaluated.values()))
    check("eval entries are variant 0, window [0, W] only", not bad_eval, str(bad_eval[:3]))
    check("eval entries use the eval sample, not a slice of the training sample", not wrong_tier)
    check("train entries carry every variant and every window start", not bad_train, str(bad_train[:3]))


def test_refusals(tmp: Path, manifest: dict, cache_dir: Path) -> None:
    """Schema 1 and a missing eval window fail with the fix in the message."""
    old = tmp / "old-cache"
    old.mkdir()
    (old / "manifest.json").write_text(json.dumps({"train": [], "val": [], "window": 8}))
    for name, fn in (
        ("load_manifest refuses schema 1", lambda: C.load_manifest(old)),
        ("split_entries refuses schema 1", lambda: C.split_entries({"train": [], "val": [], "window": 8}, [], [])),
        ("build_cache will not overwrite a schema-1 cache",
         lambda: X.build_cache(cache_dir=old, labels_json=tmp / "labels.json", recordings_dir=tmp,
                               backbone=StubBackbone())),
    ):
        try:
            fn()
            check(name, False, "did not raise")
        except SystemExit as err:
            check(name, "schema 1" in str(err) or "schema-1" in str(err), str(err)[:120])
    try:
        C.split_entries(manifest, [0], [1], window=12, cache_dir=cache_dir)
        check("an uncached eval window is refused", False, "did not raise")
    except SystemExit as err:
        check("an uncached eval window is refused", "--eval-windows" in str(err), str(err)[:120])


def test_legacy_split_for_train_py(manifest: dict, cache_dir: Path) -> None:
    """train.py's load_entries still works, on a deterministic legacy split."""
    with patch.object(T, "CACHE_DIR", cache_dir):
        train = T.load_entries(manifest, "train")
        val = T.load_entries(manifest, "val")
    tr, va = {e["filename"] for e in train}, {e["filename"] for e in val}
    check("legacy split has both sides", bool(tr) and bool(va), f"{len(tr)} train, {len(va)} val")
    check("legacy split is disjoint", not (tr & va))
    idx_train, idx_val = C.legacy_split_indices(manifest)
    names = [c["filename"] for c in manifest["clips"]]
    check("legacy split is the one legacy_split_indices names",
          tr == {names[i] for i in idx_train} and va == {names[i] for i in idx_val})


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="bunnycache-"))
    try:
        rec, labels_json, cache_dir = setup(tmp)
        print("\nSingle-process decode")
        test_decode_parity(rec)
        calls: list = []
        manifest = build(rec, labels_json, cache_dir, calls)
        check("first run decodes each clip once per tier",
              len(calls) == 2 * len(CLIPS), f"{len(calls)} calls for {len(CLIPS)} clips")
        for fn in (
            lambda: test_manifest(manifest, labels_json, cache_dir),
            lambda: test_frames_tier_reuse(rec, labels_json, cache_dir, manifest),
            lambda: test_fold_invariant(manifest, cache_dir),
            lambda: test_refusals(tmp, manifest, cache_dir),
            lambda: test_legacy_split_for_train_py(manifest, cache_dir),
        ):
            fn()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    finish("cache v2")


if __name__ == "__main__":
    main()
