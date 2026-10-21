"""Regression gate for the crop view.

What this pins down:
  - estimate_roi finds a moving object, falls back to full frame for a still
    clip, and ignores a global brightness shift and a scene-wide change.
  - The background for a clip comes from its own source group only, so no
    pixel from another fold's clips can shape its crop.
  - extract_frames --view crop writes its own manifest, training variants and
    eval embeddings, leaves the full view untouched, and a fallback clip
    embeds exactly as in the full view.
  - crossval.py runs crop only as the paired gru@crop arm, and train.py refuses
    to fit a crop arm for deployment.

Crop is UNEVALUATED; this checks that it is built correctly, not that it
helps. Hermetic: synthetic frames and clips, stub backbone, temp dirs.

Usage (from server/, training venv):
    .venv-train/bin/python training/test_crop.py
"""
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared import frames as F
from shared.labels import LABELS
from tests.checks import check, finish
from training import cache as C
from training import crop
from training import crossval as CV
from training import extract_frames as X
from training import test_cache_v2 as TC
from training import test_crossval_guards as TG
from training import train as T
from training.splits import group_key, make_folds


def square_clip(xs, *, y: int = 140, side: int = 28, n: int | None = None) -> np.ndarray:
    """Grey 224x224 frames with a white square at each x in xs."""
    frames = np.full((n or len(xs), 224, 224, 3), 90, dtype=np.uint8)
    for f, x in zip(frames, xs, strict=False):
        f[y:y + side, x:x + side] = 255
    return frames


def raises(fn, needle: str) -> tuple[bool, str]:
    try:
        fn()
    except SystemExit as err:
        return needle in str(err), str(err)[:200]
    return False, "did not raise"


def test_estimate_roi() -> None:
    """A moving square is boxed; still, relit and scene-wide clips are not."""
    xs = list(range(20, 116, 6))
    moving = square_clip(xs)
    roi = crop.estimate_roi(moving, crop.background([moving]))
    box = roi["box"]
    check("a moving square gets a box", box is not None, str(roi))
    if box:
        x0, y0, x1, y1 = (v * 224 for v in box)
        check("the box contains every position of the square",
              x0 <= min(xs) and x1 >= max(xs) + 28 and y0 <= 140 and y1 >= 168, str(box))
        check("the box is square and inside the frame",
              abs((box[2] - box[0]) - (box[3] - box[1])) < 1e-3 and 0 <= min(box) and max(box) <= 1, str(box))

    still = square_clip([60] * 16)
    roi = crop.estimate_roi(still, crop.background([still]))
    check("a still clip falls back to full frame", roi["box"] is None and "no motion blob" in roi["fallback"],
          str(roi))

    # Compensation is per frame against a fixed background (crop.py docstring:
    # a lighting change inside the background window is not compensated).
    relit = np.clip(moving.astype(np.int16) + 40, 0, 255).astype(np.uint8)
    check("a uniform brightness shift against the background does not change the box",
          crop.estimate_roi(relit, crop.background([moving]))["box"] == box)

    bumped = np.broadcast_to(np.linspace(0, 255, 224, dtype=np.uint8)[None, None, :, None],
                             moving.shape).copy()  # a different scene: dark-to-bright gradient
    blobs = [crop.frame_blob(f, crop.background([moving])) for f in bumped]
    check("a frame unlike the background everywhere (camera bumped) counts as no blob",
          all(b is None for b in blobs))

    big = square_clip(list(range(0, 165, 11)), y=80, side=60)
    roi = crop.estimate_roi(big, crop.background([big]))
    check("a box covering most of the frame falls back", roi["box"] is None and "box side" in roi["fallback"],
          str(roi))


def test_crop_frames() -> None:
    """crop_frames takes the boxed region and resizes it to the backbone's input."""
    hi = np.zeros((2, crop.HI_RES, crop.HI_RES, 3), dtype=np.uint8)
    half = crop.HI_RES // 2
    hi[:, half:, half:] = 200  # bottom-right quadrant
    out = crop.crop_frames(hi, [0.5, 0.5, 1.0, 1.0])
    check("crop output is [T,224,224,3] uint8", out.shape == (2, 224, 224, 3) and out.dtype == np.uint8)
    check("crop takes the boxed region", (out == 200).all())


def make_moving_clip(path: Path, x0: int, speed: int) -> None:
    subprocess.run(
        [F.FFMPEG, "-f", "lavfi", "-i", "color=c=0x406080:size=320x240:rate=15:duration=2",
         "-f", "lavfi", "-i", "color=c=white:size=40x30:rate=15:duration=2",
         "-filter_complex", f"[0][1]overlay=x='{x0}+t*{speed}':y=150",
         "-c:v", "libvpx", "-b:v", "800k", "-loglevel", "error", "-y", str(path)],
        check=True,
    )


def test_build_crop_view(tmp: Path) -> None:
    """--view crop shares the frames tier and leaves the full view alone."""
    rec = tmp / "crop-rec"
    rec.mkdir()
    moving = ["recording-clip-2026-08-01T10-00-00-000Z.webm", "recording-clip-2026-08-01T10-05-00-000Z.webm"]
    still = "recording-clip-2026-08-02T10-00-00-000Z.webm"
    make_moving_clip(rec / moving[0], 40, 60)
    make_moving_clip(rec / moving[1], 180, -50)
    make_moving_clip(rec / still, 100, 0)
    labels_json = tmp / "crop-labels.json"
    labels_json.write_text(json.dumps({moving[0]: LABELS[0], moving[1]: LABELS[1], still: LABELS[0]}))
    cache = tmp / "crop-cache"
    kw = dict(cache_dir=cache, labels_json=labels_json, recordings_dir=rec,
              backbone=TC.StubBackbone().eval(), backbone_name="stub")
    full = X.build_cache(**kw)
    full_bytes = {p.relative_to(cache): p.read_bytes() for p in cache.rglob("*") if p.is_file()}
    cropped = X.build_cache(**kw, view="crop")

    check("crop has its own manifest", C.manifest_path(cache, view="crop").name == "manifest.crop.json"
          and C.load_manifest(cache, view="crop")["view"] == "crop")
    check("full manifest records view full", full["view"] == "full")
    check("crop manifest records the crop parameters", cropped.get("crop_params") == crop.CROP_PARAMS)
    check("full-view files untouched by the crop build",
          all((cache / rel).read_bytes() == data for rel, data in full_bytes.items()))
    names = [n for c in cropped["clips"] for n in [*c["variants"], *c["eval"].values()]]
    check("crop embeddings are separate files",
          all(".crop." in n for n in names) and not {Path("emb") / n for n in names} & set(full_bytes))

    rois = cropped["roi"]
    check("every moving clip is cropped", all(rois[m]["box"] is not None for m in moving), str(rois)[:300])
    check("the still singleton falls back to full frame", rois[still]["box"] is None, str(rois[still]))
    raws = {fn: np.load(cache / "frames" / f"{fn}.npy") for fn in [*moving, still]}
    group_bg = crop.background([raws[m] for m in moving])
    check("both moving clips share a source group", group_key(moving[0]) == group_key(moving[1]))
    check("a clip's box comes from its own group's background only",
          all(rois[m] == crop.estimate_roi(raws[m], group_bg) for m in moving))

    by_name = {c["filename"]: c for c in cropped["clips"]}
    full_by_name = {c["filename"]: c for c in full["clips"]}
    emb = cache / "emb"

    def same(a: str, b: str) -> bool:
        return np.array_equal(np.load(emb / a), np.load(emb / b))

    fb, ff = by_name[still], full_by_name[still]
    check("a fallback clip embeds exactly as in the full view",
          all(same(a, b) for a, b in zip(fb["variants"], ff["variants"], strict=True))
          and same(fb["eval"]["8"], ff["eval"]["8"]))
    mc, mf = by_name[moving[0]], full_by_name[moving[0]]
    check("a cropped clip embeds differently from the full view",
          not same(mc["variants"][0], mf["variants"][0]) and not same(mc["eval"]["8"], mf["eval"]["8"]))
    CV.check_caches_comparable(full, cropped, "gru", "gru@crop")
    check("full and crop caches are a valid paired comparison", True)


def _crop_manifest(cache: Path, manifest: dict, seed: int) -> dict:
    """A synthetic crop-view manifest: same clips, different training and eval
    embeddings, under the crop view's file names."""
    rng = np.random.default_rng(seed)
    emb = cache / "emb"
    other = json.loads(json.dumps(manifest))
    for clip in other["clips"]:
        fn = clip["filename"]
        new = []
        for k, old in enumerate(clip["variants"]):
            x = np.load(emb / old)
            np.save(emb / C.variant_name(fn, k, view="crop"), (x + rng.normal(0, 0.3, x.shape)).astype(np.float32))
            new.append(C.variant_name(fn, k, view="crop"))
        clip["variants"] = new
        for w, old in list(clip["eval"].items()):
            x = np.load(emb / old)
            np.save(emb / C.eval_name(fn, int(w), "crop"), (x + rng.normal(0, 0.3, x.shape)).astype(np.float32))
            clip["eval"][w] = C.eval_name(fn, int(w), "crop")
    other["view"] = "crop"
    C.manifest_path(cache, view="crop").write_text(json.dumps(other))
    return other


def test_crossval_crop_arm(tmp: Path) -> None:
    """gru@crop runs paired on identical folds, never as primary, never deployed."""
    cache = tmp / "cv-cache"
    manifest = TG.build_cache(cache)
    crop_manifest = _crop_manifest(cache, manifest, seed=2)

    folds = make_folds([c["filename"] for c in manifest["clips"]], [c["label"] for c in manifest["clips"]],
                       n_splits=TG.CFG["n_splits"], n_inner=TG.CFG["n_inner"], repeats=TG.CFG["repeats"],
                       seed=TG.CFG["seed"])
    leak_free, eval_cropped = True, True
    for f in folds:
        for tr_idx, ev_idx in [(f.train, f.eval), *f.inner]:
            tr, ev = C.split_entries(crop_manifest, tr_idx, ev_idx, cache_dir=cache)
            leak_free &= not ({e["filename"] for e in tr} & {e["filename"] for e in ev})
        eval_cropped &= all(crop_manifest["clips"][i]["eval"][str(TG.WINDOW)].endswith(
            f".crop.eval{TG.WINDOW}.npy") for i in f.eval)
    check("no crop variant of an eval clip is trained on, outer or inner", leak_free)
    check("crop eval clips are scored on their crop eval sample", eval_cropped)

    kw = dict(cache_dir=cache, quick=True, arms=["gru"], **TG.CFG)
    base = CV.crossval(out_dir=tmp / "cv-base", **kw)
    paired = CV.crossval(out_dir=tmp / "cv-crop", views=["full", "crop"], **kw)
    check("paired run has gru and gru@crop", set(paired["arms"]) == {"gru", "gru@crop"}, str(sorted(paired["arms"])))
    check("primary gru result unchanged by adding the crop arm",
          paired["arms"]["gru"]["pooled_oof"]["macro_f1"]["mean"] == base["arms"]["gru"]["pooled_oof"]["macro_f1"]["mean"])
    pc = paired.get("paired_view", {}).get("crop", {})
    check("corrected paired test gru@crop vs gru reported",
          "p_two_sided" in pc.get("corrected_resampled_t", {})
          and len(pc.get("differences", [])) == TG.CFG["repeats"] * TG.CFG["n_splits"], str(pc)[:200])
    check("arm_cache records gru@crop on the crop view",
          paired["arm_cache"]["gru@crop"] == {"aug_policy": "legacy", "view": "crop"})
    check("every arm carries a grouped bootstrap CI on macro-F1",
          all("ci95" in a["pooled_oof"]["macro_f1"].get("group_bootstrap", {}) for a in paired["arms"].values()))
    lines = [json.loads(x) for x in (tmp / "cv-crop" / "runs" / "index.jsonl").read_text().splitlines()]
    check("index.jsonl records each arm's view", {(x["arm"], x["view"]) for x in lines}
          == {("gru", "full"), ("gru@crop", "crop")})

    ok, msg = raises(lambda: CV.crossval(out_dir=tmp / "cv-bad", views=["crop"], **kw), "primary view")
    check("crossval refuses crop as the primary view", ok, msg)

    out = tmp / "cv-crop"
    ok, msg = raises(lambda: T.train_all_clips(cache, out, seed=0, arm="gru@crop"), "agent feeds full frames")
    check("train --all-clips refuses a crop arm", ok, msg)
    ok, msg = raises(lambda: T.train_fold(cache, out, 0, 0, seed=0, arm="gru@crop"), "agent feeds full frames")
    check("train --fold refuses a crop arm", ok, msg)
    ok, msg = raises(lambda: T.train_all_clips(cache, out, seed=0, arm="gru@nope"), "no arm 'gru@nope'")
    check("train refuses an arm the CV run does not have", ok, msg)
    tr = T.train_all_clips(cache, out, seed=0)
    check("the default arm still trains from the same run", tr["cv_arm"] == "gru" and tr["view"] == "full")

    stale = json.loads((out / "cv_result.json").read_text())
    del stale["arm_cache"]
    (out / "cv_result.json").write_text(json.dumps(stale))
    ok, msg = raises(lambda: T.train_all_clips(cache, out, seed=0), "Re-run crossval.py")
    check("train refuses a cv_result without per-arm cache records", ok, msg)


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="bunnycrop-"))
    try:
        for fn, needs_tmp in (
            (test_estimate_roi, False),
            (test_crop_frames, False),
            (test_build_crop_view, True),
            (test_crossval_crop_arm, True),
        ):
            print(f"\n{fn.__doc__.splitlines()[0]}")
            fn(tmp) if needs_tmp else fn()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    finish("crop view")


if __name__ == "__main__":
    main()
