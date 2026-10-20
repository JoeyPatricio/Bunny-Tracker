"""Regression gate for the Phase 4 augmentation policies (docs/upgrade-plan.md).

What this pins down:
  - `legacy` is byte-identical to the pre-Phase-4 make_variants, so an old
    cache and a new legacy cache mean the same thing.
  - `albu_v1` samples one transform per variant and applies it to every frame
    of the clip: a clip of identical frames must stay identical frame to frame.
  - Both are deterministic per seed, and two policies coexist in one cache dir
    without touching each other's embeddings or re-decoding the frames tier.
  - crossval.py runs the paired gru@<policy> arm on identical folds without
    changing the primary arms' results.

albu_v1 is UNEVALUATED; this checks that it is built correctly, not that it
helps. Hermetic: synthetic clips, stub backbone, temp dirs.

Usage (from server_py/, training venv):
    .venv-train/bin/python training/test_augment.py
"""
import hashlib
import json
import random
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.checks import check, finish
from training import cache as C
from training import crossval as CV
from training import test_cache_v2 as TC
from training import test_crossval_guards as TG
from training.augment import POLICIES, replay_clip
from training.dataset import AUG_COPIES


# ── the pre-Phase-4 implementation, copied verbatim as the reference ─────

def _ref_flip_frames(frames):
    return frames[:, :, ::-1, :].copy()


def _ref_jitter_frames(frames, brightness, contrast):
    x = frames.astype(np.float32)
    x = (x - 128.0) * contrast + 128.0 + brightness
    return np.clip(x, 0, 255).astype(np.uint8)


def _ref_make_variants(frames, rng):
    variants = [frames, _ref_flip_frames(frames)]
    while len(variants) < AUG_COPIES + 1:
        brightness = rng.random() * 50 - 25
        contrast = 0.85 + rng.random() * 0.35
        also_flip = rng.random() < 0.5
        v = _ref_jitter_frames(frames, brightness, contrast)
        if also_flip:
            v = _ref_flip_frames(v)
        variants.append(v)
    return variants


def random_clip(seed: int, n: int = 6) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (n, 224, 224, 3), dtype=np.uint8)


def same_list(a, b) -> bool:
    return len(a) == len(b) and all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))


def test_legacy_is_byte_identical() -> None:
    """legacy must reproduce the old make_variants exactly."""
    ok = True
    for s in range(5):
        clip = random_clip(s)
        key = f"0:clip{s}.webm"
        ok &= same_list(POLICIES["legacy"].make(clip, random.Random(key)), _ref_make_variants(clip, random.Random(key)))
    check("legacy variants byte-identical to pre-Phase-4 make_variants (5 clips)", ok)
    check("legacy keeps the original emb names", C.variant_name("a.webm", 1) == "a.webm.v1.npy")
    check("legacy keeps the original manifest name", C.manifest_path(Path("x")).name == "manifest.json")


def test_variant_counts() -> None:
    """Each policy produces the variant count it declares, k=0 the untouched clip."""
    clip = random_clip(0)
    for name, policy in POLICIES.items():
        out = policy.make(clip, random.Random("0:x"))
        check(f"{name}: {policy.n_variants} variants", len(out) == policy.n_variants, f"got {len(out)}")
        check(f"{name}: variant 0 is the untouched clip", np.array_equal(out[0], clip))
        check(f"{name}: every variant keeps shape and dtype",
              all(v.shape == clip.shape and v.dtype == np.uint8 for v in out))
    check("albu_v1 has about 8 variants", POLICIES["albu_v1"].n_variants == 8)


def test_albu_same_transform_every_frame() -> None:
    """One sampled transform per variant, applied identically to every frame."""
    frame = random_clip(3, n=1)[0]
    still = np.stack([frame] * 6)  # a clip whose frames are all the same
    variants = POLICIES["albu_v1"].make(still, random.Random("0:still.webm"))
    per_frame_same = all(all(np.array_equal(v[0], v[i]) for i in range(1, len(v))) for v in variants)
    check("a still clip stays still under every albu_v1 variant", per_frame_same)
    changed = sum(not np.array_equal(v, still) for v in variants[2:])
    check("albu_v1 variants actually change the clip", changed >= 4, f"{changed} of {len(variants) - 2} changed")
    # The replay record is what made the frames agree: re-applying it to frame
    # i of a moving clip reproduces output frame i exactly.
    import albumentations as A

    from training.augment import _albu_v1_transform
    moving = random_clip(4)
    out, replay = replay_clip(_albu_v1_transform(), moving, seed=11)
    again = [A.ReplayCompose.replay(replay, image=f)["image"] for f in moving]
    check("replaying the recorded params reproduces every frame", same_list(list(out), again))


def test_deterministic_per_seed() -> None:
    """Same per-clip seed, same variants; different seed, different variants."""
    clip = random_clip(5)
    a = POLICIES["albu_v1"].make(clip, random.Random("0:c.webm"))
    b = POLICIES["albu_v1"].make(clip, random.Random("0:c.webm"))
    c = POLICIES["albu_v1"].make(clip, random.Random("1:c.webm"))
    check("albu_v1 is deterministic for a seed", same_list(a, b))
    check("albu_v1 changes with the seed", not same_list(a[2:], c[2:]))
    random.seed(123)
    np.random.seed(123)
    d = POLICIES["albu_v1"].make(clip, random.Random("0:c.webm"))
    check("albu_v1 ignores the global RNG state", same_list(a, d))


def _hashes(paths) -> dict:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def test_policies_coexist(tmp: Path) -> None:
    """legacy and albu_v1 share one cache dir, frames tier and eval tier."""
    rec, labels_json, cache_dir = TC.setup(tmp)
    calls: list = []
    legacy = TC.build(rec, labels_json, cache_dir, calls)
    legacy_files = _hashes(sorted((cache_dir / "emb").glob("*.v*.npy")))
    legacy_manifest = (cache_dir / "manifest.json").read_bytes()
    decodes_before = len(calls)

    from unittest.mock import patch

    from training import extract_frames as X
    real = X.ffmpeg_extract_frames

    def counting(path, n):
        calls.append((Path(path).name, n))
        return real(path, n=n)

    with patch.object(X, "ffmpeg_extract_frames", counting):
        albu = X.build_cache(cache_dir=cache_dir, labels_json=labels_json, recordings_dir=rec,
                             backbone=TC.StubBackbone().eval(), backbone_name="stub", aug_policy="albu_v1")

    check("manifest records aug_policy legacy", legacy["aug_policy"] == "legacy", legacy["aug_policy"])
    check("manifest records aug_policy albu_v1", albu["aug_policy"] == "albu_v1", albu["aug_policy"])
    check("aug_policy_detail recorded", "UNEVALUATED" in albu.get("aug_policy_detail", ""))
    check("albu_v1 has its own manifest", (cache_dir / "manifest.albu_v1.json").is_file())
    check("legacy manifest untouched by the albu_v1 build",
          (cache_dir / "manifest.json").read_bytes() == legacy_manifest)
    check("legacy variant embeddings untouched by the albu_v1 build",
          _hashes(cache_dir / "emb" / n for n in legacy_files) == legacy_files)
    names = [n for c in albu["clips"] for n in c["variants"]]
    check("albu_v1 variant files are distinct from legacy ones",
          not set(names) & set(legacy_files) and all(".albu_v1.v" in n for n in names))
    check("albu_v1 build re-decoded nothing (frames tier shared)", len(calls) == decodes_before,
          f"{len(calls) - decodes_before} decodes")
    check("both load through cache.load_manifest",
          C.load_manifest(cache_dir)["aug_policy"] == "legacy"
          and C.load_manifest(cache_dir, "albu_v1")["aug_policy"] == "albu_v1")
    CV.check_policies_comparable(legacy, albu, "legacy", "albu_v1")
    check("the two policies are a valid paired comparison", True)


def _second_policy(cache: Path, manifest: dict, name: str, seed: int) -> dict:
    """A synthetic second-policy manifest: same clips and eval tier, different
    training-variant embeddings."""
    rng = np.random.default_rng(seed)
    emb = cache / "emb"
    other = json.loads(json.dumps(manifest))
    for clip in other["clips"]:
        new = []
        for k, old in enumerate(clip["variants"]):
            x = np.load(emb / old) + rng.normal(0, 0.3, np.load(emb / old).shape).astype(np.float32)
            n = C.variant_name(clip["filename"], k, name)
            np.save(emb / n, x.astype(np.float32))
            new.append(n)
        clip["variants"] = new
    other["aug_policy"] = name
    C.manifest_path(cache, name).write_text(json.dumps(other))
    return other


def test_crossval_paired_policies(tmp: Path) -> None:
    """crossval adds gru@<policy> on identical folds and leaves gru unchanged."""
    cache = tmp / "cv-cache"
    manifest = TG.build_cache(cache)
    _second_policy(cache, manifest, "albu_v1", seed=1)
    kw = dict(cache_dir=cache, quick=True, arms=["gru"], **TG.CFG)
    base = CV.crossval(out_dir=tmp / "cv-base", **kw)
    paired = CV.crossval(out_dir=tmp / "cv-paired", aug_policies=["legacy", "albu_v1"], **kw)

    check("paired run has gru and gru@albu_v1", set(paired["arms"]) == {"gru", "gru@albu_v1"},
          str(sorted(paired["arms"])))
    check("primary gru result unchanged by adding a policy",
          paired["arms"]["gru"]["pooled_oof"] == base["arms"]["gru"]["pooled_oof"])
    pc = paired.get("paired_aug_policy", {}).get("albu_v1", {})
    check("corrected paired test gru@albu_v1 vs gru reported",
          "p_two_sided" in pc.get("corrected_resampled_t", {}), str(pc)[:200])
    check("both arms scored on the same folds", len(pc.get("differences", [])) ==
          TG.CFG["repeats"] * TG.CFG["n_splits"])
    check("aug_policy_arms maps each arm to its policy",
          paired.get("aug_policy_arms") == {"gru": "legacy", "gru@albu_v1": "albu_v1"})
    oof = np.load(tmp / "cv-paired" / "cv_oof.npz")
    check("OOF saved for the extra arm", "gru@albu_v1_P" in oof.files, str(oof.files))
    lines = [json.loads(x) for x in (tmp / "cv-paired" / "runs" / "index.jsonl").read_text().splitlines()]
    check("index.jsonl records each arm's policy",
          {(x["arm"], x["aug_policy"]) for x in lines} == {("gru", "legacy"), ("gru@albu_v1", "albu_v1")})

    bad = json.loads(C.manifest_path(cache, "albu_v1").read_text())
    bad["labels_sha256"] = "cd" * 32
    C.manifest_path(cache, "albu_v2").write_text(json.dumps(bad))
    try:
        CV.crossval(out_dir=tmp / "cv-bad", aug_policies=["legacy", "albu_v2"], **kw)
        check("policies built from different labels are refused", False, "did not raise")
    except SystemExit as err:
        check("policies built from different labels are refused", "labels_sha256" in str(err), str(err))


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="bunnyaug-"))
    try:
        for fn, needs_tmp in (
            (test_legacy_is_byte_identical, False),
            (test_variant_counts, False),
            (test_albu_same_transform_every_frame, False),
            (test_deterministic_per_seed, False),
            (test_policies_coexist, True),
            (test_crossval_paired_policies, True),
        ):
            print(f"\n{fn.__doc__.splitlines()[0]}")
            fn(tmp) if needs_tmp else fn()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    finish("augmentation policy")


if __name__ == "__main__":
    main()
