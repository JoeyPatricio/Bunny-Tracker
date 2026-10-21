"""Embedding cache v2: split-agnostic, two tiers.

Layout under the cache dir (default models/cache/):

    frames/<clip>.npy          uint8 [TRAIN_FRAMES,224,224,3]  training sample
    frames/<clip>.eval<W>.npy  uint8 [W,224,224,3]             eval sample, window W
    emb/<clip>.v<k>.npy        float32 [TRAIN_FRAMES,1280]     variant k (k=0 identity), legacy policy
    emb/<clip>.<key>.v<k>.npy                                  same, any other cache key (below)
    emb/<clip>.eval<W>.npy     float32 [W,1280]                eval sample, identity, full view
    emb/<clip>.crop.eval<W>.npy                                same, crop view (training/crop.py)
    manifest.json              schema 2, legacy policy, full view, see below
    manifest.<key>.json        schema 2, any other cache key

A cache key names one (view, aug policy) pair: "" for full/legacy, otherwise
the non-default parts joined by dots, e.g. "albu_v1", "crop", "crop.albu_v1".

Schema 1 baked the train/val split in at extraction time, because augmented
variants were only generated for training clips. The backbone is frozen, so an
embedding does not depend on the split; only the choice of which variants to
generate did. Schema 2 generates everything for every clip and picks at fold
time: train clips contribute every variant and every window start, eval clips
contribute only their identity eval sample.

Why a separate eval sample rather than window [0, W] of variant 0: schema 1
evaluated each held-out clip on W frames spread evenly across the whole clip
(extract_frames(n=W)), while training clips were sampled at TRAIN_FRAMES.
Window [0, W] of the TRAIN_FRAMES sample covers only the first half of the clip
at different timestamps, so it would have silently changed the eval protocol
and made new numbers incomparable with old ones. The eval tier reproduces the
old protocol exactly. One eval tier per window length in `eval_windows`.

Manifest:
    {"schema": 2, "clips": [{"filename", "label", "group",
                             "variants": [emb names, index = k],
                             "eval": {"<W>": emb name}}],
     "window", "eval_windows", "train_frames", "backbone", "labels",
     "labels_sha256", "ethogram_version", "aug_policy", "view",
     crop view only: "roi": {clip: training.crop.estimate_roi result}}

No torch import: the loaders here are plain numpy so anything can use them.
"""
import json
from pathlib import Path

import numpy as np

SCHEMA = 2
DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent / "models" / "cache"


# Augmentation policies (training/augment.py) and views (full frame, or the
# training/crop.py ROI) share one cache dir. The full-view frames tier and
# eval embeddings do not depend on the policy, so they are shared; each
# (view, policy) gets its own manifest and its own training-variant
# embeddings, and the crop view its own eval embeddings. Full/legacy keeps the
# original names, so a cache built before Phase 4 is still a valid one.
LEGACY_POLICY = "legacy"
FULL_VIEW = "full"
VIEWS = (FULL_VIEW, "crop")


def cache_key(policy: str = LEGACY_POLICY, view: str = FULL_VIEW) -> str:
    return ".".join(p for p in (view != FULL_VIEW and view, policy != LEGACY_POLICY and policy) if p)


def manifest_path(cache_dir: Path, policy: str = LEGACY_POLICY, view: str = FULL_VIEW) -> Path:
    key = cache_key(policy, view)
    return Path(cache_dir) / (f"manifest.{key}.json" if key else "manifest.json")


def variant_name(filename: str, k: int, policy: str = LEGACY_POLICY, view: str = FULL_VIEW) -> str:
    """emb/ filename for training variant k of a clip under (policy, view)."""
    key = cache_key(policy, view)
    return f"{filename}.{key}.v{k}.npy" if key else f"{filename}.v{k}.npy"


def eval_name(filename: str, window: int, view: str = FULL_VIEW) -> str:
    """emb/ filename for a clip's identity eval sample. Depends on the view
    only: augmentation never touches eval clips."""
    return f"{filename}.eval{window}.npy" if view == FULL_VIEW else f"{filename}.{view}.eval{window}.npy"


def check_schema(manifest: dict, where: str = "the embedding cache") -> dict:
    """Refuse anything but schema 2, with the fix in the message."""
    schema = manifest.get("schema", 1)
    if schema != SCHEMA:
        raise SystemExit(
            f"{where} is schema {schema}; this code reads schema {SCHEMA}. Schema 1 "
            f"baked a train/val split into the cache and cannot be reused under "
            f"cross-validation. Archive the old cache (docs/pipeline.md, 'After an "
            f"ethogram change') and regenerate with training/extract_frames.py."
        )
    return manifest


def load_manifest(cache_dir: Path = DEFAULT_CACHE_DIR, policy: str = LEGACY_POLICY,
                  view: str = FULL_VIEW) -> dict:
    path = manifest_path(cache_dir, policy, view)
    if not path.is_file():
        hint = "".join(f" --{flag} {v}" for flag, v, default in
                       (("aug-policy", policy, LEGACY_POLICY), ("view", view, FULL_VIEW)) if v != default)
        raise SystemExit(f"No embedding cache at {path}. Run training/extract_frames.py{hint} first.")
    return check_schema(json.loads(path.read_text()), str(path))


def train_windows(n_frames: int, window: int) -> list[tuple[int, int]]:
    """Up to three evenly spaced window starts over a clip's training sample.
    For the default 8-of-16 this is [0, 4, 8], schema 1's TRAIN_WINDOW_STARTS."""
    if window >= n_frames:
        return [(0, n_frames)]
    starts = sorted({int(s) for s in np.linspace(0, n_frames - window, num=min(3, n_frames - window + 1))})
    return [(s, s + window) for s in starts]


def split_entries(manifest: dict, train_idx, eval_idx, *, window: int | None = None,
                  cache_dir: Path = DEFAULT_CACHE_DIR) -> tuple[list[dict], list[dict]]:
    """Train and eval entries for one split of manifest["clips"].

    Indices are positions in manifest["clips"], the same order make_folds sees.
    Train: every variant of every train clip, each with all window starts.
    Eval: the identity eval sample of each eval clip, one window [0, W].
    Entries carry filename and variant so the no-leakage invariant is checkable.
    """
    check_schema(manifest)
    window = manifest["window"] if window is None else window
    eval_key = str(window)
    if eval_key not in {str(w) for w in manifest["eval_windows"]}:
        raise SystemExit(
            f"No eval sample for window={window} in the cache (has {manifest['eval_windows']}). "
            f"Regenerate with training/extract_frames.py --eval-windows including {window}."
        )
    train_set, eval_set = set(map(int, train_idx)), set(map(int, eval_idx))
    if train_set & eval_set:
        raise ValueError(f"clips on both sides of the split: {sorted(train_set & eval_set)}")

    emb_dir = Path(cache_dir) / "emb"
    clips = manifest["clips"]
    train_entries = []
    for i in sorted(train_set):
        clip = clips[i]
        for k, name in enumerate(clip["variants"]):
            emb = np.load(emb_dir / name)
            train_entries.append({
                "filename": clip["filename"], "label": clip["label"], "variant": k,
                "embeddings": emb, "windows": train_windows(emb.shape[0], window),
            })
    eval_entries = []
    for i in sorted(eval_set):
        clip = clips[i]
        emb = np.load(emb_dir / clip["eval"][eval_key])
        eval_entries.append({
            "filename": clip["filename"], "label": clip["label"], "variant": 0,
            "embeddings": emb, "windows": [(0, window)],
        })
    return train_entries, eval_entries


def fold_entries(manifest: dict, fold, *, window: int | None = None,
                 cache_dir: Path = DEFAULT_CACHE_DIR) -> tuple[list[dict], list[dict]]:
    """split_entries for a training.splits.Fold's outer train/eval."""
    return split_entries(manifest, fold.train, fold.eval, window=window, cache_dir=cache_dir)


def legacy_split_indices(manifest: dict, val_fraction: float = 0.2, seed: int = 0):
    """The pre-CV single train/val split (dataset.split_by_group), as indices into
    manifest["clips"]. Kept so train.py, evaluate.py and verify_predictor_parity.py
    keep working until crossval.py replaces them; every number from this split is
    a selection score, not a result."""
    from training.dataset import split_by_group  # torch import, so deferred

    clips = [(c["filename"], c["label"]) for c in manifest["clips"]]
    split = split_by_group(clips, val_fraction=val_fraction, seed=seed)
    pos = {c["filename"]: i for i, c in enumerate(manifest["clips"])}
    return [pos[f] for f, _ in split["train"]], [pos[f] for f, _ in split["val"]]
