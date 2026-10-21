"""Intra-observer reliability: Cohen's kappa for the blind 25% re-code.

Compares first-pass labels (data/labels.json) with the blind re-code
(data/labels.reliability.json) over the frozen sample
(data/labels.reliability_sample.json). Human repeatability and model OOF
agreement describe different samples. The re-code includes out_of_view, which
the classifier excludes, so it is not an upper bound on classifier performance.

Reported:
- overall kappa and raw percent agreement over every coding code, including
  `out_of_view` (a coding category even though it is not a model class);
- per-class kappa, one-vs-rest: each code is reduced to "this code / any
  other" for both passes and kappa computed on that 2x2 table. It answers "how
  consistently is this code applied", which the overall figure averages away;
- kappa with uncertain-flagged clips excluded, and how many were flagged
  (from labels.meta.json), so the methods section can report both.

Refuses if the re-code is incomplete: a kappa over part of a pre-registered
sample is a different, self-selected sample.

Runtime or training venv: numpy only, via training/metrics.py.

Usage (from server/):
    .venv/bin/python -m training.reliability [--data-dir DIR] [--out PATH]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.labels import ALL_CODES, ETHOGRAM_VERSION
from training.metrics import cohen_kappa, confusion_matrix

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
OUT_PATH = Path(__file__).resolve().parent.parent / "models" / "reliability.json"


class ReliabilityError(Exception):
    pass


def _load(path: Path, default):
    if not path.is_file():
        return default
    text = path.read_text(encoding="utf-8").lstrip("﻿").strip()
    return json.loads(text) if text else default


def _kappa(first: list[str], second: list[str], codes: list[str]) -> float:
    index = {c: i for i, c in enumerate(codes)}
    cm = confusion_matrix([index[a] for a in first], [index[b] for b in second], n_classes=len(codes))
    return cohen_kappa(cm)


def compute(labels: dict, recode: dict, sample: dict, meta: dict) -> dict:
    clips = list(sample.get("clips") or [])
    if not clips:
        raise ReliabilityError("No reliability sample has been drawn yet (labels.reliability_sample.json).")
    missing_first = [fn for fn in clips if not labels.get(fn)]
    if missing_first:
        raise ReliabilityError(
            f"{len(missing_first)} sampled clip(s) have no first-pass label, e.g. {missing_first[0]}."
        )
    done = [fn for fn in clips if (recode.get(fn) or {}).get("label")]
    if len(done) < len(clips):
        raise ReliabilityError(
            f"Re-code incomplete: {len(done)} of {len(clips)} sampled clips recoded. "
            f"Finish the blind re-code in the Labeling Studio's reliability mode first."
        )

    first = [labels[fn] for fn in clips]
    second = [recode[fn]["label"] for fn in clips]
    codes = list(ALL_CODES) + sorted({c for c in first + second if c not in ALL_CODES})
    uncertain = [fn for fn in clips if (meta.get(fn) or {}).get("uncertain")]
    certain = [i for i, fn in enumerate(clips) if fn not in set(uncertain)]

    per_class = {}
    for code in codes:
        a = ["y" if x == code else "n" for x in first]
        b = ["y" if x == code else "n" for x in second]
        # A code neither pass used has no agreement to measure; the 2x2 table
        # is all "other" and kappa's chance correction degenerates to 1.0.
        used = "y" in a or "y" in b
        per_class[code] = {
            "kappa_one_vs_rest": _kappa(a, b, ["y", "n"]) if used else None,
            "first_pass_n": a.count("y"),
            "recode_n": b.count("y"),
        }

    return {
        "ethogram_version": ETHOGRAM_VERSION,
        "n": len(clips),
        "percent_agreement": sum(x == y for x, y in zip(first, second, strict=True)) / len(clips),
        "kappa": _kappa(first, second, codes),
        "kappa_excluding_uncertain": (
            _kappa([first[i] for i in certain], [second[i] for i in certain], codes) if certain else None
        ),
        "n_uncertain_in_sample": len(uncertain),
        "n_uncertain_overall": sum(1 for v in meta.values() if isinstance(v, dict) and v.get("uncertain")),
        "per_class": per_class,
        "coders": sorted({(recode[fn].get("coder") or "") for fn in clips} - {""}),
        "sample": {k: sample.get(k) for k in ("seed", "fraction", "pool_size", "drawn_at", "ethogram_version")},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    args = parser.parse_args()
    d = args.data_dir
    try:
        result = compute(
            _load(d / "labels.json", {}),
            _load(d / "labels.reliability.json", {}),
            _load(d / "labels.reliability_sample.json", {}),
            _load(d / "labels.meta.json", {}),
        )
    except ReliabilityError as err:
        raise SystemExit(f"reliability: {err}") from None
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"n={result['n']}  kappa={result['kappa']:.3f}  agreement={result['percent_agreement'] * 100:.1f}%  "
          f"uncertain in sample={result['n_uncertain_in_sample']}")
    for code, r in result["per_class"].items():
        k = r["kappa_one_vs_rest"]
        shown = "   n/a" if k is None else f"{k:6.3f}"
        print(f"  {code:18s} kappa={shown}  first={r['first_pass_n']:3d}  recode={r['recode_n']:3d}")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
