"""Cross-tabulate the v1 labels against the v2 recode: what `normal` was hiding.

Rows are v1 labels (data/labels.v1.json, nested under "labels"), columns are
v2 codes (data/labels.json, flat) plus `unlabeled` for clips not yet recoded.
Counts only. Zero compute, and the strongest empirical support for the
migration: it shows where each v1 class went once the definitions were
operational.

Clips present only in labels.json (captured after the migration) have no v1
row and are counted separately as `new_in_v2`.

Runtime or training venv: standard library only.

Usage (from server/):
    .venv/bin/python -m training.recoding_report [--v1 PATH] [--v2 PATH] [--out PATH]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.labels import ALL_CODES, ETHOGRAM_VERSION

ROOT = Path(__file__).resolve().parent.parent.parent
V1_PATH = ROOT / "data" / "labels.v1.json"
V2_PATH = ROOT / "data" / "labels.json"
OUT_PATH = Path(__file__).resolve().parent.parent / "models" / "recoding_report.json"
UNLABELED = "unlabeled"


def crosstab(v1: dict[str, str], v2: dict[str, str]) -> dict:
    v1_labels = sorted(set(v1.values()))
    extra = sorted({lab for lab in v2.values() if lab and lab not in ALL_CODES})
    columns = list(ALL_CODES) + extra + [UNLABELED]
    table = {row: dict.fromkeys(columns, 0) for row in v1_labels}
    for fn, old in v1.items():
        new = v2.get(fn) or UNLABELED
        table[old][new] += 1
    return {
        "ethogram_version": ETHOGRAM_VERSION,
        "rows": v1_labels,
        "columns": columns,
        "table": table,
        "n_v1": len(v1),
        "n_recoded": sum(1 for fn in v1 if v2.get(fn)),
        "new_in_v2": sum(1 for fn, lab in v2.items() if lab and fn not in v1),
    }


def print_table(report: dict) -> None:
    cols = report["columns"]
    width = max(len(c) for c in cols + report["rows"]) + 1
    print("v1 \\ v2".ljust(width) + "".join(c[:width - 1].rjust(width) for c in cols))
    for row in report["rows"]:
        print(row.ljust(width) + "".join(str(report["table"][row][c]).rjust(width) for c in cols))
    print(f"\n{report['n_recoded']}/{report['n_v1']} v1 clips recoded; "
          f"{report['new_in_v2']} labeled clips are new since v1.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--v1", type=Path, default=V1_PATH)
    parser.add_argument("--v2", type=Path, default=V2_PATH)
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    args = parser.parse_args()
    # labels.v1.json is nested ({"labels": {...}, ...}); labels.json is flat.
    v1 = json.loads(args.v1.read_text(encoding="utf-8"))["labels"]
    v2 = json.loads(args.v2.read_text(encoding="utf-8"))
    report = crosstab(v1, v2)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print_table(report)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
