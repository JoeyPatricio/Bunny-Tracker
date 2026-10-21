"""The PASS/FAIL helper every check-script uses.

It used to be copy-pasted verbatim into six files. The scripts all do
`sys.path.insert(0, <server>)` before importing project code, so
`from tests.checks import check, skip, finish` resolves from both tests/ and
training/.

Three outcomes, not two. `skip` exists for a measurement that cannot be made on
this machine (a timing check on a box too fast to stall the loop, say). That is
not a regression and must not exit 1, but it must not print PASS either, or a
check that never ran reads like one that held.
"""
import sys

failures: list[str] = []
skipped: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    if ok:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}{': ' + detail if detail else ''}")
        failures.append(name)


def skip(name: str, reason: str) -> None:
    print(f"  SKIP  {name}: {reason}")
    skipped.append(name)


def finish(what: str) -> None:
    """Print the summary line and exit 1 if anything failed."""
    if failures:
        print(f"\nFAIL: {len(failures)} check(s) failed")
        sys.exit(1)
    note = f" ({len(skipped)} skipped)" if skipped else ""
    print(f"\nPASS: all {what} checks passed{note}")
