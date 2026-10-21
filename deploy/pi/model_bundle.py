"""Pack or verify a promoted model for transfer to the Pi.

The agent refuses to classify without models/promoted.json, and an ONNX graph
saved with external data fails to load without its .onnx.data sidecar. The
promotion record's manifest lists every artifact with its sha256, so it is the
transfer list: pack refuses unless every listed file is present and matches,
and verify repeats the check after extraction.

Standard library only, so it runs on the Windows PC and on the Pi.

    python deploy/pi/model_bundle.py pack server_py/models model-bundle.tar.gz
    python deploy/pi/model_bundle.py verify server_py/models
"""
import argparse
import hashlib
import json
import sys
import tarfile
from pathlib import Path

PROMOTED_NAME = "promoted.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check(models_dir: Path) -> list[str]:
    """Return the bundle's file names, or exit naming what is wrong."""
    promoted = models_dir / PROMOTED_NAME
    if not promoted.is_file():
        sys.exit(f"{promoted} does not exist. Promote a candidate before transferring a model.")
    artifacts = json.loads(promoted.read_text(encoding="utf-8"))["manifest"].get("artifacts") or {}
    if not artifacts:
        sys.exit(f"{promoted} lists no artifacts.")
    problems = []
    for name, expected in sorted(artifacts.items()):
        path = models_dir / name
        if not path.is_file():
            problems.append(f"missing {name}")
        elif sha256_file(path) != expected:
            problems.append(f"sha256 mismatch for {name}")
    if problems:
        sys.exit(f"{models_dir}: " + "; ".join(problems))
    return [PROMOTED_NAME, *sorted(artifacts)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    pack = sub.add_parser("pack")
    pack.add_argument("models_dir", type=Path)
    pack.add_argument("out", type=Path)
    verify = sub.add_parser("verify")
    verify.add_argument("models_dir", type=Path)
    args = parser.parse_args()

    names = check(args.models_dir)
    if args.command == "pack":
        with tarfile.open(args.out, "w:gz") as tar:
            for name in names:
                tar.add(args.models_dir / name, arcname=name)
        print(f"wrote {args.out}: {', '.join(names)}")
    else:
        print(f"{args.models_dir}: {len(names) - 1} artifacts match {PROMOTED_NAME}")


if __name__ == "__main__":
    main()
