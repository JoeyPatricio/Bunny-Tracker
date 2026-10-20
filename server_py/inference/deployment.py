import json
from pathlib import Path

from shared.labels import LABELS


def read_promoted(models_dir: Path) -> dict | None:
    try:
        record = json.loads((models_dir / "promoted.json").read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    manifest = record["manifest"]
    if manifest.get("labels") != LABELS:
        raise ValueError("Promoted model vocabulary differs from the current ethogram. Retrain before use.")
    if manifest.get("view", "full") != "full":
        raise ValueError("The promoted crop model needs ROI preprocessing that the agent does not implement.")
    window = manifest.get("window")
    if type(window) is not int or window <= 0:
        raise ValueError("Promoted model must record a positive integer window.")
    return record
