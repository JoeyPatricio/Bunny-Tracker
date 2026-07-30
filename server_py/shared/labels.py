"""Canonical behavior vocabulary, ported from client/src/labels.js.

LABELS order is significant: it matches server/model/labels.json, so a
classifier output index maps to LABELS[index]. Do not reorder without
retraining the model.
"""

LABELS = ["grooming", "normal", "standing", "yawn", "zoomies"]

LABEL_PHRASE = {
    "grooming": "Bunny is grooming",
    "normal": "Bunny is resting",
    "standing": "Bunny is standing up",
    "yawn": "Bunny is yawning",
    "zoomies": "Bunny has the zoomies",
}
