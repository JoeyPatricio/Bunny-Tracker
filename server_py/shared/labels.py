"""Canonical behavior vocabulary (ethogram v2). Mirrors client/src/labels.js.

Definitions, and the published sources they come from, are in
docs/ethogram.md. Do not add, rename, or reorder a class here without
updating that document first: the ethogram is the specification, this file is
the implementation of it.

LABELS order is significant: it matches the classifier's output axis, so
output index i means LABELS[i]. Reordering requires a retrain.

LABELS holds the model's classes only. `out_of_view` is a coding category, not
a behavior - clips with no visible rabbit have to be recordable so they stop
being silently coded as resting, but there is nothing for the classifier to
learn from them, so they are excluded from the output space and the training
loaders drop them by filtering on LABELS.
"""

LABELS = [
    "feeding",
    "grooming",
    "locomotion",
    "locomotion_rapid",
    "rearing",
    "resting_lying",
    "resting_sitting",
]

# Coding categories that are valid labels but not model classes.
CODING_ONLY = ["out_of_view"]

# Everything the label API will accept.
ALL_CODES = LABELS + CODING_ONLY

# The baseline states. An alert gate asks "is this something other than the
# rabbit being at rest", and v2 has two resting postures where v1 had one
# catch-all `normal`, so every consumer that used to compare against a single
# index now has to compare against this set.
RESTING_LABELS = ["resting_lying", "resting_sitting"]
RESTING_INDICES = frozenset(LABELS.index(l) for l in RESTING_LABELS)

# Codes that must never surface on the public demo page. v1 excluded the single
# `normal` label; a rabbit sitting still is no more of a highlight than one
# lying down, and a clip with no rabbit in it is not a highlight at all.
NON_HIGHLIGHT = frozenset(RESTING_LABELS) | frozenset(CODING_ONLY)

LABEL_PHRASE = {
    "feeding": "Bunny is eating",
    "grooming": "Bunny is grooming",
    "locomotion": "Bunny is hopping about",
    "locomotion_rapid": "Bunny has the zoomies",
    "rearing": "Bunny is rearing up",
    "resting_lying": "Bunny is lying down",
    "resting_sitting": "Bunny is sitting",
    "out_of_view": "No bunny in view",
}
