"""Port of server/lib/labelStore.js."""
from app.config import SERVER_DIR
from app.lib.json_store import create_store

# Human-confirmed labels — the ONLY source of training data. guard_wipe on so a
# bug can't shrink the whole training set to empty in one write.
_label_store = create_store(SERVER_DIR, "labels.json", {}, guard_wipe=True)
read_labels = _label_store.read
update_labels = _label_store.update

# Agent-predicted labels awaiting review in Label Studio. Kept separate
# from labels.json so the model's own (possibly wrong) predictions can never
# feed back into training.
_autolabel_store = create_store(SERVER_DIR, "autolabels.json", {}, guard_wipe=True)
read_auto_labels = _autolabel_store.read
update_auto_labels = _autolabel_store.update

# Coding provenance, one entry per first-pass label:
#   {filename: {ethogram_version, coder, coded_at, uncertain}}
# A sidecar rather than a richer labels.json, because eight consumers read
# labels.json as a flat {filename: label} map and docs/ethogram.md committed to
# that schema. Nothing that trains or serves reads this file; it exists so the
# methods section can say who coded what, under which definitions, and how many
# clips the coder flagged as uncertain.
_meta_store = create_store(SERVER_DIR, "labels.meta.json", {}, guard_wipe=True)
read_label_meta = _meta_store.read
update_label_meta = _meta_store.update

# Blind re-code for intra-observer agreement:
#   {filename: {label, coder, coded_at, ethogram_version}}
# Separate from labels.json so a re-code can never overwrite the first pass it
# is being compared against. training/ reads both to compute Cohen's kappa.
_reliability_store = create_store(SERVER_DIR, "labels.reliability.json", {}, guard_wipe=True)
read_reliability = _reliability_store.read
update_reliability = _reliability_store.update

# Which clips the re-code covers. Drawn once and then frozen: see
# app/lib/reliability_sample.py. Not guard_wipe'd as a map; it is written once
# and the route refuses to redraw.
_reliability_sample_store = create_store(SERVER_DIR, "labels.reliability_sample.json", {})
read_reliability_sample = _reliability_sample_store.read
update_reliability_sample = _reliability_sample_store.update

# Every store above, for tests that need to repoint them all at a temp dir.
ALL_STORES = (
    _label_store,
    _autolabel_store,
    _meta_store,
    _reliability_store,
    _reliability_sample_store,
)
