"""Port of server/lib/labelStore.js."""
from app.config import SERVER_DIR
from app.lib.json_store import create_store

# Human-confirmed labels — the ONLY source of training data. guard_wipe on so a
# bug can't shrink the whole training set to empty in one write.
_label_store = create_store(SERVER_DIR, "labels.json", {}, guard_wipe=True)
read_labels = _label_store.read
update_labels = _label_store.update

# Agent-predicted labels awaiting human review in Label Studio. Kept separate
# from labels.json so the model's own (possibly wrong) predictions can never
# feed back into training.
_autolabel_store = create_store(SERVER_DIR, "autolabels.json", {}, guard_wipe=True)
read_auto_labels = _autolabel_store.read
update_auto_labels = _autolabel_store.update
