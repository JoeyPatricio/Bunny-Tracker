import { createStore } from './jsonStore.js'

// Human-confirmed labels — the ONLY source of training data. guardWipe on so a
// bug can't shrink the whole training set to empty in one write.
const labelStore = createStore('labels.json', { defaultValue: {}, guardWipe: true })
export const readLabels   = labelStore.read
export const updateLabels = labelStore.update

// Agent-predicted labels awaiting human review in Label Studio. Kept separate
// from labels.json so the model's own (possibly wrong) predictions can never
// feed back into training — see uploadAndAlert in agent/capture.mjs.
const autoLabelStore = createStore('autolabels.json', { defaultValue: {}, guardWipe: true })
export const readAutoLabels   = autoLabelStore.read
export const updateAutoLabels = autoLabelStore.update
