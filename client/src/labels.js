// Canonical behavior vocabulary for the client. Import from here instead of
// re-declaring these maps per component.
//
// LABELS order is significant: it matches server/model/labels.json, so a
// classifier output index maps to LABELS[index]. Do not reorder without
// retraining the model.
export const LABELS = ['grooming', 'normal', 'standing', 'yawn', 'zoomies']

export const LABEL_COLOR = {
  grooming: '#dc82ff',
  normal:   '#88aaff',
  standing: '#ff9f3c',
  yawn:     '#ffd264',
  zoomies:  '#7dff7d',
}

export const LABEL_PHRASE = {
  grooming: 'Bunny is grooming',
  normal:   'Bunny is resting',
  standing: 'Bunny is standing up',
  yawn:     'Bunny is yawning',
  zoomies:  'Bunny has the zoomies',
}
