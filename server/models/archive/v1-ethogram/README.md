# v1 ethogram artifacts (archived 2026-09-25)

Everything here was produced under ethogram v1
(`grooming`, `normal`, `standing`, `yawn`, `zoomies`) and is unreadable by v2
code: `LABELS.index('normal')` raises, and `head_state.pt` has a 5-wide `fc`.

Archived rather than deleted because these are the only surviving record of the
v1 model's behaviour, and the paper cites numbers derived from them.

| File | Note |
|---|---|
| `cache/` | Embedding cache + `manifest.json`, all v1 label strings. Schema 1 (has `train`/`val` keys). |
| `head_state.pt` | 5-class torch checkpoint (`fc: 5x32`). |
| `train_result.json` | `val_acc 0.6154`. A selection score, not held-out. Do not cite as accuracy. |
| `gate_result.json` | `gate_passed: false`. The model shipped anyway; there was no promote step. |
| `js_baseline.json` | tfjs baseline, `accuracy 0.7436`. Same caveat as above. |

`scripts/legacy/js_baseline_v1.mjs --manifest` defaults to the manifest in
here, so the v1 number stays reproducible.
