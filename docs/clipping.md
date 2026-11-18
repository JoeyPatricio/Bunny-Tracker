# Clipping Mode

Back to the [project overview](../README.md#using-it).

Clipping Mode saves a clip whenever motion crosses a threshold. It collects
unlabeled footage for a training set and works without a trained model.
Enable Clipping in the dashboard header.

By default, the classifier doesn't run in this mode. There are no predictions,
email alerts, or label suggestions, so a quiet Activity Log is expected.
Clips are 8 seconds long, with motion sampled every 0.5 seconds. The saved
clip is the segment containing the motion. Normal monitoring uses 12 second
clips and samples every 1.5 seconds.

Clips land in `data/recordings/` as `recording-clip-*.webm`. Use the Unlabeled
filter in Label Studio to work through them, following the
[ethogram](ethogram.md).

## Controls

The panel under the live feed contains the controls:

| Control | Effect |
|---|---|
| Sensitivity (1-10) | Maps to a motion threshold from 8.0 down to 0.4. Higher values catch more motion and more empty footage. |
| Cooldown | Sets the minimum gap between clips. At or above the clip length, it allows at most one clip per segment. |
| Dry run | Reports what would be captured without saving anything. |
| Live motion trace | Shows the agent's motion level with the threshold drawn across it. |

Turn on dry run and watch the trace for ten minutes. Place the threshold just
below the spikes caused by rabbit movement. A low threshold can fill the clip
budget overnight with footage of nothing.

## Storage limits

Harvesting stops at `clippingMaxUnlabeled` unlabeled clips (default 500) or
when free disk drops below `clippingMinFreeGb` (default 10). Labeled clips
don't count toward the cap, so labeling frees room in the budget.

Reaching a limit doesn't delete clips. The dashboard shows why harvesting
stopped and waits for you to re-arm it. Environment variables provide initial
defaults. Dashboard changes are saved in `data/monitor.json` and take
precedence afterward.

## Collect across several days

`server/training/splits.py::group_key` groups agent clips by capture hour so
clips from one session can't straddle the training and validation split.
Collecting 500 clips in one overnight run gives only a handful of groups.
Spreading collection over several days gives the split more independent
groups. See the [training pipeline](pipeline.md) for the next steps.
