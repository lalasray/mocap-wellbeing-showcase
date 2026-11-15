# Published gait-expression model integration

The app runs the **ST-GCN baseline checkpoint distributed with Take an Emotion
Walk (ECCV 2020)**. It is not the flagship TAEW hierarchical-attention architecture,
and no accuracy reported for that flagship model is claimed for this integration.
No model is trained on the user's recording, and no heuristic generates emotion scores.

## Source and reproducibility

- Authors' repository: https://github.com/PeterZs/take_an_emotion_walk
- Pinned source commit: `881b90611ca6a964794997f15e9721eade502803`.
- Download URL is taken from the authors' `download_data_weights.sh`.
- Checkpoint: `weights/stgcn_500_5.pt`, SHA-256
  `c9cae6d55337db35238f78a4125a387232c1a2f93fc045aef0ea5fc2f24f30bf`.
- The three files in `vendor/taew_stgcn` are unmodified upstream model code;
  their MIT license is retained. `models/model_info.json` records configuration.
- Loaded on CPU with `weights_only=True`, strict state-dict matching, and eval mode.
  Double precision and the original forward pass are preserved.
- The authors' evaluation loader takes `datasets/elmd_dag.npz`, subsamples every
  fifth frame and reshapes to `(batch,48,21,3,1)`. Reference data has 240 frames.
- Class order Happy, Sad, Angry, Neutral follows `taew_net/utils/loader.py`'s
  `load_edin_labels`; the baseline loader uses the first four label columns.
  The released evaluation uses binary cross-entropy with logits. We apply sigmoid
  independently to each output. These scores need not sum to one and are not
  calibrated probabilities of a person's emotional state.

## Approximate marker-to-joint mapping

| Model joints | Source in this recording |
|---|---|
| Root | Mean of four waist front/back markers |
| Left/right hip | Mean of respective front/back waist markers |
| Knees | Outer knee markers |
| Heels / toes | Heel / toe-tip markers (no ankle replacement needed) |
| Lower back / spine | One-third / two-thirds of root-to-neck vector |
| Neck | Mean ChestTop and BackTop |
| Head | Mean four head markers |
| Shoulders / elbows | ShoulderTop / ElbowOut |
| Hands (wrist nodes) | Mean inner/outer wrist markers |
| Hand-index nodes | Mean inner/outer hand markers; approximate hand endpoint |

These are proxies for joint centres, not calibrated anatomical joint centres.
Each window is aligned to Y-up, left shoulder along +X, forward along +Z; horizontal
root translation starts at zero. This uses the initial heading and preserves turns.
Bone directions are retargeted to median reference lengths measured from the first
three released ELMD clips. Root translation is scaled by pelvis-to-heel height and
starts at the reference root height. This geometric template is an adapter choice,
not a preprocessing procedure validated by the model authors on OptiTrack markers.

The selected 3–15 s window is linearly resampled to 48 frames. This time normalization
changes effective motion timing as window length changes and is another departure
from the reference evaluation. Comparing predictions across window lengths does
not provide a confidence estimate. Initial partial windows do not produce scores.

## Missing data and display

Inference requires at least 98% observed mapped joint samples. Only bounded gaps
of at most 150 ms are interpolated; missing endpoints and longer gaps suppress
scores. Mean horizontal pelvis speed below 0.15 m/s also suppresses scores because
this is a gait model. These are engineering gates, not published clinical thresholds.
The original measurement charts continue to leave missing samples unfilled.

Inference runs in a background worker, at up to 2 Hz in playback. The model panel
shows the actual inference interval. Paused scrubbing uses the exact selected
frame. An unavailable or pending result displays dashes instead of stale scores.
Export includes `model_prediction.json` with checkpoint identity, interval, scores
and status, or a pending/unavailable status when appropriate.

## What has and has not been established

Verified: the downloaded checkpoint loads exactly; reference clips and mapped
studio data produce finite outputs; inference is deterministic; geometric mapping,
gap rejection, and room-rotation/translation invariance have tests.

**Accuracy on this studio's data is unknown.** No labelled local evaluation or
calibration was performed. Tested windows often strongly favour Neutral, and some
change considerably with window size. Saturated outputs are not evidence of accuracy.
The adapter and differing task/recording domain may affect predictions substantially.

Present this as a demonstration of transferring a published **perceived gait
expression** model. It does not detect stress, depression, diagnose mental health,
or establish what the recorded person felt. The StressPose repository publishes
analysis/training code, but its `.gitignore` excludes pickled trained pipelines;
we did not find a ready-to-load stress checkpoint in that repository checkout.
EMOKINE provides feature extraction, not a plug-in trained stress classifier.
