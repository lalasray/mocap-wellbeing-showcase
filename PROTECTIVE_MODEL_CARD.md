# Protective movement model

The demo uses the **pose branch (`suboutput1`)** of the published EmoPain-Fusion BI-BA-CNN coordinate checkpoint. Its two softmax outputs are **Non-protective** (0) and **Protective** (1). These are movement-class scores, not pain intensity or stress/anxiety measurements.

- Source: https://github.com/981526092/EmoPain-Fusion
- Pinned commit: `219e88c02ed65daaa4cd772d53745204e0b07048`
- Checkpoint: `Best_Model_Weights/HOT_bi_cnn_banet_coordinate.hdf5`
- SHA-256: `916cf28e65042910baa7c41e96c5a27073446f78a87de0b877450589fc2800f7`
- License: MIT; attribution in `vendor/emopain_fusion/LICENSE`.
- Provenance: university coursework implementation using the EmoPain dataset, not a checkpoint released by the original BANet authors.

## What runs

The original model averages pose and EMG classifier heads. This integration extracts only the trained pose head: shared temporal convolutions, batch normalization, temporal attention per joint, body attention, and its two-class classifier. The weights are unchanged. HDF5 arrays are read without deserializing executable Keras Lambda layers. PyTorch inference is checked against independent NumPy channels-last equations. The original full ensemble's reported performance does **not** apply to the extracted head.

## Input adapter and rolling windows

The release consumes raw XYZ in coordinate blocks for 22 joints, with 180 frames at 60 Hz. The original 26-node EmoPain ordering excludes the left/right heel and toe nodes. Anatomical ordering is documented in the [EmoPain multi-rater paper](https://discovery.ucl.ac.uk/id/eprint/10173489/1/TAC_MRCL_MutliTask-1.pdf); the released validation sample `CoordinateData/validation/C256N.mat` corroborates the ordering, Y-up coordinates, and centimetre-scale limb dimensions.

The optical markers approximate these joints: waist markers for hips, knee surface markers for knees, heels for ankles (the sample's left ankle marker is largely missing), shoulder markers for shoulders, interpolated clavicles/spine, wrist averages, hand averages for fingertips, and head markers. These are **not recovered anatomical joint centres**. Metres Z-up are rotated to centimetres Y-up as `(x,z,-y)*100`. The original raw-coordinate preprocessing is retained: no undocumented body normalization or synthetic EMG is added.

Each native three-second subwindow is sampled at 60 Hz, without stretching time. For a selected 3–15-second interval, starts are spaced one second apart and scores are averaged. All selected data must pass 98% coverage; only bounded gaps up to 150 ms are interpolated. A full selected window is required.

## Limits

This optical-marker transfer has no target-domain validation. Sensor type, joint proxies, body proportions, activity, heading, and room origin can affect scores. In particular the raw-coordinate classifier is not translation/rotation invariant. Successful inference and numerical parity establish software correctness, **not** psychological or clinical validity. The sample recording has no protective-behaviour labels. No accuracy claim is made for it.

## Other requested output

No verified compatible pretrained stress/fatigue/anxiety checkpoint was found in the inspected releases: StressPose (`empkins/stresspose-analysis` and OSF `va6t3`), PsyMo (`cosmaadrian/psymo`), or the IMU-based `AmirBGitHub/fatigue-gait-classification`. StressPose includes research/training code and data; the inspected OSF extras contain condition order, not model weights. Training a new classifier would be a separate task and must be described as newly trained. The demo does not substitute emotion probabilities or heuristic motion measures for these outputs.
