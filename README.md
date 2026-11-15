# Motion studio showcase

Windows desktop app for local C3D recordings. It shows 3D marker playback,
gait speed, trunk lean, movement measurements, gait-expression scores, and
protective-movement scores. Predictions use published weights and run locally.
Scores animate smoothly; exports retain the original model outputs.

## Install from the ZIP (Windows 10/11, 64-bit)

1. Install **64-bit Python 3.13** from https://www.python.org/downloads/windows/.
   Include **pip**, **Tcl/Tk**, and the **Python launcher** in the installer.
2. Right-click `Pose_stress.zip` and choose **Extract All**. Do not run inside the ZIP.
   Use a writable folder such as `Documents\Pose_stress`.
3. Open the extracted folder and double-click **setup_demo.bat**.
   Internet is required to install dependencies. Wait for `Setup complete`.
   This creates a local `.venv`; no compiler, GPU, or separate build step is needed.
4. Double-click **launch_demo.bat**. If no C3D is in the app folder, a file picker opens.

The shareable ZIP includes both model checkpoints. If you received source without
weights, setup downloads them automatically (the gait archive is about 181 MiB).
After setup, normal playback works offline. No recording is uploaded.
If setup was interrupted, run `setup_demo.bat` again.

## Open your C3D

- Use **Open C3D...** inside the app.
- Or drag a C3D onto `launch_demo.bat` in Windows Explorer.
- Or use PowerShell from the extracted folder (quote paths containing spaces):

```powershell
.\launch_demo.bat "C:\Recordings\My recording.c3d"
```

With no argument, the app opens the first `.c3d` in its own folder, or prompts
for one if there is none. Use an explicit path when several recordings exist.

### Supported recordings

This is configured for the studio's **single-subject OptiTrack/Motive full-body
marker layout**, with **Z up**. It is not a universal C3D skeleton importer.
Marker labels must have a subject prefix, for example `Antonin_FullBody:ChestTop`.
The prefix may change; the short marker names must match. Units may be m, cm, or mm.
Unlabelled trajectories are ignored and multiple subjects are rejected.

Required short names for all measurements and both models:

```text
ChestTop BackTop HeadFront HeadLeft HeadRight HeadTop
WaistLFront WaistLBack WaistRFront WaistRBack
LShoulderTop LElbowOut LWristIn LWristOut LHandIn LHandOut LHeel LKneeOut LToeTip
RShoulderTop RElbowOut RWristIn RWristOut RHandIn RHandOut RHeel RKneeOut RToeTip
```

Additional markers improve the drawn skeleton. Missing observations produce
plot gaps or unavailable model scores. A different marker layout requires a
mapping change in the Python adapters; renaming arbitrary joints is insufficient.

## Controls and exports

- Play/Pause: button or Space. Restart returns to the beginning.
- Drag the timeline to seek. Choose a trailing **3-15 second** window.
- Drag the 3D view to rotate. F11 toggles fullscreen; Escape exits it.
- Models wait until a full selected window is available.
- **Export** writes `measurements.csv`, `rolling_window.csv`,
  `model_prediction.json`, `protective_prediction.json`, and `showcase.png`
  into `outputs/`. Later exports overwrite those files.

Trunk lean is the pelvis-to-upper-trunk angle from vertical, not a direct
explanation of the protective classifier. Model transfer to optical markers is
approximate. See [MODEL_CARD.md](MODEL_CARD.md) and
[PROTECTIVE_MODEL_CARD.md](PROTECTIVE_MODEL_CARD.md) for provenance and limits.

## Manual setup / run

From PowerShell in the extracted folder:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-model.txt
.\.venv\Scripts\python.exe setup_model.py
.\.venv\Scripts\python.exe setup_protective_model.py
.\.venv\Scripts\python.exe demo.py "C:\Recordings\My recording.c3d"
```

Activation of the virtual environment is unnecessary. If `py` is not recognized,
repair the Python installation and enable its launcher. If dependencies are
missing, rerun setup. A model checksum error requires rerunning its setup script.

## Create a shareable ZIP

```powershell
py -3.13 package_demo.py
```

Creates `dist/Pose_stress.zip`, including runtime code, setup/launch scripts,
requirements, model checkpoints, model cards, and third-party licenses.
It excludes `.venv`, research downloads, outputs, caches, and recordings.
To deliberately bundle the studio sample too:

```powershell
py -3.13 package_demo.py --include-sample
```

Recipients install Python and follow the ZIP instructions above. Do not copy
`.venv` between computers. Keep `models/` and `vendor/` beside `demo.py`.

## Developer checks

The tests use the original studio sample C3D in the repository root (not included
in the default distribution ZIP):

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```
