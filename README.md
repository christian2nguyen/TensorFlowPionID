# PionID

A TensorFlow baseline for binary charged-pion identification from tabular detector
features (for example momentum, dE/dx, time of flight, calorimeter energy, and
Cherenkov response).

## Expected data

Supply either CSV files or ROOT files containing a flat TTree/RNTuple with one
track per entry, numeric feature branches, and an `is_pion` label (`1` for pion,
`0` for background):

```text
momentum,dedx,tof,ecal_energy,is_pion
1.42,2.11,7.36,0.31,1
0.88,1.07,7.91,0.82,0
```

Choose detector observables that are available at inference time. Do not include
truth-level variables, particle ID codes, or columns derived from the label.

## Setup and training

This project is compatible with Python 3.8–3.10 and pinned to TensorFlow 2.10.0.
TensorFlow 2.10 wheels are not available for Python 3.11. On Apple silicon, the
dependency file selects `tensorflow-macos`; elsewhere it selects `tensorflow`.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -c "import tensorflow as tf; print(tf.__version__)"

python train.py tracks.csv \
  --features momentum dedx tof ecal_energy \
  --output artifacts/pion_classifier.h5
```

For ROOT files, give the tree path and branch names:

```bash
python train.py run_001.root run_002.root \
  --tree events/tracks \
  --features momentum dedx tof ecal_energy \
  --label is_pion \
  --output artifacts/pion_classifier.h5
```

To discover the tree path and available branches:

```bash
python inspect_root.py run_001.root
```

If a ROOT file contains exactly one TTree/RNTuple, `--tree` may be omitted.
Jagged event-level branches are intentionally rejected: flatten the track arrays
and broadcast event-level variables first so every feature and label refers to
the same track.

The command prints validation and test metrics and writes:

- the trained `.h5` model (including its normalization layer), and
- a neighboring `.json` metadata file containing the ordered feature names and
  classification threshold.

For real collision data, create train/validation/test CSV files by splitting on
event, run, and preferably data-taking period first. Then pass them with
`--validation-data` and `--test-data`; this avoids leakage between tracks from the
same event.

## Prediction

```bash
python predict.py artifacts/pion_classifier.h5 new_tracks.csv \
  --keep event_id track_id \
  --output pion_scores.csv
```

ROOT input uses the same `--tree events/tracks` option. Predictions are written
to CSV; use `--keep` to retain stable event/track keys for joining the scores
back to the source data. Identifier columns are not passed to the model.

The output adds `pion_score` (the model probability) and `pion_prediction`.
Tune the threshold on validation data for the physics goal: use a lower threshold
for pion efficiency or a higher one for sample purity.

## ANNIE `hitPE` pion model

`train_annie_pion.py` is the dedicated path for the jagged ANNIE event branches.
It follows the feature construction in `stv-analysis-Joint/stv_BDT_ANNIE` and
summarizes both PE vectors using the six BDT variables:

```text
hitPE_sum                    hitPE_tankcluster_sum
hitPE_mean                   hitPE_tankcluster_mean
hitPE_std                    hitPE_tankcluster_std
```

The tank-cluster branch is auto-detected as either `hitPE_tankcluster` or
`hitPE_tankclusters`. The default label is 1 when any of `truePiPlusCher`,
`truePiMinusCher`, or `truePi0` is positive. This is intentionally the inverse
of the old BDT's `tank_bdt_no_pion_score`, so a high `pion_score` means that a
pion is present.

The earlier BDT preselection is enabled by default. Train with:

```bash
python train_annie_pion.py simulation_*.root \
  --tree phaseIITriggerTree \
  --output artifacts/annie_pion.h5
```

Useful variations:

```bash
# Explicitly select the branch spelling used by a file
python train_annie_pion.py simulation.root \
  --tankcluster-branch hitPE_tankcluster

# Charged pions only (ignore pi0 when constructing the target)
python train_annie_pion.py simulation.root --charged-only

# Diagnostic training without the historical BDT event cuts
python train_annie_pion.py simulation.root --no-bdt-cuts
```

Apply the trained network to ROOT files:

```bash
python score_annie_pion.py artifacts/annie_pion.h5 sample.root \
  --output annie_pion_scores.csv
```

The score table contains the source filename, original TTree entry number,
`pion_score`, and thresholded prediction. Reading and feature extraction happen
in bounded-memory chunks; training arrays are held in memory after selection.

## PMT ring-image model

The recommended model for recognizing Cherenkov ring structure is
`train_annie_ring.py`. Unlike the six-variable model above, it keeps PMT spatial
information:

1. `hitDetID` and `hitDetID_tankcluster` are mapped to the bundled ANNIE PMT
   geometry.
2. PMT directions are calculated from `simpleRecoVtxX/Y/Z`.
3. Hits are projected into an elevation-versus-azimuth image.
4. A second image unfolds the physical detector into bottom endcap, barrel, and
   top endcap regions, following the geometry in
   `Draw_ANNIE_Single_Event_Detector_Image.cpp`.
5. Full-event `hitPE` and tank-cluster `hitPE` form two channels in both views.
6. Two convolutional towers combine the ring-centered and detector-layout
   information before making one event-level pion prediction.

Before training, render several pion and no-pion events and confirm that the
projection is sensible:

```bash
python view_annie_ring.py simulation.root 42 \
  --tree phaseIITriggerTree \
  --output ring_event_42.png
```

To build a complete, inspectable Model A image dataset from one or more ROOT
files, use:

```bash
python build_annie_training_images.py simulation_*.root \
  --tree phaseIITriggerTree \
  --preview-count 20 \
  --output-dir model_a_images
```

For a quick preprocessing test, add `--max-events 500`. The output directory
contains compressed `shard_*.npz` tensors, a `manifest.json` describing every
branch and preprocessing choice, and PNG files under `previews/`. Each shard
contains `angular_images` with shape `(N, 16, 34, 2)` and `detector_images` with
shape `(N, 48, 32, 2)`. The two channels are full-event PE and tank-cluster PE.
The angular width contains 32 physical bins plus two periodic edge-copy columns.
Preview creation targets an even split between pion and no-pion truth labels.

Train and score the ring model:

```bash
python train_annie_ring.py simulation_*.root \
  --tree phaseIITriggerTree \
  --output artifacts/annie_ring_pion.h5

python score_annie_ring.py artifacts/annie_ring_pion.h5 sample.root \
  --output annie_ring_scores.csv
```

This command is **Model A**, the PE-only, dual-view baseline. Each view uses
exactly two channels (`hitPE` and `hitPE_tankcluster`) and no charge or timing
inputs. Training also writes `annie_ring_pion.history.csv`, containing the
per-epoch training and validation metrics used to identify the best stopping
point.

The unfolded tensor deliberately does not copy presentation-only objects from
the C++ event display. It excludes PMT number labels, titles, axes, legends,
event/run text, charge-histogram insets, detector outlines, region labels, and
truth information. Only the PMT position and accumulated PE are retained. This
prevents the network from learning annotations or other information that will
not be available when scoring data.

The angular map uses 16 × 32 bins by default and duplicates the periodic
azimuth edge before convolution. PE is accumulated per angular bin and
transformed with `log(1 + PE)`. Events with misaligned PE and detector-ID
vectors are rejected and counted in the model metadata.

Bad PMTs are masked before either image channel is filled. The default
`--pmt-mask bdt` reproduces the historical BDT policy: known shutoff PMTs are
excluded, except PMT 358, which that workflow deliberately retains. The exact
excluded IDs are saved in the model JSON. Alternative policies are:

```bash
# Keep only PMTs whose geometry-table status is ON
python train_annie_ring.py simulation.root --pmt-mask on

# Include every geometry PMT (diagnostics only)
python view_annie_ring.py simulation.root 42 --pmt-mask all
```

This is an event-level, weakly supervised ring classifier: the available truth
labels say whether a pion is present, but do not give a ring center or radius.
Localizing or counting individual rings requires corresponding per-ring truth
targets (for example particle direction and vertex) or manually labeled images.

## Moving the project

Copy the complete project directory to the destination system. The source uses
only relative paths, so it does not depend on its current location. On the target:

```bash
cd PionID
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -c "import tensorflow as tf; print(tf.__version__)"
```

Each final model consists of two files that must remain together, for example:

```text
artifacts/pion_classifier.h5
artifacts/pion_classifier.json

artifacts/annie_ring_pion.h5
artifacts/annie_ring_pion.json
```

The `.h5` file contains the network and normalization layer. The `.json` file
stores preprocessing settings and the classification threshold. Keep the
bundled `PMT_position_id_info.cvs` beside the Python files when moving the ring
model. ROOT itself is not required on the target because `uproot` reads the
files in Python.
