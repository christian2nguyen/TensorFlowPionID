# PionID

This repository contains a generic tabular pion-ID baseline and the recommended
ANNIE Model C workflow. Model C is a supervised, two-head TensorFlow classifier
that predicts both pion presence and fiducial-volume membership from PMT images
and reconstructed MRD information.

## Expected data for the generic tabular baseline

This section describes `train.py`, not the ANNIE ring-image Model C workflow.
Model C branch requirements are listed later in this README.

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

The current environment is pinned to TensorFlow/Keras 2.13.1. It supports the
Python 3.8.13 environment used for ANNIE training and uses NumPy 1.22–1.24.3;
NumPy 1.23.5 is known to work. On Apple silicon the dependency file selects
`tensorflow-macos`; elsewhere it selects `tensorflow`.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -c "import tensorflow as tf, numpy, uproot, awkward, sklearn; print(tf.__version__)"
python -m pip check

python train.py tracks.csv \
  --features momentum dedx tof ecal_energy \
  --output artifacts/pion_classifier.h5
```

The TensorFlow message saying that AVX2/FMA could be enabled by rebuilding is
informational; it does not mean training failed. `pip check` should not report a
TensorFlow dependency conflict. In particular, TensorFlow 2.13.1 requires
`gast <= 0.4.0`, which is pinned in `requirements.txt`.

### ANNIE shared Python packages

On the ANNIE system, first activate the Python 3.9/TensorFlow environment and
then source the project setup:

```bash
cd /exp/annie/app/users/cnguyen/tensorflow_PionID/TensorFlowPionID
source setup.sh
```

The script makes this shared directory available:

```text
/exp/annie/app/users/dajana/myboy/lib/python3.9/site-packages
```

It explicitly selects shared Awkward 2.8.12 and Uproot 5.6.9, adds the rest of
the directory as a fallback after the active environment's own site-packages,
and prints the resolved version and path of every requested package. This lets
it replace an older LCG Awkward 1.x without replacing the compatible NumPy.
Do not prepend the complete shared path directly to `PYTHONPATH`: it contains
NumPy 1.26.4, which would override the TensorFlow-2.13-compatible NumPy 1.23.5
environment.

The currently observed shared versions are:

| Package | Shared version | Requested range | Setup behavior |
| --- | ---: | --- | --- |
| NumPy | 1.26.4 | `>=1.22,<=1.24.3` | Critical incompatibility if selected; setup fails |
| pandas | 2.3.3 | `>=1.5,<2.1` | Warning when selected |
| scikit-learn | 1.6.1 | `>=1.1,<1.4` | Warning when selected |
| Uproot | 5.6.9 | `>=5.0,<6` | Compatible |
| Awkward | 2.8.12 | `>=2.0,<3` | Compatible |
| Matplotlib | 3.9.4 | `>=3.5,<3.8` | Warning when selected |

The shared directory is built for Python 3.9, so `setup.sh` rejects a different
Python minor version. NumPy, Uproot, and Awkward are treated as critical because
they directly affect TensorFlow compatibility and ROOT/jagged-array reading.
The other out-of-range packages are reported as warnings so their shared builds
can still be tested deliberately. The checker also requires TensorFlow 2.13.1
and `gast <= 0.4.0`. The original NumPy range is narrowed to
`>=1.22,<=1.24.3` to match TensorFlow 2.13.1. The environment checker is also
available on its own:

```bash
python3 verify_python_environment.py
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
   information with the MRD inputs before making independent event-level
   `pion_score` and `fv_score` predictions.

Before training, render several pion and no-pion events and confirm that the
projection is sensible:

```bash
python view_annie_ring.py simulation.root 42 \
  --tree phaseIITriggerTree \
  --output ring_event_42.png
```

To see exactly what the per-PMT response tune changes for one simulated event,
draw raw PE, tuned PE, and their difference with a shared color scale:

```bash
python view_annie_ring.py simulation.root 42 \
  --compare-pmt-response \
  --pmt-response-calibration individual_pmt_fit_output.root \
  --pmt-tune-variant final \
  --output ring_event_42_raw_vs_tuned.png
```

The four figure rows cover angular full-event PE, angular tank-cluster PE,
unfolded full-event PE, and unfolded tank-cluster PE. Each row reports raw and
tuned total accumulated PE and their percentage change. Simulation pion truth
counts are added to the title when those branches are present.

To automatically find and draw ten truth-pion events that pass the same event
selection used for training:

```bash
python view_annie_ring.py simulation.root -n 10 \
  --output-dir pion_event_plots
```

To search the ordered MC list used for training and compare raw and tuned PE:

```bash
python view_annie_ring.py \
  --file-list training_mc_files.txt \
  -n 10 \
  --compare-pmt-response \
  --pmt-response-calibration individual_pmt_fit_output.root \
  --pmt-tune-variant response \
  --output-dir pion_response_comparisons
```

By default, π⁺, π⁻, or π⁰ makes an event a pion event. Add `--charged-only` to
require π⁺ or π⁻, or `--no-event-cuts` to scan truth pions without the training
selection. The output directory also contains `pion_plots.csv` with each PNG's
source file, local tree entry, chain entry, and three truth-pion counts.

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
The shards also retain `truth_pion_counts` in the order `truePiPlusCher`,
`truePiMinusCher`, `truePi0`, plus source and tree-entry provenance. The angular
width contains 32 physical bins plus two periodic edge-copy columns. Preview
creation targets an even split between pion and no-pion truth labels, and each
preview title reports the three pion counts. When an odd number is requested,
the extra preview is assigned to the pion class; therefore `--preview-count 1`
requests one pion example.

`build_annie_training_images.py` is the inspectable Model A image-builder path;
it does not train the two-head Model C network. Model C reads the ROOT files
directly with `train_annie_ring.py` so it can also load MRD inputs and `trueFV`.

### Model C ROOT branches

Model C reads the following branches. Training-only truth branches are not
requested when applying a saved model to detector data.

| Purpose | Branches | Expected layout |
| --- | --- | --- |
| Full-event PMT image | `hitPE`, `hitDetID` | Aligned per-event vectors |
| Tank-cluster PMT image | `hitPE_tankcluster` or `hitPE_tankclusters`, `hitDetID_tankcluster` | Aligned per-event vectors |
| Image direction | `simpleRecoVtxX`, `simpleRecoVtxY`, `simpleRecoVtxZ` | Event scalars |
| Default physics selection | `sel_nu_mu_cc` | Boolean or 0/1 event scalar |
| MRD multiplicity | `numMRDTracks` | Event scalar |
| MRD track starts | `MRDTrackStartX`, `MRDTrackStartY`, `MRDTrackStartZ` | Per-event vectors; all three lengths must agree with `numMRDTracks` |
| MRD track properties | `MRDEnergyLoss`, `MRDTrackLength`, `MRDTrackAngle` | Per-track vectors or event scalars |
| Pion supervision | `truePiPlusCher`, `truePiMinusCher`, `truePi0` | Training-only event scalars |
| FV supervision | `trueFV` | Training-only Boolean or 0/1 event scalar |

The default pion label is true when any of the three pion truth counts is
positive. `--charged-only` changes it to require `truePiPlusCher` or
`truePiMinusCher`; it does not alter the FV label. The FV label is true exactly
when `trueFV` is true. Both pion classes and both FV classes must remain after
selection so the two heads can be trained and evaluated.

Truth momentum or opening-angle branches are optional. When available, they
are used only to annotate held-out diagnostic figures and never enter either
network head.

Train and score the ring model:

```bash
python train_annie_ring.py simulation_*.root \
  --tree phaseIITriggerTree \
  --pion-threshold 0.70 \
  --fv-threshold 0.50 \
  --fv-loss-weight 0.30 \
  --epochs 100 \
  --batch-size 128 \
  --output artifacts/annie_ring_pion.h5

python score_annie_ring.py artifacts/annie_ring_pion.h5 sample.root \
  --pion-threshold 0.70 \
  --fv-threshold 0.50 \
  --output annie_ring_scores.csv
```

`--epochs` is a maximum: early stopping restores the best weights after 12
epochs without improvement in validation pion PR AUC. The learning rate is
halved after five stagnant epochs. `--seed` controls the reproducible 70/15/15
split and defaults to `20260620`. `--chunk-size` controls bounded ROOT reading,
but the selected tensors are concatenated in memory before model fitting, so it
does not cap total training memory.

### Independent pion and fiducial-volume predictions

The two output heads share the image and MRD feature extractor but have
separate dense layers, binary-cross-entropy losses, truth labels, metrics, and
decision thresholds:

```text
pion_prediction = pion_score >= pion_threshold
fv_prediction   = fv_score   >= fv_threshold
total_loss      = pion_loss + fv_loss_weight * fv_loss
```

`--fv-loss-weight` controls how strongly learning the FV task changes the
shared representation; its default is `0.30`. It does not multiply either
output score. The training sample weights balance pion/no-pion and
inside/outside-FV classes independently. Early stopping and learning-rate
reduction continue to monitor validation pion PR AUC, so the primary training
objective remains pion identification.

No correlation penalty, score multiplication, or conditional gate is applied.
The scores can naturally be correlated if the underlying event populations are
correlated, but the model is not required to make them agree. A later physics
selection may require both predictions, for example:

```python
selected = (pion_score >= 0.70) & (fv_score >= 0.50)
```

Omit either scoring threshold option to use its value stored in the model JSON.
Changing a threshold changes only the binary decision and does not require
retraining. The two threshold options control evaluation and inference; they do
not change either training loss. `trueFV` is required while training because it
supplies the FV target, but it is not a network input and is not needed in
detector data during inference.

By default, scoring reapplies the saved `sel_nu_mu_cc == true` selection. Use
`--all-events` only when the input was already selected elsewhere or when scores
are intentionally needed for every event. The scoring CSV records both scores,
both applied thresholds, and both independent decisions for every written
event.

For `--output artifacts/annie_ring_pion.h5`, the main artifacts are:

| Artifact | Contents |
| --- | --- |
| `annie_ring_pion.h5` | Keras Model C network and adapted normalization layers |
| `annie_ring_pion.json` | Inputs, PMT response provenance, both thresholds, split/class counts, metrics, and diagnostic filenames |
| `annie_ring_pion.history.csv` | Per-epoch training and validation losses and metrics for both heads |
| `annie_ring_pion.test_predictions.csv` | Held-out provenance, truth, model inputs, both scores, both thresholds, and both decisions |
| `annie_ring_pion.training_curves.png` | Total loss plus pion/FV ROC-AUC and PR-AUC histories |
| `annie_ring_pion.misclassified_events.pdf` | Up to 20 annotated pion-head false positives and false negatives in one PDF |
| `annie_ring_pion.gradcam_examples.png` | Pion-head Grad-CAM examples for both image towers |

The pion and FV validation figures described below are written beside these
files with the same model stem.

Export the trained hybrid model for C++ inference with the exporter in this
project. Its serving signature includes both image tensors, the ordered
`event_features` tensor, and the padded MRD track tensors. Its outputs are the
continuous `pion_score` and `fv_score` tensors; downstream C++ should read the
two thresholds from the neighboring JSON and apply them independently:

```bash
python export_annie_ring_saved_model.py \
  artifacts/annie_ring_pion.h5 \
  artifacts/annie_ring_pion_saved_model
```

The scalar feature order is saved in the JSON metadata and currently contains
`numMRDTracks`. The exact MRD start positions are a separate
`mrd_track_starts` tensor with shape `(batch, 4, 3)` in X/Y/Z order. A
`mrd_track_properties` tensor with shape `(batch, 4, 3)` contains
`MRDEnergyLoss`, `MRDTrackLength`, and `MRDTrackAngle` for those same tracks. A
`mrd_track_mask` tensor with shape `(batch, 4)` is 1 for a real track and 0 for
padding. Therefore a one-track event retains its exact start position without
inventing three additional tracks.

The three MRD property branches may be stored either as per-track vectors or as
one scalar per event. Vector values are aligned with their corresponding track.
When a property is an event scalar, that same value is broadcast to every valid
retained track slot; the mask still removes padded slots. A scalar ROOT branch
cannot provide distinct property values for multiple tracks, so the model uses
it as shared event-level context for the track encoder.

Compare the MRD schemas of a working and failing ROOT file with:

```bash
python inspect_annie_mrd_schema.py working.root different.root \
  --tree phaseIITriggerTree
```

The report shows each ROOT type and whether Awkward reads it as an event scalar
or a per-event vector. It also reports sampled vector-length ranges and their
mismatch counts relative to `numMRDTracks`. The MRD start X/Y/Z inputs must be
per-event vectors; an event-scalar start coordinate does not contain enough
information to construct a multi-track set without an explicit interpretation.

For long input lists, place one ROOT path per line in a text file. Blank lines
and comments beginning with `#` are ignored; relative paths are resolved from
the list file's directory:

```bash
python train_annie_ring.py --file-list simulation_files.txt \
  --output artifacts/annie_ring_pion.h5
```

To train with two explicitly different detector-response groups, use one list
for MC that should be convolved on the fly and one list that should remain raw:

```bash
python train_annie_ring.py \
  --tuned-file-list mc_to_convolve.txt \
  --raw-file-list mc_to_leave_raw.txt \
  --pmt-response-calibration fitted_calibration.root \
  --pmt-tune-variant response \
  --output artifacts/annie_ring_pion_mixed_response.h5
```

`--convolved-file-list` is an alias for `--tuned-file-list`, and
`--unconvolved-file-list` is an alias for `--raw-file-list`. Files in the tuned
list are convolved during image construction; therefore, a ROOT file whose PE
branches are already convolved belongs in the raw list to avoid applying the
response twice. The `response` tune variant applies the Gain/Delta/Sigma
response, while `final` also replays residual hits. The saved JSON records each
source group, its response mode, and its selected-event count.

Do not put the same events in both lists merely as raw and tuned copies. The
current event-level random split could place one copy in training and the other
in validation or testing, producing overly optimistic metrics. Use independent
event samples in the two lists.

This command is **Model C**, a hybrid two-head classifier. Its two image views each use
exactly two channels (`hitPE` and `hitPE_tankcluster`). A scalar input contains
`numMRDTracks`, while a second input preserves up to four exact start positions
from `MRDTrackStartX`, `MRDTrackStartY`, and `MRDTrackStartZ`. A third input
provides the aligned per-track `MRDEnergyLoss`, `MRDTrackLength`, and
`MRDTrackAngle` values. Starts and properties are normalized separately, then
combined track by track. The same dense encoder is applied independently to
every valid track, padding is removed with the mask, and global max pooling
makes the result insensitive to track order. Normalization is adapted only to
valid tracks in the training subset. Empty events contain four zero-padded
positions, four zero-padded property rows, and an all-zero mask.
Events with inconsistent X/Y/Z vector lengths or lengths inconsistent with
`numMRDTracks` are rejected. Events with more than four tracks retain the first
four entries in stored branch order; their full multiplicity remains available
through `numMRDTracks`, and the truncated-event count is saved in the model JSON
metadata. `clusterChargeBalance_tankcluster_pmt_filtered` is neither a model
input nor an event-selection cut.

The shared representation feeds two independent head-specific dense layers:
`pion_score` is supervised by the pion truth label, while `fv_score` is
supervised by `trueFV`. `trueFV` is never supplied as a model input. No
correlation loss, score multiplication, or conditional gating is applied. The
two decisions use separate options, `--pion-threshold` (also accepted as
`--threshold`) for pion ID and
`--fv-threshold` for FV classification. The FV auxiliary loss has relative
weight `--fv-loss-weight` (default `0.3`). Scoring CSVs and the exported
SavedModel contain both scores.

Training also writes `annie_ring_pion.history.csv`, containing the per-epoch
training and validation metrics used to identify the best stopping point.
Events are split reproducibly and stratified by the joint pion/`trueFV` truth
category into 70% training, 15% validation, and 15% testing. Joint
stratification preserves the four possible pion/FV label combinations when
they are present. Evaluation includes count and true-class-normalized confusion
matrices at the requested `--pion-threshold`. A separate confidence-band matrix
classifies `score < 0.20` as non-pion-like and `score > 0.80` as pion-like;
events in the middle band are excluded and their count is printed on the plot.
This confidence-band matrix is written in count and true-class-normalized forms.
A second count and normalized pair uses the looser confidence bands
`score < 0.30` for non-pion-like and `score > 0.70` for pion-like, excluding
the middle `0.30 <= score <= 0.70` region.
It also writes an efficiency/purity threshold scan as both PNG and CSV. Here,
pion efficiency is `TP/(TP+FN)`, pion purity is `TP/(TP+FP)`, and the third
curve is their product. The plot highlights the `0.20` and `0.80` confidence
boundaries and the configured `--threshold`. The CSV and JSON metadata also
record specificity, balanced accuracy, ordinary accuracy, F1, and the Matthews
correlation coefficient at each evaluated operating point. A Brier score
provides a threshold-independent check of probability calibration.

### Fiducial-volume validation outputs

The held-out test set produces a separate FV validation suite:

- `annie_ring_pion.fv_score_distribution.png` compares normalized `fv_score`
  distributions for `trueFV = 0` and `trueFV = 1`, with `--fv-threshold` drawn
  as a dashed line. This is the main FV signal-separation figure.
- `annie_ring_pion.fv_roc_curve.png` reports the outside-FV false-positive rate,
  inside-FV efficiency, and ROC AUC.
- `annie_ring_pion.fv_precision_recall_curve.png` reports inside-FV efficiency,
  FV purity, and average precision.
- `annie_ring_pion.fv_confusion_matrix.png` and
  `annie_ring_pion.fv_confusion_matrix_normalized.png` show count and
  truth-class-normalized decisions at the configured FV threshold.
- The model JSON records `fv_roc_auc`, `fv_average_precision`, `fv_brier_score`,
  and the FV operating point. The operating point includes efficiency, purity,
  efficiency times purity, specificity, balanced accuracy, accuracy, F1, MCC,
  and the four confusion-matrix counts.

The FV plots are independent of the pion confidence-band plots. The FV
confusion matrix always uses `fv_score >= fv_threshold`; it does not use the
special pion score bands of 0.20/0.80 or 0.30/0.70.

Four held-out truth-pion image pages are produced for visual validation: a
dedicated charged-pion page, the highest-scoring pions, pions nearest the
configured decision threshold, and the lowest-scoring pions. Each page shows up
to three events with all four PMT image views and pion-truth annotations.
The annotations include the truth muon and leading-pion kinetic energies in GeV
when their truth momentum vectors are available.

By default, Model C applies only this physics event-selection cut:

```text
sel_nu_mu_cc == true
```

Use `--no-event-cuts` to disable this selection. The older spelling
`--no-bdt-cuts` remains available as an alias. Image construction also follows
the calibration feature ranges: full-event PE must be finite and non-negative,
while tank-cluster PE must be finite and in the half-open range `[0, 350)` PE.
Misaligned PMT charge/ID vectors, non-finite model inputs, and inconsistent MRD
track-vector lengths are rejected as input-quality requirements; they are not
additional physics selection cuts.

Training also writes test-set diagnostic products beside the model: ROC and
precision-recall curves, score distributions, efficiency/background rejection
versus threshold, a confusion matrix, a probability-calibration curve, training
history plots, the most confident misclassified events, Grad-CAM examples for
both image towers, and four pion-example pages. The pion-example pages show the
full-event and tank-cluster PE channels in both angular and unfolded views for
charged-pion, high-scoring, threshold-boundary, and low-scoring held-out pion
events; each row includes the source file, tree entry, model score, and the
truth counts for pi+, pi-, and pi0. Every row also includes angular and unfolded
Grad-CAM heatmaps;
these tower-level maps use both PE channels and are overlaid on full-event
`hitPE` for detector-coordinate context. The annotation also reports the truth
opening angle in degrees between the muon and leading pion. This is calculated
from `mc_p3_mu` and `mc_p3_lead_pi` (or `mc_p3_lead_pion`) when available and is
never supplied to the CNN. The same truth vectors provide the momentum
magnitudes used to calculate relativistic kinetic energy as
`T = sqrt(p^2 + m^2) - m`. The code uses the muon rest mass and selects the
charged- or neutral-pion rest mass from the event's pion truth counts. Momentum
is interpreted in GeV/c, matching the source analysis, and kinetic energy is
reported in GeV. If the ROOT file instead contains a scalar angle in
degrees, select it explicitly with
`--muon-pion-opening-angle-deg-branch BRANCH_NAME`. The test CSV contains the
same truth counts, opening angle, muon kinetic energy, and leading-pion kinetic
energy together with each event's source and entry. It also records the pion
truth label, `trueFV`, all scalar and padded MRD inputs, `pion_score`,
`pion_threshold`, `pion_prediction`, `fv_score`, `fv_threshold`, and
`fv_prediction`. Its filename and all summary metrics are stored under
`evaluation` in the model JSON.

The misclassified-event output keeps the combined gallery and writes the
selected events to one multipage PDF named
`annie_ring_pion.misclassified_events.pdf` for the default model output. It
does not write separate PNG files for each event. Each event occupies one PDF
page. The selection initially takes up to 10 of the most confident false
positives and 10 of the most confident false negatives, then fills unused slots
from the other category.
Every example includes its source file, tree entry, true and predicted classes,
pion score, truth pion counts, truth muon-pion opening angle, and truth muon and
leading-pion kinetic energies. Unavailable truth values are shown as `n/a`.

The unfolded tensor deliberately does not copy presentation-only objects from
the C++ event display. It excludes PMT number labels, titles, axes, legends,
event/run text, charge-histogram insets, detector outlines, region labels, and
truth information. Only the PMT position and accumulated PE are retained. This
prevents the network from learning annotations or other information that will
not be available when scoring data. Pion truth counts are used only for labels,
metadata, and diagnostic figure annotations; truth opening angles and kinetic
energies are also diagnostic-only. None of these truth quantities are image
channels or model inputs.

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

### Optional per-PMT response tuning

The ring model uses raw PE by default. For simulated events it can replay the
`final_pmt_tuning_parameters` payload written by
`Fit_indivdiualPMT_Gaussian_Convolution.cpp`. This payload has independent
regional parameters for full-event `hitPE` (`branch_kind=0`) and tank-cluster
`hitPE` (`branch_kind=1`):

```bash
python build_annie_training_images.py simulation.root \
  --pmt-response tuned \
  --pmt-response-calibration individual_pmt_fit_output.root \
  --pmt-tune-variant final \
  --output-dir model_a_tuned_images

python train_annie_ring.py simulation.root \
  --pmt-response tuned \
  --pmt-response-calibration individual_pmt_fit_output.root \
  --pmt-tune-variant final \
  --output artifacts/annie_ring_pion_tuned.h5
```

For each hit, Python selects the saved row matching PMT ID, branch kind, and
original PE region `[region_min, region_max_exclusive)`, then applies
`Gain*PE + Delta + Normal(0,Sigma)`. The deterministic Gaussian stream matches
the C++ `ordered-field-mixing-v2` implementation. `--pmt-tune-variant response`
stops there. The default, `--pmt-tune-variant final`, additionally replays the
stored residual hit weights as deterministic hit rejection or duplication;
the weights are never multiplied into PE as scale factors. In the image,
duplicated hits are equivalent to accumulating the transformed PE repeatedly
in the same PMT pixel.

For multiple ROOT inputs, the deterministic random stream uses the files in
the exact command-line or `--file-list` order, matching their TChain entry
ordering. Keep that order fixed when reproducing the same tuned simulation.

Older calibration ROOT files containing
`PMT_Calibrations/PMT_<id>/response_map` TGraphs remain supported. Those files
have one shared per-PMT map rather than separate full/tank payloads, and the
tune variant has no effect on them.

The calibration filename, SHA-256 digest, payload format, tune variant, random
stream version, source-file list, and mapped PMT count are saved in the
manifest/model metadata. This permits the exact calibration file to be checked
when tuned simulation is scored later.

The response map transforms **simulation toward detector data**. Therefore:

- train or preview simulated input with `--pmt-response tuned`;
- score simulated input with the same tuned calibration; and
- score beam-on detector data with the default `--pmt-response raw`.

For example, scoring tuned simulation requires:

```bash
python score_annie_ring.py artifacts/annie_ring_pion_tuned.h5 simulation.root \
  --pmt-response tuned \
  --pmt-response-calibration individual_pmt_fit_output.root \
  --output tuned_mc_scores.csv
```

Scoring defaults to the tune variant stored with the model and rejects a
different calibration checksum or tune variant. Detector data remains raw.
The stored `sel_nu_mu_cc` decision is not recomputed after tuning, so this
replay changes the CNN images but does not model migration across that
preselection boundary.

This is an event-level, weakly supervised, multi-task classifier: one truth
label says whether a pion is present and the other says whether the event is in
the fiducial volume, but neither provides a ring center or radius. Localizing or
counting individual rings requires corresponding per-ring truth targets (for
example particle direction and vertex) or manually labeled images.

## Moving the project

Copy the complete project directory to the destination system. The source uses
only relative paths, so it does not depend on its current location. On the target:

```bash
cd PionID
python3 -m venv .venv
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

The `.h5` file contains the network and normalization layers. The `.json` file
stores preprocessing settings, both output-head definitions, both decision
thresholds, the FV loss weight, response-tuning provenance, class counts, data
split, test metrics, and diagnostic filenames. Keep the
bundled `PMT_position_id_info.cvs` beside the Python files when moving the ring
model. ROOT itself is not required on the target because `uproot` reads the
files in Python.
