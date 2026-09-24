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

Train and score the ring model:

```bash
python train_annie_ring.py simulation_*.root \
  --tree phaseIITriggerTree \
  --output artifacts/annie_ring_pion.h5

python score_annie_ring.py artifacts/annie_ring_pion.h5 sample.root \
  --output annie_ring_scores.csv
```

For long input lists, place one ROOT path per line in a text file. Blank lines
and comments beginning with `#` are ignored; relative paths are resolved from
the list file's directory:

```bash
python train_annie_ring.py --file-list simulation_files.txt \
  --output artifacts/annie_ring_pion.h5
```

This command is **Model A**, the PE-only, dual-view baseline. Each view uses
exactly two channels (`hitPE` and `hitPE_tankcluster`) and no charge or timing
inputs. Training also writes `annie_ring_pion.history.csv`, containing the
per-epoch training and validation metrics used to identify the best stopping
point. Evaluation includes count and true-class-normalized confusion matrices
at the requested `--threshold`, plus corresponding matrices at a fixed pion
score threshold of `0.20` (`score < 0.20` is classified as no pion).
It also writes an efficiency/purity threshold scan as both PNG and CSV. Here,
pion efficiency is `TP/(TP+FN)`, pion purity is `TP/(TP+FP)`, and the third
curve is their product. The plot highlights score `0.20` and the configured
`--threshold`.

By default, Model A uses the reconstructed-event selection from
`Fit_indivdiualPMT_Gaussian_Convolution.cpp`, with the charge-balance threshold
intentionally loosened from `0.30` to `0.25` for training:

```text
sel_CC0pi_wc
&& sel_promptMuonTotalPE_pmt_filtered
&& clusterChargeBalance_tankcluster_pmt_filtered > 0.25
&& sel_clusterHist_tankcluster_branch
&& clusterHits_tankcluster > 55
&& match_found
```

Use `--no-event-cuts` to disable this selection. The older spelling
`--no-bdt-cuts` remains available as an alias. Image construction also follows
the calibration feature ranges: full-event PE must be finite and non-negative,
while tank-cluster PE must be finite and in the half-open range `[0, 350)` PE.

Training also writes test-set diagnostic products beside the model: ROC and
precision-recall curves, score distributions, efficiency/background rejection
versus threshold, a confusion matrix, a probability-calibration curve, training
history plots, the most confident misclassified events, Grad-CAM examples for
both image towers, and `annie_ring_pion.pion_examples.png`. The pion-example
figure shows the full-event and tank-cluster PE channels in both angular and
unfolded views for high-scoring held-out pion events; each row includes the
source file, tree entry, model score, and the truth counts for pi+, pi-, and pi0.
The test CSV contains the same truth counts together with each event's source,
entry, binary truth label, and pion score. Their filenames and summary metrics
are stored under `evaluation` in the model JSON.

The unfolded tensor deliberately does not copy presentation-only objects from
the C++ event display. It excludes PMT number labels, titles, axes, legends,
event/run text, charge-histogram insets, detector outlines, region labels, and
truth information. Only the PMT position and accumulated PE are retained. This
prevents the network from learning annotations or other information that will
not be available when scoring data. Pion truth counts are used only for labels,
metadata, and diagnostic figure annotations; they are never image channels or
model inputs.

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

Model A uses raw PE by default. For simulated events it can replay the
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
The stored six event-selection branches are not recomputed after tuning, so
this replay changes the CNN images but does not model migration across those
preselection boundaries.

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
