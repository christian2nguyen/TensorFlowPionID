#!/usr/bin/env python3
"""Train an ANNIE CC-inclusive classifier from detector PMT images and MRD inputs."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import awkward as ak
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    auc,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)
from sklearn.model_selection import train_test_split
import tensorflow as tf
import uproot

from annie_features import (
    MAX_MRD_TRACKS,
    MRD_TRACK_PROPERTY_BRANCHES,
    MRD_TRACK_START_BRANCHES,
    MRD_TRACK_STOP_BRANCHES,
    build_ring_event_features,
    resolve_tankcluster_branch,
)
from annie_pmt_response import (
    PMT_RESPONSE_CHOICES,
    PMT_TUNE_VARIANT_CHOICES,
    PMTResponse,
)
from annie_ring_images import (
    DEFAULT_DETECTOR_HEIGHT,
    DEFAULT_DETECTOR_WIDTH,
    DEFAULT_GEOMETRY,
    PMT_MASK_CHOICES,
    PMTGeometry,
    build_unfolded_detector_images,
)


TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15
TEST_FRACTION = 0.15
TRUTH_BRANCHES = ("trueCC", "trueFSLPdg")
FV_BRANCH_ALIASES = ("simpleRecoFV", "recoFV", "sel_vertex_in_FV")
SELECTION_INPUT_SPECS = (
    ("sel_trigword", "trigword", 5),
    ("sel_HasTank", "HasTank", 1),
    ("sel_HasMRD", "HasMRD", 1),
    ("sel_TankMRDCoinc", "TankMRDCoinc", 1),
    ("sel_noVeto", "NoVeto", 1),
    ("sel_simpleRecoFlag", "simpleRecoFlag", 1),
)
DEFAULT_PROMPT_PE_BRANCH = "promptMuonTotalPE_filtered"


def _to_numpy(values, dtype=np.float64) -> np.ndarray:
    return ak.to_numpy(values).astype(dtype, copy=False)


def _read_file_list(parser, option: str, path: Path) -> List[Path]:
    if not path.is_file():
        parser.error(f"{option} {path} does not exist")
    paths = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        listed_path = Path(stripped).expanduser()
        if not listed_path.is_absolute():
            listed_path = path.parent / listed_path
        paths.append(listed_path)
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root_files", nargs="*", type=Path)
    parser.add_argument("--file-list", type=Path)
    parser.add_argument(
        "--tuned-file-list",
        "--convolved-file-list",
        dest="tuned_file_list",
        type=Path,
    )
    parser.add_argument(
        "--raw-file-list",
        "--unconvolved-file-list",
        dest="raw_file_list",
        type=Path,
    )
    parser.add_argument("--tree", default="phaseIITriggerTree")
    parser.add_argument("--geometry", type=Path, default=DEFAULT_GEOMETRY)
    parser.add_argument("--pmt-mask", choices=PMT_MASK_CHOICES, default="bdt")
    parser.add_argument("--hitpe-branch", default="hitPE")
    parser.add_argument("--hitid-branch", default="hitDetID")
    parser.add_argument("--tankcluster-branch")
    parser.add_argument("--tankcluster-id-branch", default="hitDetID_tankcluster")
    parser.add_argument("--prompt-pe-branch", default=DEFAULT_PROMPT_PE_BRANCH)
    parser.add_argument(
        "--fv-branch",
        help=(
            "reconstructed FV branch; by default auto-detect simpleRecoFV, "
            "recoFV, or sel_vertex_in_FV"
        ),
    )
    parser.add_argument(
        "--no-fv-cut",
        action="store_true",
        help="disable the default reconstructed fiducial-volume requirement",
    )
    parser.add_argument("--detector-height", type=int, default=DEFAULT_DETECTOR_HEIGHT)
    parser.add_argument("--detector-width", type=int, default=DEFAULT_DETECTOR_WIDTH)
    parser.add_argument(
        "--pmt-response", choices=PMT_RESPONSE_CHOICES, default="raw"
    )
    parser.add_argument("--pmt-response-calibration", type=Path)
    parser.add_argument(
        "--pmt-tune-variant",
        choices=PMT_TUNE_VARIANT_CHOICES,
        default="final",
    )
    parser.add_argument("--chunk-size", default="50 MB")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20261012)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/annie_CCinclusive.keras"),
    )
    args = parser.parse_args()

    if args.file_list is not None:
        args.root_files.extend(_read_file_list(parser, "--file-list", args.file_list))
    args.tuned_root_files = (
        _read_file_list(parser, "--tuned-file-list", args.tuned_file_list)
        if args.tuned_file_list is not None
        else []
    )
    args.raw_root_files = (
        _read_file_list(parser, "--raw-file-list", args.raw_file_list)
        if args.raw_file_list is not None
        else []
    )
    all_paths = [*args.root_files, *args.tuned_root_files, *args.raw_root_files]
    if not all_paths:
        parser.error("Supply ROOT files or one of the file-list options")
    duplicates = [path for path, count in Counter(all_paths).items() if count > 1]
    if duplicates:
        parser.error(
            "The same ROOT path appears more than once: "
            + ", ".join(str(path) for path in sorted(duplicates, key=str))
        )
    if not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be between 0 and 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    return args


def _resolve_fv_branch(available: set, requested: Optional[str]) -> str:
    if requested:
        if requested not in available:
            raise ValueError(f"Requested FV branch {requested!r} was not found")
        return requested
    matches = [name for name in FV_BRANCH_ALIASES if name in available]
    if not matches:
        raise ValueError(
            "No reconstructed FV branch was found. Tried: "
            + ", ".join(FV_BRANCH_ALIASES)
        )
    return matches[0]


def _build_scalar_features(
    arrays, prompt_pe_branch: str, mrd_count_features: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    columns = []
    valid = np.ones(len(arrays), dtype=bool)
    for _, branch, required_value in SELECTION_INPUT_SPECS:
        values = _to_numpy(arrays[branch])
        if values.ndim != 1:
            raise ValueError(f"Scalar input branch {branch!r} is not event-level")
        valid &= np.isfinite(values)
        columns.append((values == required_value).astype(np.float32))
    prompt_pe = _to_numpy(arrays[prompt_pe_branch])
    if prompt_pe.ndim != 1:
        raise ValueError(f"{prompt_pe_branch!r} is not an event-level scalar")
    valid &= np.isfinite(prompt_pe)
    columns.append(prompt_pe.astype(np.float32))
    columns.append(mrd_count_features[:, 0].astype(np.float32))
    features = np.column_stack(columns).astype(np.float32, copy=False)
    valid &= np.isfinite(features).all(axis=1)
    return features, valid


def _build_track_stops(arrays) -> Tuple[np.ndarray, np.ndarray]:
    """Return padded MRD stop coordinates and a per-event validity mask."""
    num_tracks = _to_numpy(arrays["numMRDTracks"])
    valid = np.isfinite(num_tracks) & (num_tracks >= 0.0)
    valid &= np.isclose(num_tracks, np.rint(num_tracks))
    padded = []
    for branch in MRD_TRACK_STOP_BRANCHES:
        values = arrays[branch]
        try:
            counts = _to_numpy(ak.num(values, axis=1))
        except ValueError as error:
            raise ValueError(
                f"MRD stop branch {branch!r} must be a per-event vector, but "
                "this ROOT file stores it as a scalar"
            ) from error
        valid &= counts == np.rint(num_tracks)
        valid &= ak.to_numpy(ak.all(np.isfinite(values), axis=1)).astype(bool)
        padded.append(
            _to_numpy(
                ak.fill_none(
                    ak.pad_none(values, MAX_MRD_TRACKS, axis=1, clip=True),
                    0.0,
                )
            )
        )
    stops = np.stack(padded, axis=-1).astype(np.float32, copy=False)
    valid &= np.isfinite(stops).all(axis=(1, 2))
    return stops, valid


def _input_groups(args: argparse.Namespace):
    needs_tuned = bool(args.tuned_root_files) or (
        bool(args.root_files) and args.pmt_response == "tuned"
    )
    needs_raw = bool(args.raw_root_files) or (
        bool(args.root_files) and args.pmt_response == "raw"
    )
    responses = {}
    if needs_tuned:
        responses["tuned"] = PMTResponse(
            "tuned", args.pmt_response_calibration, args.pmt_tune_variant
        )
    if needs_raw:
        responses["raw"] = PMTResponse(
            "raw", tune_variant=args.pmt_tune_variant
        )
    groups = []
    if args.root_files:
        groups.append(("configured", args.root_files, responses[args.pmt_response]))
    if args.tuned_root_files:
        groups.append(("tuned", args.tuned_root_files, responses["tuned"]))
    if args.raw_root_files:
        groups.append(("raw", args.raw_root_files, responses["raw"]))
    return groups, responses


def load_data(args: argparse.Namespace) -> Dict[str, object]:
    geometry = PMTGeometry(args.geometry, args.pmt_mask)
    groups, responses = _input_groups(args)
    for mode, response in responses.items():
        print(f"{mode.capitalize()} input — {response.describe()}")

    detector_chunks = []
    scalar_chunks = []
    start_chunks = []
    stop_chunks = []
    property_chunks = []
    mask_chunks = []
    label_chunks = []
    source_chunks = []
    entry_chunks = []
    fv_branches = set()
    tank_branches = set()
    group_summaries = []
    total_read = 0
    total_selected = 0
    total_misaligned = 0
    total_invalid_features = 0
    total_outside_fv = 0
    total_mrd_overflow = 0
    chain_entry_offset = 0

    for group_name, paths, response in groups:
        group_read = 0
        group_selected = 0
        for path in paths:
            with uproot.open(path) as root_file:
                if args.tree not in root_file:
                    raise ValueError(f"Tree {args.tree!r} was not found in {path}")
                tree = root_file[args.tree]
                available = set(tree.keys())
                tank_branch = resolve_tankcluster_branch(
                    tree, args.tankcluster_branch
                )
                fv_branch = ""
                if not args.no_fv_cut:
                    fv_branch = _resolve_fv_branch(available, args.fv_branch)
                    fv_branches.add(fv_branch)
                tank_branches.add(tank_branch)
                requested = [
                    tank_branch,
                    args.tankcluster_id_branch,
                    *(branch for _, branch, _ in SELECTION_INPUT_SPECS),
                    args.prompt_pe_branch,
                    "numMRDTracks",
                    *MRD_TRACK_START_BRANCHES,
                    *MRD_TRACK_STOP_BRANCHES,
                    *MRD_TRACK_PROPERTY_BRANCHES,
                    *TRUTH_BRANCHES,
                ]
                if fv_branch:
                    requested.append(fv_branch)
                missing = [name for name in requested if name not in available]
                if missing:
                    raise ValueError(
                        f"{path} is missing branches: {', '.join(missing)}"
                    )

                entry_offset = 0
                for arrays in tree.iterate(
                    list(dict.fromkeys(requested)),
                    step_size=args.chunk_size,
                    library="ak",
                ):
                    n_entries = len(arrays)
                    entries = np.arange(
                        entry_offset, entry_offset + n_entries, dtype=np.int64
                    )
                    tuning_entries = entries + chain_entry_offset
                    detector, detector_aligned = build_unfolded_detector_images(
                        arrays,
                        geometry,
                        args.hitpe_branch,
                        args.hitid_branch,
                        tank_branch,
                        args.tankcluster_id_branch,
                        args.detector_height,
                        args.detector_width,
                        response,
                        tuning_entries,
                        tankcluster_only=True,
                    )
                    (
                        mrd_count_features,
                        track_starts,
                        track_properties,
                        track_mask,
                        valid_mrd,
                        mrd_overflow,
                    ) = build_ring_event_features(arrays, ["numMRDTracks"])
                    track_stops, valid_stops = _build_track_stops(arrays)
                    scalar_features, valid_scalars = _build_scalar_features(
                        arrays, args.prompt_pe_branch, mrd_count_features
                    )
                    aligned = detector_aligned
                    valid_features = valid_mrd & valid_stops & valid_scalars
                    selected = aligned & valid_features
                    outside_fv = np.zeros(n_entries, dtype=bool)
                    if fv_branch:
                        fv_values = _to_numpy(arrays[fv_branch])
                        valid_fv = np.isfinite(fv_values)
                        inside_fv = valid_fv & (fv_values != 0)
                        outside_fv = valid_fv & ~inside_fv
                        selected &= inside_fv
                    true_cc = _to_numpy(arrays["trueCC"])
                    true_fs_lepton = _to_numpy(arrays["trueFSLPdg"])
                    valid_truth = np.isfinite(true_cc) & np.isfinite(true_fs_lepton)
                    selected &= valid_truth
                    labels = ((true_cc == 1) & (true_fs_lepton == 13)).astype(
                        np.float32
                    )

                    detector_chunks.append(
                        detector[selected].astype(np.float16, copy=False)
                    )
                    scalar_chunks.append(scalar_features[selected])
                    start_chunks.append(track_starts[selected])
                    stop_chunks.append(track_stops[selected])
                    property_chunks.append(track_properties[selected])
                    mask_chunks.append(track_mask[selected])
                    label_chunks.append(labels[selected])
                    entry_chunks.append(entries[selected])
                    source_chunks.append(
                        np.full(int(selected.sum()), str(path), dtype=object)
                    )

                    selected_count = int(selected.sum())
                    total_read += n_entries
                    group_read += n_entries
                    total_selected += selected_count
                    group_selected += selected_count
                    total_misaligned += int((~aligned).sum())
                    total_invalid_features += int((~valid_features).sum())
                    total_outside_fv += int(outside_fv.sum())
                    total_mrd_overflow += int((selected & mrd_overflow).sum())
                    print(
                        f"Read {total_read}; selected {total_selected}",
                        end="\r",
                        flush=True,
                    )
                    entry_offset += n_entries
                chain_entry_offset += int(tree.num_entries)
        group_summaries.append(
            {
                "name": group_name,
                "pmt_response": response.mode,
                "sources": [str(path) for path in paths],
                "events_read": group_read,
                "events_selected": group_selected,
            }
        )
    print()
    if total_selected == 0:
        raise ValueError("No events passed image, feature, and FV requirements")

    detector_images = np.concatenate(detector_chunks)
    detector_chunks.clear()
    return {
        "detector_images": detector_images,
        "scalar_features": np.concatenate(scalar_chunks),
        "track_starts": np.concatenate(start_chunks),
        "track_stops": np.concatenate(stop_chunks),
        "track_properties": np.concatenate(property_chunks),
        "track_mask": np.concatenate(mask_chunks),
        "labels": np.concatenate(label_chunks),
        "scalar_feature_names": [
            *(name for name, _, _ in SELECTION_INPUT_SPECS),
            args.prompt_pe_branch,
            "numMRDTracks",
        ],
        "source_paths": np.concatenate(source_chunks),
        "entries": np.concatenate(entry_chunks),
        "geometry": geometry,
        "responses": responses,
        "groups": group_summaries,
        "tank_branches": sorted(tank_branches),
        "fv_branches": sorted(fv_branches),
        "events_read": total_read,
        "events_selected": total_selected,
        "misaligned_events": total_misaligned,
        "invalid_feature_events": total_invalid_features,
        "outside_fv_events": total_outside_fv,
        "mrd_overflow_events": total_mrd_overflow,
    }


def _channel_moments(
    values: np.ndarray, indices: np.ndarray, batch_size: int = 512
) -> Tuple[np.ndarray, np.ndarray]:
    channels = values.shape[-1]
    sums = np.zeros(channels, dtype=np.float64)
    squared_sums = np.zeros(channels, dtype=np.float64)
    count = 0
    for start in range(0, len(indices), batch_size):
        batch = values[indices[start : start + batch_size]].reshape(-1, channels)
        sums += np.sum(batch, axis=0, dtype=np.float64)
        squared_sums += np.sum(
            np.square(batch, dtype=np.float32), axis=0, dtype=np.float64
        )
        count += len(batch)
    means = sums / count
    variances = np.maximum(squared_sums / count - means * means, 1e-7)
    return means.astype(np.float32), variances.astype(np.float32)


class CCInclusiveSequence(tf.keras.utils.Sequence):
    def __init__(
        self,
        arrays: Dict[str, np.ndarray],
        indices: np.ndarray,
        batch_size: int,
        shuffle: bool,
        seed: int,
        class_weights: Optional[Dict[int, float]] = None,
    ):
        super().__init__()
        self.arrays = arrays
        self.indices = np.asarray(indices, dtype=np.int64)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.class_weights = class_weights
        self.order = np.arange(len(indices), dtype=np.int64)
        self.rng = np.random.default_rng(seed)
        if self.shuffle:
            self.rng.shuffle(self.order)

    def __len__(self) -> int:
        return int(np.ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, batch_index: int):
        start = batch_index * self.batch_size
        positions = self.order[start : start + self.batch_size]
        indices = self.indices[positions]
        inputs = {
            "pmt_unfolded_image": self.arrays["detector_images"][indices],
            "event_features": self.arrays["scalar_features"][indices],
            "mrd_track_starts": self.arrays["track_starts"][indices],
            "mrd_track_stops": self.arrays["track_stops"][indices],
            "mrd_track_properties": self.arrays["track_properties"][indices],
            "mrd_track_mask": self.arrays["track_mask"][indices],
        }
        targets = {"ccinclusive_score": self.arrays["labels"][indices]}
        if self.class_weights is None:
            return inputs, targets
        weights = np.where(
            targets["ccinclusive_score"] == 0,
            self.class_weights[0],
            self.class_weights[1],
        ).astype(np.float32)
        return inputs, targets, {"ccinclusive_score": weights}

    def on_epoch_end(self) -> None:
        if self.shuffle:
            self.rng.shuffle(self.order)


def _image_tower(inputs, normalizer, prefix: str):
    x = normalizer(inputs)
    x = tf.keras.layers.Conv2D(
        16, 3, padding="same", activation="relu", name=f"{prefix}_conv1"
    )(x)
    x = tf.keras.layers.MaxPooling2D(2, name=f"{prefix}_pool1")(x)
    x = tf.keras.layers.Conv2D(
        32, 3, padding="same", activation="relu", name=f"{prefix}_conv2"
    )(x)
    x = tf.keras.layers.MaxPooling2D(2, name=f"{prefix}_pool2")(x)
    x = tf.keras.layers.Conv2D(
        64, 3, padding="same", activation="relu", name=f"{prefix}_conv3"
    )(x)
    return tf.keras.layers.GlobalAveragePooling2D(name=f"{prefix}_features")(x)


def build_model(
    arrays: Dict[str, np.ndarray], train_indices: np.ndarray
) -> tf.keras.Model:
    detector = arrays["detector_images"]
    scalars = arrays["scalar_features"]
    starts = arrays["track_starts"]
    stops = arrays["track_stops"]
    properties = arrays["track_properties"]
    mask = arrays["track_mask"]
    detector_mean, detector_variance = _channel_moments(detector, train_indices)
    detector_normalizer = tf.keras.layers.Normalization(
        axis=-1,
        mean=detector_mean,
        variance=detector_variance,
        name="detector_tankcluster_normalization",
    )
    scalar_normalizer = tf.keras.layers.Normalization(
        axis=-1, name="event_feature_normalization"
    )
    start_normalizer = tf.keras.layers.Normalization(
        axis=-1, name="mrd_track_start_normalization"
    )
    stop_normalizer = tf.keras.layers.Normalization(
        axis=-1, name="mrd_track_stop_normalization"
    )
    property_normalizer = tf.keras.layers.Normalization(
        axis=-1, name="mrd_track_property_normalization"
    )
    scalar_normalizer.adapt(scalars[train_indices])
    valid_starts = starts[train_indices][mask[train_indices].astype(bool)]
    valid_stops = stops[train_indices][mask[train_indices].astype(bool)]
    valid_properties = properties[train_indices][mask[train_indices].astype(bool)]
    if len(valid_starts) == 0:
        raise ValueError("The training split contains no valid MRD tracks")
    start_normalizer.adapt(valid_starts)
    stop_normalizer.adapt(valid_stops)
    property_normalizer.adapt(valid_properties)

    detector_input = tf.keras.Input(
        shape=detector.shape[1:], name="pmt_unfolded_image"
    )
    scalar_input = tf.keras.Input(
        shape=(scalars.shape[1],), name="event_features"
    )
    start_input = tf.keras.Input(
        shape=(MAX_MRD_TRACKS, 3), name="mrd_track_starts"
    )
    stop_input = tf.keras.Input(
        shape=(MAX_MRD_TRACKS, 3), name="mrd_track_stops"
    )
    property_input = tf.keras.Input(
        shape=(MAX_MRD_TRACKS, 3), name="mrd_track_properties"
    )
    mask_input = tf.keras.Input(shape=(MAX_MRD_TRACKS,), name="mrd_track_mask")

    detector_features = _image_tower(
        detector_input, detector_normalizer, "detector"
    )
    scalar_features = scalar_normalizer(scalar_input)
    scalar_features = tf.keras.layers.Dense(
        16, activation="relu", name="event_feature_embedding"
    )(scalar_features)
    track_features = tf.keras.layers.Concatenate(axis=-1)(
        [
            start_normalizer(start_input),
            stop_normalizer(stop_input),
            property_normalizer(property_input),
        ]
    )
    track_features = tf.keras.layers.Dense(
        16, activation="relu", name="shared_mrd_track_embedding"
    )(track_features)
    expanded_mask = tf.keras.layers.Reshape((MAX_MRD_TRACKS, 1))(mask_input)
    track_features = tf.keras.layers.Multiply()([track_features, expanded_mask])
    track_features = tf.keras.layers.GlobalMaxPooling1D(
        name="mrd_track_set_features"
    )(track_features)
    combined = tf.keras.layers.Concatenate(name="combined_ccinclusive_inputs")(
        [detector_features, scalar_features, track_features]
    )
    combined = tf.keras.layers.Dense(64, activation="relu")(combined)
    combined = tf.keras.layers.Dropout(0.30)(combined)
    combined = tf.keras.layers.Dense(32, activation="relu")(combined)
    output = tf.keras.layers.Dense(
        1, activation="sigmoid", name="ccinclusive_score"
    )(combined)
    model = tf.keras.Model(
        [
            detector_input,
            scalar_input,
            start_input,
            stop_input,
            property_input,
            mask_input,
        ],
        {"ccinclusive_score": output},
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss={"ccinclusive_score": "binary_crossentropy"},
        metrics={
            "ccinclusive_score": [
                tf.keras.metrics.AUC(name="roc_auc"),
                tf.keras.metrics.AUC(name="pr_auc", curve="PR"),
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
            ]
        },
    )
    return model


def _operating_point(labels, scores, threshold: float) -> Dict[str, float]:
    predictions = scores >= threshold
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    efficiency = tp / (tp + fn) if tp + fn else 0.0
    purity = tp / (tp + fp) if tp + fp else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {
        "threshold": float(threshold),
        "efficiency": float(efficiency),
        "purity": float(purity),
        "efficiency_x_purity": float(efficiency * purity),
        "specificity": float(specificity),
        "balanced_accuracy": float((efficiency + specificity) / 2.0),
        "true_positive": int(tp),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_negative": int(tn),
    }


def evaluate_and_plot(
    model: tf.keras.Model,
    arrays: Dict[str, np.ndarray],
    test_indices: np.ndarray,
    history_path: Path,
    output: Path,
    threshold: float,
) -> Dict[str, object]:
    inputs = {
        "pmt_unfolded_image": arrays["detector_images"][test_indices],
        "event_features": arrays["scalar_features"][test_indices],
        "mrd_track_starts": arrays["track_starts"][test_indices],
        "mrd_track_stops": arrays["track_stops"][test_indices],
        "mrd_track_properties": arrays["track_properties"][test_indices],
        "mrd_track_mask": arrays["track_mask"][test_indices],
    }
    labels = arrays["labels"][test_indices].astype(np.int64)
    outputs = model.predict(inputs, batch_size=64, verbose=0)
    scores = np.asarray(outputs["ccinclusive_score"]).reshape(-1)
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = float(auc(fpr, tpr))
    precision, recall, _ = precision_recall_curve(labels, scores)
    average_precision = float(average_precision_score(labels, scores))
    brier = float(brier_score_loss(labels, scores))
    operating_point = _operating_point(labels, scores, threshold)

    fig, axis = plt.subplots(figsize=(6, 5))
    axis.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.4f}")
    axis.plot([0, 1], [0, 1], "--", color="gray")
    axis.set(xlabel="False-positive rate", ylabel="Signal efficiency")
    axis.legend()
    fig.tight_layout()
    roc_path = output.with_suffix(".roc_curve.png")
    fig.savefig(roc_path, dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(6, 5))
    axis.plot(recall, precision, label=f"Average precision = {average_precision:.4f}")
    axis.set(xlabel="Signal efficiency", ylabel="Signal purity")
    axis.legend()
    fig.tight_layout()
    pr_path = output.with_suffix(".precision_recall_curve.png")
    fig.savefig(pr_path, dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7, 5))
    bins = np.linspace(0.0, 1.0, 41)
    axis.hist(scores[labels == 0], bins=bins, density=True, histtype="step", label="background")
    axis.hist(scores[labels == 1], bins=bins, density=True, histtype="step", label="CC-inclusive signal")
    axis.axvline(threshold, linestyle="--", color="black", label=f"cut = {threshold:.2f}")
    axis.set(xlabel="CC-inclusive score", ylabel="Normalized events")
    axis.legend()
    fig.tight_layout()
    score_path = output.with_suffix(".score_distribution.png")
    fig.savefig(score_path, dpi=160)
    plt.close(fig)

    matrix = confusion_matrix(labels, scores >= threshold, labels=[0, 1])
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(
        matrix,
        row_sums,
        out=np.zeros_like(matrix, dtype=float),
        where=row_sums != 0,
    )
    matrix_paths = []
    for values, suffix, value_format in (
        (matrix, ".confusion_matrix.png", "d"),
        (normalized, ".confusion_matrix_normalized.png", ".3f"),
    ):
        fig, axis = plt.subplots(figsize=(5.5, 5))
        image = axis.imshow(values, cmap="Blues", vmin=0)
        fig.colorbar(image, ax=axis)
        for row in range(2):
            for column in range(2):
                axis.text(
                    column,
                    row,
                    format(values[row, column], value_format),
                    ha="center",
                    va="center",
                )
        axis.set_xticks([0, 1], ["background", "CC inclusive"])
        axis.set_yticks([0, 1], ["background", "CC inclusive"])
        axis.set(xlabel="Predicted", ylabel="Truth")
        fig.tight_layout()
        path = output.with_suffix(suffix)
        fig.savefig(path, dpi=160)
        plt.close(fig)
        matrix_paths.append(path.name)

    scan_path = output.with_suffix(".efficiency_purity.csv")
    scan_rows = []
    for scan_threshold in np.linspace(0.0, 1.0, 201):
        scan_rows.append(_operating_point(labels, scores, float(scan_threshold)))
    with scan_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(scan_rows[0]))
        writer.writeheader()
        writer.writerows(scan_rows)
    fig, axis = plt.subplots(figsize=(7, 5))
    thresholds = [row["threshold"] for row in scan_rows]
    axis.plot(thresholds, [row["efficiency"] for row in scan_rows], label="efficiency")
    axis.plot(thresholds, [row["purity"] for row in scan_rows], label="purity")
    axis.plot(
        thresholds,
        [row["efficiency_x_purity"] for row in scan_rows],
        label="efficiency × purity",
    )
    axis.axvline(threshold, linestyle="--", color="black")
    axis.set(xlabel="CC-inclusive threshold", ylabel="Metric", ylim=(0, 1.05))
    axis.legend()
    fig.tight_layout()
    scan_plot_path = output.with_suffix(".efficiency_purity.png")
    fig.savefig(scan_plot_path, dpi=160)
    plt.close(fig)

    predictions_path = output.with_suffix(".test_predictions.csv")
    with predictions_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "source_file",
                "tree_entry",
                "true_ccinclusive",
                *arrays["scalar_feature_names"],
                "ccinclusive_score",
                "threshold",
                "prediction",
            ]
        )
        scalar_values = arrays["scalar_features"][test_indices]
        for row, event_index in enumerate(test_indices):
            writer.writerow(
                [
                    arrays["source_paths"][event_index],
                    int(arrays["entries"][event_index]),
                    int(labels[row]),
                    *scalar_values[row].astype(float),
                    float(scores[row]),
                    float(threshold),
                    int(scores[row] >= threshold),
                ]
            )

    history_plot = ""
    if history_path.is_file():
        with history_path.open(newline="", encoding="utf-8") as handle:
            history_rows = list(csv.DictReader(handle))
        if history_rows:
            fig, axes = plt.subplots(1, 2, figsize=(11, 4))
            epochs = np.arange(1, len(history_rows) + 1)
            axes[0].plot(epochs, [float(row["loss"]) for row in history_rows], label="train")
            axes[0].plot(epochs, [float(row["val_loss"]) for row in history_rows], label="validation")
            axes[0].set(xlabel="Epoch", ylabel="Loss")
            axes[0].legend()
            metric_key = "ccinclusive_score_pr_auc"
            validation_key = f"val_{metric_key}"
            if metric_key not in history_rows[0]:
                metric_key = "pr_auc"
                validation_key = "val_pr_auc"
            axes[1].plot(epochs, [float(row[metric_key]) for row in history_rows], label="train")
            axes[1].plot(epochs, [float(row[validation_key]) for row in history_rows], label="validation")
            axes[1].set(xlabel="Epoch", ylabel="PR AUC")
            axes[1].legend()
            fig.tight_layout()
            history_plot_path = output.with_suffix(".training_curves.png")
            fig.savefig(history_plot_path, dpi=160)
            plt.close(fig)
            history_plot = history_plot_path.name

    return {
        "roc_auc": roc_auc,
        "average_precision": average_precision,
        "brier_score": brier,
        "operating_point": operating_point,
        "roc_curve": roc_path.name,
        "precision_recall_curve": pr_path.name,
        "score_distribution": score_path.name,
        "confusion_matrix": matrix_paths[0],
        "confusion_matrix_normalized": matrix_paths[1],
        "efficiency_purity_csv": scan_path.name,
        "efficiency_purity_plot": scan_plot_path.name,
        "test_predictions": predictions_path.name,
        "training_curves": history_plot,
    }


def main() -> None:
    args = parse_args()
    if args.detector_height < 24 or args.detector_width < 16:
        raise ValueError("The unfolded image must be at least 24 x 16 bins")
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    arrays = load_data(args)
    labels = arrays["labels"]
    class_counts = np.bincount(labels.astype(np.int64), minlength=2)
    if np.any(class_counts == 0):
        raise ValueError(
            "Both CC-inclusive target classes are required after the FV cut; "
            f"counts are {class_counts.tolist()}"
        )
    indices = np.arange(len(labels))
    train_indices, temporary_indices = train_test_split(
        indices,
        test_size=VALIDATION_FRACTION + TEST_FRACTION,
        random_state=args.seed,
        stratify=labels,
    )
    validation_indices, test_indices = train_test_split(
        temporary_indices,
        test_size=TEST_FRACTION / (VALIDATION_FRACTION + TEST_FRACTION),
        random_state=args.seed,
        stratify=labels[temporary_indices],
    )
    train_counts = np.bincount(labels[train_indices].astype(np.int64), minlength=2)
    class_weights = {
        0: len(train_indices) / (2.0 * train_counts[0]),
        1: len(train_indices) / (2.0 * train_counts[1]),
    }
    model = build_model(arrays, train_indices)
    train_sequence = CCInclusiveSequence(
        arrays,
        train_indices,
        args.batch_size,
        True,
        args.seed,
        class_weights,
    )
    validation_sequence = CCInclusiveSequence(
        arrays, validation_indices, args.batch_size, False, args.seed
    )
    test_sequence = CCInclusiveSequence(
        arrays, test_indices, args.batch_size, False, args.seed
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    history_path = args.output.with_suffix(".history.csv")
    model.fit(
        train_sequence,
        validation_data=validation_sequence,
        epochs=args.epochs,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_pr_auc",
                mode="max",
                patience=12,
                restore_best_weights=True,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_pr_auc",
                mode="max",
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1,
            ),
            tf.keras.callbacks.CSVLogger(str(history_path)),
        ],
        verbose=2,
    )
    test_metrics = model.evaluate(test_sequence, return_dict=True, verbose=0)
    print("Test metrics:")
    for name, value in test_metrics.items():
        print(f"  {name}: {value:.5f}")
    model.save(args.output)
    evaluation = evaluate_and_plot(
        model, arrays, test_indices, history_path, args.output, args.threshold
    )

    responses = arrays["responses"]
    response_modes = sorted(responses)
    training_response = response_modes[0] if len(response_modes) == 1 else "mixed"
    metadata_response = responses.get("tuned", responses.get("raw"))
    metadata = {
        "model_type": "annie_ccinclusive_detector_tankcluster_cnn",
        "tensorflow_version": tf.__version__,
        "tree": args.tree,
        "training_sources": [
            source for group in arrays["groups"] for source in group["sources"]
        ],
        "training_source_groups": arrays["groups"],
        "target": "trueCC == 1 && trueFSLPdg == 13",
        "truth_branches": list(TRUTH_BRANCHES),
        "truth_used_as_input": False,
        "vertex_information_used_as_input": False,
        "threshold": args.threshold,
        "fv_cut_applied": not args.no_fv_cut,
        "fv_cut_branches": arrays["fv_branches"],
        "fv_cut_expression": (
            "resolved_reconstructed_fv_branch != 0"
            if not args.no_fv_cut
            else "none"
        ),
        "tankcluster_branches": arrays["tank_branches"],
        "tankcluster_id_branch": args.tankcluster_id_branch,
        "image_channels": ["log1p_hitPE_tankcluster"],
        "image_projections": [
            "annotation_free_unfolded_barrel_top_bottom",
        ],
        "detector_height": args.detector_height,
        "detector_width": args.detector_width,
        "image_storage_dtype": str(arrays["detector_images"].dtype),
        "scalar_input_name": "event_features",
        "scalar_feature_names": [
            *(name for name, _, _ in SELECTION_INPUT_SPECS),
            args.prompt_pe_branch,
            "numMRDTracks",
        ],
        "scalar_feature_definitions": {
            name: f"{branch} == {required_value}"
            for name, branch, required_value in SELECTION_INPUT_SPECS
        },
        "prompt_pe_branch": args.prompt_pe_branch,
        "selection_flags_are_inputs_not_cuts": True,
        "mrd_track_start_branches": MRD_TRACK_START_BRANCHES,
        "mrd_track_stop_branches": list(MRD_TRACK_STOP_BRANCHES),
        "mrd_track_property_branches": MRD_TRACK_PROPERTY_BRANCHES,
        "max_mrd_tracks": MAX_MRD_TRACKS,
        "pmt_mask": args.pmt_mask,
        "excluded_pmt_ids": arrays["geometry"].excluded_ids,
        "training_pmt_response": training_response,
        "pmt_response_calibration_file": (
            metadata_response.calibration_path.name
            if metadata_response.calibration_path
            else None
        ),
        "pmt_response_calibration_sha256": metadata_response.calibration_sha256,
        "pmt_response_payload_format": metadata_response.payload_format,
        "pmt_tune_variant": metadata_response.tune_variant,
        "batch_size": args.batch_size,
        "root_chunk_size": args.chunk_size,
        "data_split": {
            "training_fraction": TRAIN_FRACTION,
            "validation_fraction": VALIDATION_FRACTION,
            "testing_fraction": TEST_FRACTION,
            "training_events": int(len(train_indices)),
            "validation_events": int(len(validation_indices)),
            "testing_events": int(len(test_indices)),
            "stratified_by_target": True,
            "random_seed": args.seed,
        },
        "class_counts": {
            "background": int(class_counts[0]),
            "ccinclusive_signal": int(class_counts[1]),
        },
        "events_read": arrays["events_read"],
        "events_selected": arrays["events_selected"],
        "outside_fv_events_rejected": arrays["outside_fv_events"],
        "misaligned_events_rejected": arrays["misaligned_events"],
        "invalid_feature_events_rejected": arrays["invalid_feature_events"],
        "mrd_track_overflow_events": arrays["mrd_overflow_events"],
        "test_metrics": {name: float(value) for name, value in test_metrics.items()},
        "evaluation": evaluation,
        "training_history": history_path.name,
    }
    metadata_path = args.output.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {args.output} and {metadata_path}")
    print(
        f"Held-out ROC AUC={evaluation['roc_auc']:.5f}; "
        f"average precision={evaluation['average_precision']:.5f}"
    )


if __name__ == "__main__":
    main()
