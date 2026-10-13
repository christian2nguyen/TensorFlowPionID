#!/usr/bin/env python3
"""Train ANNIE CC-inclusive and FV classifiers from detector and MRD inputs."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import gc
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import awkward as ak
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
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
CCINCLUSIVE_SIGNAL_BRANCH = "sel_nu_mu_cc"
SUPERVISION_BRANCHES = (CCINCLUSIVE_SIGNAL_BRANCH, "trueFV")
FV_BRANCH_ALIASES = ("simpleRecoFV", "recoFV", "sel_vertex_in_FV")
TRAINING_SELECTION_SPECS = (
    ("sel_clusterHist_tankcluster_branch", "==", 1),
    ("match_found", "==", 1),
    ("sel_simpleRecoFlag", "==", 1),
    ("trigword", "==", 5),
)
TRAINING_SELECTION_EXPRESSION = (
    "sel_clusterHist_tankcluster_branch == true && match_found == true && "
    "sel_simpleRecoFlag == true && trigword == 5"
)
SELECTION_INPUT_SPECS = (
    ("sel_trigword", "trigword", 5),
    ("sel_HasTank", "HasTank", 1),
    ("sel_HasMRD", "HasMRD", 1),
    ("sel_TankMRDCoinc", "TankMRDCoinc", 1),
    ("sel_noVeto", "NoVeto", 1),
    ("sel_simpleRecoFlag", "sel_simpleRecoFlag", 1),
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
            "reconstructed FV branch used only with --apply-reco-fv-cut; "
            "otherwise no reconstructed FV branch is read"
        ),
    )
    fv_cut_group = parser.add_mutually_exclusive_group()
    fv_cut_group.add_argument(
        "--apply-reco-fv-cut",
        action="store_true",
        help=(
            "optionally require a nonzero reconstructed FV flag; disabled by "
            "default so the FV head sees both target classes"
        ),
    )
    fv_cut_group.add_argument(
        "--no-fv-cut",
        dest="apply_reco_fv_cut",
        action="store_false",
        help="explicit compatibility spelling for the default no-FV-cut behavior",
    )
    parser.set_defaults(apply_reco_fv_cut=False)
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
    parser.add_argument(
        "--diagnostic-batch-size",
        type=int,
        default=32,
        help="streaming batch size for post-training predictions (default: 32)",
    )
    parser.add_argument(
        "--input-importance-repeats",
        "--mrd-branch-importance-repeats",
        dest="input_importance_repeats",
        type=int,
        default=1,
        help=(
            "number of held-out permutations per input for the complete "
            "CC/FV feature-impact ranking (default: 1)"
        ),
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--fv-threshold",
        type=float,
        default=0.5,
        help="decision threshold for the true-FV output (default: 0.5)",
    )
    parser.add_argument(
        "--fv-loss-weight",
        type=float,
        default=1.0,
        help="relative training-loss weight for fv_score (default: 1.0)",
    )
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
    if not 0.0 <= args.fv_threshold <= 1.0:
        parser.error("--fv-threshold must be between 0 and 1")
    if args.fv_loss_weight <= 0.0:
        parser.error("--fv-loss-weight must be positive")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.diagnostic_batch_size < 1:
        parser.error("--diagnostic-batch-size must be positive")
    if args.input_importance_repeats < 1:
        parser.error("--input-importance-repeats must be positive")
    if args.fv_branch and not args.apply_reco_fv_cut:
        parser.error("--fv-branch requires --apply-reco-fv-cut")
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


def _build_training_selection(arrays) -> Tuple[np.ndarray, Dict[str, int]]:
    """Build the requested hard event-selection mask and failure counts."""
    selected = np.ones(len(arrays), dtype=bool)
    failed_counts = {}
    for branch, operator, required_value in TRAINING_SELECTION_SPECS:
        values = _to_numpy(arrays[branch])
        if values.ndim != 1:
            raise ValueError(f"Selection branch {branch!r} is not event-level")
        finite = np.isfinite(values)
        if operator == "==":
            passed = finite & (values == required_value)
        else:
            passed = finite & (values != required_value)
        failed_counts[branch] = int((~passed).sum())
        selected &= passed
    return selected, failed_counts


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
    fv_label_chunks = []
    source_chunks = []
    entry_chunks = []
    fv_branches = set()
    tank_branches = set()
    group_summaries = []
    total_read = 0
    total_selected = 0
    total_misaligned = 0
    total_invalid_features = 0
    total_training_cut_rejected = 0
    training_cut_failed_counts = Counter()
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
                if args.apply_reco_fv_cut:
                    fv_branch = _resolve_fv_branch(available, args.fv_branch)
                    fv_branches.add(fv_branch)
                tank_branches.add(tank_branch)
                requested = [
                    args.hitpe_branch,
                    args.hitid_branch,
                    tank_branch,
                    args.tankcluster_id_branch,
                    *(branch for branch, _, _ in TRAINING_SELECTION_SPECS),
                    *(branch for _, branch, _ in SELECTION_INPUT_SPECS),
                    args.prompt_pe_branch,
                    "numMRDTracks",
                    *MRD_TRACK_START_BRANCHES,
                    *MRD_TRACK_STOP_BRANCHES,
                    *MRD_TRACK_PROPERTY_BRANCHES,
                    *SUPERVISION_BRANCHES,
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
                        tankcluster_only=False,
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
                    training_selection, cut_failed_counts = (
                        _build_training_selection(arrays)
                    )
                    aligned = detector_aligned
                    valid_features = valid_mrd & valid_stops & valid_scalars
                    selected = aligned & valid_features & training_selection
                    outside_fv = np.zeros(n_entries, dtype=bool)
                    if fv_branch:
                        fv_values = _to_numpy(arrays[fv_branch])
                        valid_fv = np.isfinite(fv_values)
                        inside_fv = valid_fv & (fv_values != 0)
                        outside_fv = valid_fv & ~inside_fv
                        selected &= inside_fv
                    ccinclusive_signal = _to_numpy(
                        arrays[CCINCLUSIVE_SIGNAL_BRANCH]
                    )
                    true_fv = _to_numpy(arrays["trueFV"])
                    valid_truth = (
                        np.isfinite(ccinclusive_signal) & np.isfinite(true_fv)
                    )
                    selected &= valid_truth
                    labels = (ccinclusive_signal == 1).astype(np.float32)
                    fv_labels = (true_fv != 0).astype(np.float32)

                    detector_chunks.append(
                        detector[selected].astype(np.float16, copy=False)
                    )
                    scalar_chunks.append(scalar_features[selected])
                    start_chunks.append(track_starts[selected])
                    stop_chunks.append(track_stops[selected])
                    property_chunks.append(track_properties[selected])
                    mask_chunks.append(track_mask[selected])
                    label_chunks.append(labels[selected])
                    fv_label_chunks.append(fv_labels[selected])
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
                    total_training_cut_rejected += int((~training_selection).sum())
                    training_cut_failed_counts.update(cut_failed_counts)
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
    print(f"Training selection: {TRAINING_SELECTION_EXPRESSION}")
    print(
        f"Rejected by at least one training cut: "
        f"{total_training_cut_rejected} / {total_read}"
    )
    for branch, failed_count in training_cut_failed_counts.items():
        print(f"  failed {branch}: {failed_count}")
    if total_selected == 0:
        raise ValueError(
            "No events passed the requested event selection, image, feature, "
            "supervision-label, and optional FV requirements"
        )

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
        "fv_labels": np.concatenate(fv_label_chunks),
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
        "training_cut_rejected_events": total_training_cut_rejected,
        "training_cut_failed_counts": dict(training_cut_failed_counts),
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
        class_weights: Optional[Dict[str, Dict[int, float]]] = None,
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
        targets = {
            "ccinclusive_score": self.arrays["labels"][indices],
            "fv_score": self.arrays["fv_labels"][indices],
        }
        if self.class_weights is None:
            return inputs, targets
        sample_weights = {}
        for output_name, output_targets in targets.items():
            weights = self.class_weights[output_name]
            sample_weights[output_name] = np.where(
                output_targets == 0, weights[0], weights[1]
            ).astype(np.float32)
        return inputs, targets, sample_weights

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
    arrays: Dict[str, np.ndarray],
    train_indices: np.ndarray,
    fv_loss_weight: float,
) -> tf.keras.Model:
    detector = arrays["detector_images"]
    scalars = arrays["scalar_features"]
    starts = arrays["track_starts"]
    stops = arrays["track_stops"]
    properties = arrays["track_properties"]
    mask = arrays["track_mask"]
    detector_mean, detector_variance = _channel_moments(detector, train_indices)
    if detector.shape[-1] != 2:
        raise ValueError(
            "CC-inclusive training expects unfolded image channels in "
            "[hitPE, hitPE_tankcluster] order"
        )
    fv_detector_normalizer = tf.keras.layers.Normalization(
        axis=-1,
        mean=detector_mean,
        variance=detector_variance,
        name="fv_detector_both_channel_normalization",
    )
    cc_detector_normalizer = tf.keras.layers.Normalization(
        axis=-1,
        mean=detector_mean[1:2],
        variance=detector_variance[1:2],
        name="cc_detector_tankcluster_normalization",
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

    cc_detector_input = detector_input[..., 1:2]
    cc_detector_features = _image_tower(
        cc_detector_input, cc_detector_normalizer, "cc_detector"
    )
    fv_detector_features = _image_tower(
        detector_input, fv_detector_normalizer, "fv_detector"
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
    cc_combined = tf.keras.layers.Concatenate(name="combined_ccinclusive_inputs")(
        [cc_detector_features, scalar_features, track_features]
    )
    cc_combined = tf.keras.layers.Dense(
        64, activation="relu", name="ccinclusive_dense"
    )(cc_combined)
    cc_combined = tf.keras.layers.Dropout(
        0.30, name="ccinclusive_dropout"
    )(cc_combined)
    cc_features = tf.keras.layers.Dense(
        32, activation="relu", name="ccinclusive_features"
    )(cc_combined)
    fv_combined = tf.keras.layers.Concatenate(name="combined_fv_inputs")(
        [fv_detector_features, scalar_features, track_features]
    )
    fv_combined = tf.keras.layers.Dense(64, activation="relu", name="fv_dense")(
        fv_combined
    )
    fv_combined = tf.keras.layers.Dropout(0.30, name="fv_dropout")(fv_combined)
    fv_features = tf.keras.layers.Dense(32, activation="relu", name="fv_features")(
        fv_combined
    )
    cc_output = tf.keras.layers.Dense(
        1, activation="sigmoid", name="ccinclusive_score"
    )(cc_features)
    fv_output = tf.keras.layers.Dense(1, activation="sigmoid", name="fv_score")(
        fv_features
    )
    model = tf.keras.Model(
        [
            detector_input,
            scalar_input,
            start_input,
            stop_input,
            property_input,
            mask_input,
        ],
        {"ccinclusive_score": cc_output, "fv_score": fv_output},
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss={
            "ccinclusive_score": "binary_crossentropy",
            "fv_score": "binary_crossentropy",
        },
        loss_weights={"ccinclusive_score": 1.0, "fv_score": fv_loss_weight},
        metrics={
            "ccinclusive_score": [
                tf.keras.metrics.AUC(name="roc_auc"),
                tf.keras.metrics.AUC(name="pr_auc", curve="PR"),
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
            ],
            "fv_score": [
                tf.keras.metrics.AUC(name="roc_auc"),
                tf.keras.metrics.AUC(name="pr_auc", curve="PR"),
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
            ],
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


def _model_inputs(
    arrays: Dict[str, np.ndarray], indices: np.ndarray
) -> Dict[str, np.ndarray]:
    return {
        "pmt_unfolded_image": arrays["detector_images"][indices],
        "event_features": arrays["scalar_features"][indices],
        "mrd_track_starts": arrays["track_starts"][indices],
        "mrd_track_stops": arrays["track_stops"][indices],
        "mrd_track_properties": arrays["track_properties"][indices],
        "mrd_track_mask": arrays["track_mask"][indices],
    }


def _predict_scores_in_batches(
    model: tf.keras.Model,
    arrays: Dict[str, np.ndarray],
    indices: np.ndarray,
    batch_size: int,
    override_input: Optional[str] = None,
    override_column: Optional[int] = None,
    override_values: Optional[np.ndarray] = None,
    override_source_indices: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Predict without materializing the complete held-out image tensor."""
    cc_scores = np.empty(len(indices), dtype=np.float32)
    fv_scores = np.empty(len(indices), dtype=np.float32)
    for start in range(0, len(indices), batch_size):
        stop = min(start + batch_size, len(indices))
        batch_indices = indices[start:stop]
        inputs = _model_inputs(arrays, batch_indices)
        if override_input is not None:
            changed = np.array(inputs[override_input], copy=True)
            if override_input == "pmt_unfolded_image":
                source_indices = override_source_indices[start:stop]
                changed[:, :, :, override_column] = arrays["detector_images"][
                    source_indices, :, :, override_column
                ]
            elif override_input == "event_features":
                changed[:, override_column] = override_values[start:stop]
            else:
                changed[:, :, override_column] = override_values[start:stop]
            inputs[override_input] = changed
        outputs = model(inputs, training=False)
        cc_scores[start:stop] = np.asarray(
            outputs["ccinclusive_score"]
        ).reshape(-1)
        fv_scores[start:stop] = np.asarray(outputs["fv_score"]).reshape(-1)
    return cc_scores, fv_scores


def _format_event_details(arrays: Dict[str, np.ndarray], event_index: int) -> str:
    scalar_values = arrays["scalar_features"][event_index]
    scalar_parts = [
        f"{name}={float(value):.4g}"
        for name, value in zip(arrays["scalar_feature_names"], scalar_values)
    ]
    mask = arrays["track_mask"][event_index].astype(bool)
    starts = arrays["track_starts"][event_index][mask]
    stops = arrays["track_stops"][event_index][mask]
    properties = arrays["track_properties"][event_index][mask]
    track_lines = []
    for track_index, (start, stop, props) in enumerate(
        zip(starts, stops, properties)
    ):
        track_lines.append(
            f"track {track_index}: start=({start[0]:.3g}, {start[1]:.3g}, "
            f"{start[2]:.3g}), stop=({stop[0]:.3g}, {stop[1]:.3g}, "
            f"{stop[2]:.3g}), dE={props[0]:.3g}, L={props[1]:.3g}, "
            f"angle={props[2]:.3g}"
        )
    if not track_lines:
        track_lines.append("MRD tracks: none")
    return "Scalar inputs: " + ", ".join(scalar_parts) + "\n" + "\n".join(
        track_lines
    )


def _misclassified_examples(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    maximum: int,
) -> List[Tuple[str, int]]:
    false_positives = np.flatnonzero((labels == 0) & (scores >= threshold))
    false_negatives = np.flatnonzero((labels == 1) & (scores < threshold))
    false_positives = false_positives[np.argsort(scores[false_positives])[::-1]]
    false_negatives = false_negatives[np.argsort(scores[false_negatives])]
    per_class = maximum // 2
    selected = [
        ("False positive", int(index)) for index in false_positives[:per_class]
    ]
    selected.extend(
        ("False negative", int(index))
        for index in false_negatives[:per_class]
    )
    remaining = [
        ("False positive", int(index))
        for index in false_positives[per_class:]
    ]
    remaining.extend(
        ("False negative", int(index))
        for index in false_negatives[per_class:]
    )
    remaining.sort(key=lambda item: abs(float(scores[item[1]]) - threshold), reverse=True)
    selected.extend(remaining[: maximum - len(selected)])
    return selected


def plot_misclassified_events(
    arrays: Dict[str, np.ndarray],
    test_indices: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    fv_labels: np.ndarray,
    fv_scores: np.ndarray,
    output: Path,
    threshold: float,
    fv_threshold: float,
) -> Dict[str, object]:
    """Write a compact gallery and a PDF containing up to 20 mistakes."""
    selected = _misclassified_examples(labels, scores, threshold, 20)
    gallery_selected = _misclassified_examples(labels, scores, threshold, 6)
    gallery_path = output.with_suffix(".misclassified_gallery.png")
    if not gallery_selected:
        fig, axis = plt.subplots(figsize=(8, 4))
        axis.axis("off")
        axis.text(
            0.5,
            0.5,
            f"No test events were misclassified at threshold {threshold:.3f}.",
            ha="center",
            va="center",
        )
    else:
        fig, axes = plt.subplots(
            len(gallery_selected),
            2,
            figsize=(12, 3.4 * len(gallery_selected)),
            squeeze=False,
            gridspec_kw={"width_ratios": [1.1, 1.9]},
        )
        for row, (category, local_index) in enumerate(gallery_selected):
            event_index = int(test_indices[local_index])
            image = arrays["detector_images"][event_index, :, :, 1]
            axes[row, 0].imshow(image, origin="lower", aspect="equal", cmap="magma")
            axes[row, 0].set_title(
                f"{category}: signal={int(labels[local_index])}, "
                f"CC score={scores[local_index]:.3f}, "
                f"trueFV={int(fv_labels[local_index])}, "
                f"FV score={fv_scores[local_index]:.3f}",
                fontsize=9,
            )
            axes[row, 0].set_axis_off()
            axes[row, 1].axis("off")
            source = Path(str(arrays["source_paths"][event_index])).name
            details = _format_event_details(arrays, event_index)
            axes[row, 1].text(
                0.0,
                1.0,
                f"{source}\nentry={int(arrays['entries'][event_index])}\n{details}",
                ha="left",
                va="top",
                fontsize=8,
                wrap=True,
            )
    fig.suptitle("CC-inclusive held-out misclassifications", fontsize=12)
    fig.tight_layout()
    fig.savefig(gallery_path, dpi=160)
    plt.close(fig)

    pdf_path = output.with_suffix(".misclassified_events.pdf")
    with PdfPages(
        pdf_path,
        metadata={
            "Title": "ANNIE CC-inclusive misclassified events",
            "Subject": "Held-out false-positive and false-negative events",
            "Author": "train_annie_CCinclusive.py",
        },
    ) as pdf:
        if not selected:
            fig, axis = plt.subplots(figsize=(11, 6))
            axis.axis("off")
            axis.text(
                0.5,
                0.5,
                f"No test events were misclassified at threshold {threshold:.3f}.",
                ha="center",
                va="center",
                fontsize=14,
            )
            pdf.savefig(fig)
            plt.close(fig)
        for category, local_index in selected:
            event_index = int(test_indices[local_index])
            source = Path(str(arrays["source_paths"][event_index])).name
            fig = plt.figure(figsize=(11, 8.5))
            full_axis = fig.add_axes([0.05, 0.32, 0.42, 0.58])
            full_axis.imshow(
                arrays["detector_images"][event_index, :, :, 0],
                origin="lower",
                aspect="equal",
                cmap="magma",
            )
            full_axis.set_title("Full-event hitPE")
            tank_axis = fig.add_axes([0.53, 0.32, 0.42, 0.58])
            tank_axis.imshow(
                arrays["detector_images"][event_index, :, :, 1],
                origin="lower",
                aspect="equal",
                cmap="magma",
            )
            tank_axis.set_title("Tank-cluster hitPE")
            text_axis = fig.add_axes([0.06, 0.04, 0.88, 0.23])
            text_axis.axis("off")
            text_axis.text(
                0.0,
                1.0,
                (
                    f"{category} | signal={int(labels[local_index])} | "
                    f"CC score={scores[local_index]:.5f} | CC threshold={threshold:.3f}\n"
                    f"FV score={fv_scores[local_index]:.5f} | "
                    f"FV threshold={fv_threshold:.3f} | "
                    f"trueFV={int(fv_labels[local_index])}\n"
                    f"{source} | entry={int(arrays['entries'][event_index])}\n"
                    f"{_format_event_details(arrays, event_index)}"
                ),
                ha="left",
                va="top",
                fontsize=9,
                wrap=True,
            )
            fig.suptitle("ANNIE CC-inclusive misclassification review", fontsize=14)
            pdf.savefig(fig)
            plt.close(fig)
    return {
        "misclassified_gallery": gallery_path.name,
        "misclassified_events_pdf": pdf_path.name,
        "misclassified_events_shown": int(len(selected)),
    }


def _grad_cam_heatmaps(
    model: tf.keras.Model,
    conv_layer_name: str,
    output_layer_name: str,
    inputs: Dict[str, np.ndarray],
    target_shape: Tuple[int, int],
) -> np.ndarray:
    grad_model = tf.keras.Model(
        model.inputs,
        [
            model.get_layer(conv_layer_name).output,
            model.get_layer(output_layer_name).output,
        ],
    )
    tensors = {name: tf.convert_to_tensor(value) for name, value in inputs.items()}
    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(tensors, training=False)
        target = predictions[:, 0]
    gradients = tape.gradient(target, conv_output)
    pooled_gradients = tf.reduce_mean(gradients, axis=(1, 2))
    heatmaps = tf.nn.relu(tf.einsum("bhwc,bc->bhw", conv_output, pooled_gradients))
    heatmaps = tf.image.resize(heatmaps[..., tf.newaxis], target_shape)[..., 0]
    heatmaps = heatmaps.numpy()
    peaks = heatmaps.reshape(len(heatmaps), -1).max(axis=1)
    peaks[peaks == 0.0] = 1.0
    return heatmaps / peaks[:, None, None]


def plot_gradcam_examples(
    model: tf.keras.Model,
    arrays: Dict[str, np.ndarray],
    test_indices: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    fv_labels: np.ndarray,
    fv_scores: np.ndarray,
    output: Path,
    threshold: float,
) -> str:
    predictions = (scores >= threshold).astype(np.int64)
    categories = (
        ("True positive", (labels == 1) & (predictions == 1)),
        ("True negative", (labels == 0) & (predictions == 0)),
        ("False positive", (labels == 0) & (predictions == 1)),
        ("False negative", (labels == 1) & (predictions == 0)),
    )
    selected = []
    for title, category_mask in categories:
        candidates = np.flatnonzero(category_mask)
        if len(candidates):
            confidence = np.abs(scores[candidates] - threshold)
            selected.append((title, int(candidates[np.argmax(confidence)])))
    selected_indices = np.asarray(
        [test_indices[local_index] for _, local_index in selected], dtype=np.int64
    )
    inputs = _model_inputs(arrays, selected_indices)
    detector_batch = inputs["pmt_unfolded_image"]
    cc_heatmaps = _grad_cam_heatmaps(
        model,
        "cc_detector_conv3",
        "ccinclusive_score",
        inputs,
        (detector_batch.shape[1], detector_batch.shape[2]),
    )
    fv_heatmaps = _grad_cam_heatmaps(
        model,
        "fv_detector_conv3",
        "fv_score",
        inputs,
        (detector_batch.shape[1], detector_batch.shape[2]),
    )
    fig, axes = plt.subplots(
        len(selected), 4, figsize=(17, 3.6 * len(selected)), squeeze=False
    )
    for row, (title, local_index) in enumerate(selected):
        full_visible = detector_batch[row, :, :, 0]
        tank_visible = detector_batch[row, :, :, 1]
        axes[row, 0].imshow(
            full_visible, origin="lower", aspect="equal", cmap="magma"
        )
        axes[row, 0].set_title(
            f"{title}: signal={int(labels[local_index])}, "
            f"CC score={scores[local_index]:.3f}, "
            f"trueFV={int(fv_labels[local_index])}, "
            f"FV score={fv_scores[local_index]:.3f}",
            fontsize=9,
        )
        axes[row, 1].imshow(
            tank_visible, origin="lower", aspect="equal", cmap="magma"
        )
        axes[row, 1].set_title("Tank-cluster channel", fontsize=9)
        axes[row, 2].imshow(
            tank_visible, origin="lower", aspect="equal", cmap="gray"
        )
        axes[row, 2].imshow(
            cc_heatmaps[row],
            origin="lower",
            aspect="equal",
            cmap="jet",
            alpha=0.55,
        )
        axes[row, 2].set_title("CC Grad-CAM (tank-cluster tower)", fontsize=9)
        axes[row, 3].imshow(
            full_visible, origin="lower", aspect="equal", cmap="gray"
        )
        axes[row, 3].imshow(
            fv_heatmaps[row],
            origin="lower",
            aspect="equal",
            cmap="jet",
            alpha=0.55,
        )
        axes[row, 3].set_title("FV Grad-CAM (two-channel tower)", fontsize=9)
        for axis in axes[row]:
            axis.set_axis_off()
    fig.suptitle(
        "CC-inclusive and FV detector-image Grad-CAM (red = strongest influence)",
        fontsize=12,
    )
    fig.tight_layout()
    path = output.with_suffix(".gradcam_examples.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path.name


def _masked_event_mean(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    valid = mask.astype(bool) & np.isfinite(values)
    counts = valid.sum(axis=1)
    sums = np.sum(np.where(valid, values, 0.0), axis=1)
    return np.divide(
        sums,
        counts,
        out=np.full(len(values), np.nan, dtype=np.float64),
        where=counts > 0,
    )


def _permute_valid_track_values(
    values: np.ndarray, mask: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    shuffled = np.array(values, copy=True)
    for track_index in range(values.shape[1]):
        valid_indices = np.flatnonzero(mask[:, track_index] != 0)
        if len(valid_indices) > 1:
            shuffled[valid_indices, track_index] = values[
                rng.permutation(valid_indices), track_index
            ]
    return shuffled


def _image_channel_event_sums(
    arrays: Dict[str, np.ndarray],
    indices: np.ndarray,
    channel: int,
    batch_size: int,
) -> np.ndarray:
    sums = np.empty(len(indices), dtype=np.float64)
    for start in range(0, len(indices), batch_size):
        stop = min(start + batch_size, len(indices))
        batch = arrays["detector_images"][indices[start:stop], :, :, channel]
        sums[start:stop] = np.sum(batch, axis=(1, 2), dtype=np.float64)
    return sums


def evaluate_input_importance(
    model: tf.keras.Model,
    arrays: Dict[str, np.ndarray],
    test_indices: np.ndarray,
    cc_labels: np.ndarray,
    fv_labels: np.ndarray,
    full_cc_scores: np.ndarray,
    full_fv_scores: np.ndarray,
    output: Path,
    seed: int,
    repeats: int,
    batch_size: int,
) -> Dict[str, object]:
    """Rank every inference input for both outputs using held-out permutations."""
    array_name_by_model_input = {
        "mrd_track_starts": "track_starts",
        "mrd_track_stops": "track_stops",
        "mrd_track_properties": "track_properties",
    }
    scalar_names = list(arrays["scalar_feature_names"])
    branch_specs = [
        ("image: hitPE", "pmt_unfolded_image", 0),
        ("image: hitPE_tankcluster", "pmt_unfolded_image", 1),
        *[
            (name, "event_features", column)
            for column, name in enumerate(scalar_names)
        ],
        *[
            (name, "mrd_track_starts", column)
            for column, name in enumerate(MRD_TRACK_START_BRANCHES)
        ],
        *[
            (name, "mrd_track_stops", column)
            for column, name in enumerate(MRD_TRACK_STOP_BRANCHES)
        ],
        *[
            (name, "mrd_track_properties", column)
            for column, name in enumerate(MRD_TRACK_PROPERTY_BRANCHES)
        ],
    ]
    cc_fpr, cc_tpr, _ = roc_curve(cc_labels, full_cc_scores)
    fv_fpr, fv_tpr, _ = roc_curve(fv_labels, full_fv_scores)
    full_cc_roc_auc = float(auc(cc_fpr, cc_tpr))
    full_fv_roc_auc = float(auc(fv_fpr, fv_tpr))
    full_cc_ap = float(average_precision_score(cc_labels, full_cc_scores))
    full_fv_ap = float(average_precision_score(fv_labels, full_fv_scores))
    rng = np.random.default_rng(seed + 130363)
    results = []
    cc_impacts_by_branch = {}
    fv_impacts_by_branch = {}
    values_by_branch = {}
    mask = arrays["track_mask"][test_indices]
    for branch_name, input_name, column in branch_specs:
        cc_repeat_impacts = []
        fv_repeat_impacts = []
        cc_repeat_roc_auc = []
        fv_repeat_roc_auc = []
        cc_repeat_ap = []
        fv_repeat_ap = []
        for _ in range(repeats):
            predict_kwargs = {}
            if input_name == "pmt_unfolded_image":
                source_positions = rng.permutation(len(test_indices))
                predict_kwargs["override_source_indices"] = test_indices[
                    source_positions
                ]
            elif input_name == "event_features":
                original_values = arrays["scalar_features"][test_indices, column]
                changed_values = original_values[rng.permutation(len(original_values))]
                predict_kwargs["override_values"] = changed_values
            else:
                array_name = array_name_by_model_input[input_name]
                original_values = arrays[array_name][test_indices, :, column]
                changed_values = _permute_valid_track_values(
                    original_values, mask, rng
                )
                predict_kwargs["override_values"] = changed_values
            permuted_cc_scores, permuted_fv_scores = _predict_scores_in_batches(
                model,
                arrays,
                test_indices,
                batch_size,
                override_input=input_name,
                override_column=column,
                **predict_kwargs,
            )
            cc_repeat_impacts.append(full_cc_scores - permuted_cc_scores)
            fv_repeat_impacts.append(full_fv_scores - permuted_fv_scores)
            cc_fpr, cc_tpr, _ = roc_curve(cc_labels, permuted_cc_scores)
            fv_fpr, fv_tpr, _ = roc_curve(fv_labels, permuted_fv_scores)
            cc_repeat_roc_auc.append(float(auc(cc_fpr, cc_tpr)))
            fv_repeat_roc_auc.append(float(auc(fv_fpr, fv_tpr)))
            cc_repeat_ap.append(
                float(average_precision_score(cc_labels, permuted_cc_scores))
            )
            fv_repeat_ap.append(
                float(average_precision_score(fv_labels, permuted_fv_scores))
            )
        cc_event_impacts = np.mean(np.stack(cc_repeat_impacts), axis=0)
        fv_event_impacts = np.mean(np.stack(fv_repeat_impacts), axis=0)
        cc_impacts_by_branch[branch_name] = cc_event_impacts
        fv_impacts_by_branch[branch_name] = fv_event_impacts
        if input_name == "pmt_unfolded_image":
            display_values = _image_channel_event_sums(
                arrays, test_indices, column, batch_size
            )
        elif input_name == "event_features":
            display_values = arrays["scalar_features"][
                test_indices, column
            ].astype(np.float64)
        else:
            array_name = array_name_by_model_input[input_name]
            display_values = _masked_event_mean(
                arrays[array_name][test_indices, :, column], mask
            )
        values_by_branch[branch_name] = display_values
        results.append(
            {
                "branch": branch_name,
                "input_group": input_name,
                "cc_mean_abs_score_impact": float(
                    np.mean(np.abs(cc_event_impacts))
                ),
                "cc_mean_signed_score_impact": float(np.mean(cc_event_impacts)),
                "cc_roc_auc_drop": full_cc_roc_auc
                - float(np.mean(cc_repeat_roc_auc)),
                "cc_roc_auc_drop_std": float(np.std(cc_repeat_roc_auc)),
                "cc_average_precision_drop": full_cc_ap
                - float(np.mean(cc_repeat_ap)),
                "cc_average_precision_drop_std": float(np.std(cc_repeat_ap)),
                "fv_mean_abs_score_impact": float(
                    np.mean(np.abs(fv_event_impacts))
                ),
                "fv_mean_signed_score_impact": float(np.mean(fv_event_impacts)),
                "fv_roc_auc_drop": full_fv_roc_auc
                - float(np.mean(fv_repeat_roc_auc)),
                "fv_roc_auc_drop_std": float(np.std(fv_repeat_roc_auc)),
                "fv_average_precision_drop": full_fv_ap
                - float(np.mean(fv_repeat_ap)),
                "fv_average_precision_drop_std": float(np.std(fv_repeat_ap)),
            }
        )
    cc_order = sorted(
        results, key=lambda row: row["cc_mean_abs_score_impact"], reverse=True
    )
    fv_order = sorted(
        results, key=lambda row: row["fv_mean_abs_score_impact"], reverse=True
    )
    cc_ranks = {row["branch"]: rank for rank, row in enumerate(cc_order, start=1)}
    fv_ranks = {row["branch"]: rank for rank, row in enumerate(fv_order, start=1)}
    for row in results:
        row["cc_rank"] = cc_ranks[row["branch"]]
        row["fv_rank"] = fv_ranks[row["branch"]]
    csv_rows = sorted(results, key=lambda row: row["cc_rank"])
    csv_path = output.with_suffix(".input_importance.csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)

    plot_rng = np.random.default_rng(seed + 155921)
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(16, 16),
        gridspec_kw={"width_ratios": [2.3, 1.0]},
        sharey="row",
    )
    head_specs = (
        (
            "CC-inclusive",
            cc_order,
            cc_impacts_by_branch,
            "cc_mean_abs_score_impact",
        ),
        ("FV", fv_order, fv_impacts_by_branch, "fv_mean_abs_score_impact"),
    )
    for plot_row, (head_name, ranking, impact_map, metric_name) in enumerate(
        head_specs
    ):
        impact_axis, rank_axis = axes[plot_row]
        y_positions = np.arange(len(ranking))
        for y_position, result in zip(y_positions, ranking):
            branch = str(result["branch"])
            impacts = impact_map[branch]
            values = values_by_branch[branch]
            chosen = np.arange(len(impacts))
            if len(chosen) > 2000:
                chosen = plot_rng.choice(chosen, size=2000, replace=False)
            plot_values = values[chosen]
            finite = plot_values[np.isfinite(plot_values)]
            if len(finite):
                low, high = np.nanpercentile(finite, [5.0, 95.0])
                scale = high - low
                colors = (
                    np.clip((plot_values - low) / scale, 0.0, 1.0)
                    if scale > 0.0
                    else np.full(len(plot_values), 0.5)
                )
                colors = np.where(np.isfinite(colors), colors, 0.5)
            else:
                colors = np.full(len(plot_values), 0.5)
            impact_axis.scatter(
                impacts[chosen],
                y_position + plot_rng.normal(0.0, 0.10, len(chosen)),
                c=colors,
                cmap="coolwarm",
                vmin=0.0,
                vmax=1.0,
                s=9,
                alpha=0.5,
                linewidths=0,
            )
        impact_axis.axvline(0.0, color="black", linewidth=1)
        impact_axis.set_yticks(
            y_positions, [str(result["branch"]) for result in ranking]
        )
        impact_axis.invert_yaxis()
        impact_axis.set_xlabel(
            f"Original {head_name} score − score after permutation"
        )
        impact_axis.set_title(f"{head_name}: held-out event impacts")
        rank_axis.barh(
            y_positions,
            [result[metric_name] for result in ranking],
            color="tab:blue" if plot_row == 0 else "tab:green",
            alpha=0.8,
        )
        rank_axis.set_xlabel("Mean |score impact|")
        rank_axis.set_title(f"{head_name}: global ranking")
        color_scale = plt.cm.ScalarMappable(cmap="coolwarm")
        color_scale.set_array([])
        colorbar = fig.colorbar(color_scale, ax=impact_axis, pad=0.01)
        colorbar.set_ticks([0.0, 1.0], labels=["low", "high"])
        colorbar.set_label("Event input value (within-input scale)")
    fig.suptitle(
        "Complete held-out permutation importance\n"
        "Image colors use channel sum; vector colors use valid-track mean",
        fontsize=12,
    )
    fig.tight_layout()
    plot_path = output.with_suffix(".input_importance_beeswarm.png")
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)
    return {
        "input_importance_csv": csv_path.name,
        "input_importance_beeswarm": plot_path.name,
        "input_importance": {
            "method": (
                "One inference input at a time was permuted across held-out "
                "events. Image channels were permuted as complete detector "
                "maps. Per-event impact is the original score minus the score "
                "after permutation. This is SHAP-style permutation impact, "
                "not an exact SHAP value."
            ),
            "repeats": int(repeats),
            "cc_ranking_metric": "cc_mean_abs_score_impact",
            "fv_ranking_metric": "fv_mean_abs_score_impact",
            "cc_ranking": [row["branch"] for row in cc_order],
            "fv_ranking": [row["branch"] for row in fv_order],
            "rows": csv_rows,
        },
    }


def evaluate_fv_prediction(
    labels: np.ndarray,
    scores: np.ndarray,
    output: Path,
    threshold: float,
) -> Dict[str, object]:
    """Create the held-out performance suite for the true-FV output."""
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = float(auc(fpr, tpr))
    precision, recall, _ = precision_recall_curve(labels, scores)
    average_precision = float(average_precision_score(labels, scores))
    brier = float(brier_score_loss(labels, scores))
    operating_point = _operating_point(labels, scores, threshold)

    fig, axis = plt.subplots(figsize=(6, 5))
    axis.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.4f}")
    axis.plot([0, 1], [0, 1], "--", color="gray")
    axis.set(xlabel="Outside-FV false-positive rate", ylabel="Inside-FV efficiency")
    axis.legend()
    fig.tight_layout()
    roc_path = output.with_suffix(".fv_roc_curve.png")
    fig.savefig(roc_path, dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(6, 5))
    axis.plot(recall, precision, label=f"Average precision = {average_precision:.4f}")
    axis.set(xlabel="Inside-FV efficiency", ylabel="Inside-FV purity")
    axis.legend()
    fig.tight_layout()
    pr_path = output.with_suffix(".fv_precision_recall_curve.png")
    fig.savefig(pr_path, dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7, 5))
    bins = np.linspace(0.0, 1.0, 41)
    axis.hist(
        scores[labels == 0],
        bins=bins,
        density=True,
        histtype="step",
        label="trueFV = 0",
    )
    axis.hist(
        scores[labels == 1],
        bins=bins,
        density=True,
        histtype="step",
        label="trueFV = 1",
    )
    axis.axvline(threshold, linestyle="--", color="black", label=f"cut = {threshold:.2f}")
    axis.set(xlabel="FV score", ylabel="Normalized events")
    axis.legend()
    fig.tight_layout()
    score_path = output.with_suffix(".fv_score_distribution.png")
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
        (matrix, ".fv_confusion_matrix.png", "d"),
        (normalized, ".fv_confusion_matrix_normalized.png", ".3f"),
    ):
        fig, axis = plt.subplots(figsize=(5.5, 5))
        image = axis.imshow(values, cmap="Greens", vmin=0)
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
        axis.set_xticks([0, 1], ["outside FV", "inside FV"])
        axis.set_yticks([0, 1], ["outside FV", "inside FV"])
        axis.set(xlabel="Predicted", ylabel="Truth")
        fig.tight_layout()
        path = output.with_suffix(suffix)
        fig.savefig(path, dpi=160)
        plt.close(fig)
        matrix_paths.append(path.name)

    scan_rows = [
        _operating_point(labels, scores, float(scan_threshold))
        for scan_threshold in np.linspace(0.0, 1.0, 201)
    ]
    scan_path = output.with_suffix(".fv_efficiency_purity.csv")
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
    axis.set(xlabel="FV threshold", ylabel="Metric", ylim=(0, 1.05))
    axis.legend()
    fig.tight_layout()
    scan_plot_path = output.with_suffix(".fv_efficiency_purity.png")
    fig.savefig(scan_plot_path, dpi=160)
    plt.close(fig)
    return {
        "target": "trueFV != 0",
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
    }


def evaluate_and_plot(
    model: tf.keras.Model,
    arrays: Dict[str, np.ndarray],
    test_indices: np.ndarray,
    history_path: Path,
    output: Path,
    threshold: float,
    fv_threshold: float,
    seed: int,
    input_importance_repeats: int,
    batch_size: int,
) -> Dict[str, object]:
    labels = arrays["labels"][test_indices].astype(np.int64)
    scores, fv_scores = _predict_scores_in_batches(
        model, arrays, test_indices, batch_size
    )
    fv_labels = arrays["fv_labels"][test_indices].astype(np.int64)
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = float(auc(fpr, tpr))
    precision, recall, _ = precision_recall_curve(labels, scores)
    average_precision = float(average_precision_score(labels, scores))
    brier = float(brier_score_loss(labels, scores))
    operating_point = _operating_point(labels, scores, threshold)
    fv_evaluation = evaluate_fv_prediction(
        fv_labels, fv_scores, output, fv_threshold
    )

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
                "ccinclusive_signal",
                "true_fv",
                *arrays["scalar_feature_names"],
                "ccinclusive_score",
                "ccinclusive_threshold",
                "ccinclusive_prediction",
                "fv_score",
                "fv_threshold",
                "fv_prediction",
            ]
        )
        scalar_values = arrays["scalar_features"][test_indices]
        for row, event_index in enumerate(test_indices):
            writer.writerow(
                [
                    arrays["source_paths"][event_index],
                    int(arrays["entries"][event_index]),
                    int(labels[row]),
                    int(fv_labels[row]),
                    *scalar_values[row].astype(float),
                    float(scores[row]),
                    float(threshold),
                    int(scores[row] >= threshold),
                    float(fv_scores[row]),
                    float(fv_threshold),
                    int(fv_scores[row] >= fv_threshold),
                ]
            )

    history_plot = ""
    if history_path.is_file():
        with history_path.open(newline="", encoding="utf-8") as handle:
            history_rows = list(csv.DictReader(handle))
        if history_rows:
            fig, axes = plt.subplots(1, 3, figsize=(16, 4))
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
            fv_metric_key = "fv_score_pr_auc"
            fv_validation_key = f"val_{fv_metric_key}"
            axes[2].plot(
                epochs,
                [float(row[fv_metric_key]) for row in history_rows],
                label="train",
            )
            axes[2].plot(
                epochs,
                [float(row[fv_validation_key]) for row in history_rows],
                label="validation",
            )
            axes[2].set(xlabel="Epoch", ylabel="FV PR AUC")
            axes[2].legend()
            fig.tight_layout()
            history_plot_path = output.with_suffix(".training_curves.png")
            fig.savefig(history_plot_path, dpi=160)
            plt.close(fig)
            history_plot = history_plot_path.name

    misclassified = plot_misclassified_events(
        arrays,
        test_indices,
        labels,
        scores,
        fv_labels,
        fv_scores,
        output,
        threshold,
        fv_threshold,
    )
    gradcam_path = plot_gradcam_examples(
        model,
        arrays,
        test_indices,
        labels,
        scores,
        fv_labels,
        fv_scores,
        output,
        threshold,
    )
    input_importance = evaluate_input_importance(
        model,
        arrays,
        test_indices,
        labels,
        fv_labels,
        scores,
        fv_scores,
        output,
        seed,
        input_importance_repeats,
        batch_size,
    )

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
        "fv_prediction": fv_evaluation,
        "gradcam_examples": gradcam_path,
        **misclassified,
        **input_importance,
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
    fv_labels = arrays["fv_labels"]
    class_counts = np.bincount(labels.astype(np.int64), minlength=2)
    fv_class_counts = np.bincount(fv_labels.astype(np.int64), minlength=2)
    if np.any(class_counts == 0):
        raise ValueError(
            "Both sel_nu_mu_cc target classes are required after selection; "
            f"counts are {class_counts.tolist()}"
        )
    if np.any(fv_class_counts == 0):
        raise ValueError(
            "Both true-FV target classes are required after the reconstructed "
            "FV selection; counts are "
            f"{fv_class_counts.tolist()}. Remove --apply-reco-fv-cut if the "
            "selection removes all outside-true-FV events."
        )
    indices = np.arange(len(labels))
    joint_targets = labels.astype(np.int64) * 2 + fv_labels.astype(np.int64)
    train_indices, temporary_indices = train_test_split(
        indices,
        test_size=VALIDATION_FRACTION + TEST_FRACTION,
        random_state=args.seed,
        stratify=joint_targets,
    )
    validation_indices, test_indices = train_test_split(
        temporary_indices,
        test_size=TEST_FRACTION / (VALIDATION_FRACTION + TEST_FRACTION),
        random_state=args.seed,
        stratify=joint_targets[temporary_indices],
    )
    train_counts = np.bincount(labels[train_indices].astype(np.int64), minlength=2)
    fv_train_counts = np.bincount(
        fv_labels[train_indices].astype(np.int64), minlength=2
    )
    class_weights = {
        "ccinclusive_score": {
            0: len(train_indices) / (2.0 * train_counts[0]),
            1: len(train_indices) / (2.0 * train_counts[1]),
        },
        "fv_score": {
            0: len(train_indices) / (2.0 * fv_train_counts[0]),
            1: len(train_indices) / (2.0 * fv_train_counts[1]),
        },
    }
    model = build_model(arrays, train_indices, args.fv_loss_weight)
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
                monitor="val_ccinclusive_score_pr_auc",
                mode="max",
                patience=12,
                restore_best_weights=True,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_ccinclusive_score_pr_auc",
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
    del train_sequence, validation_sequence, test_sequence
    gc.collect()
    evaluation = evaluate_and_plot(
        model,
        arrays,
        test_indices,
        history_path,
        args.output,
        args.threshold,
        args.fv_threshold,
        args.seed,
        args.input_importance_repeats,
        args.diagnostic_batch_size,
    )

    responses = arrays["responses"]
    response_modes = sorted(responses)
    training_response = response_modes[0] if len(response_modes) == 1 else "mixed"
    metadata_response = responses.get("tuned", responses.get("raw"))
    metadata = {
        "model_type": "annie_ccinclusive_fv_detector_dual_channel_cnn",
        "tensorflow_version": tf.__version__,
        "tree": args.tree,
        "training_sources": [
            source for group in arrays["groups"] for source in group["sources"]
        ],
        "training_source_groups": arrays["groups"],
        "outputs": {
            "ccinclusive_score": {
                "target": "sel_nu_mu_cc == true",
                "threshold": args.threshold,
                "loss_weight": 1.0,
            },
            "fv_score": {
                "target": "trueFV != 0",
                "threshold": args.fv_threshold,
                "loss_weight": args.fv_loss_weight,
            },
        },
        "target": "sel_nu_mu_cc == true",
        "ccinclusive_signal_branch": CCINCLUSIVE_SIGNAL_BRANCH,
        "supervision_branches": list(SUPERVISION_BRANCHES),
        "truth_branches": ["trueFV"],
        "supervision_used_as_input": False,
        "truth_used_as_input": False,
        "vertex_information_used_as_input": False,
        "threshold": args.threshold,
        "fv_threshold": args.fv_threshold,
        "fv_loss_weight": args.fv_loss_weight,
        "fv_cut_applied": args.apply_reco_fv_cut,
        "fv_cut_branches": arrays["fv_branches"],
        "fv_cut_expression": (
            "resolved_reconstructed_fv_branch != 0"
            if args.apply_reco_fv_cut
            else "none"
        ),
        "hitpe_branch": args.hitpe_branch,
        "hitid_branch": args.hitid_branch,
        "tankcluster_branches": arrays["tank_branches"],
        "tankcluster_id_branch": args.tankcluster_id_branch,
        "image_channels": ["log1p_hitPE", "log1p_hitPE_tankcluster"],
        "image_channel_routing": {
            "ccinclusive_score": ["log1p_hitPE_tankcluster"],
            "fv_score": ["log1p_hitPE", "log1p_hitPE_tankcluster"],
        },
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
        "training_selection_expression": TRAINING_SELECTION_EXPRESSION,
        "training_selection_branches": [
            branch for branch, _, _ in TRAINING_SELECTION_SPECS
        ],
        "selection_flags_are_inputs_not_cuts": False,
        "selection_input_notes": {
            "sel_trigword": "also a hard cut; constant 1 after selection",
            "sel_simpleRecoFlag": "also a hard cut; constant 1 after selection",
            "other_selection_flags": "model inputs, not hard cuts",
        },
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
        "diagnostic_batch_size": args.diagnostic_batch_size,
        "root_chunk_size": args.chunk_size,
        "data_split": {
            "training_fraction": TRAIN_FRACTION,
            "validation_fraction": VALIDATION_FRACTION,
            "testing_fraction": TEST_FRACTION,
            "training_events": int(len(train_indices)),
            "validation_events": int(len(validation_indices)),
            "testing_events": int(len(test_indices)),
            "stratified_by_joint_ccinclusive_and_fv_targets": True,
            "random_seed": args.seed,
        },
        "class_counts": {
            "ccinclusive_score": {
                "background": int(class_counts[0]),
                "signal": int(class_counts[1]),
            },
            "fv_score": {
                "outside_fv": int(fv_class_counts[0]),
                "inside_fv": int(fv_class_counts[1]),
            },
        },
        "events_read": arrays["events_read"],
        "events_selected": arrays["events_selected"],
        "outside_fv_events_rejected": arrays["outside_fv_events"],
        "misaligned_events_rejected": arrays["misaligned_events"],
        "invalid_feature_events_rejected": arrays["invalid_feature_events"],
        "training_cut_events_rejected": arrays[
            "training_cut_rejected_events"
        ],
        "training_cut_failed_counts": arrays["training_cut_failed_counts"],
        "mrd_track_overflow_events": arrays["mrd_overflow_events"],
        "test_metrics": {name: float(value) for name, value in test_metrics.items()},
        "evaluation": evaluation,
        "training_history": history_path.name,
    }
    metadata_path = args.output.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {args.output} and {metadata_path}")
    print(
        f"Held-out CC ROC AUC={evaluation['roc_auc']:.5f}; "
        f"average precision={evaluation['average_precision']:.5f}"
    )
    print(
        f"Held-out FV ROC AUC={evaluation['fv_prediction']['roc_auc']:.5f}; "
        "average precision="
        f"{evaluation['fv_prediction']['average_precision']:.5f}"
    )


if __name__ == "__main__":
    main()
