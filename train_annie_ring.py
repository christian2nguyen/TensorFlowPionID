#!/usr/bin/env python3
"""Train a CNN to identify pion-like ANNIE PMT ring patterns."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import tensorflow as tf
import awkward as ak
import uproot
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    auc,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)
from sklearn.model_selection import train_test_split

from annie_pmt_response import (
    PMT_RESPONSE_CHOICES,
    PMT_TUNE_VARIANT_CHOICES,
    PMTResponse,
)
from annie_features import (
    FIT_INDIVIDUAL_PMT_SELECTION_EXPRESSION,
    MAX_MRD_TRACKS,
    MRD_TRACK_PROPERTY_BRANCHES,
    MRD_TRACK_START_BRANCHES,
    RING_EVENT_FEATURE_NAMES,
    RING_EVENT_SCALAR_BRANCHES,
)
from annie_ring_images import (
    DEFAULT_GEOMETRY,
    DEFAULT_DETECTOR_HEIGHT,
    DEFAULT_DETECTOR_WIDTH,
    DEFAULT_HEIGHT,
    DEFAULT_WIDTH,
    PMT_MASK_CHOICES,
    PMTGeometry,
    iterate_ring_images,
)

TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15
TEST_FRACTION = 0.15

OPENING_ANGLE_DEGREE_BRANCHES = (
    "trueMuonPionOpeningAngleDeg",
    "trueMuonPionOpeningAngle_deg",
    "mcMuonPionOpeningAngleDeg",
    "mc_muon_pion_opening_angle_deg",
)
OPENING_ANGLE_RADIAN_BRANCHES = (
    "trueMuonPionOpeningAngle",
    "mcMuonPionOpeningAngle",
    "mc_muon_pion_opening_angle",
)
TRUTH_MUON_VECTOR_BASES = ("mc_p3_mu",)
TRUTH_PION_VECTOR_BASES = ("mc_p3_lead_pi", "mc_p3_lead_pion")
MUON_MASS_GEV = 0.1056583755
CHARGED_PION_MASS_GEV = 0.13957039
NEUTRAL_PION_MASS_GEV = 0.1349768


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root_files",
        nargs="*",
        type=Path,
        help="ROOT files; optional when --file-list is supplied",
    )
    parser.add_argument(
        "--file-list",
        type=Path,
        help="Text file containing one ROOT file path per line",
    )
    parser.add_argument("--tree", default="phaseIITriggerTree")
    parser.add_argument("--geometry", type=Path, default=DEFAULT_GEOMETRY)
    parser.add_argument(
        "--pmt-mask",
        choices=PMT_MASK_CHOICES,
        default="bdt",
        help="bdt: historical shutoff mask; on: CSV status ON only; all: no mask",
    )
    parser.add_argument("--hitpe-branch", default="hitPE")
    parser.add_argument("--hitid-branch", default="hitDetID")
    parser.add_argument("--tankcluster-branch")
    parser.add_argument("--tankcluster-id-branch", default="hitDetID_tankcluster")
    parser.add_argument("--image-height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--image-width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--detector-height", type=int, default=DEFAULT_DETECTOR_HEIGHT)
    parser.add_argument("--detector-width", type=int, default=DEFAULT_DETECTOR_WIDTH)
    parser.add_argument(
        "--pmt-response",
        choices=PMT_RESPONSE_CHOICES,
        default="raw",
        help="raw: unchanged PE; tuned: map both MC PE channels toward beam-on data",
    )
    parser.add_argument(
        "--pmt-response-calibration",
        type=Path,
        help="all-PMT calibration ROOT file required for --pmt-response tuned",
    )
    parser.add_argument(
        "--pmt-tune-variant",
        choices=PMT_TUNE_VARIANT_CHOICES,
        default="final",
        help="response: Gain/Delta/Sigma only; final: also replay residual hits",
    )
    parser.add_argument("--charged-only", action="store_true")
    parser.add_argument(
        "--muon-pion-opening-angle-deg-branch",
        help=(
            "optional scalar truth branch containing the leading-pion/muon "
            "opening angle in degrees; otherwise auto-detect a scalar branch "
            "or derive it from mc_p3_mu and mc_p3_lead_pi(on)"
        ),
    )
    parser.add_argument(
        "--no-event-cuts",
        "--no-bdt-cuts",
        dest="no_event_cuts",
        action="store_true",
        help=(
            "Disable the Fit_indivdiualPMT_Gaussian_Convolution event selection; "
            "--no-bdt-cuts is retained as a compatibility alias"
        ),
    )
    parser.add_argument("--chunk-size", default="100 MB")
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/annie_ring_pion.h5")
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260620)
    args = parser.parse_args()
    if args.file_list is not None:
        if not args.file_list.is_file():
            parser.error(f"--file-list {args.file_list} does not exist")
        listed_files = []
        for line in args.file_list.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            listed_path = Path(stripped).expanduser()
            if not listed_path.is_absolute():
                listed_path = args.file_list.parent / listed_path
            listed_files.append(listed_path)
        args.root_files.extend(listed_files)
    if not args.root_files:
        parser.error("No ROOT files given: use positional paths or --file-list")
    return args


def load_data(
    args: argparse.Namespace,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    str,
    int,
    int,
    PMTGeometry,
    PMTResponse,
]:
    geometry = PMTGeometry(args.geometry, args.pmt_mask)
    response = PMTResponse(
        args.pmt_response,
        args.pmt_response_calibration,
        args.pmt_tune_variant,
    )
    print(response.describe())
    image_chunks: List[np.ndarray] = []
    detector_chunks: List[np.ndarray] = []
    event_feature_chunks: List[np.ndarray] = []
    mrd_track_start_chunks: List[np.ndarray] = []
    mrd_track_property_chunks: List[np.ndarray] = []
    mrd_track_mask_chunks: List[np.ndarray] = []
    label_chunks: List[np.ndarray] = []
    truth_pion_chunks: List[np.ndarray] = []
    source_path_chunks: List[np.ndarray] = []
    entry_chunks: List[np.ndarray] = []
    tank_branch = ""
    n_read = 0
    n_selected = 0
    n_misaligned = 0
    n_mrd_tracks_truncated = 0
    for chunk in iterate_ring_images(
        args.root_files,
        args.tree,
        geometry,
        args.hitpe_branch,
        args.hitid_branch,
        args.tankcluster_branch,
        args.tankcluster_id_branch,
        args.image_height,
        args.image_width,
        args.detector_height,
        args.detector_width,
        response,
        not args.no_event_cuts,
        True,
        args.charged_only,
        args.chunk_size,
        include_ring_event_features=True,
    ):
        image_chunks.append(chunk["images"])
        detector_chunks.append(chunk["detector_images"])
        event_feature_chunks.append(chunk["event_features"])
        mrd_track_start_chunks.append(chunk["mrd_track_starts"])
        mrd_track_property_chunks.append(chunk["mrd_track_properties"])
        mrd_track_mask_chunks.append(chunk["mrd_track_mask"])
        label_chunks.append(chunk["labels"])
        truth_pion_chunks.append(chunk["truth_pion_counts"])
        entries = chunk["entries"]
        entry_chunks.append(entries)
        source_path_chunks.append(
            np.full(len(entries), str(chunk["path"]), dtype=object)
        )
        tank_branch = str(chunk["tankcluster_branch"])
        n_read += int(chunk["n_read"])
        n_selected += int(chunk["n_selected"])
        n_misaligned += int(chunk["n_misaligned"])
        n_mrd_tracks_truncated += int(chunk["n_mrd_tracks_truncated"])
        print(f"Read {n_read}; selected {n_selected}", end="\r", flush=True)
    print()
    if not image_chunks or n_selected == 0:
        raise ValueError("No events passed ring-image construction and selection")
    return (
        np.concatenate(image_chunks),
        np.concatenate(detector_chunks),
        np.concatenate(event_feature_chunks),
        np.concatenate(mrd_track_start_chunks),
        np.concatenate(mrd_track_property_chunks),
        np.concatenate(mrd_track_mask_chunks),
        np.concatenate(label_chunks),
        np.concatenate(source_path_chunks),
        np.concatenate(entry_chunks),
        np.concatenate(truth_pion_chunks),
        tank_branch,
        n_misaligned,
        n_mrd_tracks_truncated,
        geometry,
        response,
    )


def _resolve_vector3_spec(available, base_names: Tuple[str, ...]) -> Tuple[str, ...]:
    component_suffixes = (
        (".fX", ".fY", ".fZ"),
        ("_fX", "_fY", "_fZ"),
        (".x", ".y", ".z"),
        ("_x", "_y", "_z"),
    )
    for base_name in base_names:
        for suffixes in component_suffixes:
            names = tuple(f"{base_name}{suffix}" for suffix in suffixes)
            if all(name in available for name in names):
                return names
    for base_name in base_names:
        if base_name in available:
            return (base_name,)
    return ()


def _read_vector3(tree, spec: Tuple[str, ...]) -> np.ndarray:
    if len(spec) == 3:
        return np.column_stack(
            [np.asarray(tree[name].array(library="np"), dtype=np.float64) for name in spec]
        )
    if len(spec) != 1:
        raise ValueError("A truth-vector branch was not resolved")

    values = tree[spec[0]].array(library="ak")
    fields = set(ak.fields(values))
    for names in (("fX", "fY", "fZ"), ("x", "y", "z"), ("X", "Y", "Z")):
        if all(name in fields for name in names):
            return np.column_stack(
                [np.asarray(ak.to_numpy(values[name]), dtype=np.float64) for name in names]
            )
    converted = np.asarray(ak.to_numpy(values))
    if converted.dtype.names:
        for names in (("fX", "fY", "fZ"), ("x", "y", "z"), ("X", "Y", "Z")):
            if all(name in converted.dtype.names for name in names):
                return np.column_stack([converted[name] for name in names]).astype(
                    np.float64, copy=False
                )
    if converted.ndim == 2 and converted.shape[1] >= 3:
        return converted[:, :3].astype(np.float64, copy=False)
    raise ValueError(
        f"Could not extract x/y/z components from truth-vector branch {spec[0]!r}"
    )


def _truth_kinematics_for_tree(
    tree, scalar_degree_branch: str = ""
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Read truth momenta and obtain the leading-pion/muon opening angle."""
    available = set(tree.keys(recursive=True))
    entry_count = int(tree.num_entries)
    muon_vectors = None
    pion_vectors = None
    muon_momentum = np.full(entry_count, np.nan, dtype=np.float64)
    pion_momentum = np.full(entry_count, np.nan, dtype=np.float64)

    muon_spec = _resolve_vector3_spec(available, TRUTH_MUON_VECTOR_BASES)
    pion_spec = _resolve_vector3_spec(available, TRUTH_PION_VECTOR_BASES)
    vector_descriptions = []
    if muon_spec:
        try:
            muon_vectors = _read_vector3(tree, muon_spec)
            if len(muon_vectors) != entry_count:
                raise ValueError("Muon truth-vector length differs from tree length")
            muon_momentum = np.linalg.norm(muon_vectors, axis=1)
            muon_momentum[~np.isfinite(muon_vectors).all(axis=1)] = np.nan
            vector_descriptions.append(f"muon momentum from {muon_spec[0]}")
        except Exception as error:
            vector_descriptions.append(f"muon momentum unavailable ({error})")
            muon_vectors = None
    else:
        vector_descriptions.append("muon momentum unavailable")
    if pion_spec:
        try:
            pion_vectors = _read_vector3(tree, pion_spec)
            if len(pion_vectors) != entry_count:
                raise ValueError(
                    "Leading-pion truth-vector length differs from tree length"
                )
            pion_momentum = np.linalg.norm(pion_vectors, axis=1)
            pion_momentum[~np.isfinite(pion_vectors).all(axis=1)] = np.nan
            vector_descriptions.append(f"leading-pion momentum from {pion_spec[0]}")
        except Exception as error:
            vector_descriptions.append(f"leading-pion momentum unavailable ({error})")
            pion_vectors = None
    else:
        vector_descriptions.append("leading-pion momentum unavailable")

    if scalar_degree_branch:
        if scalar_degree_branch not in available:
            raise ValueError(
                f"Requested opening-angle branch {scalar_degree_branch!r} was not found"
            )
        angles = np.asarray(
            tree[scalar_degree_branch].array(library="np"), dtype=np.float64
        )
        description = f"degree branch {scalar_degree_branch}"
    else:
        degree_branch = next(
            (name for name in OPENING_ANGLE_DEGREE_BRANCHES if name in available),
            "",
        )
        radian_branch = next(
            (name for name in OPENING_ANGLE_RADIAN_BRANCHES if name in available),
            "",
        )
        if degree_branch:
            angles = np.asarray(
                tree[degree_branch].array(library="np"), dtype=np.float64
            )
            description = f"degree branch {degree_branch}"
        elif radian_branch:
            radians = np.asarray(
                tree[radian_branch].array(library="np"), dtype=np.float64
            )
            angles = np.degrees(radians)
            description = f"radian branch {radian_branch} converted to degrees"
        else:
            if muon_vectors is None or pion_vectors is None:
                angles = np.full(entry_count, np.nan, dtype=np.float64)
                description = "opening angle unavailable"
            else:
                denominator = muon_momentum * pion_momentum
                angles = np.full(entry_count, np.nan, dtype=np.float64)
                valid = (
                    np.isfinite(muon_vectors).all(axis=1)
                    & np.isfinite(pion_vectors).all(axis=1)
                    & (denominator > 0.0)
                )
                cosine = np.zeros(entry_count, dtype=np.float64)
                cosine[valid] = (
                    np.einsum("ij,ij->i", muon_vectors[valid], pion_vectors[valid])
                    / denominator[valid]
                )
                angles[valid] = np.degrees(
                    np.arccos(np.clip(cosine[valid], -1.0, 1.0))
                )
                description = (
                    f"opening angle derived from {muon_spec[0]} and "
                    f"{pion_spec[0]} truth vectors"
                )

    angles = np.asarray(angles, dtype=np.float64).reshape(-1)
    valid_angle = np.isfinite(angles) & (angles >= 0.0) & (angles <= 180.0)
    muon_momentum = np.where(muon_momentum >= 0.0, muon_momentum, np.nan)
    pion_momentum = np.where(pion_momentum >= 0.0, pion_momentum, np.nan)
    full_description = "; ".join([description, *vector_descriptions])
    return (
        np.where(valid_angle, angles, np.nan),
        muon_momentum,
        pion_momentum,
        full_description,
    )


def load_truth_muon_pion_kinematics(
    source_paths: np.ndarray,
    entries: np.ndarray,
    tree_name: str,
    scalar_degree_branch: str = "",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Load truth angle and momenta for selected events in multiple ROOT files."""
    angles = np.full(len(entries), np.nan, dtype=np.float64)
    muon_momenta = np.full(len(entries), np.nan, dtype=np.float64)
    pion_momenta = np.full(len(entries), np.nan, dtype=np.float64)
    descriptions = set()
    source_strings = np.asarray(source_paths, dtype=str)
    for source in dict.fromkeys(source_strings.tolist()):
        mask = source_strings == source
        try:
            with uproot.open(source) as root_file:
                tree = root_file[tree_name]
                (
                    file_angles,
                    file_muon_momenta,
                    file_pion_momenta,
                    description,
                ) = _truth_kinematics_for_tree(tree, scalar_degree_branch)
            local_entries = entries[mask].astype(np.int64, copy=False)
            valid_entries = (local_entries >= 0) & (local_entries < len(file_angles))
            target_indices = np.flatnonzero(mask)
            selected_targets = target_indices[valid_entries]
            selected_entries = local_entries[valid_entries]
            angles[selected_targets] = file_angles[selected_entries]
            muon_momenta[selected_targets] = file_muon_momenta[selected_entries]
            pion_momenta[selected_targets] = file_pion_momenta[selected_entries]
            descriptions.add(description)
        except Exception as error:
            if scalar_degree_branch:
                raise
            print(
                f"Warning: could not load truth muon-pion kinematics from "
                f"{Path(source).name}: {error}"
            )
            descriptions.add("unavailable")
    return (
        angles,
        muon_momenta,
        pion_momenta,
        "; ".join(sorted(descriptions)) or "unavailable",
    )


def _kinetic_energy(momentum: np.ndarray, mass: np.ndarray) -> np.ndarray:
    """Return relativistic kinetic energy for GeV/c momentum and GeV/c^2 mass."""
    momentum = np.asarray(momentum, dtype=np.float64)
    mass = np.asarray(mass, dtype=np.float64)
    kinetic_energy = np.sqrt(np.square(momentum) + np.square(mass)) - mass
    valid = np.isfinite(momentum) & np.isfinite(mass) & (momentum >= 0.0)
    return np.where(valid, kinetic_energy, np.nan)


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
    angular_shape: Tuple[int, ...],
    detector_shape: Tuple[int, ...],
    angular_train: np.ndarray,
    detector_train: np.ndarray,
    event_features_train: np.ndarray,
    mrd_track_starts_train: np.ndarray,
    mrd_track_properties_train: np.ndarray,
    mrd_track_mask_train: np.ndarray,
) -> tf.keras.Model:
    angular_normalizer = tf.keras.layers.Normalization(
        axis=-1, name="angular_channel_normalization"
    )
    detector_normalizer = tf.keras.layers.Normalization(
        axis=-1, name="detector_channel_normalization"
    )
    event_feature_normalizer = tf.keras.layers.Normalization(
        axis=-1, name="event_feature_normalization"
    )
    mrd_track_normalizer = tf.keras.layers.Normalization(
        axis=-1, name="mrd_track_coordinate_normalization"
    )
    mrd_property_normalizer = tf.keras.layers.Normalization(
        axis=-1, name="mrd_track_property_normalization"
    )
    angular_normalizer.adapt(angular_train)
    detector_normalizer.adapt(detector_train)
    event_feature_normalizer.adapt(event_features_train)
    valid_track_starts = mrd_track_starts_train[
        mrd_track_mask_train.astype(bool)
    ]
    valid_track_properties = mrd_track_properties_train[
        mrd_track_mask_train.astype(bool)
    ]
    if len(valid_track_starts) == 0:
        raise ValueError("At least one MRD track is required in the training split")
    mrd_track_normalizer.adapt(valid_track_starts)
    mrd_property_normalizer.adapt(valid_track_properties)
    angular_inputs = tf.keras.Input(shape=angular_shape, name="pmt_angular_image")
    detector_inputs = tf.keras.Input(shape=detector_shape, name="pmt_unfolded_image")
    event_feature_inputs = tf.keras.Input(
        shape=(event_features_train.shape[1],), name="event_features"
    )
    mrd_track_start_inputs = tf.keras.Input(
        shape=(MAX_MRD_TRACKS, 3), name="mrd_track_starts"
    )
    mrd_track_property_inputs = tf.keras.Input(
        shape=(MAX_MRD_TRACKS, 3), name="mrd_track_properties"
    )
    mrd_track_mask_inputs = tf.keras.Input(
        shape=(MAX_MRD_TRACKS,), name="mrd_track_mask"
    )
    angular_features = _image_tower(angular_inputs, angular_normalizer, "angular")
    detector_features = _image_tower(detector_inputs, detector_normalizer, "detector")
    event_features = event_feature_normalizer(event_feature_inputs)
    event_features = tf.keras.layers.Dense(
        8, activation="relu", name="event_feature_embedding"
    )(event_features)
    normalized_track_starts = mrd_track_normalizer(mrd_track_start_inputs)
    normalized_track_properties = mrd_property_normalizer(
        mrd_track_property_inputs
    )
    mrd_track_features = tf.keras.layers.Concatenate(
        axis=-1, name="combined_mrd_track_values"
    )([normalized_track_starts, normalized_track_properties])
    mrd_track_features = tf.keras.layers.Dense(
        16, activation="relu", name="shared_mrd_track_embedding"
    )(mrd_track_features)
    expanded_track_mask = tf.keras.layers.Reshape(
        (MAX_MRD_TRACKS, 1), name="expanded_mrd_track_mask"
    )(mrd_track_mask_inputs)
    mrd_track_features = tf.keras.layers.Multiply(
        name="masked_mrd_track_embeddings"
    )([mrd_track_features, expanded_track_mask])
    mrd_track_features = tf.keras.layers.GlobalMaxPooling1D(
        name="mrd_track_set_features"
    )(mrd_track_features)
    x = tf.keras.layers.Concatenate(name="combined_views_and_event_features")(
        [
            angular_features,
            detector_features,
            event_features,
            mrd_track_features,
        ]
    )
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.30)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="pion_score")(x)
    model = tf.keras.Model(
        [
            angular_inputs,
            detector_inputs,
            event_feature_inputs,
            mrd_track_start_inputs,
            mrd_track_property_inputs,
            mrd_track_mask_inputs,
        ],
        outputs,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.AUC(name="roc_auc"),
            tf.keras.metrics.AUC(curve="PR", name="pr_auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def plot_roc_curve(y_test: np.ndarray, test_scores: np.ndarray, output: Path):
    fpr, tpr, thresholds = roc_curve(y_test, test_scores)
    roc_auc_value = float(auc(fpr, tpr))
    fig, axis = plt.subplots(figsize=(6, 6))
    axis.plot(fpr, tpr, color="C0", label=f"ROC (AUC = {roc_auc_value:.4f})")
    axis.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Chance")
    axis.set_xlabel("False positive rate (no-pion misclassified as pion)")
    axis.set_ylabel("True positive rate (pion efficiency)")
    axis.set_title("Test-set ROC curve")
    axis.legend(loc="lower right")
    fig.tight_layout()
    path = output.with_suffix(".roc_curve.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path.name, roc_auc_value, fpr, tpr, thresholds


def plot_score_distribution(
    y_test: np.ndarray, test_scores: np.ndarray, output: Path
) -> str:
    fig, axis = plt.subplots(figsize=(7, 5))
    bins = np.linspace(0.0, 1.0, 41)
    axis.hist(
        test_scores[y_test == 0],
        bins=bins,
        alpha=0.6,
        density=True,
        label="No pion (truth)",
        color="C0",
    )
    axis.hist(
        test_scores[y_test == 1],
        bins=bins,
        alpha=0.6,
        density=True,
        label="Pion (truth)",
        color="C1",
    )
    axis.set_xlabel("Model pion_score")
    axis.set_ylabel("Normalized event density")
    axis.set_title("Test-set score separation by truth class")
    axis.legend()
    fig.tight_layout()
    path = output.with_suffix(".score_distribution.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path.name


def plot_precision_recall(
    y_test: np.ndarray, test_scores: np.ndarray, output: Path
):
    precision, recall, _ = precision_recall_curve(y_test, test_scores)
    average_precision = float(average_precision_score(y_test, test_scores))
    fig, axis = plt.subplots(figsize=(6, 6))
    axis.plot(
        recall,
        precision,
        color="C2",
        label=f"PR (AP = {average_precision:.4f})",
    )
    axis.set_xlabel("Recall (pion efficiency)")
    axis.set_ylabel("Precision (purity)")
    axis.set_title("Test-set precision-recall curve")
    axis.legend(loc="lower left")
    fig.tight_layout()
    path = output.with_suffix(".precision_recall_curve.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path.name, average_precision


def plot_efficiency_rejection_vs_threshold(
    fpr: np.ndarray, tpr: np.ndarray, thresholds: np.ndarray, output: Path
) -> str:
    finite = np.isfinite(thresholds)
    order = np.argsort(thresholds[finite])
    x = thresholds[finite][order]
    efficiency = tpr[finite][order]
    rejection = 1.0 - fpr[finite][order]
    fig, axis = plt.subplots(figsize=(7, 5))
    axis.plot(x, efficiency, label="Signal efficiency (pion)")
    axis.plot(x, rejection, label="Background rejection (no-pion)")
    axis.set_xlabel("pion_score threshold")
    axis.set_ylabel("Rate")
    axis.set_title("Efficiency / rejection vs. threshold")
    axis.legend()
    fig.tight_layout()
    path = output.with_suffix(".efficiency_rejection_vs_threshold.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path.name


def pion_operating_point(
    y_true: np.ndarray, scores: np.ndarray, threshold: float
) -> Dict[str, object]:
    """Return pion efficiency and purity for score >= threshold."""
    predictions = scores >= threshold
    truth_pion = y_true.astype(bool)
    true_positive = int(np.count_nonzero(predictions & truth_pion))
    false_positive = int(np.count_nonzero(predictions & ~truth_pion))
    false_negative = int(np.count_nonzero(~predictions & truth_pion))
    true_negative = int(np.count_nonzero(~predictions & ~truth_pion))
    efficiency_denominator = true_positive + false_negative
    purity_denominator = true_positive + false_positive
    specificity_denominator = true_negative + false_positive
    total = true_positive + false_positive + false_negative + true_negative
    efficiency = (
        true_positive / efficiency_denominator
        if efficiency_denominator
        else 0.0
    )
    purity = true_positive / purity_denominator if purity_denominator else 0.0
    specificity = (
        true_negative / specificity_denominator
        if specificity_denominator
        else 0.0
    )
    f1_denominator = 2 * true_positive + false_positive + false_negative
    f1_score = 2 * true_positive / f1_denominator if f1_denominator else 0.0
    mcc_denominator = np.sqrt(
        (true_positive + false_positive)
        * (true_positive + false_negative)
        * (true_negative + false_positive)
        * (true_negative + false_negative)
    )
    matthews_correlation = (
        (true_positive * true_negative - false_positive * false_negative)
        / mcc_denominator
        if mcc_denominator
        else 0.0
    )
    return {
        "threshold": float(threshold),
        "efficiency": float(efficiency),
        "purity": float(purity),
        "efficiency_x_purity": float(efficiency * purity),
        "specificity": float(specificity),
        "balanced_accuracy": float((efficiency + specificity) / 2.0),
        "accuracy": float(
            (true_positive + true_negative) / total if total else 0.0
        ),
        "f1_score": float(f1_score),
        "matthews_correlation_coefficient": float(matthews_correlation),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
    }


def plot_efficiency_purity_vs_threshold(
    y_test: np.ndarray,
    test_scores: np.ndarray,
    output: Path,
    evaluation_threshold: float,
) -> Tuple[str, str, Dict[str, object], Dict[str, object]]:
    """Plot and tabulate pion efficiency, purity, and their product."""
    scan_thresholds = np.unique(
        np.concatenate(
            [
                np.linspace(0.0, 1.0, 201),
                np.array([evaluation_threshold, 0.20, 0.80], dtype=np.float64),
            ]
        )
    )
    operating_points = [
        pion_operating_point(y_test, test_scores, value)
        for value in scan_thresholds
    ]
    efficiencies = np.array(
        [point["efficiency"] for point in operating_points], dtype=np.float64
    )
    purities = np.array(
        [point["purity"] for point in operating_points], dtype=np.float64
    )
    products = np.array(
        [point["efficiency_x_purity"] for point in operating_points],
        dtype=np.float64,
    )

    csv_path = output.with_suffix(".efficiency_purity_vs_threshold.csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(operating_points[0]))
        writer.writeheader()
        writer.writerows(operating_points)

    fig, axis = plt.subplots(figsize=(7, 5))
    axis.plot(scan_thresholds, efficiencies, label="Pion efficiency")
    axis.plot(scan_thresholds, purities, label="Pion purity")
    axis.plot(scan_thresholds, products, label="Efficiency x purity")
    axis.axvline(
        0.20,
        color="gray",
        linestyle="--",
        alpha=0.7,
        label="non-pion boundary = 0.20",
    )
    axis.axvline(
        0.80,
        color="black",
        linestyle="--",
        alpha=0.7,
        label="pion boundary = 0.80",
    )
    if not (
        np.isclose(evaluation_threshold, 0.20)
        or np.isclose(evaluation_threshold, 0.80)
    ):
        axis.axvline(
            evaluation_threshold,
            color="gray",
            linestyle=":",
            alpha=0.8,
            label=f"configured score = {evaluation_threshold:.2f}",
        )
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.05)
    axis.set_xlabel("Minimum pion_score (pion if score >= threshold)")
    axis.set_ylabel("Fraction")
    axis.set_title("Pion efficiency and purity vs. score threshold")
    axis.legend()
    fig.tight_layout()
    plot_path = output.with_suffix(".efficiency_purity_vs_threshold.png")
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)

    return (
        plot_path.name,
        csv_path.name,
        pion_operating_point(y_test, test_scores, evaluation_threshold),
        pion_operating_point(y_test, test_scores, 0.80),
    )


def plot_confusion_matrix(
    y_test: np.ndarray,
    test_scores: np.ndarray,
    threshold: float,
    output: Path,
    *,
    normalize: bool = False,
    filename_suffix: str = ".confusion_matrix.png",
) -> str:
    predictions = (test_scores >= threshold).astype(int)
    rule_text = (
        f"non-pion-like if score < {threshold:.2f}; "
        f"pion-like if score >= {threshold:.2f}"
    )
    counts = confusion_matrix(y_test, predictions, labels=[0, 1])
    if normalize:
        denominators = counts.sum(axis=1, keepdims=True)
        matrix = np.divide(
            counts,
            denominators,
            out=np.zeros_like(counts, dtype=np.float64),
            where=denominators != 0,
        )
    else:
        matrix = counts
    fig, axis = plt.subplots(figsize=(5, 5))
    image = axis.imshow(
        matrix,
        cmap="Blues",
        vmin=0.0,
        vmax=1.0 if normalize else None,
    )
    axis.set_xticks([0, 1])
    axis.set_xticklabels(["no pion", "pion"])
    axis.set_yticks([0, 1])
    axis.set_yticklabels(["no pion", "pion"])
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Truth")
    title = (
        "True-class-normalized confusion matrix"
        if normalize
        else "Confusion matrix"
    )
    axis.set_title(f"{title}\n({rule_text})", fontsize=10)
    for row in range(2):
        for column in range(2):
            value = matrix[row, column]
            label = f"{value:.1%}" if normalize else str(int(value))
            axis.text(
                column,
                row,
                label,
                ha="center",
                va="center",
                color=(
                    "white"
                    if value > (0.5 if normalize else matrix.max() / 2)
                    else "black"
                ),
            )
    colorbar_label = "Fraction within truth class" if normalize else "Event count"
    fig.colorbar(image, ax=axis, label=colorbar_label)
    fig.tight_layout()
    path = output.with_suffix(filename_suffix)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path.name


def plot_score_band_confusion_matrix(
    y_test: np.ndarray,
    test_scores: np.ndarray,
    output: Path,
    *,
    normalize: bool = False,
    low_score: float = 0.20,
    high_score: float = 0.80,
    filename_tag: str = "",
) -> str:
    """Plot a matrix using only confident low- and high-score events."""
    selected = (test_scores < low_score) | (test_scores > high_score)
    selected_truth = y_test[selected]
    selected_predictions = (test_scores[selected] > high_score).astype(int)
    counts = confusion_matrix(
        selected_truth, selected_predictions, labels=[0, 1]
    )
    if normalize:
        denominators = counts.sum(axis=1, keepdims=True)
        matrix = np.divide(
            counts,
            denominators,
            out=np.zeros_like(counts, dtype=np.float64),
            where=denominators != 0,
        )
    else:
        matrix = counts

    fig, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(
        matrix,
        cmap="Blues",
        vmin=0.0,
        vmax=1.0 if normalize else None,
    )
    axis.set_xticks([0, 1])
    axis.set_xticklabels(
        [f"non-pion-like\nscore < {low_score:.2f}", f"pion-like\nscore > {high_score:.2f}"]
    )
    axis.set_yticks([0, 1])
    axis.set_yticklabels(["no pion", "pion"])
    axis.set_xlabel("Predicted score category")
    axis.set_ylabel("Truth")
    title = (
        "True-class-normalized confidence-band matrix"
        if normalize
        else "Confidence-band confusion matrix"
    )
    excluded_count = int(np.count_nonzero(~selected))
    axis.set_title(
        f"{title}\n{low_score:.2f} <= score <= {high_score:.2f} excluded "
        f"({excluded_count}/{len(test_scores)} test events)",
        fontsize=10,
    )
    for row in range(2):
        for column in range(2):
            value = matrix[row, column]
            label = f"{value:.1%}" if normalize else str(int(value))
            axis.text(
                column,
                row,
                label,
                ha="center",
                va="center",
                color=(
                    "white"
                    if value > (0.5 if normalize else matrix.max() / 2)
                    else "black"
                ),
            )
    colorbar_label = "Fraction within selected truth class" if normalize else "Event count"
    fig.colorbar(image, ax=axis, label=colorbar_label)
    fig.tight_layout()
    tag = f"_{filename_tag}" if filename_tag else ""
    normalized_tag = "_normalized" if normalize else ""
    suffix = f".confusion_matrix_score_bands{tag}{normalized_tag}.png"
    path = output.with_suffix(suffix)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path.name


def plot_calibration_curve(
    y_test: np.ndarray,
    test_scores: np.ndarray,
    output: Path,
    n_bins: int = 10,
) -> str:
    fraction_positive, mean_predicted = calibration_curve(
        y_test, test_scores, n_bins=n_bins
    )
    fig, axis = plt.subplots(figsize=(6, 6))
    axis.plot(
        mean_predicted,
        fraction_positive,
        marker="o",
        color="C3",
        label="Model",
    )
    axis.plot(
        [0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration"
    )
    axis.set_xlabel("Mean predicted pion_score in bin")
    axis.set_ylabel("Observed pion fraction in bin")
    axis.set_title("Calibration (reliability) curve")
    axis.legend()
    fig.tight_layout()
    path = output.with_suffix(".calibration_curve.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path.name


def plot_training_curves(history_path: Path, output: Path) -> str:
    with history_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return ""
    epochs = np.array([float(row["epoch"]) for row in rows])
    pairs = [
        ("loss", "val_loss"),
        ("roc_auc", "val_roc_auc"),
        ("pr_auc", "val_pr_auc"),
    ]
    pairs = [pair for pair in pairs if pair[0] in rows[0] and pair[1] in rows[0]]
    if not pairs:
        return ""
    fig, axes = plt.subplots(1, len(pairs), figsize=(5 * len(pairs), 4))
    axes = np.atleast_1d(axes)
    for axis, (train_key, validation_key) in zip(axes, pairs):
        axis.plot(epochs, [float(row[train_key]) for row in rows], label="train")
        axis.plot(
            epochs,
            [float(row[validation_key]) for row in rows],
            label="validation",
        )
        axis.set_xlabel("Epoch")
        axis.set_ylabel(train_key)
        axis.legend()
    fig.suptitle("Training history")
    fig.tight_layout()
    path = output.with_suffix(".training_curves.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path.name


def _format_event_features(
    feature_values: np.ndarray,
    track_starts: np.ndarray,
    track_properties: np.ndarray,
    track_mask: np.ndarray,
) -> Tuple[str, str]:
    """Format the MRD multiplicity and retained exact start positions."""
    num_mrd_tracks = feature_values[0]
    scalar_text = f"numMRDTracks={num_mrd_tracks:.0f}"
    valid_positions = track_starts[track_mask.astype(bool)]
    valid_properties = track_properties[track_mask.astype(bool)]
    if len(valid_positions) == 0:
        position_text = "MRD tracks: none"
    else:
        position_text = "MRD tracks: " + "; ".join(
            (
                f"{index}: start=({x:.3f}, {y:.3f}, {z:.3f}), "
                f"dE={energy_loss:.3f}, L={length:.3f}, angle={angle:.3f}"
            )
            for index, ((x, y, z), (energy_loss, length, angle)) in enumerate(
                zip(valid_positions, valid_properties)
            )
        )
        if num_mrd_tracks > MAX_MRD_TRACKS:
            position_text += f"; first {MAX_MRD_TRACKS} shown"
    return scalar_text, position_text


def plot_misclassified_gallery(
    images_test: np.ndarray,
    detector_images_test: np.ndarray,
    event_features_test: np.ndarray,
    mrd_track_starts_test: np.ndarray,
    mrd_track_properties_test: np.ndarray,
    mrd_track_mask_test: np.ndarray,
    y_test: np.ndarray,
    test_scores: np.ndarray,
    source_paths_test: np.ndarray,
    entries_test: np.ndarray,
    truth_pion_counts_test: np.ndarray,
    truth_muon_pion_angles_test: np.ndarray,
    truth_muon_kinetic_energy_test: np.ndarray,
    truth_pion_kinetic_energy_test: np.ndarray,
    output: Path,
    threshold: float,
    per_category: int = 3,
) -> str:
    """Plot the most confident false positives and false negatives."""
    categories = [
        (
            "Confident false positive (no-pion scored high)",
            (y_test == 0) & (test_scores >= threshold),
            True,
        ),
        (
            "Confident false negative (pion scored low)",
            (y_test == 1) & (test_scores < threshold),
            False,
        ),
    ]
    rows: List[Tuple[str, np.ndarray]] = []
    for title, mask, descending in categories:
        indices = np.flatnonzero(mask)
        if len(indices) == 0:
            continue
        order = indices[np.argsort(test_scores[indices])]
        if descending:
            order = order[::-1]
        rows.append((title, order[:per_category]))
    if not rows:
        return ""

    row_count = sum(len(order) for _, order in rows)
    fig, axes = plt.subplots(
        row_count, 2, figsize=(9, 3.6 * row_count), squeeze=False
    )
    row_index = 0
    for title, order in rows:
        for event_index in order:
            angular = images_test[event_index][:, 1:-1, 0]
            detector = detector_images_test[event_index][:, :, 0]
            source_name = Path(str(source_paths_test[event_index])).name
            true_label = "pion" if y_test[event_index] == 1 else "non-pion"
            predicted_label = (
                "pion" if test_scores[event_index] >= threshold else "non-pion"
            )
            pi_plus, pi_minus, pi_zero = (
                int(value) for value in truth_pion_counts_test[event_index]
            )
            opening_angle = truth_muon_pion_angles_test[event_index]
            angle_text = (
                f"θ(π,μ)={opening_angle:.1f}°"
                if np.isfinite(opening_angle)
                else "θ(π,μ)=n/a"
            )
            muon_kinetic_energy = truth_muon_kinetic_energy_test[event_index]
            pion_kinetic_energy = truth_pion_kinetic_energy_test[event_index]
            muon_energy_text = (
                f"Tμ={muon_kinetic_energy:.3f} GeV"
                if np.isfinite(muon_kinetic_energy)
                else "Tμ=n/a"
            )
            pion_energy_text = (
                f"Tπ={pion_kinetic_energy:.3f} GeV"
                if np.isfinite(pion_kinetic_energy)
                else "Tπ=n/a"
            )
            scalar_text, position_text = _format_event_features(
                event_features_test[event_index],
                mrd_track_starts_test[event_index],
                mrd_track_properties_test[event_index],
                mrd_track_mask_test[event_index],
            )
            caption = (
                f"{title}\n{source_name} — entry {int(entries_test[event_index])}\n"
                f"truth={true_label}, prediction={predicted_label}, "
                f"score={test_scores[event_index]:.3f}\n"
                f"π⁺={pi_plus}, π⁻={pi_minus}, π⁰={pi_zero}; {angle_text}\n"
                f"{scalar_text}\n{position_text}\n"
                f"truth {muon_energy_text}, {pion_energy_text}"
            )
            axes[row_index, 0].imshow(
                angular, origin="lower", aspect="auto", cmap="magma"
            )
            axes[row_index, 0].set_title(caption, fontsize=8)
            axes[row_index, 1].imshow(
                detector, origin="lower", aspect="equal", cmap="magma"
            )
            axes[row_index, 1].set_title("Unfolded view", fontsize=8)
            row_index += 1
    fig.tight_layout()
    path = output.with_suffix(".misclassified_gallery.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path.name


def plot_misclassified_event_pdf(
    images_test: np.ndarray,
    detector_images_test: np.ndarray,
    event_features_test: np.ndarray,
    mrd_track_starts_test: np.ndarray,
    mrd_track_properties_test: np.ndarray,
    mrd_track_mask_test: np.ndarray,
    y_test: np.ndarray,
    test_scores: np.ndarray,
    source_paths_test: np.ndarray,
    entries_test: np.ndarray,
    truth_pion_counts_test: np.ndarray,
    truth_muon_pion_angles_test: np.ndarray,
    truth_muon_kinetic_energy_test: np.ndarray,
    truth_pion_kinetic_energy_test: np.ndarray,
    output: Path,
    threshold: float,
    max_examples: int = 20,
) -> str:
    """Write up to 20 annotated misclassified events to one PDF."""
    false_positive_indices = np.flatnonzero(
        (y_test == 0) & (test_scores >= threshold)
    )
    false_negative_indices = np.flatnonzero(
        (y_test == 1) & (test_scores < threshold)
    )
    false_positive_indices = false_positive_indices[
        np.argsort(test_scores[false_positive_indices])[::-1]
    ]
    false_negative_indices = false_negative_indices[
        np.argsort(test_scores[false_negative_indices])
    ]

    per_class_target = max_examples // 2
    selected: List[Tuple[str, int]] = [
        ("false_positive", int(index))
        for index in false_positive_indices[:per_class_target]
    ]
    selected.extend(
        ("false_negative", int(index))
        for index in false_negative_indices[:per_class_target]
    )
    remaining = [
        ("false_positive", int(index))
        for index in false_positive_indices[per_class_target:]
    ]
    remaining.extend(
        ("false_negative", int(index))
        for index in false_negative_indices[per_class_target:]
    )
    remaining.sort(
        key=lambda item: abs(float(test_scores[item[1]]) - threshold),
        reverse=True,
    )
    selected.extend(remaining[: max_examples - len(selected)])

    pdf_path = output.with_suffix(".misclassified_events.pdf")
    with PdfPages(
        pdf_path,
        metadata={
            "Title": "ANNIE pion-ID misclassified events",
            "Subject": "Annotated false-positive and false-negative PMT images",
            "Author": "train_annie_ring.py",
        },
    ) as pdf:
        if not selected:
            fig, axis = plt.subplots(figsize=(11, 5))
            axis.axis("off")
            axis.text(
                0.5,
                0.5,
                f"No events were misclassified at pion-score threshold {threshold:.3f}.",
                ha="center",
                va="center",
                fontsize=14,
            )
            fig.suptitle("ANNIE pion-ID misclassification review")
            pdf.savefig(fig)
            plt.close(fig)

        for category, event_index in selected:
            source_name = Path(str(source_paths_test[event_index])).name
            true_label = "pion" if y_test[event_index] == 1 else "non-pion"
            predicted_label = (
                "pion" if test_scores[event_index] >= threshold else "non-pion"
            )
            pi_plus, pi_minus, pi_zero = (
                int(value) for value in truth_pion_counts_test[event_index]
            )
            opening_angle = truth_muon_pion_angles_test[event_index]
            angle_text = (
                f"θ(π,μ)={opening_angle:.1f}°"
                if np.isfinite(opening_angle)
                else "θ(π,μ)=n/a"
            )
            muon_kinetic_energy = truth_muon_kinetic_energy_test[event_index]
            pion_kinetic_energy = truth_pion_kinetic_energy_test[event_index]
            muon_energy_text = (
                f"Tμ={muon_kinetic_energy:.3f} GeV"
                if np.isfinite(muon_kinetic_energy)
                else "Tμ=n/a"
            )
            pion_energy_text = (
                f"Tπ={pion_kinetic_energy:.3f} GeV"
                if np.isfinite(pion_kinetic_energy)
                else "Tπ=n/a"
            )
            scalar_text, position_text = _format_event_features(
                event_features_test[event_index],
                mrd_track_starts_test[event_index],
                mrd_track_properties_test[event_index],
                mrd_track_mask_test[event_index],
            )
            category_title = category.replace("_", " ").title()
            annotation = (
                f"{category_title}: truth={true_label}, prediction={predicted_label}, "
                f"score={test_scores[event_index]:.3f}\n"
                f"{source_name} - entry {int(entries_test[event_index])}\n"
                f"π⁺={pi_plus}, π⁻={pi_minus}, π⁰={pi_zero}; {angle_text}; "
                f"{scalar_text}\n{position_text}\n"
                f"truth {muon_energy_text}, {pion_energy_text}"
            )

            fig, axes = plt.subplots(1, 2, figsize=(11, 5))
            axes[0].imshow(
                images_test[event_index][:, 1:-1, 0],
                origin="lower",
                aspect="auto",
                cmap="magma",
            )
            axes[0].set_title("Angular full hitPE")
            axes[1].imshow(
                detector_images_test[event_index][:, :, 0],
                origin="lower",
                aspect="equal",
                cmap="magma",
            )
            axes[1].set_title("Unfolded full hitPE")
            fig.suptitle(annotation, fontsize=10)
            fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.88))
            pdf.savefig(fig)
            plt.close(fig)
    return pdf_path.name


def plot_pion_example_pages(
    model: tf.keras.Model,
    images_test: np.ndarray,
    detector_images_test: np.ndarray,
    event_features_test: np.ndarray,
    mrd_track_starts_test: np.ndarray,
    mrd_track_properties_test: np.ndarray,
    mrd_track_mask_test: np.ndarray,
    y_test: np.ndarray,
    test_scores: np.ndarray,
    source_paths_test: np.ndarray,
    entries_test: np.ndarray,
    truth_pion_counts_test: np.ndarray,
    truth_muon_pion_angles_test: np.ndarray,
    truth_muon_kinetic_energy_test: np.ndarray,
    truth_pion_kinetic_energy_test: np.ndarray,
    output: Path,
    threshold: float,
    examples_per_page: int = 3,
) -> List[str]:
    """Draw charged-, high-, boundary-, and low-score truth-pion pages."""
    pion_indices = np.flatnonzero(y_test == 1)
    if len(pion_indices) == 0:
        return []

    charged_pion_indices = pion_indices[
        (truth_pion_counts_test[pion_indices, 0]
         + truth_pion_counts_test[pion_indices, 1]) > 0
    ]
    ranking_options = [
        (
            "charged_pions",
            "Held-out truth charged-pion events",
            charged_pion_indices[
                np.argsort(test_scores[charged_pion_indices])[::-1]
            ],
        ),
        (
            "high_score",
            "Highest-scoring held-out truth-pion events",
            pion_indices[np.argsort(test_scores[pion_indices])[::-1]],
        ),
        (
            "boundary_score",
            f"Truth-pion events closest to score threshold {threshold:.2f}",
            pion_indices[
                np.argsort(np.abs(test_scores[pion_indices] - threshold))
            ],
        ),
        (
            "low_score",
            "Lowest-scoring held-out truth-pion events",
            pion_indices[np.argsort(test_scores[pion_indices])],
        ),
    ]
    raw_column_titles = [
        "Angular full hitPE",
        "Angular tank-cluster hitPE",
        "Unfolded full hitPE",
        "Unfolded tank-cluster hitPE",
    ]
    used_indices = set()
    output_names = []
    for page_tag, page_title, candidates in ranking_options:
        order = []
        for event_index in candidates:
            integer_index = int(event_index)
            if integer_index in used_indices:
                continue
            order.append(integer_index)
            used_indices.add(integer_index)
            if len(order) == examples_per_page:
                break
        if not order:
            continue

        angular_batch = images_test[order]
        detector_batch = detector_images_test[order]
        model_inputs = {
            "pmt_angular_image": angular_batch,
            "pmt_unfolded_image": detector_batch,
            "event_features": event_features_test[order],
            "mrd_track_starts": mrd_track_starts_test[order],
            "mrd_track_properties": mrd_track_properties_test[order],
            "mrd_track_mask": mrd_track_mask_test[order],
        }
        angular_heatmaps = _grad_cam_heatmaps(
            model,
            "angular_conv3",
            model_inputs,
            (angular_batch.shape[1], angular_batch.shape[2]),
        )
        detector_heatmaps = _grad_cam_heatmaps(
            model,
            "detector_conv3",
            model_inputs,
            (detector_batch.shape[1], detector_batch.shape[2]),
        )

        fig, axes = plt.subplots(
            len(order), 6, figsize=(23, 3.8 * len(order)), squeeze=False
        )
        for row_index, event_index in enumerate(order):
            angular = angular_batch[row_index][:, 1:-1, :]
            detector = detector_batch[row_index]
            panels = [
                (angular[:, :, 0], "auto"),
                (angular[:, :, 1], "auto"),
                (detector[:, :, 0], "equal"),
                (detector[:, :, 1], "equal"),
            ]
            for column, ((panel, aspect), title) in enumerate(
                zip(panels, raw_column_titles)
            ):
                axes[row_index, column].imshow(
                    panel, origin="lower", aspect=aspect, cmap="magma"
                )
                axes[row_index, column].set_title(title, fontsize=9)

            angular_heat = angular_heatmaps[row_index][:, 1:-1]
            axes[row_index, 4].imshow(
                angular[:, :, 0], origin="lower", aspect="auto", cmap="gray"
            )
            axes[row_index, 4].imshow(
                angular_heat,
                origin="lower",
                aspect="auto",
                cmap="jet",
                alpha=0.55,
                vmin=0.0,
                vmax=1.0,
            )
            axes[row_index, 4].set_title(
                "Angular Grad-CAM\n(red = strongest influence)", fontsize=9
            )

            axes[row_index, 5].imshow(
                detector[:, :, 0], origin="lower", aspect="equal", cmap="gray"
            )
            axes[row_index, 5].imshow(
                detector_heatmaps[row_index],
                origin="lower",
                aspect="equal",
                cmap="jet",
                alpha=0.55,
                vmin=0.0,
                vmax=1.0,
            )
            axes[row_index, 5].set_title(
                "Unfolded Grad-CAM\n(red = strongest influence)", fontsize=9
            )
            pi_plus, pi_minus, pi_zero = (
                int(value) for value in truth_pion_counts_test[event_index]
            )
            source_name = Path(str(source_paths_test[event_index])).name
            opening_angle = truth_muon_pion_angles_test[event_index]
            angle_text = (
                f"truth leading π–μ angle={opening_angle:.1f}°"
                if np.isfinite(opening_angle)
                else "truth leading π–μ angle=n/a"
            )
            muon_kinetic_energy = truth_muon_kinetic_energy_test[event_index]
            pion_kinetic_energy = truth_pion_kinetic_energy_test[event_index]
            muon_energy_text = (
                f"Tμ={muon_kinetic_energy:.3f} GeV"
                if np.isfinite(muon_kinetic_energy)
                else "Tμ=n/a"
            )
            pion_energy_text = (
                f"Tπ={pion_kinetic_energy:.3f} GeV"
                if np.isfinite(pion_kinetic_energy)
                else "Tπ=n/a"
            )
            scalar_text, position_text = _format_event_features(
                event_features_test[event_index],
                mrd_track_starts_test[event_index],
                mrd_track_properties_test[event_index],
                mrd_track_mask_test[event_index],
            )
            axes[row_index, 0].set_ylabel(
                f"{source_name}\nentry {int(entries_test[event_index])}\n"
                f"score={test_scores[event_index]:.3f}\n"
                f"π⁺={pi_plus}, π⁻={pi_minus}, π⁰={pi_zero}\n{angle_text}\n"
                f"{scalar_text}\n{position_text}\n"
                f"truth {muon_energy_text}, {pion_energy_text}",
                fontsize=8,
            )
        fig.suptitle(
            f"{page_title} (truth information is annotation only)",
            fontsize=12,
        )
        fig.tight_layout()
        path = output.with_suffix(f".pion_examples_{page_tag}.png")
        fig.savefig(path, dpi=160)
        plt.close(fig)
        output_names.append(path.name)
    return output_names


def _grad_cam_heatmaps(
    model: tf.keras.Model,
    conv_layer_name: str,
    inputs: Dict[str, np.ndarray],
    target_shape: Tuple[int, int],
) -> np.ndarray:
    """Calculate batched Grad-CAM maps for one convolutional tower."""
    grad_model = tf.keras.Model(
        model.inputs, [model.get_layer(conv_layer_name).output, model.output]
    )
    tensors = {name: tf.convert_to_tensor(value) for name, value in inputs.items()}
    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(tensors, training=False)
        loss = predictions[:, 0]
    gradients = tape.gradient(loss, conv_output)
    pooled_gradients = tf.reduce_mean(gradients, axis=(1, 2))
    heatmaps = tf.nn.relu(
        tf.einsum("bhwc,bc->bhw", conv_output, pooled_gradients)
    )
    heatmaps = tf.image.resize(heatmaps[..., tf.newaxis], target_shape)[..., 0]
    heatmaps = heatmaps.numpy()
    peaks = heatmaps.reshape(len(heatmaps), -1).max(axis=1)
    peaks[peaks == 0] = 1.0
    return heatmaps / peaks[:, None, None]


def plot_gradcam_examples(
    model: tf.keras.Model,
    images_test: np.ndarray,
    detector_images_test: np.ndarray,
    event_features_test: np.ndarray,
    mrd_track_starts_test: np.ndarray,
    mrd_track_properties_test: np.ndarray,
    mrd_track_mask_test: np.ndarray,
    y_test: np.ndarray,
    test_scores: np.ndarray,
    output: Path,
    threshold: float,
) -> str:
    """Plot a confident TP, TN, FP, and FN with Grad-CAM overlays."""
    predictions = (test_scores >= threshold).astype(int)
    categories = {
        "True positive": (y_test == 1) & (predictions == 1),
        "True negative": (y_test == 0) & (predictions == 0),
        "False positive": (y_test == 0) & (predictions == 1),
        "False negative": (y_test == 1) & (predictions == 0),
    }
    selected: List[Tuple[str, int]] = []
    for title, mask in categories.items():
        indices = np.flatnonzero(mask)
        if len(indices) == 0:
            continue
        confidence = np.abs(test_scores[indices] - threshold)
        selected.append((title, int(indices[np.argmax(confidence)])))
    if not selected:
        return ""

    selected_indices = [index for _, index in selected]
    angular_batch = images_test[selected_indices]
    detector_batch = detector_images_test[selected_indices]
    model_inputs = {
        "pmt_angular_image": angular_batch,
        "pmt_unfolded_image": detector_batch,
        "event_features": event_features_test[selected_indices],
        "mrd_track_starts": mrd_track_starts_test[selected_indices],
        "mrd_track_properties": mrd_track_properties_test[selected_indices],
        "mrd_track_mask": mrd_track_mask_test[selected_indices],
    }
    angular_heatmaps = _grad_cam_heatmaps(
        model,
        "angular_conv3",
        model_inputs,
        (angular_batch.shape[1], angular_batch.shape[2]),
    )
    detector_heatmaps = _grad_cam_heatmaps(
        model,
        "detector_conv3",
        model_inputs,
        (detector_batch.shape[1], detector_batch.shape[2]),
    )

    fig, axes = plt.subplots(
        len(selected), 2, figsize=(9, 3.6 * len(selected)), squeeze=False
    )
    for row_index, (title, event_index) in enumerate(selected):
        angular_visible = angular_batch[row_index][:, 1:-1, 0]
        heat_visible = angular_heatmaps[row_index][:, 1:-1]
        axes[row_index, 0].imshow(
            angular_visible, origin="lower", aspect="auto", cmap="gray"
        )
        axes[row_index, 0].imshow(
            heat_visible, origin="lower", aspect="auto", cmap="jet", alpha=0.5
        )
        axes[row_index, 0].set_title(
            f"{title} (score={test_scores[event_index]:.3f}) — angular Grad-CAM",
            fontsize=8,
        )
        axes[row_index, 1].imshow(
            detector_batch[row_index][:, :, 0],
            origin="lower",
            aspect="equal",
            cmap="gray",
        )
        axes[row_index, 1].imshow(
            detector_heatmaps[row_index],
            origin="lower",
            aspect="equal",
            cmap="jet",
            alpha=0.5,
        )
        axes[row_index, 1].set_title("Unfolded Grad-CAM", fontsize=8)
    fig.tight_layout()
    path = output.with_suffix(".gradcam_examples.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path.name


def evaluate_and_plot(
    model: tf.keras.Model,
    images_test: np.ndarray,
    detector_images_test: np.ndarray,
    event_features_test: np.ndarray,
    mrd_track_starts_test: np.ndarray,
    mrd_track_properties_test: np.ndarray,
    mrd_track_mask_test: np.ndarray,
    y_test: np.ndarray,
    source_paths_test: np.ndarray,
    entries_test: np.ndarray,
    truth_pion_counts_test: np.ndarray,
    history_path: Path,
    output: Path,
    threshold: float,
    tree_name: str,
    opening_angle_degree_branch: str = "",
) -> Dict[str, object]:
    test_scores = model.predict(
        {
            "pmt_angular_image": images_test,
            "pmt_unfolded_image": detector_images_test,
            "event_features": event_features_test,
            "mrd_track_starts": mrd_track_starts_test,
            "mrd_track_properties": mrd_track_properties_test,
            "mrd_track_mask": mrd_track_mask_test,
        },
        verbose=0,
    ).reshape(-1)
    (
        truth_muon_pion_angles,
        truth_muon_momenta,
        truth_pion_momenta,
        truth_kinematics_source,
    ) = load_truth_muon_pion_kinematics(
        source_paths_test,
        entries_test,
        tree_name,
        opening_angle_degree_branch,
    )
    truth_muon_kinetic_energy = _kinetic_energy(
        truth_muon_momenta, MUON_MASS_GEV
    )
    charged_pion_present = (
        truth_pion_counts_test[:, 0] + truth_pion_counts_test[:, 1]
    ) > 0
    neutral_pion_present = truth_pion_counts_test[:, 2] > 0
    pion_mass = np.where(
        charged_pion_present,
        CHARGED_PION_MASS_GEV,
        np.where(neutral_pion_present, NEUTRAL_PION_MASS_GEV, np.nan),
    )
    truth_pion_kinetic_energy = _kinetic_energy(truth_pion_momenta, pion_mass)

    predictions_path = output.with_suffix(".test_predictions.csv")
    track_column_names = [
        name
        for track_index in range(MAX_MRD_TRACKS)
        for name in (
            f"MRDTrackStartX_{track_index}",
            f"MRDTrackStartY_{track_index}",
            f"MRDTrackStartZ_{track_index}",
            f"MRDEnergyLoss_{track_index}",
            f"MRDTrackLength_{track_index}",
            f"MRDTrackAngle_{track_index}",
            f"MRDTrackValid_{track_index}",
        )
    ]
    with predictions_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "source_file",
                "tree_entry",
                "true_label",
                "truePiPlusCher",
                "truePiMinusCher",
                "truePi0",
                "truth_leading_pion_muon_opening_angle_deg",
                "truth_muon_kinetic_energy_GeV",
                "truth_leading_pion_kinetic_energy_GeV",
                *RING_EVENT_FEATURE_NAMES,
                *track_column_names,
                "pion_score",
            ]
        )
        for event_index in range(len(y_test)):
            track_values = []
            for track_index in range(MAX_MRD_TRACKS):
                track_values.extend(
                    [
                        *mrd_track_starts_test[event_index, track_index].astype(
                            float
                        ),
                        *mrd_track_properties_test[
                            event_index, track_index
                        ].astype(float),
                        int(mrd_track_mask_test[event_index, track_index] != 0),
                    ]
                )
            writer.writerow(
                [
                    source_paths_test[event_index],
                    int(entries_test[event_index]),
                    int(y_test[event_index]),
                    *truth_pion_counts_test[event_index].astype(int),
                    float(truth_muon_pion_angles[event_index]),
                    float(truth_muon_kinetic_energy[event_index]),
                    float(truth_pion_kinetic_energy[event_index]),
                    *event_features_test[event_index].astype(float),
                    *track_values,
                    float(test_scores[event_index]),
                ]
            )

    roc_name, roc_auc_value, fpr, tpr, thresholds = plot_roc_curve(
        y_test, test_scores, output
    )
    pr_name, average_precision = plot_precision_recall(
        y_test, test_scores, output
    )
    (
        efficiency_purity_plot,
        efficiency_purity_csv,
        configured_operating_point,
        threshold_0p80_operating_point,
    ) = plot_efficiency_purity_vs_threshold(
        y_test, test_scores, output, threshold
    )
    pion_example_pages = plot_pion_example_pages(
        model,
        images_test,
        detector_images_test,
        event_features_test,
        mrd_track_starts_test,
        mrd_track_properties_test,
        mrd_track_mask_test,
        y_test,
        test_scores,
        source_paths_test,
        entries_test,
        truth_pion_counts_test,
        truth_muon_pion_angles,
        truth_muon_kinetic_energy,
        truth_pion_kinetic_energy,
        output,
        threshold,
    )
    return {
        "test_predictions": predictions_path.name,
        "truth_muon_pion_opening_angle_source": truth_kinematics_source,
        "truth_muon_pion_kinematics_source": truth_kinematics_source,
        "roc_curve": roc_name,
        "roc_auc_sklearn": roc_auc_value,
        "precision_recall_curve": pr_name,
        "average_precision": average_precision,
        "brier_score": float(brier_score_loss(y_test, test_scores)),
        "score_distribution": plot_score_distribution(
            y_test, test_scores, output
        ),
        "efficiency_rejection_vs_threshold": (
            plot_efficiency_rejection_vs_threshold(
                fpr, tpr, thresholds, output
            )
        ),
        "efficiency_purity_vs_threshold": efficiency_purity_plot,
        "efficiency_purity_vs_threshold_csv": efficiency_purity_csv,
        "pion_metrics_at_configured_threshold": configured_operating_point,
        "pion_metrics_at_threshold_0p80": threshold_0p80_operating_point,
        "confusion_matrix": plot_confusion_matrix(
            y_test, test_scores, threshold, output
        ),
        "confusion_matrix_normalized": plot_confusion_matrix(
            y_test,
            test_scores,
            threshold,
            output,
            normalize=True,
            filename_suffix=".confusion_matrix_normalized.png",
        ),
        "confusion_matrix_score_bands": plot_score_band_confusion_matrix(
            y_test, test_scores, output
        ),
        "confusion_matrix_score_bands_normalized": (
            plot_score_band_confusion_matrix(
                y_test, test_scores, output, normalize=True
            )
        ),
        "confusion_matrix_score_bands_0p30_0p70": (
            plot_score_band_confusion_matrix(
                y_test,
                test_scores,
                output,
                low_score=0.30,
                high_score=0.70,
                filename_tag="0p30_0p70",
            )
        ),
        "confusion_matrix_score_bands_0p30_0p70_normalized": (
            plot_score_band_confusion_matrix(
                y_test,
                test_scores,
                output,
                normalize=True,
                low_score=0.30,
                high_score=0.70,
                filename_tag="0p30_0p70",
            )
        ),
        "calibration_curve": plot_calibration_curve(y_test, test_scores, output),
        "training_curves": plot_training_curves(history_path, output),
        "misclassified_gallery": plot_misclassified_gallery(
            images_test,
            detector_images_test,
            event_features_test,
            mrd_track_starts_test,
            mrd_track_properties_test,
            mrd_track_mask_test,
            y_test,
            test_scores,
            source_paths_test,
            entries_test,
            truth_pion_counts_test,
            truth_muon_pion_angles,
            truth_muon_kinetic_energy,
            truth_pion_kinetic_energy,
            output,
            threshold,
        ),
        "misclassified_events_pdf": plot_misclassified_event_pdf(
            images_test,
            detector_images_test,
            event_features_test,
            mrd_track_starts_test,
            mrd_track_properties_test,
            mrd_track_mask_test,
            y_test,
            test_scores,
            source_paths_test,
            entries_test,
            truth_pion_counts_test,
            truth_muon_pion_angles,
            truth_muon_kinetic_energy,
            truth_pion_kinetic_energy,
            output,
            threshold,
        ),
        "pion_example_pages": pion_example_pages,
        "gradcam_examples": plot_gradcam_examples(
            model,
            images_test,
            detector_images_test,
            event_features_test,
            mrd_track_starts_test,
            mrd_track_properties_test,
            mrd_track_mask_test,
            y_test,
            test_scores,
            output,
            threshold,
        ),
    }


def main() -> None:
    args = parse_args()
    if args.image_height < 4 or args.image_width < 8:
        raise ValueError("The PMT image must be at least 4 x 8 bins")
    if args.detector_height < 24 or args.detector_width < 16:
        raise ValueError("The unfolded detector image must be at least 24 x 16 bins")
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    (
        images,
        detector_images,
        event_features,
        mrd_track_starts,
        mrd_track_properties,
        mrd_track_mask,
        labels,
        source_paths,
        entries,
        truth_pion_counts,
        tank_branch,
        n_misaligned,
        n_mrd_tracks_truncated,
        geometry,
        response,
    ) = load_data(args)
    if n_mrd_tracks_truncated:
        print(
            f"Retained the first {MAX_MRD_TRACKS} MRD starts for "
            f"{n_mrd_tracks_truncated} selected events with more tracks."
        )
    counts = np.bincount(labels.astype(np.int64), minlength=2)
    if np.any(counts == 0):
        raise ValueError(f"Both classes are required; class counts are {counts.tolist()}")
    indices = np.arange(len(labels))
    train_indices, temp_indices = train_test_split(
        indices,
        test_size=VALIDATION_FRACTION + TEST_FRACTION,
        random_state=args.seed,
        stratify=labels,
    )
    val_indices, test_indices = train_test_split(
        temp_indices,
        test_size=TEST_FRACTION / (VALIDATION_FRACTION + TEST_FRACTION),
        random_state=args.seed,
        stratify=labels[temp_indices],
    )
    y_train = labels[train_indices]
    y_val = labels[val_indices]
    y_test = labels[test_indices]
    train_counts = np.bincount(y_train.astype(np.int64), minlength=2)
    class_weight = {
        0: len(y_train) / (2.0 * train_counts[0]),
        1: len(y_train) / (2.0 * train_counts[1]),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    history_path = args.output.with_suffix(".history.csv")
    model = build_model(
        tuple(images.shape[1:]),
        tuple(detector_images.shape[1:]),
        images[train_indices],
        detector_images[train_indices],
        event_features[train_indices],
        mrd_track_starts[train_indices],
        mrd_track_properties[train_indices],
        mrd_track_mask[train_indices],
    )
    model.fit(
        {
            "pmt_angular_image": images[train_indices],
            "pmt_unfolded_image": detector_images[train_indices],
            "event_features": event_features[train_indices],
            "mrd_track_starts": mrd_track_starts[train_indices],
            "mrd_track_properties": mrd_track_properties[train_indices],
            "mrd_track_mask": mrd_track_mask[train_indices],
        },
        y_train,
        validation_data=(
            {
                "pmt_angular_image": images[val_indices],
                "pmt_unfolded_image": detector_images[val_indices],
                "event_features": event_features[val_indices],
                "mrd_track_starts": mrd_track_starts[val_indices],
                "mrd_track_properties": mrd_track_properties[val_indices],
                "mrd_track_mask": mrd_track_mask[val_indices],
            },
            y_val,
        ),
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight=class_weight,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_pr_auc", mode="max", patience=12, restore_best_weights=True
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
    metrics = model.evaluate(
        {
            "pmt_angular_image": images[test_indices],
            "pmt_unfolded_image": detector_images[test_indices],
            "event_features": event_features[test_indices],
            "mrd_track_starts": mrd_track_starts[test_indices],
            "mrd_track_properties": mrd_track_properties[test_indices],
            "mrd_track_mask": mrd_track_mask[test_indices],
        },
        y_test,
        return_dict=True,
        verbose=0,
    )
    print("Test metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.5f}")

    model.save(args.output)
    evaluation_files = evaluate_and_plot(
        model,
        images[test_indices],
        detector_images[test_indices],
        event_features[test_indices],
        mrd_track_starts[test_indices],
        mrd_track_properties[test_indices],
        mrd_track_mask[test_indices],
        y_test,
        source_paths[test_indices],
        entries[test_indices],
        truth_pion_counts[test_indices],
        history_path,
        args.output,
        args.threshold,
        args.tree,
        args.muon_pion_opening_angle_deg_branch or "",
    )
    print(
        "Test ROC AUC cross-check: "
        f"{evaluation_files['roc_auc_sklearn']:.5f}; "
        f"average precision: {evaluation_files['average_precision']:.5f}"
    )
    metadata = {
        "model_type": "annie_pion_ring_cnn",
        "model_variant": "B_PE_dual_view_plus_mrd_track_set",
        "tensorflow_version": tf.__version__,
        "training_sources": [str(path) for path in args.root_files],
        "tree": args.tree,
        "geometry_file": args.geometry.name,
        "pmt_mask": args.pmt_mask,
        "pmt_count_included": geometry.included_count,
        "excluded_pmt_ids": geometry.excluded_ids,
        "training_pmt_response": response.mode,
        "pmt_response_scope": (
            "separate_branch_kind_0_hitPE_and_branch_kind_1_tankcluster"
            if response.payload_format == "final_pmt_tuning_parameters"
            else "shared_legacy_map_hitPE_and_hitPE_tankcluster"
        ),
        "pmt_response_calibration_file": (
            response.calibration_path.name if response.calibration_path else None
        ),
        "pmt_response_calibration_sha256": response.calibration_sha256,
        "pmt_response_payload_format": response.payload_format,
        "pmt_tune_variant": response.tune_variant,
        "pmt_tune_random_stream_version": response.random_stream_version,
        "pmt_response_mapped_pmt_count": response.mapped_pmt_count,
        "hitpe_branch": args.hitpe_branch,
        "hitid_branch": args.hitid_branch,
        "tankcluster_branch": tank_branch,
        "tankcluster_id_branch": args.tankcluster_id_branch,
        "image_height": args.image_height,
        "image_width": args.image_width,
        "detector_height": args.detector_height,
        "detector_width": args.detector_width,
        "channels": [
            (
                "log1p_tuned_hitPE"
                if response.mode == "tuned"
                else "log1p_hitPE"
            ),
            (
                "log1p_tuned_hitPE_tankcluster"
                if response.mode == "tuned"
                else "log1p_hitPE_tankcluster"
            ),
        ],
        "projections": [
            "vertex_centered_equirectangular_with_azimuth_wrap",
            "annotation_free_unfolded_barrel_top_bottom",
        ],
        "event_feature_input_name": "event_features",
        "event_feature_names": RING_EVENT_FEATURE_NAMES,
        "event_feature_scalar_branches": RING_EVENT_SCALAR_BRANCHES,
        "mrd_track_start_input_name": "mrd_track_starts",
        "mrd_track_property_input_name": "mrd_track_properties",
        "mrd_track_mask_input_name": "mrd_track_mask",
        "mrd_track_start_branches": MRD_TRACK_START_BRANCHES,
        "mrd_track_property_branches": MRD_TRACK_PROPERTY_BRANCHES,
        "mrd_track_coordinate_order": ["X", "Y", "Z"],
        "mrd_track_property_order": MRD_TRACK_PROPERTY_BRANCHES,
        "max_mrd_tracks": MAX_MRD_TRACKS,
        "mrd_track_overflow_policy": "first_four_in_stored_branch_order",
        "mrd_track_overflow_events": n_mrd_tracks_truncated,
        "event_feature_preprocessing": (
            "numMRDTracks_scalar_plus_zero_padded_mrd_xyz_and_energy_loss_length_"
            "angle_track_set_and_mask;_separate_normalization_adapted_on_valid_"
            "training_tracks_only;_shared_dense_encoder_plus_masked_"
            "permutation_invariant_max_pooling"
        ),
        "label": "charged_pion_present" if args.charged_only else "any_pion_present",
        "truth_branches": ["truePiPlusCher", "truePiMinusCher", "truePi0"],
        "truth_muon_pion_opening_angle_annotation_only": True,
        "truth_muon_pion_kinetic_energy_annotation_only": True,
        "truth_momentum_units": "GeV/c",
        "truth_kinetic_energy_units": "GeV",
        "truth_kinetic_energy_formula": "sqrt(p^2 + m^2) - m",
        "truth_particle_masses_GeV": {
            "muon": MUON_MASS_GEV,
            "charged_pion": CHARGED_PION_MASS_GEV,
            "neutral_pion": NEUTRAL_PION_MASS_GEV,
        },
        "requested_muon_pion_opening_angle_degree_branch": (
            args.muon_pion_opening_angle_deg_branch
        ),
        "event_selection": "fit_individual_pmt",
        "event_selection_source": "Fit_indivdiualPMT_Gaussian_Convolution.cpp",
        "event_selection_expression": FIT_INDIVIDUAL_PMT_SELECTION_EXPRESSION,
        "event_selection_applied": not args.no_event_cuts,
        "bdt_preselection": not args.no_event_cuts,
        "threshold": args.threshold,
        "data_split": {
            "training_fraction": TRAIN_FRACTION,
            "validation_fraction": VALIDATION_FRACTION,
            "testing_fraction": TEST_FRACTION,
            "training_events": int(len(train_indices)),
            "validation_events": int(len(val_indices)),
            "testing_events": int(len(test_indices)),
            "stratified_by_truth_label": True,
            "random_seed": args.seed,
        },
        "class_counts": {"no_pion": int(counts[0]), "pion": int(counts[1])},
        "misaligned_events_rejected": n_misaligned,
        "test_metrics": {name: float(value) for name, value in metrics.items()},
        "training_history": history_path.name,
        "evaluation": evaluation_files,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Saved {args.output}, {args.output.with_suffix('.json')}, "
        f"and {history_path}"
    )
    print("Evaluation outputs:")
    for name, value in evaluation_files.items():
        if isinstance(value, str) and value:
            print(f"  {name}: {value}")
        elif isinstance(value, list):
            for item in value:
                print(f"  {name}: {item}")
    print(
        f"Probability calibration: Brier score="
        f"{evaluation_files['brier_score']:.5f} (lower is better)"
    )
    print("Pion operating points (score >= threshold):")
    for name in (
        "pion_metrics_at_configured_threshold",
        "pion_metrics_at_threshold_0p80",
    ):
        point = evaluation_files[name]
        print(
            f"  threshold={point['threshold']:.2f}: "
            f"efficiency={point['efficiency']:.5f}, "
            f"purity={point['purity']:.5f}, "
            f"efficiency*purity={point['efficiency_x_purity']:.5f}, "
            f"specificity={point['specificity']:.5f}, "
            f"balanced_accuracy={point['balanced_accuracy']:.5f}, "
            f"F1={point['f1_score']:.5f}, "
            f"MCC={point['matthews_correlation_coefficient']:.5f}"
        )


if __name__ == "__main__":
    main()
