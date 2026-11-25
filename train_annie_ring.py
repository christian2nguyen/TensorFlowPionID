#!/usr/bin/env python3
"""Train a CNN to identify pion-like ANNIE PMT ring patterns."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
    MAX_MRD_TRACKS,
    MRD_TRACK_PROPERTY_BRANCHES,
    MRD_TRACK_START_BRANCHES,
    PION_ID_SELECTION_EXPRESSION,
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

PION_EVENT_FEATURE_NAMES = ["numMRDTracks"]

TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15
TEST_FRACTION = 0.15
AMBIGUOUS_SCORE_LOW = 0.20
AMBIGUOUS_SCORE_HIGH = 0.80

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
HADRONIC_PDGS_BRANCH = "hadronic_pdgs"
HADRONIC_P4_BRANCH = "hadronic_p4"
TRUE_FSL_MOMENTUM_BRANCHES = (
    "trueFSLMomentum_X",
    "trueFSLMomentum_Y",
    "trueFSLMomentum_Z",
)
TRUE_FSL_PDG_BRANCH = "trueFSLPdg"
MUON_MASS_GEV = 0.1056583755
CHARGED_PION_MASS_GEV = 0.13957039
NEUTRAL_PION_MASS_GEV = 0.1349768


def _read_root_file_list(parser, option: str, path: Path) -> List[Path]:
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
    parser.add_argument(
        "root_files",
        nargs="*",
        type=Path,
        help="ROOT files; optional when --file-list is supplied",
    )
    parser.add_argument(
        "--file-list",
        type=Path,
        help=(
            "ROOT paths using the global --pmt-response mode; retained for "
            "backward compatibility"
        ),
    )
    parser.add_argument(
        "--tuned-file-list",
        "--convolved-file-list",
        dest="tuned_file_list",
        type=Path,
        help=(
            "ROOT paths whose PE values should receive the configured PMT-response "
            "convolution during image construction"
        ),
    )
    parser.add_argument(
        "--raw-file-list",
        "--unconvolved-file-list",
        dest="raw_file_list",
        type=Path,
        help="ROOT paths whose PE values should remain unconvolved/raw",
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
            "Disable the CC-inclusive/MRD/FV-score pion-ID preselection; "
            "--no-bdt-cuts is retained as a compatibility alias"
        ),
    )
    parser.add_argument(
        "--chunk-size",
        default="50 MB",
        help="Uproot read chunk size (default: 50 MB)",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/annie_ring_pion.keras")
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--mrd-ablation-repeats",
        type=int,
        default=5,
        help=(
            "number of held-out MRD permutation tests used to measure how much "
            "the trained model relies on MRD information (default: 5)"
        ),
    )
    parser.add_argument(
        "--mrd-branch-importance-repeats",
        type=int,
        default=3,
        help=(
            "number of single-branch MRD permutations used for the SHAP-style "
            "impact ranking and beeswarm (default: 3)"
        ),
    )
    parser.add_argument(
        "--threshold",
        "--pion-threshold",
        dest="threshold",
        type=float,
        default=0.5,
        help="Independent pion-score decision threshold (default: 0.5)",
    )
    parser.add_argument("--seed", type=int, default=20260620)
    args = parser.parse_args()
    if args.file_list is not None:
        args.root_files.extend(
            _read_root_file_list(parser, "--file-list", args.file_list)
        )
    args.tuned_root_files = (
        _read_root_file_list(
            parser, "--tuned-file-list/--convolved-file-list", args.tuned_file_list
        )
        if args.tuned_file_list is not None
        else []
    )
    args.raw_root_files = (
        _read_root_file_list(
            parser, "--raw-file-list/--unconvolved-file-list", args.raw_file_list
        )
        if args.raw_file_list is not None
        else []
    )
    all_paths = [*args.root_files, *args.tuned_root_files, *args.raw_root_files]
    if not all_paths:
        parser.error(
            "No ROOT files given: use positional paths, --file-list, "
            "--tuned-file-list, or --raw-file-list"
        )
    path_counts = Counter(all_paths)
    duplicate_paths = sorted(
        (path for path, count in path_counts.items() if count > 1), key=str
    )
    if duplicate_paths:
        parser.error(
            "The same ROOT path appears in more than one input group: "
            + ", ".join(str(path) for path in duplicate_paths)
        )
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
    Dict[str, object],
]:
    geometry = PMTGeometry(args.geometry, args.pmt_mask)
    needs_tuned_response = bool(args.tuned_root_files) or (
        bool(args.root_files) and args.pmt_response == "tuned"
    )
    needs_raw_response = bool(args.raw_root_files) or (
        bool(args.root_files) and args.pmt_response == "raw"
    )
    responses = {}
    if needs_tuned_response:
        responses["tuned"] = PMTResponse(
            "tuned",
            args.pmt_response_calibration,
            args.pmt_tune_variant,
        )
    if needs_raw_response:
        responses["raw"] = PMTResponse(
            "raw",
            tune_variant=args.pmt_tune_variant,
        )
    input_groups = []
    if args.root_files:
        input_groups.append(
            ("configured", args.root_files, responses[args.pmt_response])
        )
    if args.tuned_root_files:
        input_groups.append(
            ("tuned", args.tuned_root_files, responses["tuned"])
        )
    if args.raw_root_files:
        input_groups.append(("raw", args.raw_root_files, responses["raw"]))
    for mode, response in responses.items():
        print(f"{mode.capitalize()} input — {response.describe()}")
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
    group_summaries = []
    file_summaries: Dict[str, Dict[str, object]] = {}
    for group_name, group_paths, response in input_groups:
        group_read = 0
        group_selected = 0
        for chunk in iterate_ring_images(
            group_paths,
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
            "none" if args.no_event_cuts else "pion_id",
            True,
            args.charged_only,
            args.chunk_size,
            include_ring_event_features=True,
            include_fv_truth=False,
            include_numu_truth=False,
            ring_event_feature_names=PION_EVENT_FEATURE_NAMES,
            include_normalized_pe_channels=True,
        ):
            image_chunks.append(
                chunk["images"].astype(np.float16, copy=False)
            )
            detector_chunks.append(
                chunk["detector_images"].astype(np.float16, copy=False)
            )
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
            chunk_read = int(chunk["n_read"])
            chunk_selected = int(chunk["n_selected"])
            source_key = str(chunk["path"])
            file_summary = file_summaries.setdefault(
                source_key,
                {
                    "source": source_key,
                    "input_group": group_name,
                    "pmt_response": response.mode,
                    "events_read": 0,
                    "events_selected": 0,
                    "no_pion_selected": 0,
                    "pion_selected": 0,
                    "misaligned_events_rejected": 0,
                },
            )
            file_summary["events_read"] += chunk_read
            file_summary["events_selected"] += chunk_selected
            file_summary["no_pion_selected"] += int(
                np.sum(np.asarray(chunk["labels"]) == 0)
            )
            file_summary["pion_selected"] += int(
                np.sum(np.asarray(chunk["labels"]) == 1)
            )
            file_summary["misaligned_events_rejected"] += int(
                chunk["n_misaligned"]
            )
            n_read += chunk_read
            n_selected += chunk_selected
            group_read += chunk_read
            group_selected += chunk_selected
            n_misaligned += int(chunk["n_misaligned"])
            n_mrd_tracks_truncated += int(chunk["n_mrd_tracks_truncated"])
            print(f"Read {n_read}; selected {n_selected}", end="\r", flush=True)
        group_summaries.append(
            {
                "name": group_name,
                "pmt_response": response.mode,
                "sources": [str(path) for path in group_paths],
                "events_read": group_read,
                "events_selected": group_selected,
            }
        )
    print()
    if not image_chunks or n_selected == 0:
        raise ValueError("No events passed ring-image construction and selection")
    images = np.concatenate(image_chunks)
    image_chunks.clear()
    detector_images = np.concatenate(detector_chunks)
    detector_chunks.clear()
    return (
        images,
        detector_images,
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
        {
            "groups": group_summaries,
            "files": list(file_summaries.values()),
            "responses": responses,
        },
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


def _read_leading_hadronic_pion(
    tree,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return leading-pion [px,py,pz], total energy, and PDG from GENIE p4."""
    pdgs = tree[HADRONIC_PDGS_BRANCH].array(library="ak")
    p4 = tree[HADRONIC_P4_BRANCH].array(library="ak")
    if len(pdgs) != len(p4):
        raise ValueError("hadronic_pdgs and hadronic_p4 event counts differ")
    pdg_counts = ak.num(pdgs, axis=1)
    p4_counts = ak.num(p4, axis=1)
    if not bool(ak.all(pdg_counts == p4_counts)):
        raise ValueError("hadronic_pdgs and hadronic_p4 particle counts differ")

    component_counts = ak.num(p4, axis=2)
    padded_p4 = ak.fill_none(
        ak.pad_none(p4, 4, axis=2, clip=True), np.nan, axis=2
    )
    valid_p4 = (component_counts >= 4) & ak.all(np.isfinite(padded_p4), axis=2)
    pion_mask = ((abs(pdgs) == 211) | (pdgs == 111)) & valid_p4
    pion_p4 = padded_p4[pion_mask]
    pion_pdgs = pdgs[pion_mask]
    momentum_squared = (
        pion_p4[:, :, 0] ** 2
        + pion_p4[:, :, 1] ** 2
        + pion_p4[:, :, 2] ** 2
    )
    leading_indices = ak.argmax(
        momentum_squared, axis=1, keepdims=True, mask_identity=True
    )

    def leading_component(values) -> np.ndarray:
        return np.asarray(
            ak.to_numpy(ak.fill_none(ak.firsts(values[leading_indices]), np.nan)),
            dtype=np.float64,
        )

    vectors = np.column_stack(
        [leading_component(pion_p4[:, :, component]) for component in range(3)]
    )
    energies = leading_component(pion_p4[:, :, 3])
    leading_pdgs = leading_component(pion_pdgs)
    return vectors, energies, leading_pdgs


def _truth_kinematics_for_tree(
    tree, scalar_degree_branch: str = ""
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    str,
]:
    """Read truth momenta and obtain the leading-pion/muon opening angle."""
    available = set(tree.keys(recursive=True))
    entry_count = int(tree.num_entries)
    muon_vectors = None
    pion_vectors = None
    muon_momentum = np.full(entry_count, np.nan, dtype=np.float64)
    pion_momentum = np.full(entry_count, np.nan, dtype=np.float64)
    pion_total_energy = np.full(entry_count, np.nan, dtype=np.float64)
    leading_pion_pdg = np.full(entry_count, np.nan, dtype=np.float64)

    muon_spec = _resolve_vector3_spec(available, TRUTH_MUON_VECTOR_BASES)
    pion_spec = _resolve_vector3_spec(available, TRUTH_PION_VECTOR_BASES)
    muon_source = ""
    pion_source = ""
    vector_descriptions = []
    if muon_spec:
        try:
            muon_vectors = _read_vector3(tree, muon_spec)
            if len(muon_vectors) != entry_count:
                raise ValueError("Muon truth-vector length differs from tree length")
            muon_momentum = np.linalg.norm(muon_vectors, axis=1)
            muon_momentum[~np.isfinite(muon_vectors).all(axis=1)] = np.nan
            muon_source = muon_spec[0]
            vector_descriptions.append(f"muon momentum from {muon_source}")
        except Exception as error:
            vector_descriptions.append(f"muon momentum unavailable ({error})")
            muon_vectors = None
    elif all(name in available for name in TRUE_FSL_MOMENTUM_BRANCHES):
        try:
            muon_vectors = np.column_stack(
                [
                    np.asarray(tree[name].array(library="np"), dtype=np.float64)
                    for name in TRUE_FSL_MOMENTUM_BRANCHES
                ]
            ) / 1000.0
            if TRUE_FSL_PDG_BRANCH in available:
                muon_pdgs = np.asarray(
                    tree[TRUE_FSL_PDG_BRANCH].array(library="np"),
                    dtype=np.float64,
                )
                muon_vectors[np.abs(muon_pdgs) != 13] = np.nan
            muon_momentum = np.linalg.norm(muon_vectors, axis=1)
            muon_momentum[~np.isfinite(muon_vectors).all(axis=1)] = np.nan
            muon_source = "trueFSLMomentum_X/Y/Z (MeV converted to GeV)"
            vector_descriptions.append(f"muon momentum from {muon_source}")
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
            pion_source = pion_spec[0]
            vector_descriptions.append(f"leading-pion momentum from {pion_source}")
        except Exception as error:
            vector_descriptions.append(f"leading-pion momentum unavailable ({error})")
            pion_vectors = None
    elif {
        HADRONIC_PDGS_BRANCH,
        HADRONIC_P4_BRANCH,
    }.issubset(available):
        try:
            (
                pion_vectors,
                pion_total_energy,
                leading_pion_pdg,
            ) = _read_leading_hadronic_pion(tree)
            pion_momentum = np.linalg.norm(pion_vectors, axis=1)
            pion_momentum[~np.isfinite(pion_vectors).all(axis=1)] = np.nan
            pion_source = "leading pion in aligned hadronic_pdgs/hadronic_p4"
            vector_descriptions.append(
                f"leading-pion momentum and energy from {pion_source}"
            )
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
                    f"opening angle derived from {muon_source} and "
                    f"{pion_source} truth vectors"
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
        pion_total_energy,
        leading_pion_pdg,
        full_description,
    )


def load_truth_muon_pion_kinematics(
    source_paths: np.ndarray,
    entries: np.ndarray,
    tree_name: str,
    scalar_degree_branch: str = "",
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    str,
]:
    """Load truth angle and momenta for selected events in multiple ROOT files."""
    angles = np.full(len(entries), np.nan, dtype=np.float64)
    muon_momenta = np.full(len(entries), np.nan, dtype=np.float64)
    pion_momenta = np.full(len(entries), np.nan, dtype=np.float64)
    pion_total_energies = np.full(len(entries), np.nan, dtype=np.float64)
    leading_pion_pdgs = np.full(len(entries), np.nan, dtype=np.float64)
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
                    file_pion_total_energies,
                    file_leading_pion_pdgs,
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
            pion_total_energies[selected_targets] = file_pion_total_energies[
                selected_entries
            ]
            leading_pion_pdgs[selected_targets] = file_leading_pion_pdgs[
                selected_entries
            ]
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
        pion_total_energies,
        leading_pion_pdgs,
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


def _channel_moments(
    values: np.ndarray,
    indices: np.ndarray,
    batch_size: int = 512,
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate per-channel moments without copying an entire split."""
    channel_count = values.shape[-1]
    sums = np.zeros(channel_count, dtype=np.float64)
    squared_sums = np.zeros(channel_count, dtype=np.float64)
    value_count = 0
    for start in range(0, len(indices), batch_size):
        batch = values[indices[start : start + batch_size]]
        flattened = batch.reshape(-1, channel_count)
        sums += np.sum(flattened, axis=0, dtype=np.float64)
        squared_sums += np.sum(
            np.square(flattened, dtype=np.float32),
            axis=0,
            dtype=np.float64,
        )
        value_count += len(flattened)
    if value_count == 0:
        raise ValueError("Cannot normalize an empty training image split")
    means = sums / value_count
    variances = np.maximum(squared_sums / value_count - means * means, 1e-7)
    return means.astype(np.float32), variances.astype(np.float32)


class RingBatchSequence(tf.keras.utils.Sequence):
    """Index the shared arrays one batch at a time to bound training RAM."""

    def __init__(
        self,
        images: np.ndarray,
        detector_images: np.ndarray,
        event_features: np.ndarray,
        mrd_track_starts: np.ndarray,
        mrd_track_properties: np.ndarray,
        mrd_track_mask: np.ndarray,
        pion_labels: np.ndarray,
        indices: np.ndarray,
        batch_size: int,
        shuffle: bool,
        seed: int,
        class_weights: Optional[Dict[str, Dict[int, float]]] = None,
    ):
        super().__init__()
        self.images = images
        self.detector_images = detector_images
        self.event_features = event_features
        self.mrd_track_starts = mrd_track_starts
        self.mrd_track_properties = mrd_track_properties
        self.mrd_track_mask = mrd_track_mask
        self.pion_labels = pion_labels
        self.indices = np.asarray(indices, dtype=np.int64)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.class_weights = class_weights
        self.order = np.arange(len(self.indices), dtype=np.int64)
        self.rng = np.random.default_rng(seed)
        if self.shuffle:
            self.rng.shuffle(self.order)

    def __len__(self) -> int:
        return int(np.ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, batch_index: int):
        offset = batch_index * self.batch_size
        order = self.order[offset : offset + self.batch_size]
        event_indices = self.indices[order]
        inputs = {
            "pmt_angular_image": self.images[event_indices],
            "pmt_unfolded_image": self.detector_images[event_indices],
            "event_features": self.event_features[event_indices],
            "mrd_track_starts": self.mrd_track_starts[event_indices],
            "mrd_track_properties": self.mrd_track_properties[event_indices],
            "mrd_track_mask": self.mrd_track_mask[event_indices],
        }
        targets = {
            "pion_score": self.pion_labels[event_indices],
        }
        if self.class_weights is None:
            return inputs, targets
        sample_weights = {
            output_name: np.where(
                targets[output_name] == 0,
                weights[0],
                weights[1],
            ).astype(np.float32)
            for output_name, weights in self.class_weights.items()
        }
        return inputs, targets, sample_weights

    def on_epoch_end(self) -> None:
        if self.shuffle:
            self.rng.shuffle(self.order)


def build_model(
    angular_shape: Tuple[int, ...],
    detector_shape: Tuple[int, ...],
    angular_mean: np.ndarray,
    angular_variance: np.ndarray,
    detector_mean: np.ndarray,
    detector_variance: np.ndarray,
    event_features_train: np.ndarray,
    mrd_track_starts_train: np.ndarray,
    mrd_track_properties_train: np.ndarray,
    mrd_track_mask_train: np.ndarray,
) -> tf.keras.Model:
    if angular_shape[-1] != 4 or detector_shape[-1] != 4:
        raise ValueError(
            "Pion image inputs require absolute full/tankcluster PE plus "
            "event-normalized full/tankcluster PE channels"
        )
    if PION_EVENT_FEATURE_NAMES != ["numMRDTracks"]:
        raise ValueError("Pion scalar input must contain only numMRDTracks")
    if event_features_train.shape[1] != 1:
        raise ValueError("The pion model requires one numMRDTracks input")
    angular_normalizer = tf.keras.layers.Normalization(
        axis=-1,
        mean=angular_mean,
        variance=angular_variance,
        name="angular_channel_normalization",
    )
    detector_normalizer = tf.keras.layers.Normalization(
        axis=-1,
        mean=detector_mean,
        variance=detector_variance,
        name="detector_channel_normalization",
    )
    mrd_count_normalizer = tf.keras.layers.Normalization(
        axis=-1, name="num_mrd_tracks_normalization"
    )
    mrd_track_normalizer = tf.keras.layers.Normalization(
        axis=-1, name="mrd_track_coordinate_normalization"
    )
    mrd_property_normalizer = tf.keras.layers.Normalization(
        axis=-1, name="mrd_track_property_normalization"
    )
    mrd_count_normalizer.adapt(event_features_train[:, 0:1])
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
    mrd_count_input = event_feature_inputs
    angular_features = _image_tower(angular_inputs, angular_normalizer, "angular")
    detector_features = _image_tower(detector_inputs, detector_normalizer, "detector")
    mrd_count_features = mrd_count_normalizer(mrd_count_input)
    mrd_count_features = tf.keras.layers.Dense(
        8, activation="relu", name="event_feature_embedding"
    )(mrd_count_features)
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
    pion_inputs = tf.keras.layers.Concatenate(
        name="pion_full_and_tankcluster_inputs"
    )(
        [
            angular_features,
            detector_features,
            mrd_count_features,
            mrd_track_features,
        ]
    )
    pion_shared_features = tf.keras.layers.Dense(
        32, activation="relu", name="pion_event_embedding"
    )(pion_inputs)
    pion_shared_features = tf.keras.layers.Dropout(
        0.30, name="pion_event_dropout"
    )(pion_shared_features)
    pion_features = tf.keras.layers.Dense(
        16, activation="relu", name="pion_head_embedding"
    )(pion_shared_features)
    pion_output = tf.keras.layers.Dense(
        1, activation="sigmoid", name="pion_score"
    )(pion_features)
    model = tf.keras.Model(
        [
            angular_inputs,
            detector_inputs,
            event_feature_inputs,
            mrd_track_start_inputs,
            mrd_track_property_inputs,
            mrd_track_mask_inputs,
        ],
        {"pion_score": pion_output},
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss={"pion_score": "binary_crossentropy"},
        metrics={
            "pion_score": [
                tf.keras.metrics.AUC(name="roc_auc"),
                tf.keras.metrics.AUC(curve="PR", name="pr_auc"),
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
            ],
        },
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


def _ablation_metrics(
    pion_labels: np.ndarray,
    pion_scores: np.ndarray,
    pion_threshold: float,
) -> Dict[str, float]:
    """Summarize the pion output for an MRD permutation trial."""
    pion_fpr, pion_tpr, _ = roc_curve(pion_labels, pion_scores)
    pion_point = pion_operating_point(pion_labels, pion_scores, pion_threshold)
    return {
        "pion_roc_auc": float(auc(pion_fpr, pion_tpr)),
        "pion_average_precision": float(
            average_precision_score(pion_labels, pion_scores)
        ),
        "pion_brier_score": float(brier_score_loss(pion_labels, pion_scores)),
        "pion_efficiency": float(pion_point["efficiency"]),
        "pion_purity": float(pion_point["purity"]),
        "pion_efficiency_x_purity": float(
            pion_point["efficiency_x_purity"]
        ),
    }


def evaluate_mrd_permutation_importance(
    model: tf.keras.Model,
    images_test: np.ndarray,
    detector_images_test: np.ndarray,
    event_features_test: np.ndarray,
    mrd_track_starts_test: np.ndarray,
    mrd_track_properties_test: np.ndarray,
    mrd_track_mask_test: np.ndarray,
    pion_labels: np.ndarray,
    full_pion_scores: np.ndarray,
    pion_threshold: float,
    output: Path,
    seed: int,
    repeats: int,
) -> Dict[str, object]:
    """Measure held-out performance loss when all MRD inputs are shuffled."""
    full_metrics = _ablation_metrics(
        pion_labels,
        full_pion_scores,
        pion_threshold,
    )
    rng = np.random.default_rng(seed + 104729)
    trial_rows = []
    for repeat in range(repeats):
        permutation = rng.permutation(len(pion_labels))
        shuffled_event_features = np.array(event_features_test, copy=True)
        shuffled_event_features[:, 0] = event_features_test[permutation, 0]
        outputs = model.predict(
            {
                "pmt_angular_image": images_test,
                "pmt_unfolded_image": detector_images_test,
                "event_features": shuffled_event_features,
                "mrd_track_starts": mrd_track_starts_test[permutation],
                "mrd_track_properties": mrd_track_properties_test[permutation],
                "mrd_track_mask": mrd_track_mask_test[permutation],
            },
            verbose=0,
        )
        trial_metrics = _ablation_metrics(
            pion_labels,
            np.asarray(outputs["pion_score"]).reshape(-1),
            pion_threshold,
        )
        trial_rows.append(
            {"mode": "mrd_permuted", "repeat": repeat + 1, **trial_metrics}
        )

    metric_names = list(full_metrics)
    shuffled_mean = {
        name: float(np.mean([row[name] for row in trial_rows]))
        for name in metric_names
    }
    shuffled_std = {
        name: float(np.std([row[name] for row in trial_rows]))
        for name in metric_names
    }
    performance_drop = {
        name: float(
            shuffled_mean[name] - full_metrics[name]
            if name.endswith("brier_score")
            else full_metrics[name] - shuffled_mean[name]
        )
        for name in metric_names
    }

    csv_path = output.with_suffix(".mrd_permutation_ablation.csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["mode", "repeat", *metric_names]
        )
        writer.writeheader()
        writer.writerow({"mode": "full_model", "repeat": 0, **full_metrics})
        writer.writerows(trial_rows)

    plotted_metrics = [
        "pion_roc_auc",
        "pion_average_precision",
        "pion_efficiency_x_purity",
    ]
    plotted_labels = [
        "Pion ROC AUC",
        "Pion average precision",
        "Pion efficiency × purity",
    ]
    positions = np.arange(len(plotted_metrics))
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].bar(
        positions - width / 2,
        [full_metrics[name] for name in plotted_metrics],
        width,
        label="Normal test data",
    )
    axes[0].bar(
        positions + width / 2,
        [shuffled_mean[name] for name in plotted_metrics],
        width,
        yerr=[shuffled_std[name] for name in plotted_metrics],
        capsize=3,
        label="MRD shuffled",
    )
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_ylabel("Metric value")
    axes[0].set_xticks(positions, plotted_labels, rotation=25, ha="right")
    axes[0].set_title("Held-out performance")
    axes[0].legend()
    axes[1].barh(
        plotted_labels,
        [performance_drop[name] for name in plotted_metrics],
        xerr=[shuffled_std[name] for name in plotted_metrics],
        capsize=3,
    )
    axes[1].axvline(0.0, color="black", linewidth=1)
    axes[1].set_xlabel("Performance decrease after shuffling MRD")
    axes[1].set_title("MRD permutation importance\n(positive means MRD is useful)")
    fig.tight_layout()
    plot_path = output.with_suffix(".mrd_permutation_ablation.png")
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)

    return {
        "mrd_permutation_ablation_csv": csv_path.name,
        "mrd_permutation_ablation_plot": plot_path.name,
        "mrd_permutation_importance": {
            "method": (
                "All MRD inputs, including numMRDTracks, track positions, track "
                "properties, and masks, were permuted together across held-out "
                "events while PMT images and truth labels stayed fixed."
            ),
            "interpretation": (
                "A positive performance_drop means performance worsened when the "
                "event-to-MRD association was broken, indicating useful MRD "
                "information. This measures model reliance, not the causal gain "
                "from retraining a separate image-only model."
            ),
            "repeats": repeats,
            "full_model": full_metrics,
            "mrd_permuted_mean": shuffled_mean,
            "mrd_permuted_std": shuffled_std,
            "performance_drop": performance_drop,
        },
    }


def _masked_event_mean(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Reduce a padded per-track branch to one display value per event."""
    valid = mask.astype(bool)
    counts = np.sum(valid, axis=1)
    sums = np.sum(np.where(valid, values, 0.0), axis=1)
    return np.divide(
        sums,
        counts,
        out=np.full(len(values), np.nan, dtype=np.float64),
        where=counts > 0,
    )


def _permute_valid_track_values(
    values: np.ndarray,
    mask: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Permute a branch within each valid padded track slot."""
    shuffled = np.array(values, copy=True)
    for track_index in range(values.shape[1]):
        valid_indices = np.flatnonzero(mask[:, track_index] != 0)
        if len(valid_indices) > 1:
            shuffled[valid_indices, track_index] = values[
                rng.permutation(valid_indices), track_index
            ]
    return shuffled


def evaluate_mrd_branch_importance(
    model: tf.keras.Model,
    images_test: np.ndarray,
    detector_images_test: np.ndarray,
    event_features_test: np.ndarray,
    mrd_track_starts_test: np.ndarray,
    mrd_track_properties_test: np.ndarray,
    mrd_track_mask_test: np.ndarray,
    pion_labels: np.ndarray,
    full_pion_scores: np.ndarray,
    output: Path,
    seed: int,
    repeats: int,
) -> Dict[str, object]:
    """Rank individual MRD branches using held-out permutation impacts."""
    branch_specs = [
        ("numMRDTracks", "event", 0),
        ("MRDTrackStartX", "starts", 0),
        ("MRDTrackStartY", "starts", 1),
        ("MRDTrackStartZ", "starts", 2),
        ("MRDEnergyLoss", "properties", 0),
        ("MRDTrackLength", "properties", 1),
        ("MRDTrackAngle", "properties", 2),
    ]
    full_fpr, full_tpr, _ = roc_curve(pion_labels, full_pion_scores)
    full_roc_auc = float(auc(full_fpr, full_tpr))
    full_average_precision = float(
        average_precision_score(pion_labels, full_pion_scores)
    )
    rng = np.random.default_rng(seed + 130363)
    results = []
    impact_by_branch = {}
    display_value_by_branch = {}

    for branch_name, input_group, column in branch_specs:
        repeat_impacts = []
        repeat_roc_auc = []
        repeat_average_precision = []
        for _ in range(repeats):
            event_features = event_features_test
            track_starts = mrd_track_starts_test
            track_properties = mrd_track_properties_test
            if input_group == "event":
                event_features = np.array(event_features_test, copy=True)
                event_features[:, column] = event_features_test[
                    rng.permutation(len(event_features_test)), column
                ]
            elif input_group == "starts":
                track_starts = np.array(mrd_track_starts_test, copy=True)
                track_starts[:, :, column] = _permute_valid_track_values(
                    mrd_track_starts_test[:, :, column],
                    mrd_track_mask_test,
                    rng,
                )
            else:
                track_properties = np.array(mrd_track_properties_test, copy=True)
                track_properties[:, :, column] = _permute_valid_track_values(
                    mrd_track_properties_test[:, :, column],
                    mrd_track_mask_test,
                    rng,
                )
            outputs = model.predict(
                {
                    "pmt_angular_image": images_test,
                    "pmt_unfolded_image": detector_images_test,
                    "event_features": event_features,
                    "mrd_track_starts": track_starts,
                    "mrd_track_properties": track_properties,
                    "mrd_track_mask": mrd_track_mask_test,
                },
                verbose=0,
            )
            permuted_scores = np.asarray(outputs["pion_score"]).reshape(-1)
            repeat_impacts.append(full_pion_scores - permuted_scores)
            fpr, tpr, _ = roc_curve(pion_labels, permuted_scores)
            repeat_roc_auc.append(float(auc(fpr, tpr)))
            repeat_average_precision.append(
                float(average_precision_score(pion_labels, permuted_scores))
            )

        mean_event_impact = np.mean(np.stack(repeat_impacts), axis=0)
        impact_by_branch[branch_name] = mean_event_impact
        if input_group == "event":
            display_values = event_features_test[:, column].astype(np.float64)
        elif input_group == "starts":
            display_values = _masked_event_mean(
                mrd_track_starts_test[:, :, column], mrd_track_mask_test
            )
        else:
            display_values = _masked_event_mean(
                mrd_track_properties_test[:, :, column], mrd_track_mask_test
            )
        display_value_by_branch[branch_name] = display_values
        permuted_roc_auc_mean = float(np.mean(repeat_roc_auc))
        permuted_ap_mean = float(np.mean(repeat_average_precision))
        results.append(
            {
                "branch": branch_name,
                "mean_abs_pion_score_impact": float(
                    np.mean(np.abs(mean_event_impact))
                ),
                "mean_signed_pion_score_impact": float(
                    np.mean(mean_event_impact)
                ),
                "pion_roc_auc_drop": full_roc_auc - permuted_roc_auc_mean,
                "pion_roc_auc_drop_std": float(np.std(repeat_roc_auc)),
                "pion_average_precision_drop": (
                    full_average_precision - permuted_ap_mean
                ),
                "pion_average_precision_drop_std": float(
                    np.std(repeat_average_precision)
                ),
            }
        )

    ranking = [
        {"rank": rank, **row}
        for rank, row in enumerate(
            sorted(
                results,
                key=lambda item: item["mean_abs_pion_score_impact"],
                reverse=True,
            ),
            start=1,
        )
    ]
    csv_path = output.with_suffix(".mrd_branch_importance.csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ranking[0]))
        writer.writeheader()
        writer.writerows(ranking)

    max_points = 2000
    plot_rng = np.random.default_rng(seed + 155921)
    fig, (swarm_axis, rank_axis) = plt.subplots(
        1,
        2,
        figsize=(15, 6),
        gridspec_kw={"width_ratios": [2.3, 1.0]},
        sharey=True,
    )
    y_positions = np.arange(len(ranking))
    for y_position, row in zip(y_positions, ranking):
        branch_name = str(row["branch"])
        impacts = impact_by_branch[branch_name]
        values = display_value_by_branch[branch_name]
        available = np.arange(len(impacts))
        if len(available) > max_points:
            available = plot_rng.choice(
                available, size=max_points, replace=False
            )
        plot_impacts = impacts[available]
        plot_values = values[available]
        finite_values = plot_values[np.isfinite(plot_values)]
        if len(finite_values):
            low, high = np.nanpercentile(finite_values, [5.0, 95.0])
            scale = high - low
            colors = (
                np.clip((plot_values - low) / scale, 0.0, 1.0)
                if scale > 0.0
                else np.full(len(plot_values), 0.5)
            )
            colors = np.where(np.isfinite(colors), colors, 0.5)
        else:
            colors = np.full(len(plot_values), 0.5)
        jitter = plot_rng.normal(0.0, 0.10, size=len(plot_impacts))
        swarm_axis.scatter(
            plot_impacts,
            y_position + jitter,
            c=colors,
            cmap="coolwarm",
            vmin=0.0,
            vmax=1.0,
            s=10,
            alpha=0.55,
            linewidths=0,
        )

    swarm_axis.axvline(0.0, color="black", linewidth=1)
    swarm_axis.set_yticks(y_positions, [str(row["branch"]) for row in ranking])
    swarm_axis.invert_yaxis()
    swarm_axis.set_xlabel(
        "Pion-score impact: original score − score after branch permutation"
    )
    swarm_axis.set_title("Event-level MRD branch impacts")
    rank_axis.barh(
        y_positions,
        [row["mean_abs_pion_score_impact"] for row in ranking],
        color="tab:blue",
        alpha=0.8,
    )
    rank_axis.set_xlabel("Mean |pion-score impact|")
    rank_axis.set_title("Global ranking")
    color_scale = plt.cm.ScalarMappable(cmap="coolwarm")
    color_scale.set_array([])
    colorbar = fig.colorbar(color_scale, ax=swarm_axis, pad=0.01)
    colorbar.set_ticks([0.0, 1.0], labels=["low", "high"])
    colorbar.set_label("Event branch value (within-branch scale)")
    fig.suptitle(
        "SHAP-style MRD permutation-impact beeswarm\n"
        "Vector branch colors show the mean across valid tracks",
        fontsize=12,
    )
    fig.tight_layout()
    plot_path = output.with_suffix(".mrd_branch_importance_beeswarm.png")
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    return {
        "mrd_branch_importance_csv": csv_path.name,
        "mrd_branch_importance_beeswarm": plot_path.name,
        "mrd_branch_importance": {
            "method": (
                "One MRD branch at a time was permuted across held-out events. "
                "Per-event impact is the original pion score minus the score "
                "after permutation. This is SHAP-style permutation impact, not "
                "an exact SHAP value."
            ),
            "repeats": repeats,
            "ranking_metric": "mean_abs_pion_score_impact",
            "ranking": ranking,
        },
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
    negative_label: str = "no pion",
    positive_label: str = "pion",
    score_name: str = "pion_score",
) -> str:
    predictions = (test_scores >= threshold).astype(int)
    rule_text = (
        f"{negative_label} if {score_name} < {threshold:.2f}; "
        f"{positive_label} if {score_name} >= {threshold:.2f}"
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
    axis.set_xticklabels([negative_label, positive_label])
    axis.set_yticks([0, 1])
    axis.set_yticklabels([negative_label, positive_label])
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


def _detector_channel_observables(
    detector_images: np.ndarray,
    channel: int,
    batch_size: int = 512,
) -> Tuple[np.ndarray, np.ndarray]:
    """Recover total PE and occupied detector bins from one log1p image channel."""
    totals = np.zeros(len(detector_images), dtype=np.float64)
    occupied = np.zeros(len(detector_images), dtype=np.float64)
    for start in range(0, len(detector_images), batch_size):
        stop = min(start + batch_size, len(detector_images))
        linear_pe = np.expm1(
            detector_images[start:stop, :, :, channel].astype(np.float32)
        )
        totals[start:stop] = np.sum(linear_pe, axis=(1, 2), dtype=np.float64)
        occupied[start:stop] = np.count_nonzero(linear_pe > 0.0, axis=(1, 2))
    return totals, occupied


def _quantile_edges(values: np.ndarray, mask: np.ndarray, bins: int = 8) -> np.ndarray:
    """Return stable equal-population bin edges for a diagnostic variable."""
    finite_values = np.asarray(values, dtype=np.float64)[
        np.asarray(mask, dtype=bool) & np.isfinite(values)
    ]
    if len(finite_values) == 0:
        return np.array([], dtype=np.float64)
    unique_values = np.unique(finite_values)
    if len(unique_values) == 1:
        value = float(unique_values[0])
        width = max(abs(value) * 1e-6, 1e-6)
        return np.array([value - width, value + width], dtype=np.float64)
    edges = np.unique(np.quantile(finite_values, np.linspace(0.0, 1.0, bins + 1)))
    edges[0] = np.nextafter(edges[0], -np.inf)
    edges[-1] = np.nextafter(edges[-1], np.inf)
    return edges


def _binned_ambiguity_rows(
    variable_name: str,
    values: np.ndarray,
    base_mask: np.ndarray,
    edges: np.ndarray,
    y_test: np.ndarray,
    test_scores: np.ndarray,
    threshold: float,
) -> List[Dict[str, object]]:
    """Summarize ambiguity and classification performance in variable bins."""
    rows = []
    predictions = test_scores >= threshold
    truth = y_test.astype(bool)
    ambiguous = (
        (test_scores >= AMBIGUOUS_SCORE_LOW)
        & (test_scores <= AMBIGUOUS_SCORE_HIGH)
    )
    for bin_index, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        selected = (
            np.asarray(base_mask, dtype=bool)
            & np.isfinite(values)
            & (values >= low)
            & (values < high)
        )
        count = int(np.count_nonzero(selected))
        pion_selected = selected & truth
        no_pion_selected = selected & ~truth
        pion_count = int(np.count_nonzero(pion_selected))
        no_pion_count = int(np.count_nonzero(no_pion_selected))
        if count:
            scores = test_scores[selected]
            ambiguous_count = int(np.count_nonzero(ambiguous[selected]))
            accuracy = float(np.mean(predictions[selected] == truth[selected]))
        else:
            scores = np.array([], dtype=np.float64)
            ambiguous_count = 0
            accuracy = float("nan")
        pion_efficiency = (
            float(np.mean(predictions[pion_selected]))
            if pion_count
            else float("nan")
        )
        no_pion_rejection = (
            float(np.mean(~predictions[no_pion_selected]))
            if no_pion_count
            else float("nan")
        )
        bin_auc = float("nan")
        if pion_count and no_pion_count:
            local_fpr, local_tpr, _ = roc_curve(
                y_test[selected], test_scores[selected]
            )
            bin_auc = float(auc(local_fpr, local_tpr))
        rows.append(
            {
                "variable": variable_name,
                "bin_index": bin_index,
                "bin_low": float(low),
                "bin_high": float(high),
                "event_count": count,
                "pion_count": pion_count,
                "no_pion_count": no_pion_count,
                "ambiguous_count": ambiguous_count,
                "ambiguous_fraction": (
                    ambiguous_count / count if count else float("nan")
                ),
                "mean_score": float(np.mean(scores)) if count else float("nan"),
                "median_score": (
                    float(np.median(scores)) if count else float("nan")
                ),
                "mean_pion_score": (
                    float(np.mean(test_scores[pion_selected]))
                    if pion_count
                    else float("nan")
                ),
                "mean_no_pion_score": (
                    float(np.mean(test_scores[no_pion_selected]))
                    if no_pion_count
                    else float("nan")
                ),
                "accuracy_at_threshold": accuracy,
                "pion_efficiency_at_threshold": pion_efficiency,
                "no_pion_rejection_at_threshold": no_pion_rejection,
                "roc_auc": bin_auc,
            }
        )
    return rows


def _plot_binned_ambiguity_page(
    pdf: PdfPages,
    title: str,
    x_label: str,
    values: np.ndarray,
    base_mask: np.ndarray,
    rows: List[Dict[str, object]],
    y_test: np.ndarray,
    test_scores: np.ndarray,
    threshold: float,
) -> None:
    """Add one four-panel variable diagnostic page to a PDF."""
    selected = np.asarray(base_mask, dtype=bool) & np.isfinite(values)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    if not rows or not np.any(selected):
        for axis in axes.ravel():
            axis.axis("off")
        axes[0, 0].text(
            0.5,
            0.5,
            f"{title}\nNo finite values were available.",
            ha="center",
            va="center",
            fontsize=14,
        )
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
        return

    selected_indices = np.flatnonzero(selected)
    if len(selected_indices) > 5000:
        selected_indices = selected_indices[
            np.linspace(0, len(selected_indices) - 1, 5000).astype(int)
        ]
    truth_colors = np.where(y_test[selected_indices] == 1, "C1", "C0")
    axes[0, 0].scatter(
        values[selected_indices],
        test_scores[selected_indices],
        c=truth_colors,
        s=10,
        alpha=0.35,
        edgecolors="none",
    )
    axes[0, 0].axhspan(
        AMBIGUOUS_SCORE_LOW,
        AMBIGUOUS_SCORE_HIGH,
        color="gray",
        alpha=0.12,
        label="ambiguous score band",
    )
    axes[0, 0].axhline(threshold, color="black", linestyle="--", linewidth=1)
    axes[0, 0].set_xlabel(x_label)
    axes[0, 0].set_ylabel("pion_score")
    axes[0, 0].set_ylim(-0.02, 1.02)
    axes[0, 0].set_title("Event scores (orange=pion, blue=no pion)")
    axes[0, 0].legend(loc="best", fontsize=8)

    centers = np.array(
        [(row["bin_low"] + row["bin_high"]) / 2.0 for row in rows]
    )
    widths = np.array([row["bin_high"] - row["bin_low"] for row in rows])
    counts = np.array([row["event_count"] for row in rows], dtype=float)
    axes[0, 1].bar(centers, counts, width=0.9 * widths, color="C7")
    axes[0, 1].set_xlabel(x_label)
    axes[0, 1].set_ylabel("Selected test events")
    axes[0, 1].set_title("Statistics per bin")

    ambiguous_fraction = np.array(
        [row["ambiguous_fraction"] for row in rows], dtype=float
    )
    accuracy = np.array(
        [row["accuracy_at_threshold"] for row in rows], dtype=float
    )
    axes[1, 0].plot(
        centers, ambiguous_fraction, marker="o", label="Ambiguous fraction"
    )
    axes[1, 0].plot(
        centers,
        accuracy,
        marker="o",
        label=f"Accuracy at score {threshold:.2f}",
    )
    axes[1, 0].set_xlabel(x_label)
    axes[1, 0].set_ylabel("Fraction")
    axes[1, 0].set_ylim(-0.02, 1.02)
    axes[1, 0].set_title(
        f"Ambiguous means {AMBIGUOUS_SCORE_LOW:.2f} ≤ score ≤ "
        f"{AMBIGUOUS_SCORE_HIGH:.2f}"
    )
    axes[1, 0].legend(fontsize=8)

    mean_pion = np.array([row["mean_pion_score"] for row in rows], dtype=float)
    mean_no_pion = np.array(
        [row["mean_no_pion_score"] for row in rows], dtype=float
    )
    mean_all = np.array([row["mean_score"] for row in rows], dtype=float)
    if np.any(np.isfinite(mean_pion)):
        axes[1, 1].plot(centers, mean_pion, marker="o", label="Pion truth")
    if np.any(np.isfinite(mean_no_pion)):
        axes[1, 1].plot(
            centers, mean_no_pion, marker="o", label="No-pion truth"
        )
    if not np.any(np.isfinite(mean_no_pion)):
        axes[1, 1].plot(centers, mean_all, marker="o", label="All selected")
    axes[1, 1].set_xlabel(x_label)
    axes[1, 1].set_ylabel("Mean pion_score")
    axes[1, 1].set_ylim(-0.02, 1.02)
    axes[1, 1].set_title("Mean score by truth class")
    axes[1, 1].legend(fontsize=8)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    pdf.savefig(fig)
    plt.close(fig)


def plot_ambiguity_diagnostics(
    y_test: np.ndarray,
    test_scores: np.ndarray,
    detector_images_test: np.ndarray,
    source_paths_test: np.ndarray,
    entries_test: np.ndarray,
    truth_pion_counts_test: np.ndarray,
    truth_muon_pion_angles: np.ndarray,
    truth_pion_kinetic_energy: np.ndarray,
    output: Path,
    threshold: float,
) -> Dict[str, object]:
    """Write plots and tables that identify difficult pion-ID populations."""
    ambiguous = (
        (test_scores >= AMBIGUOUS_SCORE_LOW)
        & (test_scores <= AMBIGUOUS_SCORE_HIGH)
    )
    charged = (
        truth_pion_counts_test[:, 0] + truth_pion_counts_test[:, 1]
    ) > 0
    neutral = truth_pion_counts_test[:, 2] > 0
    multiplicity = np.sum(truth_pion_counts_test, axis=1).astype(np.int64)
    tank_total_pe, tank_occupied_bins = _detector_channel_observables(
        detector_images_test, 1
    )

    variable_specs = [
        (
            "truth_pion_kinetic_energy_GeV",
            "Pion kinetic energy",
            "Truth leading-pion kinetic energy [GeV]",
            truth_pion_kinetic_energy,
            y_test == 1,
        ),
        (
            "truth_muon_pion_opening_angle_deg",
            "Muon–pion opening angle",
            "Truth leading π–μ opening angle [degrees]",
            truth_muon_pion_angles,
            y_test == 1,
        ),
        (
            "tankcluster_total_pe",
            "Tank-cluster charge",
            "Tank-cluster total PE reconstructed from image",
            tank_total_pe,
            np.ones(len(y_test), dtype=bool),
        ),
        (
            "tankcluster_occupied_detector_bins",
            "Tank-cluster image occupancy",
            "Occupied tank-cluster detector-image bins",
            tank_occupied_bins,
            np.ones(len(y_test), dtype=bool),
        ),
    ]
    binned_rows = []
    variable_rows = []
    for variable_name, title, x_label, values, base_mask in variable_specs:
        edges = _quantile_edges(values, base_mask)
        rows = _binned_ambiguity_rows(
            variable_name,
            values,
            base_mask,
            edges,
            y_test,
            test_scores,
            threshold,
        )
        binned_rows.extend(rows)
        variable_rows.append((title, x_label, values, base_mask, rows))

    binned_path = output.with_suffix(".ambiguity_binned_metrics.csv")
    binned_fieldnames = [
        "variable",
        "bin_index",
        "bin_low",
        "bin_high",
        "event_count",
        "pion_count",
        "no_pion_count",
        "ambiguous_count",
        "ambiguous_fraction",
        "mean_score",
        "median_score",
        "mean_pion_score",
        "mean_no_pion_score",
        "accuracy_at_threshold",
        "pion_efficiency_at_threshold",
        "no_pion_rejection_at_threshold",
        "roc_auc",
    ]
    with binned_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=binned_fieldnames)
        writer.writeheader()
        writer.writerows(binned_rows)

    ambiguous_path = output.with_suffix(".ambiguous_events.csv")
    ambiguous_indices = np.flatnonzero(ambiguous)
    ambiguous_indices = ambiguous_indices[
        np.argsort(np.abs(test_scores[ambiguous_indices] - 0.5))
    ]
    with ambiguous_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "source_file",
                "tree_entry",
                "true_label",
                "pion_score",
                "distance_from_0p5",
                "truePiPlusCher",
                "truePiMinusCher",
                "truePi0",
                "truth_pion_multiplicity",
                "truth_leading_pion_kinetic_energy_GeV",
                "truth_leading_pion_muon_opening_angle_deg",
                "tankcluster_total_pe",
                "tankcluster_occupied_detector_bins",
            ]
        )
        for event_index in ambiguous_indices:
            writer.writerow(
                [
                    source_paths_test[event_index],
                    int(entries_test[event_index]),
                    int(y_test[event_index]),
                    float(test_scores[event_index]),
                    float(abs(test_scores[event_index] - 0.5)),
                    *truth_pion_counts_test[event_index].astype(int),
                    int(multiplicity[event_index]),
                    float(truth_pion_kinetic_energy[event_index]),
                    float(truth_muon_pion_angles[event_index]),
                    float(tank_total_pe[event_index]),
                    int(tank_occupied_bins[event_index]),
                ]
            )

    pdf_path = output.with_suffix(".ambiguity_diagnostics.pdf")
    with PdfPages(
        pdf_path,
        metadata={
            "Title": "ANNIE pion-ID ambiguity diagnostics",
            "Subject": "Pion score separation sliced by truth and detector observables",
        },
    ) as pdf:
        bins = np.linspace(0.0, 1.0, 41)
        categories = [
            ("No pion", ~(charged | neutral), "C0"),
            ("Charged pion only", charged & ~neutral, "C1"),
            ("π⁰ only", ~charged & neutral, "C2"),
            ("Charged + π⁰", charged & neutral, "C3"),
        ]
        fig, axis = plt.subplots(figsize=(9, 6))
        for label, mask, color in categories:
            count = int(np.count_nonzero(mask))
            if count:
                axis.hist(
                    test_scores[mask],
                    bins=bins,
                    histtype="step",
                    linewidth=2,
                    density=True,
                    label=f"{label} (n={count})",
                    color=color,
                )
        axis.axvspan(
            AMBIGUOUS_SCORE_LOW,
            AMBIGUOUS_SCORE_HIGH,
            color="gray",
            alpha=0.12,
            label="ambiguous score band",
        )
        axis.set_xlabel("pion_score")
        axis.set_ylabel("Normalized event density")
        axis.set_title("Score separation by pion truth topology")
        axis.legend()
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        fig, axis = plt.subplots(figsize=(9, 6))
        multiplicity_categories = [
            ("0 pions", multiplicity == 0),
            ("1 pion", multiplicity == 1),
            ("2 pions", multiplicity == 2),
            ("3+ pions", multiplicity >= 3),
        ]
        for label, mask in multiplicity_categories:
            count = int(np.count_nonzero(mask))
            if count:
                axis.hist(
                    test_scores[mask],
                    bins=bins,
                    histtype="step",
                    linewidth=2,
                    density=True,
                    label=f"{label} (n={count})",
                )
        axis.axvspan(
            AMBIGUOUS_SCORE_LOW,
            AMBIGUOUS_SCORE_HIGH,
            color="gray",
            alpha=0.12,
        )
        axis.set_xlabel("pion_score")
        axis.set_ylabel("Normalized event density")
        axis.set_title("Score separation by truth pion multiplicity")
        axis.legend()
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        for title, x_label, values, base_mask, rows in variable_rows:
            _plot_binned_ambiguity_page(
                pdf,
                title,
                x_label,
                values,
                base_mask,
                rows,
                y_test,
                test_scores,
                threshold,
            )

        finite_kinematics = (
            (y_test == 1)
            & np.isfinite(truth_pion_kinetic_energy)
            & np.isfinite(truth_muon_pion_angles)
        )
        fig, axis = plt.subplots(figsize=(9, 7))
        if np.any(finite_kinematics):
            points = axis.scatter(
                truth_muon_pion_angles[finite_kinematics],
                truth_pion_kinetic_energy[finite_kinematics],
                c=test_scores[finite_kinematics],
                cmap="viridis",
                vmin=0.0,
                vmax=1.0,
                s=24,
                alpha=0.65,
                edgecolors="none",
            )
            fig.colorbar(points, ax=axis, label="pion_score")
            axis.set_xlabel("Truth leading π–μ opening angle [degrees]")
            axis.set_ylabel("Truth leading-pion kinetic energy [GeV]")
            axis.set_title("Where true-pion events receive ambiguous scores")
        else:
            axis.text(
                0.5,
                0.5,
                "No events had both finite pion kinetic energy and opening angle.",
                transform=axis.transAxes,
                ha="center",
                va="center",
            )
            axis.axis("off")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    category_counts = {
        label: int(np.count_nonzero(mask)) for label, mask, _ in categories
    }
    return {
        "ambiguity_diagnostics_pdf": pdf_path.name,
        "ambiguity_binned_metrics_csv": binned_path.name,
        "ambiguous_events_csv": ambiguous_path.name,
        "ambiguous_score_definition": (
            f"{AMBIGUOUS_SCORE_LOW:.2f} <= pion_score <= "
            f"{AMBIGUOUS_SCORE_HIGH:.2f}"
        ),
        "ambiguous_event_count": int(np.count_nonzero(ambiguous)),
        "ambiguous_event_fraction": float(np.mean(ambiguous)),
        "ambiguity_truth_topology_counts": category_counts,
    }


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
        ("pion_score_roc_auc", "val_pion_score_roc_auc"),
        ("pion_score_pr_auc", "val_pion_score_pr_auc"),
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
                f"dE={energy_loss:.3f}, L={length:.3f}, "
                f"MRD angle={np.degrees(angle):.1f}°"
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
    examples_per_category: int = 5,
    max_examples: int = 20,
) -> Tuple[List[str], int]:
    """Write up to 20 readable truth-pion examples to one multipage PDF."""
    pion_indices = np.flatnonzero(y_test == 1)
    if len(pion_indices) == 0:
        return [], 0

    charged_pion_indices = pion_indices[
        (
            truth_pion_counts_test[pion_indices, 0]
            + truth_pion_counts_test[pion_indices, 1]
        )
        > 0
    ]
    ranking_options = [
        (
            "Charged-pion example",
            charged_pion_indices[
                np.argsort(test_scores[charged_pion_indices])[::-1]
            ],
        ),
        (
            "High pion-score example",
            pion_indices[np.argsort(test_scores[pion_indices])[::-1]],
        ),
        (
            "Decision-boundary example",
            pion_indices[
                np.argsort(np.abs(test_scores[pion_indices] - threshold))
            ],
        ),
        (
            "Low pion-score example",
            pion_indices[np.argsort(test_scores[pion_indices])],
        ),
    ]

    used_indices = set()
    selected_categories = []
    for category_title, candidates in ranking_options:
        category_indices = []
        for event_index in candidates:
            integer_index = int(event_index)
            if integer_index in used_indices:
                continue
            category_indices.append(integer_index)
            used_indices.add(integer_index)
            if len(category_indices) == examples_per_category:
                break
        if category_indices:
            selected_categories.append((category_title, category_indices))
        if len(used_indices) >= max_examples:
            break

    target_count = min(max_examples, len(pion_indices))
    if len(used_indices) < target_count:
        additional_indices = []
        for event_index in pion_indices[np.argsort(test_scores[pion_indices])[::-1]]:
            integer_index = int(event_index)
            if integer_index in used_indices:
                continue
            additional_indices.append(integer_index)
            used_indices.add(integer_index)
            if len(used_indices) == target_count:
                break
        if additional_indices:
            selected_categories.append(
                ("Additional representative pion example", additional_indices)
            )

    raw_column_titles = [
        "Angular full hitPE",
        "Angular tank-cluster hitPE",
        "Unfolded full hitPE",
        "Unfolded tank-cluster hitPE",
    ]
    pdf_path = output.with_suffix(".pion_examples.pdf")
    example_number = 0
    with PdfPages(
        pdf_path,
        metadata={
            "Title": "ANNIE held-out truth-pion examples",
            "Subject": (
                "Readable event information plus PMT images and Grad-CAM maps"
            ),
        },
    ) as pdf:
        for category_title, order in selected_categories:
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

            for row_index, event_index in enumerate(order):
                example_number += 1
                angular = angular_batch[row_index][:, 1:-1, :]
                detector = detector_batch[row_index]
                fig = plt.figure(figsize=(17, 9.5))
                grid = fig.add_gridspec(
                    2,
                    6,
                    height_ratios=(1.45, 4.5),
                    left=0.035,
                    right=0.985,
                    bottom=0.07,
                    top=0.91,
                    hspace=0.20,
                    wspace=0.28,
                )
                info_axis = fig.add_subplot(grid[0, :])
                info_axis.axis("off")
                panel_axes = [fig.add_subplot(grid[1, column]) for column in range(6)]

                panels = [
                    (angular[:, :, 0], "auto"),
                    (angular[:, :, 1], "auto"),
                    (detector[:, :, 0], "equal"),
                    (detector[:, :, 1], "equal"),
                ]
                for axis, (panel, aspect), title in zip(
                    panel_axes[:4], panels, raw_column_titles
                ):
                    axis.imshow(panel, origin="lower", aspect=aspect, cmap="magma")
                    axis.set_title(title, fontsize=9)

                angular_heat = angular_heatmaps[row_index][:, 1:-1]
                panel_axes[4].imshow(
                    angular[:, :, 0], origin="lower", aspect="auto", cmap="gray"
                )
                panel_axes[4].imshow(
                    angular_heat,
                    origin="lower",
                    aspect="auto",
                    cmap="jet",
                    alpha=0.55,
                    vmin=0.0,
                    vmax=1.0,
                )
                panel_axes[4].set_title(
                    "Angular Grad-CAM\n(red = strongest influence)", fontsize=9
                )

                panel_axes[5].imshow(
                    detector[:, :, 0], origin="lower", aspect="equal", cmap="gray"
                )
                panel_axes[5].imshow(
                    detector_heatmaps[row_index],
                    origin="lower",
                    aspect="equal",
                    cmap="jet",
                    alpha=0.55,
                    vmin=0.0,
                    vmax=1.0,
                )
                panel_axes[5].set_title(
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
                track_text = position_text.replace(
                    "MRD tracks: ", "MRD tracks:\n"
                ).replace("; ", "\n")
                prediction_text = (
                    "pion-like"
                    if test_scores[event_index] >= threshold
                    else "non-pion-like"
                )
                info_text = (
                    f"File: {source_name}    Entry: {int(entries_test[event_index])}    "
                    f"pion_score={test_scores[event_index]:.3f} "
                    f"(threshold={threshold:.3f}, {prediction_text})\n"
                    f"Truth: π⁺={pi_plus}, π⁻={pi_minus}, π⁰={pi_zero}    "
                    f"{angle_text}    {muon_energy_text}    {pion_energy_text}\n"
                    f"{scalar_text}\n{track_text}"
                )
                info_axis.text(
                    0.01,
                    0.98,
                    info_text,
                    transform=info_axis.transAxes,
                    va="top",
                    ha="left",
                    fontsize=9.5,
                    linespacing=1.25,
                    family="monospace",
                    bbox={
                        "boxstyle": "round,pad=0.55",
                        "facecolor": "#f4f4f4",
                        "edgecolor": "#777777",
                        "linewidth": 0.8,
                    },
                )
                fig.suptitle(
                    (
                        f"{category_title} — event {example_number} of "
                        f"{target_count} (truth information is annotation only)"
                    ),
                    fontsize=13,
                )
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

    print(
        f"Wrote {example_number} readable pion examples to {pdf_path}"
        + (
            ""
            if example_number >= 15
            else " (fewer than 15 truth-pion test events were available)"
        )
    )
    return [pdf_path.name], example_number





def _grad_cam_heatmaps(
    model: tf.keras.Model,
    conv_layer_name: str,
    inputs: Dict[str, np.ndarray],
    target_shape: Tuple[int, int],
) -> np.ndarray:
    """Calculate batched Grad-CAM maps for one convolutional tower."""
    grad_model = tf.keras.Model(
        model.inputs,
        [
            model.get_layer(conv_layer_name).output,
            model.get_layer("pion_score").output,
        ],
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
    seed: int = 0,
    mrd_ablation_repeats: int = 5,
    mrd_branch_importance_repeats: int = 3,
) -> Dict[str, object]:
    prediction_outputs = model.predict(
        {
            "pmt_angular_image": images_test,
            "pmt_unfolded_image": detector_images_test,
            "event_features": event_features_test,
            "mrd_track_starts": mrd_track_starts_test,
            "mrd_track_properties": mrd_track_properties_test,
            "mrd_track_mask": mrd_track_mask_test,
        },
        verbose=0,
    )
    test_scores = np.asarray(prediction_outputs["pion_score"]).reshape(-1)
    mrd_ablation = evaluate_mrd_permutation_importance(
        model,
        images_test,
        detector_images_test,
        event_features_test,
        mrd_track_starts_test,
        mrd_track_properties_test,
        mrd_track_mask_test,
        y_test,
        test_scores,
        threshold,
        output,
        seed,
        mrd_ablation_repeats,
    )
    mrd_branch_importance = evaluate_mrd_branch_importance(
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
        seed,
        mrd_branch_importance_repeats,
    )
    (
        truth_muon_pion_angles,
        truth_muon_momenta,
        truth_pion_momenta,
        truth_pion_total_energy,
        truth_leading_pion_pdg,
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
    fallback_pion_mass = np.where(
        charged_pion_present,
        CHARGED_PION_MASS_GEV,
        np.where(neutral_pion_present, NEUTRAL_PION_MASS_GEV, np.nan),
    )
    leading_pion_mass = np.where(
        np.abs(truth_leading_pion_pdg) == 211,
        CHARGED_PION_MASS_GEV,
        np.where(
            truth_leading_pion_pdg == 111,
            NEUTRAL_PION_MASS_GEV,
            fallback_pion_mass,
        ),
    )
    momentum_derived_pion_ke = _kinetic_energy(
        truth_pion_momenta, leading_pion_mass
    )
    energy_derived_pion_ke = truth_pion_total_energy - leading_pion_mass
    valid_total_energy = (
        np.isfinite(truth_pion_total_energy)
        & np.isfinite(leading_pion_mass)
        & (energy_derived_pion_ke >= 0.0)
    )
    truth_pion_kinetic_energy = np.where(
        valid_total_energy,
        energy_derived_pion_ke,
        momentum_derived_pion_ke,
    )

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
                "truth_leading_pion_pdg",
                "truth_leading_pion_muon_opening_angle_deg",
                "truth_leading_pion_total_energy_GeV",
                "truth_muon_kinetic_energy_GeV",
                "truth_leading_pion_kinetic_energy_GeV",
                *PION_EVENT_FEATURE_NAMES,
                *track_column_names,
                "pion_score",
                "pion_threshold",
                "pion_prediction",
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
                    float(truth_leading_pion_pdg[event_index]),
                    float(truth_muon_pion_angles[event_index]),
                    float(truth_pion_total_energy[event_index]),
                    float(truth_muon_kinetic_energy[event_index]),
                    float(truth_pion_kinetic_energy[event_index]),
                    *event_features_test[event_index].astype(float),
                    *track_values,
                    float(test_scores[event_index]),
                    float(threshold),
                    int(test_scores[event_index] >= threshold),
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
    pion_example_pages, pion_example_event_count = plot_pion_example_pages(
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
    ambiguity_diagnostics = plot_ambiguity_diagnostics(
        y_test,
        test_scores,
        detector_images_test,
        source_paths_test,
        entries_test,
        truth_pion_counts_test,
        truth_muon_pion_angles,
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
        "pion_example_event_count": pion_example_event_count,
        **ambiguity_diagnostics,
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
        **mrd_ablation,
        **mrd_branch_importance,
    }


def write_training_summary(output: Path, summary: Dict[str, object]) -> Tuple[str, str]:
    """Write compact machine-readable and human-readable run summaries."""
    json_path = output.with_suffix(".training_summary.json")
    text_path = output.with_suffix(".training_summary.txt")
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    lines = [
        "ANNIE pion-ID training summary",
        "===============================",
        f"Model: {output}",
        f"Output: pion_score only",
        f"Selection: {summary['event_selection_expression']}",
        f"PMT response: {summary['training_pmt_response']}",
        "",
        "Input files",
        "-----------",
    ]
    for file_info in summary["input_files"]:
        lines.extend(
            [
                f"- {file_info['source']}",
                (
                    f"  response={file_info['pmt_response']}; "
                    f"read={file_info['events_read']}; "
                    f"selected={file_info['events_selected']}; "
                    f"no_pion={file_info['no_pion_selected']}; "
                    f"pion={file_info['pion_selected']}; "
                    "misaligned_rejected="
                    f"{file_info['misaligned_events_rejected']}"
                ),
            ]
        )
    split = summary["data_split"]
    classes = summary["class_counts"]
    performance = summary["test_performance"]
    operating_point = performance["configured_operating_point"]
    lines.extend(
        [
            "",
            "Event counts",
            "------------",
            f"Read: {summary['events_read']}",
            (
                f"Selected: {summary['events_selected']} "
                f"({100.0 * summary['selection_efficiency']:.2f}%)"
            ),
            (
                f"Truth classes: no pion={classes['no_pion']}, "
                f"pion={classes['pion']}"
            ),
            (
                f"Split: train={split['training_events']}, "
                f"validation={split['validation_events']}, "
                f"test={split['testing_events']}"
            ),
            "",
            "Training",
            "--------",
            (
                f"Epochs: {summary['epochs_completed']} completed / "
                f"{summary['epochs_requested']} requested"
            ),
            f"Batch size: {summary['batch_size']}",
            f"Trainable parameters: {summary['trainable_parameters']}",
            (
                "Truth pion-muon opening-angle unit: "
                f"{summary['truth_muon_pion_opening_angle_units']}"
            ),
            f"Training time: {summary['training_seconds']:.1f} s",
            f"Total run time: {summary['total_seconds']:.1f} s",
            (
                f"Pion example events: {summary['pion_example_event_count']} "
                f"({summary['pion_example_pdf']})"
            ),
            (
                f"Ambiguous-score events: {summary['ambiguous_event_count']} "
                f"({100.0 * summary['ambiguous_event_fraction']:.2f}%; "
                f"{summary['ambiguity_diagnostics_pdf']})"
            ),
            (
                "Best validation PR AUC: "
                f"{summary['best_validation_pr_auc']:.6f} at epoch "
                f"{summary['best_validation_epoch']}"
                if summary["best_validation_pr_auc"] is not None
                else "Best validation PR AUC: unavailable"
            ),
            "",
            "Held-out pion performance",
            "-------------------------",
            f"ROC AUC: {performance['roc_auc']:.6f}",
            f"Average precision: {performance['average_precision']:.6f}",
            f"Brier score: {performance['brier_score']:.6f}",
            (
                f"At pion_score >= {operating_point['threshold']:.3f}: "
                f"efficiency={operating_point['efficiency']:.6f}, "
                f"purity={operating_point['purity']:.6f}, "
                "efficiency*purity="
                f"{operating_point['efficiency_x_purity']:.6f}, "
                f"F1={operating_point['f1_score']:.6f}, "
                "balanced_accuracy="
                f"{operating_point['balanced_accuracy']:.6f}, "
                f"MCC={operating_point['matthews_correlation_coefficient']:.6f}"
            ),
        ]
    )
    text_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path.name, text_path.name


def main() -> None:
    args = parse_args()
    run_start_time = time.perf_counter()
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be between 0 and 1")
    if args.mrd_ablation_repeats < 1:
        raise ValueError("--mrd-ablation-repeats must be at least 1")
    if args.mrd_branch_importance_repeats < 1:
        raise ValueError("--mrd-branch-importance-repeats must be at least 1")
    if args.image_height < 4 or args.image_width < 8:
        raise ValueError("The PMT image must be at least 4 x 8 bins")
    if args.detector_height < 24 or args.detector_width < 16:
        raise ValueError("The unfolded detector image must be at least 24 x 16 bins")
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    data_load_start_time = time.perf_counter()
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
        response_info,
    ) = load_data(args)
    data_loading_seconds = time.perf_counter() - data_load_start_time
    responses = response_info["responses"]
    response_modes = sorted(responses)
    training_pmt_response = (
        response_modes[0] if len(response_modes) == 1 else "mixed"
    )
    metadata_response = responses.get("tuned", responses.get("raw"))
    training_sources = [
        source
        for group in response_info["groups"]
        for source in group["sources"]
    ]
    if n_mrd_tracks_truncated:
        print(
            f"Retained the first {MAX_MRD_TRACKS} MRD starts for "
            f"{n_mrd_tracks_truncated} selected events with more tracks."
        )
    counts = np.bincount(labels.astype(np.int64), minlength=2)
    if np.any(counts == 0):
        raise ValueError(f"Both classes are required; class counts are {counts.tolist()}")
    indices = np.arange(len(labels))
    try:
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
    except ValueError as error:
        raise ValueError(
            "Could not create the 70/15/15 split stratified by the pion label. "
            f"Class counts: {counts.tolist()}"
        ) from error
    y_train = labels[train_indices]
    y_test = labels[test_indices]
    train_counts = np.bincount(y_train.astype(np.int64), minlength=2)
    pion_class_weight = {
        0: len(y_train) / (2.0 * train_counts[0]),
        1: len(y_train) / (2.0 * train_counts[1]),
    }
    training_class_weights = {"pion_score": pion_class_weight}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    history_path = args.output.with_suffix(".history.csv")
    angular_mean, angular_variance = _channel_moments(images, train_indices)
    detector_mean, detector_variance = _channel_moments(
        detector_images, train_indices
    )
    model = build_model(
        tuple(images.shape[1:]),
        tuple(detector_images.shape[1:]),
        angular_mean,
        angular_variance,
        detector_mean,
        detector_variance,
        event_features[train_indices],
        mrd_track_starts[train_indices],
        mrd_track_properties[train_indices],
        mrd_track_mask[train_indices],
    )
    sequence_arguments = (
        images,
        detector_images,
        event_features,
        mrd_track_starts,
        mrd_track_properties,
        mrd_track_mask,
        labels,
    )
    train_sequence = RingBatchSequence(
        *sequence_arguments,
        train_indices,
        args.batch_size,
        True,
        args.seed,
        training_class_weights,
    )
    validation_sequence = RingBatchSequence(
        *sequence_arguments,
        val_indices,
        args.batch_size,
        False,
        args.seed,
    )
    test_sequence = RingBatchSequence(
        *sequence_arguments,
        test_indices,
        args.batch_size,
        False,
        args.seed,
    )
    image_storage_gib = (images.nbytes + detector_images.nbytes) / (1024**3)
    print(
        f"Stored PMT images as {images.dtype} using "
        f"{image_storage_gib:.2f} GiB; training batches are loaded on demand."
    )
    training_start_time = time.perf_counter()
    fit_history = model.fit(
        train_sequence,
        validation_data=validation_sequence,
        epochs=args.epochs,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_pion_score_pr_auc",
                mode="max",
                patience=12,
                restore_best_weights=True,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_pion_score_pr_auc",
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
    training_seconds = time.perf_counter() - training_start_time
    metrics = model.evaluate(
        test_sequence,
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
        args.seed,
        args.mrd_ablation_repeats,
        args.mrd_branch_importance_repeats,
    )
    print(
        "Test ROC AUC cross-check: "
        f"{evaluation_files['roc_auc_sklearn']:.5f}; "
        f"average precision: {evaluation_files['average_precision']:.5f}"
    )
    val_pr_history = fit_history.history.get("val_pion_score_pr_auc", [])
    best_validation_pr_auc = (
        float(np.max(val_pr_history)) if val_pr_history else None
    )
    best_validation_epoch = (
        int(np.argmax(val_pr_history) + 1) if val_pr_history else None
    )
    split_class_counts = {
        "training": {
            "no_pion": int(np.sum(labels[train_indices] == 0)),
            "pion": int(np.sum(labels[train_indices] == 1)),
        },
        "validation": {
            "no_pion": int(np.sum(labels[val_indices] == 0)),
            "pion": int(np.sum(labels[val_indices] == 1)),
        },
        "testing": {
            "no_pion": int(np.sum(labels[test_indices] == 0)),
            "pion": int(np.sum(labels[test_indices] == 1)),
        },
    }
    training_summary = {
        "model_file": args.output.name,
        "output_heads": ["pion_score"],
        "input_files": response_info["files"],
        "input_file_count": len(response_info["files"]),
        "events_read": int(
            sum(file_info["events_read"] for file_info in response_info["files"])
        ),
        "events_selected": int(len(labels)),
        "selection_efficiency": float(
            len(labels)
            / max(
                1,
                sum(
                    file_info["events_read"]
                    for file_info in response_info["files"]
                ),
            )
        ),
        "event_selection_expression": (
            "none" if args.no_event_cuts else PION_ID_SELECTION_EXPRESSION
        ),
        "training_pmt_response": training_pmt_response,
        "image_branches": [args.hitpe_branch, tank_branch],
        "image_channels": [
            f"log1p_{args.hitpe_branch}",
            f"log1p_{tank_branch}",
            f"{args.hitpe_branch}_over_total_event_pe",
            f"{tank_branch}_over_total_tankcluster_pe",
        ],
        "scalar_features": PION_EVENT_FEATURE_NAMES,
        "mrd_track_start_branches": MRD_TRACK_START_BRANCHES,
        "mrd_track_property_branches": MRD_TRACK_PROPERTY_BRANCHES,
        "truth_muon_pion_opening_angle_units": "degrees",
        "class_counts": {"no_pion": int(counts[0]), "pion": int(counts[1])},
        "split_class_counts": split_class_counts,
        "data_split": {
            "training_events": int(len(train_indices)),
            "validation_events": int(len(val_indices)),
            "testing_events": int(len(test_indices)),
            "fractions": [TRAIN_FRACTION, VALIDATION_FRACTION, TEST_FRACTION],
            "stratified_by": "pion label",
            "random_seed": args.seed,
        },
        "class_weights": {
            str(class_id): float(weight)
            for class_id, weight in pion_class_weight.items()
        },
        "epochs_requested": int(args.epochs),
        "epochs_completed": int(len(fit_history.epoch)),
        "stopped_early": bool(len(fit_history.epoch) < args.epochs),
        "best_validation_pr_auc": best_validation_pr_auc,
        "best_validation_epoch": best_validation_epoch,
        "batch_size": int(args.batch_size),
        "model_parameters": int(model.count_params()),
        "trainable_parameters": int(
            sum(np.prod(variable.shape) for variable in model.trainable_weights)
        ),
        "data_loading_seconds": float(data_loading_seconds),
        "training_seconds": float(training_seconds),
        "total_seconds": float(time.perf_counter() - run_start_time),
        "pion_example_event_count": int(
            evaluation_files["pion_example_event_count"]
        ),
        "pion_example_pdf": evaluation_files["pion_example_pages"][0]
        if evaluation_files["pion_example_pages"]
        else None,
        "ambiguous_event_count": int(evaluation_files["ambiguous_event_count"]),
        "ambiguous_event_fraction": float(
            evaluation_files["ambiguous_event_fraction"]
        ),
        "ambiguity_diagnostics_pdf": evaluation_files[
            "ambiguity_diagnostics_pdf"
        ],
        "test_keras_metrics": {
            name: float(value) for name, value in metrics.items()
        },
        "test_performance": {
            "roc_auc": float(evaluation_files["roc_auc_sklearn"]),
            "average_precision": float(evaluation_files["average_precision"]),
            "brier_score": float(evaluation_files["brier_score"]),
            "configured_operating_point": evaluation_files[
                "pion_metrics_at_configured_threshold"
            ],
            "threshold_0p80_operating_point": evaluation_files[
                "pion_metrics_at_threshold_0p80"
            ],
        },
    }
    summary_json_name, summary_text_name = write_training_summary(
        args.output, training_summary
    )
    metadata = {
        "model_type": "annie_pion_ring_cnn",
        "model_variant": (
            "pion_only_absolute_and_event_normalized_pe_images_plus_mrd"
        ),
        "tensorflow_version": tf.__version__,
        "training_sources": training_sources,
        "training_source_groups": response_info["groups"],
        "training_source_files": response_info["files"],
        "tree": args.tree,
        "geometry_file": args.geometry.name,
        "pmt_mask": args.pmt_mask,
        "pmt_count_included": geometry.included_count,
        "excluded_pmt_ids": geometry.excluded_ids,
        "training_pmt_response": training_pmt_response,
        "pmt_response_scope": (
            "separate_branch_kind_0_hitPE_and_branch_kind_1_tankcluster"
            if metadata_response.payload_format == "final_pmt_tuning_parameters"
            else "shared_legacy_map_hitPE_and_hitPE_tankcluster"
        ),
        "pmt_response_calibration_file": (
            metadata_response.calibration_path.name
            if metadata_response.calibration_path
            else None
        ),
        "pmt_response_calibration_sha256": metadata_response.calibration_sha256,
        "pmt_response_payload_format": metadata_response.payload_format,
        "pmt_tune_variant": metadata_response.tune_variant,
        "pmt_tune_random_stream_version": metadata_response.random_stream_version,
        "pmt_response_mapped_pmt_count": metadata_response.mapped_pmt_count,
        "hitpe_branch": args.hitpe_branch,
        "hitid_branch": args.hitid_branch,
        "tankcluster_branch": tank_branch,
        "tankcluster_id_branch": args.tankcluster_id_branch,
        "image_height": args.image_height,
        "image_width": args.image_width,
        "detector_height": args.detector_height,
        "detector_width": args.detector_width,
        "image_storage_dtype": str(images.dtype),
        "training_data_feeding": "indexed_batches_without_split_copies",
        "batch_size": args.batch_size,
        "root_chunk_size": args.chunk_size,
        "image_channel_source": (
            "full_event_and_tankcluster_absolute_and_event_normalized"
        ),
        "image_channel_count": 4,
        "include_normalized_pe_channels": True,
        "head_image_channels": {
            "pion_score": [
                f"log1p_{args.hitpe_branch}",
                f"log1p_{tank_branch}",
                f"{args.hitpe_branch}_over_total_event_pe",
                f"{tank_branch}_over_total_tankcluster_pe",
            ],
        },
        "head_scalar_features": {
            "pion_score": ["numMRDTracks"],
        },
        "channels": [
            (
                "log1p_mixed_raw_and_tuned_hitPE"
                if training_pmt_response == "mixed"
                else (
                    "log1p_tuned_hitPE"
                    if training_pmt_response == "tuned"
                    else "log1p_hitPE"
                )
            ),
            (
                "log1p_mixed_raw_and_tuned_hitPE_tankcluster"
                if training_pmt_response == "mixed"
                else (
                    "log1p_tuned_hitPE_tankcluster"
                    if training_pmt_response == "tuned"
                    else "log1p_hitPE_tankcluster"
                )
            ),
            "hitPE_over_total_event_PE",
            "hitPE_tankcluster_over_total_tankcluster_PE",
        ],
        "projections": [
            "vertex_centered_equirectangular_with_azimuth_wrap",
            "annotation_free_unfolded_barrel_top_bottom",
        ],
        "event_feature_input_name": "event_features",
        "event_feature_names": PION_EVENT_FEATURE_NAMES,
        "event_feature_scalar_branches": PION_EVENT_FEATURE_NAMES,
        "mrd_track_start_input_name": "mrd_track_starts",
        "mrd_track_property_input_name": "mrd_track_properties",
        "mrd_track_mask_input_name": "mrd_track_mask",
        "mrd_track_start_branches": MRD_TRACK_START_BRANCHES,
        "mrd_track_property_branches": MRD_TRACK_PROPERTY_BRANCHES,
        "mrd_track_coordinate_order": ["X", "Y", "Z"],
        "mrd_track_property_order": MRD_TRACK_PROPERTY_BRANCHES,
        "mrd_track_property_layout_support": (
            "per_track_vectors_or_event_scalars_broadcast_to_valid_track_slots"
        ),
        "max_mrd_tracks": MAX_MRD_TRACKS,
        "mrd_track_overflow_policy": "first_four_in_stored_branch_order",
        "mrd_track_overflow_events": n_mrd_tracks_truncated,
        "event_feature_preprocessing": (
            "numMRDTracks_scalar_plus_zero_padded_mrd_xyz_and_"
            "energy_loss_length_angle_track_set_and_mask;_separate_"
            "normalization_adapted_on_training_values_and_valid_tracks_only;_"
            "masked_permutation_invariant_max_pooling"
        ),
        "label": "charged_pion_present" if args.charged_only else "any_pion_present",
        "output_heads": ["pion_score"],
        "truth_branches": ["truePiPlusCher", "truePiMinusCher", "truePi0"],
        "truth_muon_pion_opening_angle_annotation_only": True,
        "truth_muon_pion_opening_angle_units": "degrees",
        "truth_muon_pion_kinetic_energy_annotation_only": True,
        "hadronic_p4_annotation_only": True,
        "hadronic_p4_component_order": ["px", "py", "pz", "E"],
        "hadronic_p4_units": "GeV",
        "leading_pion_definition": (
            "highest-momentum PDG 211, -211, or 111 entry in aligned "
            "hadronic_pdgs/hadronic_p4"
        ),
        "truth_muon_momentum_fallback": (
            "trueFSLMomentum_X/Y/Z in MeV, converted to GeV; requires "
            "abs(trueFSLPdg) == 13 when trueFSLPdg is present"
        ),
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
        "event_selection": "pion_id",
        "event_selection_source": "user_requested_training_preselection",
        "event_selection_expression": PION_ID_SELECTION_EXPRESSION,
        "event_selection_applied": not args.no_event_cuts,
        "bdt_preselection": False,
        "threshold": args.threshold,
        "data_split": {
            "training_fraction": TRAIN_FRACTION,
            "validation_fraction": VALIDATION_FRACTION,
            "testing_fraction": TEST_FRACTION,
            "training_events": int(len(train_indices)),
            "validation_events": int(len(val_indices)),
            "testing_events": int(len(test_indices)),
            "stratified_by_pion_label": True,
            "random_seed": args.seed,
        },
        "class_counts": {"no_pion": int(counts[0]), "pion": int(counts[1])},
        "misaligned_events_rejected": n_misaligned,
        "test_metrics": {name: float(value) for name, value in metrics.items()},
        "training_history": history_path.name,
        "evaluation": evaluation_files,
        "training_summary": training_summary,
        "training_summary_json": summary_json_name,
        "training_summary_text": summary_text_name,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Saved {args.output}, {args.output.with_suffix('.json')}, "
        f"{history_path}, {summary_json_name}, and {summary_text_name}"
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
    print(
        "Ambiguous score band "
        f"({evaluation_files['ambiguous_score_definition']}): "
        f"{evaluation_files['ambiguous_event_count']} events, "
        f"{evaluation_files['ambiguous_event_fraction']:.1%} of the test set"
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
    mrd_report = evaluation_files["mrd_permutation_importance"]
    mrd_drop = mrd_report["performance_drop"]
    print(
        "MRD permutation importance (positive means MRD was useful): "
        f"pion ROC-AUC drop={mrd_drop['pion_roc_auc']:.5f}, "
        f"pion average-precision drop="
        f"{mrd_drop['pion_average_precision']:.5f}, "
        f"pion efficiency*purity drop="
        f"{mrd_drop['pion_efficiency_x_purity']:.5f}"
    )
    print("MRD branch ranking by mean absolute pion-score impact:")
    for rank, branch_result in enumerate(
        evaluation_files["mrd_branch_importance"]["ranking"], start=1
    ):
        print(
            f"  {rank}. {branch_result['branch']}: "
            f"mean |impact|="
            f"{branch_result['mean_abs_pion_score_impact']:.6f}, "
            f"ROC-AUC drop={branch_result['pion_roc_auc_drop']:.5f}"
        )


if __name__ == "__main__":
    main()
