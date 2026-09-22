"""ANNIE feature extraction shared by training and inference."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import awkward as ak
import numpy as np
import uproot


FEATURE_NAMES = [
    "hitPE_sum",
    "hitPE_mean",
    "hitPE_std",
    "hitPE_tankcluster_sum",
    "hitPE_tankcluster_mean",
    "hitPE_tankcluster_std",
]

TRUTH_BRANCHES = ["truePiPlusCher", "truePiMinusCher", "truePi0"]

BDT_CUT_BRANCHES = [
    "simpleRecoFV",
    "promptMuonTotalPE",
    "simpleRecoEnergy",
    "simpleRecoCosTheta",
    "trigword",
    "HasTank",
    "HasMRD",
    "TankMRDCoinc",
    "NoVeto",
]

TANKCLUSTER_ALIASES = [
    "hitPE_tankcluster",
    "hitPE_tankclusters",
]


def resolve_tankcluster_branch(tree, requested: Optional[str]) -> str:
    available = set(tree.keys())
    if requested:
        if requested not in available:
            raise ValueError(f"Tank-cluster branch {requested!r} was not found")
        return requested
    matches = [name for name in TANKCLUSTER_ALIASES if name in available]
    if len(matches) != 1:
        raise ValueError(
            "Could not uniquely find the tank-cluster PE branch. Use "
            "--tankcluster-branch. Tried: " + ", ".join(TANKCLUSTER_ALIASES)
        )
    return matches[0]


def _to_numpy(values) -> np.ndarray:
    return ak.to_numpy(values).astype(np.float64, copy=False)


def summarize_pe(values, maximum: Optional[float] = None) -> Tuple[np.ndarray, ...]:
    """Return per-event sum, mean, and population standard deviation."""
    valid = np.isfinite(values) & (values >= 0.0)
    if maximum is not None:
        valid = valid & (values <= maximum)
    clean = values[valid]
    sums = _to_numpy(ak.sum(clean, axis=1))
    means = _to_numpy(ak.fill_none(ak.mean(clean, axis=1, mask_identity=True), 0.0))
    stds = _to_numpy(
        ak.fill_none(ak.std(clean, axis=1, ddof=0, mask_identity=True), 0.0)
    )
    return sums, means, stds


def build_features(arrays, hitpe_branch: str, tankcluster_branch: str) -> np.ndarray:
    hit_sum, hit_mean, hit_std = summarize_pe(arrays[hitpe_branch])
    tank_sum, tank_mean, tank_std = summarize_pe(
        arrays[tankcluster_branch], maximum=350.0
    )
    return np.column_stack(
        [hit_sum, hit_mean, hit_std, tank_sum, tank_mean, tank_std]
    ).astype(np.float32)


def build_pion_labels(arrays, charged_only: bool) -> np.ndarray:
    charged = (_to_numpy(arrays["truePiPlusCher"]) > 0) | (
        _to_numpy(arrays["truePiMinusCher"]) > 0
    )
    if charged_only:
        return charged.astype(np.float32)
    neutral = _to_numpy(arrays["truePi0"]) > 0
    return (charged | neutral).astype(np.float32)


def build_bdt_selection(arrays) -> np.ndarray:
    """Reproduce the event preselection in train_phaseII_bdt_simpler.C."""
    energy = _to_numpy(arrays["simpleRecoEnergy"])
    momentum = np.full_like(energy, -9999.0)
    physical = energy > 105.7
    momentum[physical] = (
        np.sqrt(energy[physical] ** 2 - 105.7**2) * 0.82 + 160.0
    )
    return (
        (_to_numpy(arrays["simpleRecoFV"]) != 0)
        & (_to_numpy(arrays["promptMuonTotalPE"]) > 500.0)
        & (_to_numpy(arrays["promptMuonTotalPE"]) < 3500.0)
        & (momentum > 600.0)
        & (momentum < 1200.0)
        & (_to_numpy(arrays["simpleRecoCosTheta"]) > 0.8)
        & (_to_numpy(arrays["trigword"]) == 5)
        & (_to_numpy(arrays["HasTank"]) == 1)
        & (_to_numpy(arrays["HasMRD"]) == 1)
        & (_to_numpy(arrays["TankMRDCoinc"]) == 1)
        & (_to_numpy(arrays["NoVeto"]) == 1)
    )


def iterate_root_features(
    paths: List[Path],
    tree_name: str,
    hitpe_branch: str,
    tankcluster_branch: Optional[str],
    apply_bdt_cuts: bool,
    include_truth: bool,
    charged_only: bool,
    chunk_size: str,
) -> Iterator[Dict[str, object]]:
    """Yield selected ANNIE event features in bounded-memory chunks."""
    for path in paths:
        with uproot.open(path) as root_file:
            if tree_name not in root_file:
                raise ValueError(f"Tree {tree_name!r} was not found in {path}")
            tree = root_file[tree_name]
            available = set(tree.keys())
            if hitpe_branch not in available:
                raise ValueError(f"Branch {hitpe_branch!r} was not found in {path}")
            tank_branch = resolve_tankcluster_branch(tree, tankcluster_branch)
            requested = [hitpe_branch, tank_branch]
            if apply_bdt_cuts:
                requested.extend(BDT_CUT_BRANCHES)
            if include_truth:
                requested.extend(TRUTH_BRANCHES)
            missing = [name for name in requested if name not in available]
            if missing:
                raise ValueError(f"{path} is missing branches: {', '.join(missing)}")

            entry_offset = 0
            for arrays in tree.iterate(
                list(dict.fromkeys(requested)), step_size=chunk_size, library="ak"
            ):
                n_entries = len(arrays)
                selected = (
                    build_bdt_selection(arrays)
                    if apply_bdt_cuts
                    else np.ones(n_entries, dtype=bool)
                )
                features = build_features(arrays, hitpe_branch, tank_branch)
                finite = np.isfinite(features).all(axis=1)
                selected &= finite
                result: Dict[str, object] = {
                    "path": path,
                    "entries": np.arange(
                        entry_offset, entry_offset + n_entries, dtype=np.int64
                    )[selected],
                    "features": features[selected],
                    "tankcluster_branch": tank_branch,
                    "n_read": n_entries,
                    "n_selected": int(selected.sum()),
                }
                if include_truth:
                    result["labels"] = build_pion_labels(arrays, charged_only)[selected]
                yield result
                entry_offset += n_entries

