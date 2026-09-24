"""Load and replay ANNIE per-PMT MC-to-data response calibrations."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import awkward as ak
import numpy as np
import uproot


PMT_RESPONSE_CHOICES = ("raw", "tuned")
PMT_TUNE_VARIANT_CHOICES = ("response", "final")
FIT_MINIMUM_PE = 0.0
FIT_MAXIMUM_PE = 350.0
RESIDUAL_WEIGHT_MAXIMUM = 20.0
FINAL_TUNE_TREE = "final_pmt_tuning_parameters"
_RESPONSE_KEY = re.compile(
    r"(?:^|/)PMT_Calibrations/PMT_(\d+)/response_map(?:;\d+)?$"
)


@dataclass(frozen=True)
class TuneRegion:
    low: float
    high: float
    gain: float
    delta: float
    sigma: float
    residual_edges: np.ndarray
    residual_weights: np.ndarray


def _mix_tune_bits(values: np.ndarray) -> np.ndarray:
    """Vectorized unsigned-64 implementation of C++ MixTuneBits."""
    values = np.asarray(values, dtype=np.uint64)
    with np.errstate(over="ignore"):
        values = values + np.uint64(0x9E3779B97F4A7C15)
        values = (values ^ (values >> np.uint64(30))) * np.uint64(
            0xBF58476D1CE4E5B9
        )
        values = (values ^ (values >> np.uint64(27))) * np.uint64(
            0x94D049BB133111EB
        )
    return values ^ (values >> np.uint64(31))


def deterministic_uniform(
    entries: np.ndarray,
    hit_indices: np.ndarray,
    detector_ids: np.ndarray,
    branch_kind: int,
    stream: int,
) -> np.ndarray:
    """Match ordered-field-mixing-v2 in the C++ calibration program."""
    count = len(entries)
    state = np.full(count, np.uint64(20260919), dtype=np.uint64)
    fields = (
        np.asarray(entries, dtype=np.uint64),
        np.asarray(hit_indices, dtype=np.uint64),
        np.asarray(detector_ids, dtype=np.uint64),
        np.full(count, np.uint64(branch_kind), dtype=np.uint64),
        np.full(count, np.uint64(stream), dtype=np.uint64),
    )
    for field in fields:
        state = _mix_tune_bits(state ^ field)
    return (_mix_tune_bits(state) >> np.uint64(11)).astype(np.float64) / float(
        1 << 53
    )


def deterministic_normal(
    entries: np.ndarray,
    hit_indices: np.ndarray,
    detector_ids: np.ndarray,
    branch_kind: int,
) -> np.ndarray:
    u1 = np.maximum(
        np.finfo(np.float64).tiny,
        deterministic_uniform(entries, hit_indices, detector_ids, branch_kind, 0),
    )
    u2 = deterministic_uniform(entries, hit_indices, detector_ids, branch_kind, 1)
    return np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)


class PMTResponse:
    """Apply legacy response maps or the final two-branch PMT tune payload."""

    def __init__(
        self,
        mode: str,
        calibration_path: Optional[Path] = None,
        tune_variant: str = "final",
    ):
        if mode not in PMT_RESPONSE_CHOICES:
            raise ValueError(f"Unknown PMT response mode {mode!r}")
        if tune_variant not in PMT_TUNE_VARIANT_CHOICES:
            raise ValueError(f"Unknown PMT tune variant {tune_variant!r}")
        if mode == "raw" and calibration_path is not None:
            raise ValueError(
                "--pmt-response-calibration is only valid with "
                "--pmt-response tuned"
            )
        if mode == "tuned" and calibration_path is None:
            raise ValueError(
                "--pmt-response tuned requires --pmt-response-calibration "
                "pointing to an ANNIE PMT calibration ROOT file"
            )

        self.mode = mode
        self.tune_variant = tune_variant
        self.calibration_path = calibration_path
        self.calibration_sha256: Optional[str] = None
        self.payload_format = "raw"
        self.random_stream_version: Optional[str] = None
        self.maps: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        self.regions: Dict[Tuple[int, int], List[TuneRegion]] = {}
        if calibration_path is not None:
            self._load(calibration_path)

    def _load(self, path: Path) -> None:
        if not path.is_file():
            raise ValueError(f"PMT response calibration does not exist: {path}")
        digest = hashlib.sha256()
        with path.open("rb") as calibration_bytes:
            for block in iter(lambda: calibration_bytes.read(1024 * 1024), b""):
                digest.update(block)
        self.calibration_sha256 = digest.hexdigest()
        with uproot.open(path) as calibration:
            if FINAL_TUNE_TREE in calibration:
                self._load_final_tune_tree(calibration[FINAL_TUNE_TREE])
                self.payload_format = "final_pmt_tuning_parameters"
                if "tune_random_stream_version" in calibration:
                    version = calibration["tune_random_stream_version"]
                    self.random_stream_version = str(version.member("fTitle"))
                return
            self._load_legacy_maps(calibration)
            self.payload_format = "legacy_response_map"

    def _load_final_tune_tree(self, tree) -> None:
        required = [
            "detector_num",
            "branch_kind",
            "valid",
            "excluded_from_fit",
            "region_min",
            "region_max_exclusive",
            "Gain",
            "Delta",
            "Sigma",
            "residual_bin_edges",
            "residual_weights",
        ]
        missing = [name for name in required if name not in tree.keys()]
        if missing:
            raise ValueError(
                f"{FINAL_TUNE_TREE} is missing branches: {', '.join(missing)}"
            )
        arrays = tree.arrays(required, library="ak")
        scalar = {
            name: ak.to_numpy(arrays[name])
            for name in required
            if name not in {"residual_bin_edges", "residual_weights"}
        }
        for index in range(len(arrays)):
            if not bool(scalar["valid"][index]) or bool(
                scalar["excluded_from_fit"][index]
            ):
                continue
            edges = np.asarray(
                ak.to_list(arrays["residual_bin_edges"][index]), dtype=np.float64
            )
            weights = np.asarray(
                ak.to_list(arrays["residual_weights"][index]), dtype=np.float64
            )
            if len(edges) != len(weights) + 1:
                edges = np.empty(0, dtype=np.float64)
                weights = np.empty(0, dtype=np.float64)
            key = (
                int(scalar["detector_num"][index]),
                int(scalar["branch_kind"][index]),
            )
            self.regions.setdefault(key, []).append(
                TuneRegion(
                    low=float(scalar["region_min"][index]),
                    high=float(scalar["region_max_exclusive"][index]),
                    gain=float(scalar["Gain"][index]),
                    delta=float(scalar["Delta"][index]),
                    sigma=max(0.0, float(scalar["Sigma"][index])),
                    residual_edges=edges,
                    residual_weights=weights,
                )
            )
        for regions in self.regions.values():
            regions.sort(key=lambda region: region.low)
        if not self.regions:
            raise ValueError(f"No valid regions found in {FINAL_TUNE_TREE}")

    def _load_legacy_maps(self, calibration) -> None:
        for key, classname in calibration.classnames(recursive=True).items():
            match = _RESPONSE_KEY.search(key)
            if match is None or not classname.startswith("TGraph"):
                continue
            detector_id = int(match.group(1))
            x_values, y_values = calibration[key].values()
            x = np.asarray(x_values, dtype=np.float64)
            y = np.asarray(y_values, dtype=np.float64)
            finite = np.isfinite(x) & np.isfinite(y)
            x, y = x[finite], y[finite]
            if len(x) < 2:
                continue
            order = np.argsort(x, kind="stable")
            x, y = x[order], y[order]
            unique_x, unique_indices = np.unique(x, return_index=True)
            y = y[unique_indices]
            if len(unique_x) < 2:
                continue
            self.maps[detector_id] = (unique_x, np.maximum.accumulate(y))
        if not self.maps:
            raise ValueError(
                "Calibration contains neither final_pmt_tuning_parameters nor "
                "PMT_Calibrations/PMT_<id>/response_map TGraphs"
            )

    @property
    def mapped_pmt_ids(self) -> List[int]:
        if self.regions:
            return sorted({detector_id for detector_id, _ in self.regions})
        return sorted(self.maps)

    @property
    def mapped_pmt_count(self) -> int:
        return len(self.mapped_pmt_ids)

    def describe(self) -> str:
        if self.mode == "raw":
            return "PMT response: raw"
        digest = self.calibration_sha256[:12] if self.calibration_sha256 else "unknown"
        return (
            f"PMT response: tuned; payload={self.payload_format}; "
            f"variant={self.tune_variant}; PMTs={self.mapped_pmt_count}; "
            f"sha256={digest}..."
        )

    def apply(
        self,
        values: np.ndarray,
        detector_ids: np.ndarray,
        branch_kind: int = 0,
        entries: Optional[np.ndarray] = None,
        hit_indices: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Apply the selected tune to one aligned hit collection."""
        if self.mode == "raw":
            return values
        if self.regions:
            if entries is None or hit_indices is None:
                raise ValueError(
                    "The final PMT tune requires ROOT entries and per-event hit indices"
                )
            return self._apply_final_tree(
                values, detector_ids, branch_kind, entries, hit_indices
            )
        return self._apply_legacy_map(values, detector_ids)

    def _apply_legacy_map(
        self, values: np.ndarray, detector_ids: np.ndarray
    ) -> np.ndarray:
        corrected = values.copy()
        eligible = (
            np.isfinite(values)
            & (values >= FIT_MINIMUM_PE)
            & (values <= FIT_MAXIMUM_PE)
        )
        for detector_id in np.unique(detector_ids[eligible]):
            mapping = self.maps.get(int(detector_id))
            if mapping is None:
                continue
            selected = eligible & (detector_ids == detector_id)
            x, y = mapping
            corrected[selected] = np.interp(values[selected], x, y)
        return corrected

    def _apply_final_tree(
        self,
        values: np.ndarray,
        detector_ids: np.ndarray,
        branch_kind: int,
        entries: np.ndarray,
        hit_indices: np.ndarray,
    ) -> np.ndarray:
        corrected = values.copy()
        for detector_id in np.unique(detector_ids):
            regions = self.regions.get((int(detector_id), int(branch_kind)), [])
            for region in regions:
                selected = (
                    (detector_ids == detector_id)
                    & np.isfinite(values)
                    & (values >= region.low)
                    & (values < region.high)
                )
                if not np.any(selected):
                    continue
                transformed = region.gain * values[selected] + region.delta
                if region.sigma > 0.0:
                    transformed += region.sigma * deterministic_normal(
                        entries[selected],
                        hit_indices[selected],
                        detector_ids[selected],
                        branch_kind,
                    )
                transformed = np.maximum(0.0, transformed)
                transformed[~np.isfinite(transformed)] = np.nan
                if branch_kind == 1:
                    # C++ checks the transformed tank hit before residual
                    # duplication. Multiple retained copies may legitimately
                    # sum above 350 PE in one image pixel.
                    transformed[transformed >= FIT_MAXIMUM_PE] = np.nan
                if self.tune_variant == "final":
                    residual_weight = np.ones(len(transformed), dtype=np.float64)
                    if len(region.residual_edges) == len(region.residual_weights) + 1:
                        bins = np.searchsorted(
                            region.residual_edges, transformed, side="right"
                        ) - 1
                        inside = (
                            (bins >= 0)
                            & (bins < len(region.residual_weights))
                            & (transformed >= region.residual_edges[0])
                            & (transformed < region.residual_edges[-1])
                        )
                        residual_weight[inside] = region.residual_weights[bins[inside]]
                    residual_weight = np.clip(
                        residual_weight, 0.0, RESIDUAL_WEIGHT_MAXIMUM
                    )
                    copies = np.floor(residual_weight).astype(np.int64)
                    fraction = residual_weight - copies
                    copies += (
                        deterministic_uniform(
                            entries[selected],
                            hit_indices[selected],
                            detector_ids[selected],
                            branch_kind,
                            2,
                        )
                        < fraction
                    )
                    transformed *= copies
                corrected[selected] = transformed
        return corrected
