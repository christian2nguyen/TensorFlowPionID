"""Build two-channel angular PMT images from ANNIE ROOT events."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import awkward as ak
import numpy as np
import uproot

from annie_features import (
    BDT_CUT_BRANCHES,
    TRUTH_BRANCHES,
    build_bdt_selection,
    build_pion_labels,
    resolve_tankcluster_branch,
)


DEFAULT_GEOMETRY = Path(__file__).with_name("PMT_position_id_info.cvs")
DEFAULT_HEIGHT = 16
DEFAULT_WIDTH = 32
DEFAULT_DETECTOR_HEIGHT = 48
DEFAULT_DETECTOR_WIDTH = 32
BEAM_SPLIT_Z = 1.681
VERTEX_BRANCHES = ["simpleRecoVtxX", "simpleRecoVtxY", "simpleRecoVtxZ"]

# Synchronized with PMTPositionInfo::ShutoffRecords and UsePMTForTankBDT in
# stv-analysis-Joint. PMT 358 is intentionally retained by the BDT workflow.
KNOWN_SHUTOFF_PMTS = {
    333,
    337,
    342,
    343,
    345,
    346,
    349,
    352,
    358,
    359,
    366,
    408,
    416,
    431,
    444,
    445,
}
BDT_RETAINED_SHUTOFF_PMT = 358
PMT_MASK_CHOICES = ("bdt", "on", "all")


class PMTGeometry:
    def __init__(self, path: Path, mask_mode: str = "bdt"):
        if mask_mode not in PMT_MASK_CHOICES:
            raise ValueError(f"Unknown PMT mask {mask_mode!r}")
        records = []
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.reader(handle):
                if not row or row[0].lstrip().startswith("#") or len(row) < 13:
                    continue
                try:
                    records.append(
                        (
                            int(row[0]),
                            row[2].strip().lower(),
                            float(row[4]),
                            float(row[5]),
                            float(row[6]),
                            row[12].strip().lower(),
                        )
                    )
                except ValueError:
                    continue
        if not records:
            raise ValueError(f"No PMT positions were read from {path}")
        max_id = max(record[0] for record in records)
        self.positions = np.full((max_id + 1, 3), np.nan, dtype=np.float64)
        # 0 = excluded/unknown, 1 = barrel, 2 = top, 3 = bottom.
        self.locations = np.zeros(max_id + 1, dtype=np.int8)
        self.mask_mode = mask_mode
        self.excluded_ids = []
        for detector_id, location, x, y, z, status in records:
            if mask_mode == "bdt":
                excluded = (
                    detector_id in KNOWN_SHUTOFF_PMTS
                    and detector_id != BDT_RETAINED_SHUTOFF_PMT
                )
            elif mask_mode == "on":
                excluded = status != "on"
            else:
                excluded = False
            if excluded:
                self.excluded_ids.append(detector_id)
            else:
                self.positions[detector_id] = (x, y, z)
                if "top" in location:
                    self.locations[detector_id] = 2
                elif "bottom" in location:
                    self.locations[detector_id] = 3
                else:
                    self.locations[detector_id] = 1
        self.excluded_ids.sort()
        self.included_count = len(records) - len(self.excluded_ids)

    def unfolded_bins(self, height: int, width: int) -> Tuple[np.ndarray, np.ndarray]:
        """Map active PMTs onto the barrel/top/bottom layout used by the C++ view."""
        if height < 24 or width < 16:
            raise ValueError("The unfolded detector image must be at least 24 x 16 bins")
        y_bins = np.full(len(self.positions), -1, dtype=np.int64)
        x_bins = np.full(len(self.positions), -1, dtype=np.int64)

        barrel = (self.locations == 1) & np.isfinite(self.positions).all(axis=1)
        if np.any(barrel):
            barrel_positions = self.positions[barrel]
            phi = np.arctan2(
                barrel_positions[:, 0], barrel_positions[:, 2] - BEAM_SPLIT_Z
            )
            y_values = barrel_positions[:, 1]
            y_min, y_max = float(y_values.min()), float(y_values.max())
            x_fraction = (phi + np.pi) / (2.0 * np.pi)
            y_fraction = (y_values - y_min) / max(y_max - y_min, 1e-12)
            x_bins[barrel] = np.clip(
                np.floor(x_fraction * width).astype(np.int64), 0, width - 1
            )
            barrel_y_min = 0.36 * (height - 1)
            barrel_y_max = 0.64 * (height - 1)
            y_bins[barrel] = np.rint(
                barrel_y_min + y_fraction * (barrel_y_max - barrel_y_min)
            ).astype(np.int64)

        endcap = (self.locations >= 2) & np.isfinite(self.positions).all(axis=1)
        if np.any(endcap):
            endcap_positions = self.positions[endcap]
            dx = endcap_positions[:, 0]
            dz = endcap_positions[:, 2] - BEAM_SPLIT_Z
            physical_radius = max(float(np.sqrt(dx * dx + dz * dz).max()), 1e-12)
            pixel_radius = min(0.20 * (height - 1), 0.22 * (width - 1))
            x_center = 0.5 * (width - 1)
            locations = self.locations[endcap]
            y_center = np.where(
                locations == 2, 0.855 * (height - 1), 0.145 * (height - 1)
            )
            x_bins[endcap] = np.rint(
                x_center + pixel_radius * dx / physical_radius
            ).astype(np.int64)
            y_bins[endcap] = np.rint(
                y_center + pixel_radius * dz / physical_radius
            ).astype(np.int64)

        return y_bins, x_bins


def _numpy(values, dtype=np.float64) -> np.ndarray:
    return ak.to_numpy(values).astype(dtype, copy=False)


def _accumulate_channel(
    images: np.ndarray,
    channel: int,
    pe_vectors,
    id_vectors,
    vertices: np.ndarray,
    geometry: PMTGeometry,
    maximum_pe: Optional[float],
) -> np.ndarray:
    pe_counts = _numpy(ak.num(pe_vectors, axis=1), np.int64)
    id_counts = _numpy(ak.num(id_vectors, axis=1), np.int64)
    aligned = pe_counts == id_counts
    if not np.any(aligned):
        return aligned

    selected_pe = pe_vectors[aligned]
    selected_ids = id_vectors[aligned]
    flat_pe = _numpy(ak.flatten(selected_pe))
    flat_ids = _numpy(ak.flatten(selected_ids), np.int64)
    event_indices = np.repeat(np.flatnonzero(aligned), pe_counts[aligned])

    valid_id = (flat_ids >= 0) & (flat_ids < len(geometry.positions))
    positions = np.full((len(flat_ids), 3), np.nan, dtype=np.float64)
    positions[valid_id] = geometry.positions[flat_ids[valid_id]]
    relative = positions - vertices[event_indices]
    distance = np.linalg.norm(relative, axis=1)
    valid = (
        valid_id
        & np.isfinite(flat_pe)
        & (flat_pe > 0.0)
        & np.isfinite(relative).all(axis=1)
        & (distance > 0.0)
    )
    if maximum_pe is not None:
        valid &= flat_pe <= maximum_pe
    if not np.any(valid):
        return aligned

    unit = relative[valid] / distance[valid, None]
    # ANNIE convention: y is vertical, z is beam, x is left/right.
    azimuth = np.arctan2(unit[:, 0], unit[:, 2])
    elevation = np.arcsin(np.clip(unit[:, 1], -1.0, 1.0))
    width = images.shape[2]
    height = images.shape[1]
    x_bin = np.floor((azimuth + np.pi) * width / (2.0 * np.pi)).astype(int)
    y_bin = np.floor((elevation + np.pi / 2.0) * height / np.pi).astype(int)
    x_bin = np.clip(x_bin, 0, width - 1)
    y_bin = np.clip(y_bin, 0, height - 1)
    np.add.at(
        images,
        (event_indices[valid], y_bin, x_bin, np.full(valid.sum(), channel)),
        flat_pe[valid],
    )
    return aligned


def build_ring_images(
    arrays,
    geometry: PMTGeometry,
    hitpe_branch: str,
    hitid_branch: str,
    tankcluster_branch: str,
    tankcluster_id_branch: str,
    height: int,
    width: int,
) -> Tuple[np.ndarray, np.ndarray]:
    vertices = np.column_stack([_numpy(arrays[name]) for name in VERTEX_BRANCHES])
    images = np.zeros((len(arrays), height, width, 2), dtype=np.float32)
    full_aligned = _accumulate_channel(
        images,
        0,
        arrays[hitpe_branch],
        arrays[hitid_branch],
        vertices,
        geometry,
        None,
    )
    tank_aligned = _accumulate_channel(
        images,
        1,
        arrays[tankcluster_branch],
        arrays[tankcluster_id_branch],
        vertices,
        geometry,
        350.0,
    )
    valid = full_aligned & tank_aligned & np.isfinite(vertices).all(axis=1)
    images = np.log1p(images)
    # Duplicate the periodic azimuth edge so the first convolution can see rings
    # crossing -pi/+pi as a connected pattern.
    images = np.pad(images, ((0, 0), (0, 0), (1, 1), (0, 0)), mode="wrap")
    return images, valid


def _accumulate_unfolded_channel(
    images: np.ndarray,
    channel: int,
    pe_vectors,
    id_vectors,
    geometry: PMTGeometry,
    y_bins: np.ndarray,
    x_bins: np.ndarray,
    maximum_pe: Optional[float],
) -> np.ndarray:
    pe_counts = _numpy(ak.num(pe_vectors, axis=1), np.int64)
    id_counts = _numpy(ak.num(id_vectors, axis=1), np.int64)
    aligned = pe_counts == id_counts
    if not np.any(aligned):
        return aligned

    flat_pe = _numpy(ak.flatten(pe_vectors[aligned]))
    flat_ids = _numpy(ak.flatten(id_vectors[aligned]), np.int64)
    event_indices = np.repeat(np.flatnonzero(aligned), pe_counts[aligned])
    valid_id = (flat_ids >= 0) & (flat_ids < len(geometry.positions))
    hit_y = np.full(len(flat_ids), -1, dtype=np.int64)
    hit_x = np.full(len(flat_ids), -1, dtype=np.int64)
    hit_y[valid_id] = y_bins[flat_ids[valid_id]]
    hit_x[valid_id] = x_bins[flat_ids[valid_id]]
    valid = (
        valid_id
        & np.isfinite(flat_pe)
        & (flat_pe > 0.0)
        & (hit_y >= 0)
        & (hit_x >= 0)
    )
    if maximum_pe is not None:
        valid &= flat_pe <= maximum_pe
    if np.any(valid):
        np.add.at(
            images,
            (
                event_indices[valid],
                hit_y[valid],
                hit_x[valid],
                np.full(valid.sum(), channel),
            ),
            flat_pe[valid],
        )
    return aligned


def build_unfolded_detector_images(
    arrays,
    geometry: PMTGeometry,
    hitpe_branch: str,
    hitid_branch: str,
    tankcluster_branch: str,
    tankcluster_id_branch: str,
    height: int = DEFAULT_DETECTOR_HEIGHT,
    width: int = DEFAULT_DETECTOR_WIDTH,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build annotation-free PE rasters matching the C++ unfolded detector layout."""
    images = np.zeros((len(arrays), height, width, 2), dtype=np.float32)
    y_bins, x_bins = geometry.unfolded_bins(height, width)
    full_aligned = _accumulate_unfolded_channel(
        images,
        0,
        arrays[hitpe_branch],
        arrays[hitid_branch],
        geometry,
        y_bins,
        x_bins,
        None,
    )
    tank_aligned = _accumulate_unfolded_channel(
        images,
        1,
        arrays[tankcluster_branch],
        arrays[tankcluster_id_branch],
        geometry,
        y_bins,
        x_bins,
        350.0,
    )
    return np.log1p(images), full_aligned & tank_aligned


def iterate_ring_images(
    paths: List[Path],
    tree_name: str,
    geometry: PMTGeometry,
    hitpe_branch: str,
    hitid_branch: str,
    tankcluster_branch: Optional[str],
    tankcluster_id_branch: str,
    height: int,
    width: int,
    detector_height: int,
    detector_width: int,
    apply_bdt_cuts: bool,
    include_truth: bool,
    charged_only: bool,
    chunk_size: str,
) -> Iterator[Dict[str, object]]:
    for path in paths:
        with uproot.open(path) as root_file:
            if tree_name not in root_file:
                raise ValueError(f"Tree {tree_name!r} was not found in {path}")
            tree = root_file[tree_name]
            available = set(tree.keys())
            tank_branch = resolve_tankcluster_branch(tree, tankcluster_branch)
            requested = [
                hitpe_branch,
                hitid_branch,
                tank_branch,
                tankcluster_id_branch,
                *VERTEX_BRANCHES,
            ]
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
                images, aligned = build_ring_images(
                    arrays,
                    geometry,
                    hitpe_branch,
                    hitid_branch,
                    tank_branch,
                    tankcluster_id_branch,
                    height,
                    width,
                )
                detector_images, detector_aligned = build_unfolded_detector_images(
                    arrays,
                    geometry,
                    hitpe_branch,
                    hitid_branch,
                    tank_branch,
                    tankcluster_id_branch,
                    detector_height,
                    detector_width,
                )
                aligned &= detector_aligned
                selected = aligned
                if apply_bdt_cuts:
                    selected &= build_bdt_selection(arrays)
                n_entries = len(arrays)
                result: Dict[str, object] = {
                    "path": path,
                    "entries": np.arange(
                        entry_offset, entry_offset + n_entries, dtype=np.int64
                    )[selected],
                    "images": images[selected],
                    "detector_images": detector_images[selected],
                    "tankcluster_branch": tank_branch,
                    "n_read": n_entries,
                    "n_selected": int(selected.sum()),
                    "n_misaligned": int((~aligned).sum()),
                }
                if include_truth:
                    result["labels"] = build_pion_labels(arrays, charged_only)[selected]
                yield result
                entry_offset += n_entries
