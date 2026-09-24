"""Load and apply ANNIE per-PMT response maps without requiring ROOT itself."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import uproot


PMT_RESPONSE_CHOICES = ("raw", "tuned")
FIT_MINIMUM_PE = 0.0
FIT_MAXIMUM_PE = 350.0
_RESPONSE_KEY = re.compile(
    r"(?:^|/)PMT_Calibrations/PMT_(\d+)/response_map(?:;\d+)?$"
)


class PMTResponse:
    """Piecewise-linear MC-to-data response maps from the STV all-PMT tune."""

    def __init__(self, mode: str, calibration_path: Optional[Path] = None):
        if mode not in PMT_RESPONSE_CHOICES:
            raise ValueError(f"Unknown PMT response mode {mode!r}")
        if mode == "raw" and calibration_path is not None:
            raise ValueError(
                "--pmt-response-calibration is only valid with "
                "--pmt-response tuned"
            )
        if mode == "tuned" and calibration_path is None:
            raise ValueError(
                "--pmt-response tuned requires --pmt-response-calibration "
                "pointing to the all-PMT calibration ROOT file"
            )

        self.mode = mode
        self.calibration_path = calibration_path
        self.calibration_sha256: Optional[str] = None
        self.maps: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
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
                # The C++ derivation enforces a non-decreasing tuned response.
                self.maps[detector_id] = (unique_x, np.maximum.accumulate(y))
        if not self.maps:
            raise ValueError(
                f"No PMT_Calibrations/PMT_<id>/response_map TGraphs found in {path}"
            )

    @property
    def mapped_pmt_ids(self) -> List[int]:
        return sorted(self.maps)

    def apply(self, values: np.ndarray, detector_ids: np.ndarray) -> np.ndarray:
        """Apply STV interpolation inside 0--350 PE; leave other hits unchanged."""
        if self.mode == "raw":
            return values
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
