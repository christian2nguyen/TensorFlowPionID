"""Load flat tabular data from CSV files or ROOT TTrees."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd


def _find_tree(root_file, requested_tree: Optional[str]):
    if requested_tree:
        try:
            return root_file[requested_tree]
        except KeyError as error:
            available = ", ".join(root_file.keys(cycle=False))
            raise ValueError(
                f"Tree {requested_tree!r} was not found. Available objects: {available}"
            ) from error

    candidates = [
        name
        for name, class_name in root_file.classnames(recursive=True).items()
        if class_name in {"TTree", "ROOT::RNTuple"}
    ]
    if len(candidates) != 1:
        names = ", ".join(candidates) or "none"
        raise ValueError(
            "Use --tree because the ROOT file does not contain exactly one tree "
            f"(found: {names})"
        )
    return root_file[candidates[0]]


def _load_root(path: Path, columns: List[str], tree_name: Optional[str]) -> pd.DataFrame:
    try:
        import uproot
    except ImportError as error:
        raise RuntimeError("Reading ROOT files requires the 'uproot' package") from error

    with uproot.open(path) as root_file:
        tree = _find_tree(root_file, tree_name)
        missing = [column for column in columns if column not in tree.keys()]
        if missing:
            raise ValueError(f"{path} is missing branches: {', '.join(missing)}")
        arrays = tree.arrays(columns, library="np", how=dict)

    bad = [name for name, values in arrays.items() if np.asarray(values).ndim != 1]
    if bad:
        raise ValueError(
            "This baseline requires scalar, one-entry-per-track branches. "
            f"Jagged or multidimensional branches found: {', '.join(bad)}"
        )
    try:
        return pd.DataFrame(arrays, columns=columns)
    except ValueError as error:
        raise ValueError(
            f"Branches in {path} do not all have the same number of entries"
        ) from error


def load_frame(
    paths: List[Path], columns: List[str], tree_name: Optional[str] = None
) -> pd.DataFrame:
    """Load and concatenate CSV files or flat ROOT trees."""
    if not paths:
        raise ValueError("At least one input file is required")
    kinds = {path.suffix.lower() for path in paths}
    if len(kinds) != 1 or not kinds <= {".csv", ".root"}:
        raise ValueError("Inputs must be all CSV files or all ROOT files")
    if tree_name and kinds != {".root"}:
        raise ValueError("--tree can only be used with ROOT files")

    frames: List[pd.DataFrame] = []
    for path in paths:
        if path.suffix.lower() == ".root":
            frame = _load_root(path, columns, tree_name)
        else:
            frame = pd.read_csv(path, usecols=columns)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)
