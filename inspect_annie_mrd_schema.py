#!/usr/bin/env python3
"""Compare MRD branch layouts and sampled contents across ANNIE ROOT files."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import awkward as ak
import numpy as np
import uproot

from annie_features import (
    MRD_TRACK_PROPERTY_BRANCHES,
    MRD_TRACK_START_BRANCHES,
    RING_EVENT_SCALAR_BRANCHES,
)


BRANCHES = [
    *RING_EVENT_SCALAR_BRANCHES,
    *MRD_TRACK_START_BRANCHES,
    *MRD_TRACK_PROPERTY_BRANCHES,
]


def _layout(values) -> str:
    try:
        ak.num(values, axis=1)
    except ValueError:
        return "event scalar"
    return "per-event vector"


def _finite_range(values) -> str:
    array = ak.to_numpy(values).astype(np.float64, copy=False)
    finite = array[np.isfinite(array)]
    if len(finite) == 0:
        return "no finite values"
    return f"range=[{finite.min():.6g}, {finite.max():.6g}]"


def inspect_file(path: Path, tree_name: str, max_entries: int) -> Dict[str, dict]:
    report: Dict[str, dict] = {}
    with uproot.open(path) as root_file:
        if tree_name not in root_file:
            raise ValueError(f"Tree {tree_name!r} was not found in {path}")
        tree = root_file[tree_name]
        available = set(tree.keys())
        present = [branch for branch in BRANCHES if branch in available]
        arrays = tree.arrays(
            present,
            entry_stop=min(max_entries, int(tree.num_entries)),
            library="ak",
        )
        sampled = len(arrays)
        num_tracks = None
        if "numMRDTracks" in present:
            num_tracks = ak.to_numpy(arrays["numMRDTracks"]).astype(
                np.float64, copy=False
            )

        print(f"\nFile: {path}")
        print(f"Tree: {tree_name}; entries={int(tree.num_entries)}; sampled={sampled}")
        for branch in BRANCHES:
            if branch not in available:
                report[branch] = {"present": False}
                print(f"  {branch:<24} MISSING")
                continue

            branch_object = tree[branch]
            typename = str(getattr(branch_object, "typename", "unknown"))
            interpretation = str(getattr(branch_object, "interpretation", "unknown"))
            values = arrays[branch]
            layout = _layout(values)
            awkward_type = str(ak.type(values))
            details = ""
            if layout == "event scalar":
                details = _finite_range(values)
            else:
                counts = ak.to_numpy(ak.num(values, axis=1)).astype(
                    np.int64, copy=False
                )
                if len(counts):
                    details = f"lengths=[{counts.min()}, {counts.max()}]"
                else:
                    details = "lengths=[]"
                if num_tracks is not None and len(counts) == len(num_tracks):
                    valid_num_tracks = np.isfinite(num_tracks) & np.isclose(
                        num_tracks, np.rint(num_tracks)
                    )
                    mismatches = valid_num_tracks & (counts != np.rint(num_tracks))
                    details += f"; numMRDTracks mismatches={int(mismatches.sum())}"

            report[branch] = {
                "present": True,
                "typename": typename,
                "interpretation": interpretation,
                "layout": layout,
                "awkward_type": awkward_type,
            }
            print(f"  {branch:<24} {layout:<16} {typename}; {details}")
            print(f"    awkward type: {awkward_type}")
            print(f"    interpretation: {interpretation}")
    return report


def compare_reports(paths: List[Path], reports: List[Dict[str, dict]]) -> None:
    if len(reports) < 2:
        return
    reference_path = paths[0]
    reference = reports[0]
    print(f"\nDifferences relative to {reference_path}:")
    found_difference = False
    for path, report in zip(paths[1:], reports[1:]):
        for branch in BRANCHES:
            expected = reference.get(branch, {"present": False})
            observed = report.get(branch, {"present": False})
            fields = (
                "present",
                "typename",
                "layout",
                "awkward_type",
                "interpretation",
            )
            changed = [
                field
                for field in fields
                if expected.get(field) != observed.get(field)
            ]
            if not changed:
                continue
            found_difference = True
            print(f"  {path}: {branch}")
            for field in changed:
                print(
                    f"    {field}: {expected.get(field)!r} -> "
                    f"{observed.get(field)!r}"
                )
    if not found_difference:
        print("  No branch type or scalar/vector layout differences found.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root_files", nargs="*", type=Path)
    parser.add_argument("--file-list", type=Path)
    parser.add_argument("--tree", default="phaseIITriggerTree")
    parser.add_argument(
        "--max-entries",
        type=int,
        default=10000,
        help="Number of entries sampled from each file (default: 10000)",
    )
    args = parser.parse_args()
    if args.file_list is not None:
        for line in args.file_list.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            path = Path(stripped).expanduser()
            if not path.is_absolute():
                path = args.file_list.parent / path
            args.root_files.append(path)
    if not args.root_files:
        parser.error("Provide at least one ROOT file or use --file-list")
    if args.max_entries <= 0:
        parser.error("--max-entries must be positive")
    return args


def main() -> None:
    args = parse_args()
    reports = [
        inspect_file(path, args.tree, args.max_entries) for path in args.root_files
    ]
    compare_reports(args.root_files, reports)


if __name__ == "__main__":
    main()
