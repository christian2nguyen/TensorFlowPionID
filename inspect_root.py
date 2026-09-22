#!/usr/bin/env python3
"""List trees and branches in a ROOT file."""

from __future__ import annotations

import argparse
from pathlib import Path

import uproot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root_file", type=Path)
    args = parser.parse_args()

    with uproot.open(args.root_file) as root_file:
        found = False
        for name, class_name in root_file.classnames(recursive=True).items():
            if class_name not in {"TTree", "ROOT::RNTuple"}:
                continue
            found = True
            tree = root_file[name]
            print(f"{name} ({class_name}, {tree.num_entries} entries)")
            for branch in tree.keys():
                print(f"  {branch}: {tree[branch].typename}")
        if not found:
            print("No TTree or RNTuple objects found")


if __name__ == "__main__":
    main()

