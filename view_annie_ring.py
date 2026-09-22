#!/usr/bin/env python3
"""Render the angular and unfolded-detector PMT views for one ANNIE event."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import uproot

from annie_features import resolve_tankcluster_branch
from annie_ring_images import (
    DEFAULT_GEOMETRY,
    DEFAULT_DETECTOR_HEIGHT,
    DEFAULT_DETECTOR_WIDTH,
    DEFAULT_HEIGHT,
    DEFAULT_WIDTH,
    PMT_MASK_CHOICES,
    PMTGeometry,
    VERTEX_BRANCHES,
    build_ring_images,
    build_unfolded_detector_images,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root_file", type=Path)
    parser.add_argument("entry", type=int)
    parser.add_argument("--tree", default="phaseIITriggerTree")
    parser.add_argument("--geometry", type=Path, default=DEFAULT_GEOMETRY)
    parser.add_argument("--pmt-mask", choices=PMT_MASK_CHOICES, default="bdt")
    parser.add_argument("--hitpe-branch", default="hitPE")
    parser.add_argument("--hitid-branch", default="hitDetID")
    parser.add_argument("--tankcluster-branch")
    parser.add_argument("--tankcluster-id-branch", default="hitDetID_tankcluster")
    parser.add_argument("--image-height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--image-width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--detector-height", type=int, default=DEFAULT_DETECTOR_HEIGHT)
    parser.add_argument("--detector-width", type=int, default=DEFAULT_DETECTOR_WIDTH)
    parser.add_argument("--output", type=Path, default=Path("annie_ring_event.png"))
    args = parser.parse_args()

    with uproot.open(args.root_file) as root_file:
        tree = root_file[args.tree]
        tank_branch = resolve_tankcluster_branch(tree, args.tankcluster_branch)
        branches = [
            args.hitpe_branch,
            args.hitid_branch,
            tank_branch,
            args.tankcluster_id_branch,
            *VERTEX_BRANCHES,
        ]
        arrays = tree.arrays(
            branches,
            entry_start=args.entry,
            entry_stop=args.entry + 1,
            library="ak",
        )
    if len(arrays) != 1:
        raise ValueError(f"Entry {args.entry} does not exist")
    geometry = PMTGeometry(args.geometry, args.pmt_mask)
    images, valid = build_ring_images(
        arrays,
        geometry,
        args.hitpe_branch,
        args.hitid_branch,
        tank_branch,
        args.tankcluster_id_branch,
        args.image_height,
        args.image_width,
    )
    detector_images, detector_valid = build_unfolded_detector_images(
        arrays,
        geometry,
        args.hitpe_branch,
        args.hitid_branch,
        tank_branch,
        args.tankcluster_id_branch,
        args.detector_height,
        args.detector_width,
    )
    if not (valid[0] and detector_valid[0]):
        raise ValueError("PE and detector-ID vectors are not aligned for this event")
    image = images[0, :, 1:-1, :]
    detector_image = detector_images[0]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    labels = ["Full hitPE", "Tank-cluster hitPE"]
    for channel, axis in enumerate(axes[0]):
        plotted = axis.imshow(
            image[:, :, channel],
            origin="lower",
            aspect="auto",
            extent=(-180, 180, -90, 90),
            cmap="magma",
        )
        axis.set_title(labels[channel])
        axis.set_xlabel("Azimuth from +z beam direction [degrees]")
        fig.colorbar(plotted, ax=axis, label="log(1 + accumulated PE)")
    axes[0, 0].set_ylabel("Elevation [degrees]")
    for channel, axis in enumerate(axes[1]):
        plotted = axis.imshow(
            detector_image[:, :, channel],
            origin="lower",
            aspect="equal",
            cmap="magma",
            interpolation="nearest",
        )
        axis.set_title(f"Unfolded detector — {labels[channel]}")
        axis.set_xlabel("Unfolded detector x-bin")
        fig.colorbar(plotted, ax=axis, label="log(1 + accumulated PE)")
    axes[1, 0].set_ylabel("Bottom / barrel / top layout")
    fig.suptitle(f"{args.root_file.name}, {args.tree} entry {args.entry}")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
