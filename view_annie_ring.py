#!/usr/bin/env python3
"""Render one ANNIE event or scan MC truth for multiple pion-event views."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import uproot

from annie_features import TRUTH_BRANCHES, resolve_tankcluster_branch
from annie_pmt_response import (
    PMT_RESPONSE_CHOICES,
    PMT_TUNE_VARIANT_CHOICES,
    PMTResponse,
)
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
    iterate_ring_images,
)


def build_views(arrays, args, geometry, tank_branch, response):
    images, valid = build_ring_images(
        arrays,
        geometry,
        args.hitpe_branch,
        args.hitid_branch,
        tank_branch,
        args.tankcluster_id_branch,
        args.image_height,
        args.image_width,
        response,
        np.asarray([args.tune_entry], dtype=np.int64),
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
        response,
        np.asarray([args.tune_entry], dtype=np.int64),
    )
    if not (valid[0] and detector_valid[0]):
        raise ValueError("PE and detector-ID vectors are not aligned for this event")
    return images[0, :, 1:-1, :], detector_images[0]


def pion_annotation(arrays) -> str:
    fields = set(arrays.fields)
    if not all(name in fields for name in TRUTH_BRANCHES):
        return ""
    counts = [int(np.asarray(arrays[name])[0]) for name in TRUTH_BRANCHES]
    return f"; truth π⁺={counts[0]}, π⁻={counts[1]}, π⁰={counts[2]}"


def pion_counts(arrays):
    return tuple(int(np.asarray(arrays[name])[0]) for name in TRUTH_BRANCHES)


def find_pion_events(args, geometry) -> list:
    events = []
    chain_offsets = {}
    chain_offset = 0
    for path in args.root_files:
        chain_offsets[path] = chain_offset
        with uproot.open(path) as root_file:
            if args.tree not in root_file:
                raise ValueError(f"Tree {args.tree!r} was not found in {path}")
            chain_offset += int(root_file[args.tree].num_entries)
    raw_response = PMTResponse("raw", tune_variant=args.pmt_tune_variant)
    selection = "none" if args.no_event_cuts else "fit_individual_pmt"
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
        raw_response,
        selection,
        True,
        args.charged_only,
        args.chunk_size,
    ):
        pion_mask = chunk["labels"] == 1
        source = chunk["path"]
        events.extend(
            (source, int(value), chain_offsets[source] + int(value))
            for value in chunk["entries"][pion_mask]
        )
        if len(events) >= args.pion_plots:
            return events[: args.pion_plots]
    return events


def draw_response_comparison(
    raw_image,
    raw_detector_image,
    tuned_image,
    tuned_detector_image,
    args,
    truth_text,
) -> None:
    labels = [
        "Angular full hitPE",
        "Angular tank-cluster hitPE",
        "Unfolded full hitPE",
        "Unfolded tank-cluster hitPE",
    ]
    raw_panels = [
        raw_image[:, :, 0],
        raw_image[:, :, 1],
        raw_detector_image[:, :, 0],
        raw_detector_image[:, :, 1],
    ]
    tuned_panels = [
        tuned_image[:, :, 0],
        tuned_image[:, :, 1],
        tuned_detector_image[:, :, 0],
        tuned_detector_image[:, :, 1],
    ]
    fig, axes = plt.subplots(4, 3, figsize=(15, 16), squeeze=False)
    for row, (label, raw_panel, tuned_panel) in enumerate(
        zip(labels, raw_panels, tuned_panels)
    ):
        difference = tuned_panel - raw_panel
        image_maximum = max(float(raw_panel.max()), float(tuned_panel.max()), 1e-6)
        difference_maximum = max(float(np.abs(difference).max()), 1e-6)
        aspect = "auto" if row < 2 else "equal"
        raw_total = float(np.expm1(raw_panel).sum())
        tuned_total = float(np.expm1(tuned_panel).sum())
        percent = (
            100.0 * (tuned_total - raw_total) / raw_total
            if raw_total > 0.0
            else float("nan")
        )
        panels = [
            (raw_panel, "Raw", "magma", 0.0, image_maximum),
            (tuned_panel, "Tuned", "magma", 0.0, image_maximum),
            (
                difference,
                "Tuned − raw",
                "coolwarm",
                -difference_maximum,
                difference_maximum,
            ),
        ]
        for column, (panel, column_name, cmap, minimum, maximum) in enumerate(panels):
            plotted = axes[row, column].imshow(
                panel,
                origin="lower",
                aspect=aspect,
                cmap=cmap,
                vmin=minimum,
                vmax=maximum,
                interpolation="nearest",
            )
            axes[row, column].set_title(f"{label} — {column_name}", fontsize=9)
            fig.colorbar(
                plotted,
                ax=axes[row, column],
                label="log(1 + PE)" if column < 2 else "Δ log(1 + PE)",
            )
        axes[row, 0].set_ylabel(
            f"total PE: {raw_total:.1f} → {tuned_total:.1f}\n"
            f"change: {percent:+.1f}%"
        )
    fig.suptitle(
        f"{args.root_file.name}, {args.tree} entry {args.entry}{truth_text}\n"
        f"PMT response comparison: raw vs {args.pmt_response_calibration.name}; "
        f"variant={args.pmt_tune_variant}"
    )
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)


def draw_single_response(image, detector_image, args, truth_text, response) -> None:
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
    fig.suptitle(
        f"{args.root_file.name}, {args.tree} entry {args.entry} "
        f"— PMT response: {response.mode}, variant={response.tune_variant}, "
        f"payload={response.payload_format}{truth_text}"
    )
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root_file", type=Path, nargs="?")
    parser.add_argument("entry", type=int, nargs="?")
    parser.add_argument(
        "--file-list",
        type=Path,
        help="for -n, text file containing one MC ROOT path per line",
    )
    parser.add_argument(
        "-n",
        "--pion-plots",
        type=int,
        default=0,
        help="scan truth and write this many selected pion-event plots",
    )
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
    parser.add_argument(
        "--pmt-response", choices=PMT_RESPONSE_CHOICES, default="raw"
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
    parser.add_argument(
        "--compare-pmt-response",
        action="store_true",
        help="draw raw, tuned, and tuned-minus-raw views for the same event",
    )
    parser.add_argument(
        "--charged-only",
        action="store_true",
        help="for -n, require a truth pi+ or pi- rather than allowing pi0 only",
    )
    parser.add_argument(
        "--no-event-cuts",
        action="store_true",
        help="for -n, do not require the training event selection",
    )
    parser.add_argument("--chunk-size", default="100 MB")
    parser.add_argument("--output", type=Path, default=Path("annie_ring_event.png"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("pion_event_plots")
    )
    args = parser.parse_args()
    root_files = [] if args.root_file is None else [args.root_file]
    if args.file_list is not None:
        if not args.file_list.is_file():
            parser.error(f"--file-list {args.file_list} does not exist")
        for line in args.file_list.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            path = Path(stripped).expanduser()
            if not path.is_absolute():
                path = args.file_list.parent / path
            root_files.append(path)
    if not root_files:
        parser.error("Give a ROOT file or --file-list")
    args.root_files = root_files
    if args.pion_plots < 0:
        parser.error("-n/--pion-plots cannot be negative")
    if args.pion_plots and args.entry is not None:
        parser.error("Give either one entry or -n/--pion-plots, not both")
    if not args.pion_plots and args.entry is None:
        parser.error("Give a tree entry or request pion plots with -n")
    if not args.pion_plots and len(args.root_files) != 1:
        parser.error("A single entry can only be drawn from one ROOT file")
    if args.compare_pmt_response and args.pmt_response_calibration is None:
        parser.error("--compare-pmt-response requires --pmt-response-calibration")

    geometry = PMTGeometry(args.geometry, args.pmt_mask)
    if args.pion_plots:
        selected_events = find_pion_events(args, geometry)
        if not selected_events:
            raise ValueError("No truth-pion events passed image and event selection")
        if len(selected_events) < args.pion_plots:
            print(
                f"Requested {args.pion_plots} pion plots, but only "
                f"{len(selected_events)} events passed"
            )
        args.output_dir.mkdir(parents=True, exist_ok=True)
    else:
        selected_events = [
            (args.root_files[0], int(args.entry), int(args.entry))
        ]

    if args.compare_pmt_response:
        raw_response = PMTResponse("raw", tune_variant=args.pmt_tune_variant)
        tuned_response = PMTResponse(
            "tuned", args.pmt_response_calibration, args.pmt_tune_variant
        )
        print(tuned_response.describe())
        response = None
    else:
        raw_response = None
        tuned_response = None
        response = PMTResponse(
            args.pmt_response,
            args.pmt_response_calibration,
            args.pmt_tune_variant,
        )
        print(response.describe())

    manifest_rows = []
    for plot_index, (source, entry, tune_entry) in enumerate(selected_events):
        args.root_file = source
        args.entry = int(entry)
        args.tune_entry = int(tune_entry)
        with uproot.open(source) as root_file:
            if args.tree not in root_file:
                raise ValueError(f"Tree {args.tree!r} was not found in {source}")
            tree = root_file[args.tree]
            tank_branch = resolve_tankcluster_branch(tree, args.tankcluster_branch)
            available = set(tree.keys())
            branches = [
                args.hitpe_branch,
                args.hitid_branch,
                tank_branch,
                args.tankcluster_id_branch,
                *VERTEX_BRANCHES,
            ]
            branches.extend(name for name in TRUTH_BRANCHES if name in available)
            if args.pion_plots:
                suffix = "raw_vs_tuned" if args.compare_pmt_response else response.mode
                args.output = args.output_dir / (
                    f"pion_{plot_index:04d}_entry_{args.entry}_{suffix}.png"
                )
            arrays = tree.arrays(
                branches,
                entry_start=args.entry,
                entry_stop=args.entry + 1,
                library="ak",
            )
        if len(arrays) != 1:
            raise ValueError(f"Entry {args.entry} does not exist in {source}")
        truth_text = pion_annotation(arrays)
        if args.compare_pmt_response:
            raw_image, raw_detector_image = build_views(
                arrays, args, geometry, tank_branch, raw_response
            )
            tuned_image, tuned_detector_image = build_views(
                arrays, args, geometry, tank_branch, tuned_response
            )
            draw_response_comparison(
                raw_image,
                raw_detector_image,
                tuned_image,
                tuned_detector_image,
                args,
                truth_text,
            )
        else:
            image, detector_image = build_views(
                arrays, args, geometry, tank_branch, response
            )
            draw_single_response(image, detector_image, args, truth_text, response)
        counts = (
            pion_counts(arrays)
            if all(name in arrays.fields for name in TRUTH_BRANCHES)
            else (None, None, None)
        )
        manifest_rows.append(
            {
                "plot": args.output.name,
                "source_file": str(source),
                "tree_entry": args.entry,
                "chain_entry": args.tune_entry,
                "truePiPlusCher": counts[0],
                "truePiMinusCher": counts[1],
                "truePi0": counts[2],
            }
        )
        print(f"Wrote {args.output}")

    if args.pion_plots:
        manifest_path = args.output_dir / "pion_plots.csv"
        with manifest_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
            writer.writeheader()
            writer.writerows(manifest_rows)
        print(f"Wrote {len(manifest_rows)} pion plots and {manifest_path}")


if __name__ == "__main__":
    main()
