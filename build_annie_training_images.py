#!/usr/bin/env python3
"""Build Model A training tensors and PNG previews from ANNIE ROOT files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from annie_features import FIT_INDIVIDUAL_PMT_SELECTION_EXPRESSION
from annie_pmt_response import PMT_RESPONSE_CHOICES, PMTResponse
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root_files", nargs="+", type=Path)
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
    parser.add_argument("--charged-only", action="store_true")
    parser.add_argument(
        "--no-event-cuts",
        "--no-bdt-cuts",
        dest="no_event_cuts",
        action="store_true",
        help=(
            "Disable the Fit_indivdiualPMT_Gaussian_Convolution event selection; "
            "--no-bdt-cuts is retained as a compatibility alias"
        ),
    )
    parser.add_argument("--chunk-size", default="100 MB")
    parser.add_argument("--max-events", type=int)
    parser.add_argument("--preview-count", type=int, default=12)
    parser.add_argument("--output-dir", type=Path, default=Path("model_a_images"))
    return parser.parse_args()


def save_preview(
    angular_image: np.ndarray,
    detector_image: np.ndarray,
    label: int,
    truth_pion_counts: np.ndarray,
    source_name: str,
    entry: int,
    response_mode: str,
    output: Path,
) -> None:
    # The first and last image columns are periodic copies used only by the CNN.
    visible = angular_image[:, 1:-1, :]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    channel_names = ["Full-event hitPE", "Tank-cluster hitPE"]
    for channel, axis in enumerate(axes[0]):
        plotted = axis.imshow(
            visible[:, :, channel],
            origin="lower",
            aspect="auto",
            extent=(-180, 180, -90, 90),
            cmap="magma",
            vmin=0.0,
            vmax=max(float(visible[:, :, channel].max()), 1e-6),
            interpolation="nearest",
        )
        axis.set_title(channel_names[channel])
        axis.set_xlabel("Azimuth from +z beam direction [degrees]")
        fig.colorbar(plotted, ax=axis, label="log(1 + accumulated PE)")
    axes[0, 0].set_ylabel("Elevation [degrees]")
    for channel, axis in enumerate(axes[1]):
        plotted = axis.imshow(
            detector_image[:, :, channel],
            origin="lower",
            aspect="equal",
            cmap="magma",
            vmin=0.0,
            vmax=max(float(detector_image[:, :, channel].max()), 1e-6),
            interpolation="nearest",
        )
        axis.set_title(f"Unfolded detector — {channel_names[channel]}")
        axis.set_xlabel("Unfolded detector x-bin")
        fig.colorbar(plotted, ax=axis, label="log(1 + accumulated PE)")
    axes[1, 0].set_ylabel("Bottom / barrel / top layout")
    truth = "pion" if label == 1 else "no pion"
    pi_plus, pi_minus, pi_zero = (int(value) for value in truth_pion_counts)
    fig.suptitle(
        f"{source_name}, entry {entry} — truth: {truth}; "
        f"π⁺={pi_plus}, π⁻={pi_minus}, π⁰={pi_zero}; "
        f"PMT response: {response_mode}"
    )
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.image_height < 4 or args.image_width < 8:
        raise ValueError("The PMT image must be at least 4 x 8 bins")
    if args.detector_height < 24 or args.detector_width < 16:
        raise ValueError("The unfolded detector image must be at least 24 x 16 bins")
    if args.max_events is not None and args.max_events <= 0:
        raise ValueError("--max-events must be positive")
    if args.preview_count < 0:
        raise ValueError("--preview-count cannot be negative")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = args.output_dir / "previews"
    if args.preview_count:
        preview_dir.mkdir(parents=True, exist_ok=True)
    geometry = PMTGeometry(args.geometry, args.pmt_mask)
    response = PMTResponse(args.pmt_response, args.pmt_response_calibration)
    source_indices = {path: index for index, path in enumerate(args.root_files)}
    shards = []
    total_read = 0
    total_selected = 0
    total_misaligned = 0
    class_counts = np.zeros(2, dtype=np.int64)
    preview_written = 0
    preview_class_counts = np.zeros(2, dtype=np.int64)
    preview_class_limits = np.array(
        [args.preview_count // 2, (args.preview_count + 1) // 2], dtype=np.int64
    )
    tank_branch = ""

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
        response,
        not args.no_event_cuts,
        True,
        args.charged_only,
        args.chunk_size,
    ):
        total_read += int(chunk["n_read"])
        total_misaligned += int(chunk["n_misaligned"])
        images = chunk["images"]
        detector_images = chunk["detector_images"]
        labels = chunk["labels"].astype(np.int8)
        truth_pion_counts = chunk["truth_pion_counts"].astype(np.int32)
        entries = chunk["entries"]
        if args.max_events is not None:
            remaining = args.max_events - total_selected
            if remaining <= 0:
                break
            images = images[:remaining]
            detector_images = detector_images[:remaining]
            labels = labels[:remaining]
            truth_pion_counts = truth_pion_counts[:remaining]
            entries = entries[:remaining]
        if len(images) == 0:
            continue

        tank_branch = str(chunk["tankcluster_branch"])
        source_index = source_indices[chunk["path"]]
        shard_name = f"shard_{len(shards):05d}.npz"
        np.savez_compressed(
            args.output_dir / shard_name,
            angular_images=images.astype(np.float32, copy=False),
            detector_images=detector_images.astype(np.float32, copy=False),
            labels=labels,
            truth_pion_counts=truth_pion_counts,
            entries=entries.astype(np.int64, copy=False),
            source_indices=np.full(len(images), source_index, dtype=np.int32),
        )
        shards.append({"file": shard_name, "events": int(len(images))})
        total_selected += len(images)
        class_counts += np.bincount(labels, minlength=2)

        for image, detector_image, label, pion_counts, entry in zip(
            images, detector_images, labels, truth_pion_counts, entries
        ):
            if preview_written >= args.preview_count:
                break
            if preview_class_counts[int(label)] >= preview_class_limits[int(label)]:
                continue
            truth_name = "pion" if int(label) == 1 else "no_pion"
            preview_path = preview_dir / (
                f"{truth_name}_event_{preview_written:04d}_source_{source_index}_"
                f"entry_{int(entry)}.png"
            )
            save_preview(
                image,
                detector_image,
                int(label),
                pion_counts,
                Path(chunk["path"]).name,
                int(entry),
                response.mode,
                preview_path,
            )
            preview_written += 1
            preview_class_counts[int(label)] += 1

        print(
            f"Read {total_read} events; wrote {total_selected} images",
            end="\r",
            flush=True,
        )
        if args.max_events is not None and total_selected >= args.max_events:
            break
    print()
    if total_selected == 0:
        raise ValueError("No events passed image construction and selection")

    manifest = {
        "format": "annie_model_a_dual_view_shards_v2",
        "sources": [str(path) for path in args.root_files],
        "tree": args.tree,
        "geometry_file": args.geometry.name,
        "pmt_mask": args.pmt_mask,
        "excluded_pmt_ids": geometry.excluded_ids,
        "pmt_response": response.mode,
        "pmt_response_scope": "hitPE_and_hitPE_tankcluster",
        "pmt_response_calibration_file": (
            response.calibration_path.name if response.calibration_path else None
        ),
        "pmt_response_calibration_sha256": response.calibration_sha256,
        "pmt_response_mapped_pmt_count": len(response.maps),
        "hitpe_branch": args.hitpe_branch,
        "hitid_branch": args.hitid_branch,
        "tankcluster_branch": tank_branch,
        "tankcluster_id_branch": args.tankcluster_id_branch,
        "image_height": args.image_height,
        "image_width": args.image_width,
        "stored_width": args.image_width + 2,
        "detector_height": args.detector_height,
        "detector_width": args.detector_width,
        "channels": [
            (
                "log1p_tuned_hitPE"
                if response.mode == "tuned"
                else "log1p_hitPE"
            ),
            (
                "log1p_tuned_hitPE_tankcluster"
                if response.mode == "tuned"
                else "log1p_hitPE_tankcluster"
            ),
        ],
        "arrays": [
            "angular_images",
            "detector_images",
            "labels",
            "truth_pion_counts",
            "entries",
            "source_indices",
        ],
        "projections": [
            "vertex_centered_equirectangular_with_azimuth_wrap",
            "annotation_free_unfolded_barrel_top_bottom",
        ],
        "label": "charged_pion_present" if args.charged_only else "any_pion_present",
        "event_selection": "fit_individual_pmt",
        "event_selection_source": "Fit_indivdiualPMT_Gaussian_Convolution.cpp",
        "event_selection_expression": FIT_INDIVIDUAL_PMT_SELECTION_EXPRESSION,
        "event_selection_applied": not args.no_event_cuts,
        "bdt_preselection": not args.no_event_cuts,
        "events_read": total_read,
        "events_written": total_selected,
        "misaligned_events_rejected": total_misaligned,
        "class_counts": {
            "no_pion": int(class_counts[0]),
            "pion": int(class_counts[1]),
        },
        "shards": shards,
        "preview_images": preview_written,
        "preview_class_counts": {
            "no_pion": int(preview_class_counts[0]),
            "pion": int(preview_class_counts[1]),
        },
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(shards)} tensor shard(s) and {preview_written} preview PNG(s)")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
