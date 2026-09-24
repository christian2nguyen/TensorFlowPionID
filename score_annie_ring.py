#!/usr/bin/env python3
"""Apply the ANNIE PMT ring CNN to ROOT events."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import tensorflow as tf

from annie_pmt_response import (
    PMT_RESPONSE_CHOICES,
    PMT_TUNE_VARIANT_CHOICES,
    PMTResponse,
)
from annie_ring_images import DEFAULT_GEOMETRY, PMTGeometry, iterate_ring_images


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("root_files", nargs="+", type=Path)
    parser.add_argument("--geometry", type=Path, default=DEFAULT_GEOMETRY)
    parser.add_argument("--output", type=Path, default=Path("annie_ring_scores.csv"))
    parser.add_argument("--chunk-size", default="100 MB")
    parser.add_argument("--all-events", action="store_true")
    parser.add_argument(
        "--pmt-response",
        choices=PMT_RESPONSE_CHOICES,
        default="raw",
        help="Use raw for detector data; tuned may be used for simulated input",
    )
    parser.add_argument(
        "--pmt-response-calibration",
        type=Path,
        help="all-PMT calibration ROOT file required for --pmt-response tuned",
    )
    parser.add_argument(
        "--pmt-tune-variant",
        choices=PMT_TUNE_VARIANT_CHOICES,
        help="Defaults to the tune variant stored with the trained model",
    )
    args = parser.parse_args()

    metadata = json.loads(args.model.with_suffix(".json").read_text(encoding="utf-8"))
    if metadata.get("model_type") != "annie_pion_ring_cnn":
        raise ValueError("The supplied model is not an ANNIE ring CNN")
    model = tf.keras.models.load_model(args.model)
    geometry = PMTGeometry(args.geometry, metadata.get("pmt_mask", "bdt"))
    tune_variant = args.pmt_tune_variant or metadata.get("pmt_tune_variant", "final")
    response = PMTResponse(
        args.pmt_response,
        args.pmt_response_calibration,
        tune_variant,
    )
    print(response.describe())
    training_response = metadata.get("training_pmt_response", "raw")
    if response.mode == "tuned" and training_response != "tuned":
        raise ValueError(
            "Cannot apply tuned PMT response to a model trained with raw response"
        )
    training_variant = metadata.get("pmt_tune_variant", "final")
    if response.mode == "tuned" and tune_variant != training_variant:
        raise ValueError(
            "The scoring tune variant differs from the variant used for training"
        )
    expected_hash = metadata.get("pmt_response_calibration_sha256")
    if response.mode == "tuned" and expected_hash and (
        response.calibration_sha256 != expected_hash
    ):
        raise ValueError(
            "The scoring calibration differs from the calibration used for training"
        )
    if training_response == "tuned" and response.mode == "raw":
        print(
            "Using raw PMT response with a tuned-MC model; this is intended for "
            "detector data. Use --pmt-response tuned for simulated input."
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    if args.all_events:
        event_selection = "none"
    elif "event_selection" in metadata:
        event_selection = (
            str(metadata["event_selection"])
            if metadata.get("event_selection_applied", True)
            else "none"
        )
    else:
        # Models written before event_selection metadata used the old BDT cuts.
        event_selection = (
            "legacy_bdt" if metadata.get("bdt_preselection", False) else "none"
        )
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "source_file",
                "tree_entry",
                "pmt_response",
                "pmt_tune_variant",
                "pmt_response_payload_format",
                "pion_score",
                "pion_prediction",
            ],
        )
        writer.writeheader()
        for chunk in iterate_ring_images(
            args.root_files,
            metadata["tree"],
            geometry,
            metadata["hitpe_branch"],
            metadata["hitid_branch"],
            metadata["tankcluster_branch"],
            metadata["tankcluster_id_branch"],
            int(metadata["image_height"]),
            int(metadata["image_width"]),
            int(metadata["detector_height"]),
            int(metadata["detector_width"]),
            response,
            event_selection,
            False,
            False,
            args.chunk_size,
        ):
            images = chunk["images"]
            if len(images) == 0:
                continue
            scores = model.predict(
                {
                    "pmt_angular_image": images,
                    "pmt_unfolded_image": chunk["detector_images"],
                },
                verbose=0,
            ).reshape(-1)
            threshold = float(metadata["threshold"])
            for entry, score in zip(chunk["entries"], scores):
                writer.writerow(
                    {
                        "source_file": str(chunk["path"]),
                        "tree_entry": int(entry),
                        "pmt_response": response.mode,
                        "pmt_tune_variant": response.tune_variant,
                        "pmt_response_payload_format": response.payload_format,
                        "pion_score": float(score),
                        "pion_prediction": int(score >= threshold),
                    }
                )
                written += 1
    print(f"Wrote {written} selected event scores to {args.output}")


if __name__ == "__main__":
    main()
