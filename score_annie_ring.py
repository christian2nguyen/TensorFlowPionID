#!/usr/bin/env python3
"""Apply the ANNIE PMT ring CNN to ROOT events."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import tensorflow as tf

from annie_ring_images import DEFAULT_GEOMETRY, PMTGeometry, iterate_ring_images


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("root_files", nargs="+", type=Path)
    parser.add_argument("--geometry", type=Path, default=DEFAULT_GEOMETRY)
    parser.add_argument("--output", type=Path, default=Path("annie_ring_scores.csv"))
    parser.add_argument("--chunk-size", default="100 MB")
    parser.add_argument("--all-events", action="store_true")
    args = parser.parse_args()

    metadata = json.loads(args.model.with_suffix(".json").read_text(encoding="utf-8"))
    if metadata.get("model_type") != "annie_pion_ring_cnn":
        raise ValueError("The supplied model is not an ANNIE ring CNN")
    model = tf.keras.models.load_model(args.model)
    geometry = PMTGeometry(args.geometry, metadata.get("pmt_mask", "bdt"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["source_file", "tree_entry", "pion_score", "pion_prediction"],
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
            bool(metadata["bdt_preselection"]) and not args.all_events,
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
                        "pion_score": float(score),
                        "pion_prediction": int(score >= threshold),
                    }
                )
                written += 1
    print(f"Wrote {written} selected event scores to {args.output}")


if __name__ == "__main__":
    main()
