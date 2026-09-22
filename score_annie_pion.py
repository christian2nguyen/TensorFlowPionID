#!/usr/bin/env python3
"""Apply a trained ANNIE pion model to ROOT files and write event scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import tensorflow as tf

from annie_features import iterate_root_features


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("root_files", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("annie_pion_scores.csv"))
    parser.add_argument("--chunk-size", default="100 MB")
    parser.add_argument(
        "--all-events",
        action="store_true",
        help="Disable the BDT preselection stored with the model",
    )
    args = parser.parse_args()

    metadata = json.loads(args.model.with_suffix(".json").read_text(encoding="utf-8"))
    model = tf.keras.models.load_model(args.model)
    rows = []
    for chunk in iterate_root_features(
        args.root_files,
        metadata["tree"],
        metadata["hitpe_branch"],
        metadata["tankcluster_branch"],
        metadata["bdt_preselection"] and not args.all_events,
        False,
        False,
        args.chunk_size,
    ):
        features = chunk["features"]
        if len(features) == 0:
            continue
        scores = model.predict(features, verbose=0).reshape(-1)
        threshold = float(metadata["threshold"])
        path = str(chunk["path"])
        rows.extend(
            {
                "source_file": path,
                "tree_entry": int(entry),
                "pion_score": float(score),
                "pion_prediction": int(score >= threshold),
            }
            for entry, score in zip(chunk["entries"], scores)
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"Wrote {len(rows)} selected event scores to {args.output}")


if __name__ == "__main__":
    main()
