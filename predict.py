#!/usr/bin/env python3
"""Score track rows with a trained pion classifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from dataio import load_frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("data", nargs="+", type=Path, help="CSV or ROOT file(s)")
    parser.add_argument("--tree", help="TTree/RNTuple path inside ROOT files")
    parser.add_argument(
        "--keep", nargs="*", default=[], help="Identifier columns to retain in output"
    )
    parser.add_argument("--output", type=Path, default=Path("pion_scores.csv"))
    args = parser.parse_args()

    metadata_path = args.model.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    features = metadata["features"]
    output_columns = list(dict.fromkeys([*args.keep, *features]))
    frame = load_frame(args.data, output_columns, args.tree)
    missing = [column for column in features if column not in frame]
    if missing:
        raise ValueError(f"Input is missing columns: {', '.join(missing)}")
    values = frame[features].to_numpy(dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("Prediction features contain missing or infinite values")

    model = tf.keras.models.load_model(args.model)
    scores = model.predict(values, verbose=0).reshape(-1)
    frame["pion_score"] = scores
    frame["pion_prediction"] = (scores >= metadata["threshold"]).astype(np.int8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(f"Wrote {len(frame)} scored rows to {args.output}")


if __name__ == "__main__":
    main()
