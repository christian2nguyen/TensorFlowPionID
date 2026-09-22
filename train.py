#!/usr/bin/env python3
"""Train a TensorFlow classifier for binary pion identification."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from dataio import load_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", nargs="+", type=Path, help="Training CSV or ROOT file(s)")
    parser.add_argument("--features", nargs="+", required=True)
    parser.add_argument("--label", default="is_pion")
    parser.add_argument("--tree", help="TTree/RNTuple path inside ROOT files")
    parser.add_argument("--validation-data", nargs="+", type=Path)
    parser.add_argument("--test-data", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/pion_classifier.h5"))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_xy(
    paths: List[Path], features: List[str], label: str, tree: Optional[str]
) -> Tuple[np.ndarray, np.ndarray]:
    frame = load_frame(paths, [*features, label], tree)
    missing = [column for column in [*features, label] if column not in frame]
    if missing:
        raise ValueError(f"Input is missing columns: {', '.join(missing)}")
    selected = frame[[*features, label]].replace([np.inf, -np.inf], np.nan).dropna()
    if selected.empty:
        raise ValueError("Input has no complete rows for the selected columns")
    labels = selected[label].to_numpy(dtype=np.float32)
    if not np.isin(labels, [0.0, 1.0]).all():
        raise ValueError(f"{label} must contain only 0 and 1")
    return selected[features].to_numpy(dtype=np.float32), labels


def split_data(args: argparse.Namespace):
    x, y = load_xy(args.data, args.features, args.label, args.tree)
    if args.validation_data or args.test_data:
        if not (args.validation_data and args.test_data):
            raise ValueError("Provide both --validation-data and --test-data")
        x_val, y_val = load_xy(
            args.validation_data, args.features, args.label, args.tree
        )
        x_test, y_test = load_xy(args.test_data, args.features, args.label, args.tree)
        return (x, y), (x_val, y_val), (x_test, y_test)

    x_train, x_temp, y_train, y_temp = train_test_split(
        x, y, test_size=0.3, random_state=args.seed, stratify=y
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp, y_temp, test_size=0.5, random_state=args.seed, stratify=y_temp
    )
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def build_model(x_train: np.ndarray) -> tf.keras.Model:
    normalizer = tf.keras.layers.Normalization(name="feature_normalization")
    normalizer.adapt(x_train)
    inputs = tf.keras.Input(shape=(x_train.shape[1],), name="detector_features")
    x = normalizer(inputs)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="pion_score")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.AUC(name="roc_auc"),
            tf.keras.metrics.AUC(curve="PR", name="pr_auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def main() -> None:
    args = parse_args()
    if not 0.0 < args.threshold < 1.0:
        raise ValueError("--threshold must be between 0 and 1")
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    (x_train, y_train), (x_val, y_val), (x_test, y_test) = split_data(args)
    positives = float(y_train.sum())
    negatives = float(len(y_train) - positives)
    if positives == 0 or negatives == 0:
        raise ValueError("Training data must contain both pion and background rows")

    model = build_model(x_train)
    model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight={0: 1.0, 1: negatives / positives},
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_pr_auc", mode="max", patience=12, restore_best_weights=True
            )
        ],
        verbose=2,
    )
    results = model.evaluate(x_test, y_test, return_dict=True, verbose=0)
    print("Test metrics:")
    for name, value in results.items():
        print(f"  {name}: {value:.5f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.output)
    metadata = {
        "features": args.features,
        "label": args.label,
        "threshold": args.threshold,
        "test_metrics": {name: float(value) for name, value in results.items()},
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved model to {args.output}")


if __name__ == "__main__":
    main()
