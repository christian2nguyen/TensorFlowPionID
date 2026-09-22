#!/usr/bin/env python3
"""Train an ANNIE pion classifier from hitPE ROOT vector branches."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from annie_features import FEATURE_NAMES, iterate_root_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root_files", nargs="+", type=Path)
    parser.add_argument("--tree", default="phaseIITriggerTree")
    parser.add_argument("--hitpe-branch", default="hitPE")
    parser.add_argument(
        "--tankcluster-branch",
        help="Defaults to auto-detecting hitPE_tankcluster(s)",
    )
    parser.add_argument(
        "--charged-only",
        action="store_true",
        help="Label only charged pions as signal; default includes pi0",
    )
    parser.add_argument(
        "--no-bdt-cuts",
        action="store_true",
        help="Train without the preselection used by the earlier BDT",
    )
    parser.add_argument("--chunk-size", default="100 MB")
    parser.add_argument("--output", type=Path, default=Path("artifacts/annie_pion.h5"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260620)
    return parser.parse_args()


def load_training_data(args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray, str]:
    feature_chunks: List[np.ndarray] = []
    label_chunks: List[np.ndarray] = []
    tank_branch = ""
    n_read = 0
    n_selected = 0
    for chunk in iterate_root_features(
        args.root_files,
        args.tree,
        args.hitpe_branch,
        args.tankcluster_branch,
        not args.no_bdt_cuts,
        True,
        args.charged_only,
        args.chunk_size,
    ):
        feature_chunks.append(chunk["features"])
        label_chunks.append(chunk["labels"])
        tank_branch = str(chunk["tankcluster_branch"])
        n_read += int(chunk["n_read"])
        n_selected += int(chunk["n_selected"])
        print(f"Read {n_read} events; selected {n_selected}", end="\r", flush=True)
    print()
    if not feature_chunks or n_selected == 0:
        raise ValueError("No events passed feature extraction and selection")
    return np.concatenate(feature_chunks), np.concatenate(label_chunks), tank_branch


def build_model(x_train: np.ndarray) -> tf.keras.Model:
    normalizer = tf.keras.layers.Normalization(name="feature_normalization")
    normalizer.adapt(x_train)
    inputs = tf.keras.Input(shape=(len(FEATURE_NAMES),), name="pe_summary_features")
    x = normalizer(inputs)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.20)(x)
    x = tf.keras.layers.Dense(16, activation="relu")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="pion_score")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
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

    x, y, tank_branch = load_training_data(args)
    counts = np.bincount(y.astype(np.int64), minlength=2)
    if np.any(counts == 0):
        raise ValueError(f"Both classes are required; class counts are {counts.tolist()}")
    x_train, x_temp, y_train, y_temp = train_test_split(
        x, y, test_size=0.30, random_state=args.seed, stratify=y
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp, y_temp, test_size=0.50, random_state=args.seed, stratify=y_temp
    )

    train_counts = np.bincount(y_train.astype(np.int64), minlength=2)
    class_weight = {
        0: len(y_train) / (2.0 * train_counts[0]),
        1: len(y_train) / (2.0 * train_counts[1]),
    }
    model = build_model(x_train)
    model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight=class_weight,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_pr_auc", mode="max", patience=12, restore_best_weights=True
            )
        ],
        verbose=2,
    )
    metrics = model.evaluate(x_test, y_test, return_dict=True, verbose=0)
    print("Test metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.5f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.output)
    metadata = {
        "model_type": "annie_pion_pe_summary",
        "tensorflow_version": tf.__version__,
        "tree": args.tree,
        "hitpe_branch": args.hitpe_branch,
        "tankcluster_branch": tank_branch,
        "features": FEATURE_NAMES,
        "label": "charged_pion_present" if args.charged_only else "any_pion_present",
        "truth_branches": ["truePiPlusCher", "truePiMinusCher", "truePi0"],
        "bdt_preselection": not args.no_bdt_cuts,
        "threshold": args.threshold,
        "class_counts": {"no_pion": int(counts[0]), "pion": int(counts[1])},
        "test_metrics": {name: float(value) for name, value in metrics.items()},
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved {args.output} and {args.output.with_suffix('.json')}")


if __name__ == "__main__":
    main()

