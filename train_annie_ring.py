#!/usr/bin/env python3
"""Train a CNN to identify pion-like ANNIE PMT ring patterns."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

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
    parser.add_argument(
        "--pmt-mask",
        choices=PMT_MASK_CHOICES,
        default="bdt",
        help="bdt: historical shutoff mask; on: CSV status ON only; all: no mask",
    )
    parser.add_argument("--hitpe-branch", default="hitPE")
    parser.add_argument("--hitid-branch", default="hitDetID")
    parser.add_argument("--tankcluster-branch")
    parser.add_argument("--tankcluster-id-branch", default="hitDetID_tankcluster")
    parser.add_argument("--image-height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--image-width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--detector-height", type=int, default=DEFAULT_DETECTOR_HEIGHT)
    parser.add_argument("--detector-width", type=int, default=DEFAULT_DETECTOR_WIDTH)
    parser.add_argument("--charged-only", action="store_true")
    parser.add_argument("--no-bdt-cuts", action="store_true")
    parser.add_argument("--chunk-size", default="100 MB")
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/annie_ring_pion.h5")
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260620)
    return parser.parse_args()


def load_data(
    args: argparse.Namespace,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str, int, PMTGeometry]:
    geometry = PMTGeometry(args.geometry, args.pmt_mask)
    image_chunks: List[np.ndarray] = []
    detector_chunks: List[np.ndarray] = []
    label_chunks: List[np.ndarray] = []
    tank_branch = ""
    n_read = 0
    n_selected = 0
    n_misaligned = 0
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
        not args.no_bdt_cuts,
        True,
        args.charged_only,
        args.chunk_size,
    ):
        image_chunks.append(chunk["images"])
        detector_chunks.append(chunk["detector_images"])
        label_chunks.append(chunk["labels"])
        tank_branch = str(chunk["tankcluster_branch"])
        n_read += int(chunk["n_read"])
        n_selected += int(chunk["n_selected"])
        n_misaligned += int(chunk["n_misaligned"])
        print(f"Read {n_read}; selected {n_selected}", end="\r", flush=True)
    print()
    if not image_chunks or n_selected == 0:
        raise ValueError("No events passed ring-image construction and selection")
    return (
        np.concatenate(image_chunks),
        np.concatenate(detector_chunks),
        np.concatenate(label_chunks),
        tank_branch,
        n_misaligned,
        geometry,
    )


def _image_tower(inputs, normalizer, prefix: str):
    x = normalizer(inputs)
    x = tf.keras.layers.Conv2D(
        16, 3, padding="same", activation="relu", name=f"{prefix}_conv1"
    )(x)
    x = tf.keras.layers.MaxPooling2D(2, name=f"{prefix}_pool1")(x)
    x = tf.keras.layers.Conv2D(
        32, 3, padding="same", activation="relu", name=f"{prefix}_conv2"
    )(x)
    x = tf.keras.layers.MaxPooling2D(2, name=f"{prefix}_pool2")(x)
    x = tf.keras.layers.Conv2D(
        64, 3, padding="same", activation="relu", name=f"{prefix}_conv3"
    )(x)
    return tf.keras.layers.GlobalAveragePooling2D(name=f"{prefix}_features")(x)


def build_model(
    angular_shape: Tuple[int, ...],
    detector_shape: Tuple[int, ...],
    angular_train: np.ndarray,
    detector_train: np.ndarray,
) -> tf.keras.Model:
    angular_normalizer = tf.keras.layers.Normalization(
        axis=-1, name="angular_channel_normalization"
    )
    detector_normalizer = tf.keras.layers.Normalization(
        axis=-1, name="detector_channel_normalization"
    )
    angular_normalizer.adapt(angular_train)
    detector_normalizer.adapt(detector_train)
    angular_inputs = tf.keras.Input(shape=angular_shape, name="pmt_angular_image")
    detector_inputs = tf.keras.Input(shape=detector_shape, name="pmt_unfolded_image")
    angular_features = _image_tower(angular_inputs, angular_normalizer, "angular")
    detector_features = _image_tower(detector_inputs, detector_normalizer, "detector")
    x = tf.keras.layers.Concatenate(name="combined_views")(
        [angular_features, detector_features]
    )
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.30)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="pion_score")(x)
    model = tf.keras.Model([angular_inputs, detector_inputs], outputs)
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
    if args.image_height < 4 or args.image_width < 8:
        raise ValueError("The PMT image must be at least 4 x 8 bins")
    if args.detector_height < 24 or args.detector_width < 16:
        raise ValueError("The unfolded detector image must be at least 24 x 16 bins")
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    images, detector_images, labels, tank_branch, n_misaligned, geometry = load_data(args)
    counts = np.bincount(labels.astype(np.int64), minlength=2)
    if np.any(counts == 0):
        raise ValueError(f"Both classes are required; class counts are {counts.tolist()}")
    indices = np.arange(len(labels))
    train_indices, temp_indices = train_test_split(
        indices, test_size=0.30, random_state=args.seed, stratify=labels
    )
    val_indices, test_indices = train_test_split(
        temp_indices,
        test_size=0.50,
        random_state=args.seed,
        stratify=labels[temp_indices],
    )
    y_train = labels[train_indices]
    y_val = labels[val_indices]
    y_test = labels[test_indices]
    train_counts = np.bincount(y_train.astype(np.int64), minlength=2)
    class_weight = {
        0: len(y_train) / (2.0 * train_counts[0]),
        1: len(y_train) / (2.0 * train_counts[1]),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    history_path = args.output.with_suffix(".history.csv")
    model = build_model(
        tuple(images.shape[1:]),
        tuple(detector_images.shape[1:]),
        images[train_indices],
        detector_images[train_indices],
    )
    model.fit(
        {
            "pmt_angular_image": images[train_indices],
            "pmt_unfolded_image": detector_images[train_indices],
        },
        y_train,
        validation_data=(
            {
                "pmt_angular_image": images[val_indices],
                "pmt_unfolded_image": detector_images[val_indices],
            },
            y_val,
        ),
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight=class_weight,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_pr_auc", mode="max", patience=12, restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_pr_auc",
                mode="max",
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1,
            ),
            tf.keras.callbacks.CSVLogger(str(history_path)),
        ],
        verbose=2,
    )
    metrics = model.evaluate(
        {
            "pmt_angular_image": images[test_indices],
            "pmt_unfolded_image": detector_images[test_indices],
        },
        y_test,
        return_dict=True,
        verbose=0,
    )
    print("Test metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.5f}")

    model.save(args.output)
    metadata = {
        "model_type": "annie_pion_ring_cnn",
        "model_variant": "A_PE_dual_view",
        "tensorflow_version": tf.__version__,
        "tree": args.tree,
        "geometry_file": args.geometry.name,
        "pmt_mask": args.pmt_mask,
        "pmt_count_included": geometry.included_count,
        "excluded_pmt_ids": geometry.excluded_ids,
        "hitpe_branch": args.hitpe_branch,
        "hitid_branch": args.hitid_branch,
        "tankcluster_branch": tank_branch,
        "tankcluster_id_branch": args.tankcluster_id_branch,
        "image_height": args.image_height,
        "image_width": args.image_width,
        "detector_height": args.detector_height,
        "detector_width": args.detector_width,
        "channels": ["log1p_hitPE", "log1p_hitPE_tankcluster"],
        "projections": [
            "vertex_centered_equirectangular_with_azimuth_wrap",
            "annotation_free_unfolded_barrel_top_bottom",
        ],
        "label": "charged_pion_present" if args.charged_only else "any_pion_present",
        "truth_branches": ["truePiPlusCher", "truePiMinusCher", "truePi0"],
        "bdt_preselection": not args.no_bdt_cuts,
        "threshold": args.threshold,
        "class_counts": {"no_pion": int(counts[0]), "pion": int(counts[1])},
        "misaligned_events_rejected": n_misaligned,
        "test_metrics": {name: float(value) for name, value in metrics.items()},
        "training_history": history_path.name,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Saved {args.output}, {args.output.with_suffix('.json')}, "
        f"and {history_path}"
    )


if __name__ == "__main__":
    main()
