#!/usr/bin/env python3
"""Train a CNN to identify pion-like ANNIE PMT ring patterns."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)
from sklearn.model_selection import train_test_split

from annie_pmt_response import (
    PMT_RESPONSE_CHOICES,
    PMT_TUNE_VARIANT_CHOICES,
    PMTResponse,
)
from annie_features import FIT_INDIVIDUAL_PMT_SELECTION_EXPRESSION
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
    parser.add_argument(
        "root_files",
        nargs="*",
        type=Path,
        help="ROOT files; optional when --file-list is supplied",
    )
    parser.add_argument(
        "--file-list",
        type=Path,
        help="Text file containing one ROOT file path per line",
    )
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
    parser.add_argument(
        "--pmt-response",
        choices=PMT_RESPONSE_CHOICES,
        default="raw",
        help="raw: unchanged PE; tuned: map both MC PE channels toward beam-on data",
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
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/annie_ring_pion.h5")
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260620)
    args = parser.parse_args()
    if args.file_list is not None:
        if not args.file_list.is_file():
            parser.error(f"--file-list {args.file_list} does not exist")
        listed_files = []
        for line in args.file_list.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            listed_path = Path(stripped).expanduser()
            if not listed_path.is_absolute():
                listed_path = args.file_list.parent / listed_path
            listed_files.append(listed_path)
        args.root_files.extend(listed_files)
    if not args.root_files:
        parser.error("No ROOT files given: use positional paths or --file-list")
    return args


def load_data(
    args: argparse.Namespace,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    str,
    int,
    PMTGeometry,
    PMTResponse,
]:
    geometry = PMTGeometry(args.geometry, args.pmt_mask)
    response = PMTResponse(
        args.pmt_response,
        args.pmt_response_calibration,
        args.pmt_tune_variant,
    )
    print(response.describe())
    image_chunks: List[np.ndarray] = []
    detector_chunks: List[np.ndarray] = []
    label_chunks: List[np.ndarray] = []
    truth_pion_chunks: List[np.ndarray] = []
    source_path_chunks: List[np.ndarray] = []
    entry_chunks: List[np.ndarray] = []
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
        response,
        not args.no_event_cuts,
        True,
        args.charged_only,
        args.chunk_size,
    ):
        image_chunks.append(chunk["images"])
        detector_chunks.append(chunk["detector_images"])
        label_chunks.append(chunk["labels"])
        truth_pion_chunks.append(chunk["truth_pion_counts"])
        entries = chunk["entries"]
        entry_chunks.append(entries)
        source_path_chunks.append(
            np.full(len(entries), str(chunk["path"]), dtype=object)
        )
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
        np.concatenate(source_path_chunks),
        np.concatenate(entry_chunks),
        np.concatenate(truth_pion_chunks),
        tank_branch,
        n_misaligned,
        geometry,
        response,
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


def plot_roc_curve(y_test: np.ndarray, test_scores: np.ndarray, output: Path):
    fpr, tpr, thresholds = roc_curve(y_test, test_scores)
    roc_auc_value = float(auc(fpr, tpr))
    fig, axis = plt.subplots(figsize=(6, 6))
    axis.plot(fpr, tpr, color="C0", label=f"ROC (AUC = {roc_auc_value:.4f})")
    axis.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Chance")
    axis.set_xlabel("False positive rate (no-pion misclassified as pion)")
    axis.set_ylabel("True positive rate (pion efficiency)")
    axis.set_title("Test-set ROC curve")
    axis.legend(loc="lower right")
    fig.tight_layout()
    path = output.with_suffix(".roc_curve.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path.name, roc_auc_value, fpr, tpr, thresholds


def plot_score_distribution(
    y_test: np.ndarray, test_scores: np.ndarray, output: Path
) -> str:
    fig, axis = plt.subplots(figsize=(7, 5))
    bins = np.linspace(0.0, 1.0, 41)
    axis.hist(
        test_scores[y_test == 0],
        bins=bins,
        alpha=0.6,
        density=True,
        label="No pion (truth)",
        color="C0",
    )
    axis.hist(
        test_scores[y_test == 1],
        bins=bins,
        alpha=0.6,
        density=True,
        label="Pion (truth)",
        color="C1",
    )
    axis.set_xlabel("Model pion_score")
    axis.set_ylabel("Normalized event density")
    axis.set_title("Test-set score separation by truth class")
    axis.legend()
    fig.tight_layout()
    path = output.with_suffix(".score_distribution.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path.name


def plot_precision_recall(
    y_test: np.ndarray, test_scores: np.ndarray, output: Path
):
    precision, recall, _ = precision_recall_curve(y_test, test_scores)
    average_precision = float(average_precision_score(y_test, test_scores))
    fig, axis = plt.subplots(figsize=(6, 6))
    axis.plot(
        recall,
        precision,
        color="C2",
        label=f"PR (AP = {average_precision:.4f})",
    )
    axis.set_xlabel("Recall (pion efficiency)")
    axis.set_ylabel("Precision (purity)")
    axis.set_title("Test-set precision-recall curve")
    axis.legend(loc="lower left")
    fig.tight_layout()
    path = output.with_suffix(".precision_recall_curve.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path.name, average_precision


def plot_efficiency_rejection_vs_threshold(
    fpr: np.ndarray, tpr: np.ndarray, thresholds: np.ndarray, output: Path
) -> str:
    finite = np.isfinite(thresholds)
    order = np.argsort(thresholds[finite])
    x = thresholds[finite][order]
    efficiency = tpr[finite][order]
    rejection = 1.0 - fpr[finite][order]
    fig, axis = plt.subplots(figsize=(7, 5))
    axis.plot(x, efficiency, label="Signal efficiency (pion)")
    axis.plot(x, rejection, label="Background rejection (no-pion)")
    axis.set_xlabel("pion_score threshold")
    axis.set_ylabel("Rate")
    axis.set_title("Efficiency / rejection vs. threshold")
    axis.legend()
    fig.tight_layout()
    path = output.with_suffix(".efficiency_rejection_vs_threshold.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path.name


def pion_operating_point(
    y_true: np.ndarray, scores: np.ndarray, threshold: float
) -> Dict[str, object]:
    """Return pion efficiency and purity for score >= threshold."""
    predictions = scores >= threshold
    truth_pion = y_true.astype(bool)
    true_positive = int(np.count_nonzero(predictions & truth_pion))
    false_positive = int(np.count_nonzero(predictions & ~truth_pion))
    false_negative = int(np.count_nonzero(~predictions & truth_pion))
    true_negative = int(np.count_nonzero(~predictions & ~truth_pion))
    efficiency_denominator = true_positive + false_negative
    purity_denominator = true_positive + false_positive
    efficiency = (
        true_positive / efficiency_denominator
        if efficiency_denominator
        else 0.0
    )
    purity = true_positive / purity_denominator if purity_denominator else 0.0
    return {
        "threshold": float(threshold),
        "efficiency": float(efficiency),
        "purity": float(purity),
        "efficiency_x_purity": float(efficiency * purity),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
    }


def plot_efficiency_purity_vs_threshold(
    y_test: np.ndarray,
    test_scores: np.ndarray,
    output: Path,
    evaluation_threshold: float,
) -> Tuple[str, str, Dict[str, object], Dict[str, object]]:
    """Plot and tabulate pion efficiency, purity, and their product."""
    scan_thresholds = np.unique(
        np.concatenate(
            [
                np.linspace(0.0, 1.0, 201),
                np.array([evaluation_threshold, 0.20], dtype=np.float64),
            ]
        )
    )
    operating_points = [
        pion_operating_point(y_test, test_scores, value)
        for value in scan_thresholds
    ]
    efficiencies = np.array(
        [point["efficiency"] for point in operating_points], dtype=np.float64
    )
    purities = np.array(
        [point["purity"] for point in operating_points], dtype=np.float64
    )
    products = np.array(
        [point["efficiency_x_purity"] for point in operating_points],
        dtype=np.float64,
    )

    csv_path = output.with_suffix(".efficiency_purity_vs_threshold.csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(operating_points[0]))
        writer.writeheader()
        writer.writerows(operating_points)

    fig, axis = plt.subplots(figsize=(7, 5))
    axis.plot(scan_thresholds, efficiencies, label="Pion efficiency")
    axis.plot(scan_thresholds, purities, label="Pion purity")
    axis.plot(scan_thresholds, products, label="Efficiency x purity")
    axis.axvline(0.20, color="black", linestyle="--", alpha=0.7, label="score = 0.20")
    if not np.isclose(evaluation_threshold, 0.20):
        axis.axvline(
            evaluation_threshold,
            color="gray",
            linestyle=":",
            alpha=0.8,
            label=f"configured score = {evaluation_threshold:.2f}",
        )
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.05)
    axis.set_xlabel("Minimum pion_score (pion if score >= threshold)")
    axis.set_ylabel("Fraction")
    axis.set_title("Pion efficiency and purity vs. score threshold")
    axis.legend()
    fig.tight_layout()
    plot_path = output.with_suffix(".efficiency_purity_vs_threshold.png")
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)

    return (
        plot_path.name,
        csv_path.name,
        pion_operating_point(y_test, test_scores, evaluation_threshold),
        pion_operating_point(y_test, test_scores, 0.20),
    )


def plot_confusion_matrix(
    y_test: np.ndarray,
    test_scores: np.ndarray,
    threshold: float,
    output: Path,
    *,
    normalize: bool = False,
    filename_suffix: str = ".confusion_matrix.png",
) -> str:
    predictions = (test_scores >= threshold).astype(int)
    counts = confusion_matrix(y_test, predictions, labels=[0, 1])
    if normalize:
        denominators = counts.sum(axis=1, keepdims=True)
        matrix = np.divide(
            counts,
            denominators,
            out=np.zeros_like(counts, dtype=np.float64),
            where=denominators != 0,
        )
    else:
        matrix = counts
    fig, axis = plt.subplots(figsize=(5, 5))
    image = axis.imshow(
        matrix,
        cmap="Blues",
        vmin=0.0,
        vmax=1.0 if normalize else None,
    )
    axis.set_xticks([0, 1])
    axis.set_xticklabels(["no pion", "pion"])
    axis.set_yticks([0, 1])
    axis.set_yticklabels(["no pion", "pion"])
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Truth")
    title = (
        "True-class-normalized confusion matrix"
        if normalize
        else "Confusion matrix"
    )
    axis.set_title(f"{title}\n(pion if score >= {threshold:.2f})")
    for row in range(2):
        for column in range(2):
            value = matrix[row, column]
            label = f"{value:.1%}" if normalize else str(int(value))
            axis.text(
                column,
                row,
                label,
                ha="center",
                va="center",
                color=(
                    "white"
                    if value > (0.5 if normalize else matrix.max() / 2)
                    else "black"
                ),
            )
    colorbar_label = "Fraction within truth class" if normalize else "Event count"
    fig.colorbar(image, ax=axis, label=colorbar_label)
    fig.tight_layout()
    path = output.with_suffix(filename_suffix)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path.name


def plot_calibration_curve(
    y_test: np.ndarray,
    test_scores: np.ndarray,
    output: Path,
    n_bins: int = 10,
) -> str:
    fraction_positive, mean_predicted = calibration_curve(
        y_test, test_scores, n_bins=n_bins
    )
    fig, axis = plt.subplots(figsize=(6, 6))
    axis.plot(
        mean_predicted,
        fraction_positive,
        marker="o",
        color="C3",
        label="Model",
    )
    axis.plot(
        [0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration"
    )
    axis.set_xlabel("Mean predicted pion_score in bin")
    axis.set_ylabel("Observed pion fraction in bin")
    axis.set_title("Calibration (reliability) curve")
    axis.legend()
    fig.tight_layout()
    path = output.with_suffix(".calibration_curve.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path.name


def plot_training_curves(history_path: Path, output: Path) -> str:
    with history_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return ""
    epochs = np.array([float(row["epoch"]) for row in rows])
    pairs = [
        ("loss", "val_loss"),
        ("roc_auc", "val_roc_auc"),
        ("pr_auc", "val_pr_auc"),
    ]
    pairs = [pair for pair in pairs if pair[0] in rows[0] and pair[1] in rows[0]]
    if not pairs:
        return ""
    fig, axes = plt.subplots(1, len(pairs), figsize=(5 * len(pairs), 4))
    axes = np.atleast_1d(axes)
    for axis, (train_key, validation_key) in zip(axes, pairs):
        axis.plot(epochs, [float(row[train_key]) for row in rows], label="train")
        axis.plot(
            epochs,
            [float(row[validation_key]) for row in rows],
            label="validation",
        )
        axis.set_xlabel("Epoch")
        axis.set_ylabel(train_key)
        axis.legend()
    fig.suptitle("Training history")
    fig.tight_layout()
    path = output.with_suffix(".training_curves.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path.name


def plot_misclassified_gallery(
    images_test: np.ndarray,
    detector_images_test: np.ndarray,
    y_test: np.ndarray,
    test_scores: np.ndarray,
    source_paths_test: np.ndarray,
    entries_test: np.ndarray,
    output: Path,
    threshold: float,
    per_category: int = 3,
) -> str:
    """Plot the most confident false positives and false negatives."""
    categories = [
        (
            "Confident false positive (no-pion scored high)",
            (y_test == 0) & (test_scores >= threshold),
            True,
        ),
        (
            "Confident false negative (pion scored low)",
            (y_test == 1) & (test_scores < threshold),
            False,
        ),
    ]
    rows: List[Tuple[str, np.ndarray]] = []
    for title, mask, descending in categories:
        indices = np.flatnonzero(mask)
        if len(indices) == 0:
            continue
        order = indices[np.argsort(test_scores[indices])]
        if descending:
            order = order[::-1]
        rows.append((title, order[:per_category]))
    if not rows:
        return ""

    row_count = sum(len(order) for _, order in rows)
    fig, axes = plt.subplots(
        row_count, 2, figsize=(9, 3.6 * row_count), squeeze=False
    )
    row_index = 0
    for title, order in rows:
        for event_index in order:
            angular = images_test[event_index][:, 1:-1, 0]
            detector = detector_images_test[event_index][:, :, 0]
            source_name = Path(str(source_paths_test[event_index])).name
            caption = (
                f"{title}\n{source_name} entry {int(entries_test[event_index])} — "
                f"score={test_scores[event_index]:.3f}"
            )
            axes[row_index, 0].imshow(
                angular, origin="lower", aspect="auto", cmap="magma"
            )
            axes[row_index, 0].set_title(caption, fontsize=8)
            axes[row_index, 1].imshow(
                detector, origin="lower", aspect="equal", cmap="magma"
            )
            axes[row_index, 1].set_title("Unfolded view", fontsize=8)
            row_index += 1
    fig.tight_layout()
    path = output.with_suffix(".misclassified_gallery.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path.name


def plot_pion_examples(
    images_test: np.ndarray,
    detector_images_test: np.ndarray,
    y_test: np.ndarray,
    test_scores: np.ndarray,
    source_paths_test: np.ndarray,
    entries_test: np.ndarray,
    truth_pion_counts_test: np.ndarray,
    output: Path,
    example_count: int = 3,
) -> str:
    """Draw high-scoring truth-pion events with their pion composition."""
    pion_indices = np.flatnonzero(y_test == 1)
    if len(pion_indices) == 0:
        return ""
    order = pion_indices[np.argsort(test_scores[pion_indices])[::-1]][:example_count]
    fig, axes = plt.subplots(
        len(order), 4, figsize=(16, 3.8 * len(order)), squeeze=False
    )
    column_titles = [
        "Angular full hitPE",
        "Angular tank-cluster hitPE",
        "Unfolded full hitPE",
        "Unfolded tank-cluster hitPE",
    ]
    for row_index, event_index in enumerate(order):
        angular = images_test[event_index][:, 1:-1, :]
        detector = detector_images_test[event_index]
        panels = [
            (angular[:, :, 0], "auto"),
            (angular[:, :, 1], "auto"),
            (detector[:, :, 0], "equal"),
            (detector[:, :, 1], "equal"),
        ]
        for column, ((panel, aspect), title) in enumerate(
            zip(panels, column_titles)
        ):
            axes[row_index, column].imshow(
                panel, origin="lower", aspect=aspect, cmap="magma"
            )
            axes[row_index, column].set_title(title, fontsize=9)
        pi_plus, pi_minus, pi_zero = (
            int(value) for value in truth_pion_counts_test[event_index]
        )
        source_name = Path(str(source_paths_test[event_index])).name
        axes[row_index, 0].set_ylabel(
            f"{source_name}\nentry {int(entries_test[event_index])}\n"
            f"score={test_scores[event_index]:.3f}\n"
            f"π⁺={pi_plus}, π⁻={pi_minus}, π⁰={pi_zero}",
            fontsize=8,
        )
    fig.suptitle(
        "Highest-scoring held-out truth-pion events "
        "(truth information is annotation only)",
        fontsize=12,
    )
    fig.tight_layout()
    path = output.with_suffix(".pion_examples.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path.name


def _grad_cam_heatmaps(
    model: tf.keras.Model,
    conv_layer_name: str,
    inputs: Dict[str, np.ndarray],
    target_shape: Tuple[int, int],
) -> np.ndarray:
    """Calculate batched Grad-CAM maps for one convolutional tower."""
    grad_model = tf.keras.Model(
        model.inputs, [model.get_layer(conv_layer_name).output, model.output]
    )
    tensors = {name: tf.convert_to_tensor(value) for name, value in inputs.items()}
    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(tensors, training=False)
        loss = predictions[:, 0]
    gradients = tape.gradient(loss, conv_output)
    pooled_gradients = tf.reduce_mean(gradients, axis=(1, 2))
    heatmaps = tf.nn.relu(
        tf.einsum("bhwc,bc->bhw", conv_output, pooled_gradients)
    )
    heatmaps = tf.image.resize(heatmaps[..., tf.newaxis], target_shape)[..., 0]
    heatmaps = heatmaps.numpy()
    peaks = heatmaps.reshape(len(heatmaps), -1).max(axis=1)
    peaks[peaks == 0] = 1.0
    return heatmaps / peaks[:, None, None]


def plot_gradcam_examples(
    model: tf.keras.Model,
    images_test: np.ndarray,
    detector_images_test: np.ndarray,
    y_test: np.ndarray,
    test_scores: np.ndarray,
    output: Path,
    threshold: float,
) -> str:
    """Plot a confident TP, TN, FP, and FN with Grad-CAM overlays."""
    predictions = (test_scores >= threshold).astype(int)
    categories = {
        "True positive": (y_test == 1) & (predictions == 1),
        "True negative": (y_test == 0) & (predictions == 0),
        "False positive": (y_test == 0) & (predictions == 1),
        "False negative": (y_test == 1) & (predictions == 0),
    }
    selected: List[Tuple[str, int]] = []
    for title, mask in categories.items():
        indices = np.flatnonzero(mask)
        if len(indices) == 0:
            continue
        confidence = np.abs(test_scores[indices] - threshold)
        selected.append((title, int(indices[np.argmax(confidence)])))
    if not selected:
        return ""

    selected_indices = [index for _, index in selected]
    angular_batch = images_test[selected_indices]
    detector_batch = detector_images_test[selected_indices]
    model_inputs = {
        "pmt_angular_image": angular_batch,
        "pmt_unfolded_image": detector_batch,
    }
    angular_heatmaps = _grad_cam_heatmaps(
        model,
        "angular_conv3",
        model_inputs,
        (angular_batch.shape[1], angular_batch.shape[2]),
    )
    detector_heatmaps = _grad_cam_heatmaps(
        model,
        "detector_conv3",
        model_inputs,
        (detector_batch.shape[1], detector_batch.shape[2]),
    )

    fig, axes = plt.subplots(
        len(selected), 2, figsize=(9, 3.6 * len(selected)), squeeze=False
    )
    for row_index, (title, event_index) in enumerate(selected):
        angular_visible = angular_batch[row_index][:, 1:-1, 0]
        heat_visible = angular_heatmaps[row_index][:, 1:-1]
        axes[row_index, 0].imshow(
            angular_visible, origin="lower", aspect="auto", cmap="gray"
        )
        axes[row_index, 0].imshow(
            heat_visible, origin="lower", aspect="auto", cmap="jet", alpha=0.5
        )
        axes[row_index, 0].set_title(
            f"{title} (score={test_scores[event_index]:.3f}) — angular Grad-CAM",
            fontsize=8,
        )
        axes[row_index, 1].imshow(
            detector_batch[row_index][:, :, 0],
            origin="lower",
            aspect="equal",
            cmap="gray",
        )
        axes[row_index, 1].imshow(
            detector_heatmaps[row_index],
            origin="lower",
            aspect="equal",
            cmap="jet",
            alpha=0.5,
        )
        axes[row_index, 1].set_title("Unfolded Grad-CAM", fontsize=8)
    fig.tight_layout()
    path = output.with_suffix(".gradcam_examples.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path.name


def evaluate_and_plot(
    model: tf.keras.Model,
    images_test: np.ndarray,
    detector_images_test: np.ndarray,
    y_test: np.ndarray,
    source_paths_test: np.ndarray,
    entries_test: np.ndarray,
    truth_pion_counts_test: np.ndarray,
    history_path: Path,
    output: Path,
    threshold: float,
) -> Dict[str, object]:
    test_scores = model.predict(
        {
            "pmt_angular_image": images_test,
            "pmt_unfolded_image": detector_images_test,
        },
        verbose=0,
    ).reshape(-1)

    predictions_path = output.with_suffix(".test_predictions.csv")
    with predictions_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "source_file",
                "tree_entry",
                "true_label",
                "truePiPlusCher",
                "truePiMinusCher",
                "truePi0",
                "pion_score",
            ]
        )
        writer.writerows(
            zip(
                source_paths_test.tolist(),
                entries_test.astype(int).tolist(),
                y_test.astype(int).tolist(),
                truth_pion_counts_test[:, 0].astype(int).tolist(),
                truth_pion_counts_test[:, 1].astype(int).tolist(),
                truth_pion_counts_test[:, 2].astype(int).tolist(),
                test_scores.astype(float).tolist(),
            )
        )

    roc_name, roc_auc_value, fpr, tpr, thresholds = plot_roc_curve(
        y_test, test_scores, output
    )
    pr_name, average_precision = plot_precision_recall(
        y_test, test_scores, output
    )
    (
        efficiency_purity_plot,
        efficiency_purity_csv,
        configured_operating_point,
        threshold_0p20_operating_point,
    ) = plot_efficiency_purity_vs_threshold(
        y_test, test_scores, output, threshold
    )
    return {
        "test_predictions": predictions_path.name,
        "roc_curve": roc_name,
        "roc_auc_sklearn": roc_auc_value,
        "precision_recall_curve": pr_name,
        "average_precision": average_precision,
        "score_distribution": plot_score_distribution(
            y_test, test_scores, output
        ),
        "efficiency_rejection_vs_threshold": (
            plot_efficiency_rejection_vs_threshold(
                fpr, tpr, thresholds, output
            )
        ),
        "efficiency_purity_vs_threshold": efficiency_purity_plot,
        "efficiency_purity_vs_threshold_csv": efficiency_purity_csv,
        "pion_metrics_at_configured_threshold": configured_operating_point,
        "pion_metrics_at_threshold_0p20": threshold_0p20_operating_point,
        "confusion_matrix": plot_confusion_matrix(
            y_test, test_scores, threshold, output
        ),
        "confusion_matrix_normalized": plot_confusion_matrix(
            y_test,
            test_scores,
            threshold,
            output,
            normalize=True,
            filename_suffix=".confusion_matrix_normalized.png",
        ),
        "confusion_matrix_threshold_0p20": plot_confusion_matrix(
            y_test,
            test_scores,
            0.20,
            output,
            filename_suffix=".confusion_matrix_threshold_0p20.png",
        ),
        "confusion_matrix_threshold_0p20_normalized": plot_confusion_matrix(
            y_test,
            test_scores,
            0.20,
            output,
            normalize=True,
            filename_suffix=".confusion_matrix_threshold_0p20_normalized.png",
        ),
        "calibration_curve": plot_calibration_curve(y_test, test_scores, output),
        "training_curves": plot_training_curves(history_path, output),
        "misclassified_gallery": plot_misclassified_gallery(
            images_test,
            detector_images_test,
            y_test,
            test_scores,
            source_paths_test,
            entries_test,
            output,
            threshold,
        ),
        "pion_examples": plot_pion_examples(
            images_test,
            detector_images_test,
            y_test,
            test_scores,
            source_paths_test,
            entries_test,
            truth_pion_counts_test,
            output,
        ),
        "gradcam_examples": plot_gradcam_examples(
            model,
            images_test,
            detector_images_test,
            y_test,
            test_scores,
            output,
            threshold,
        ),
    }


def main() -> None:
    args = parse_args()
    if args.image_height < 4 or args.image_width < 8:
        raise ValueError("The PMT image must be at least 4 x 8 bins")
    if args.detector_height < 24 or args.detector_width < 16:
        raise ValueError("The unfolded detector image must be at least 24 x 16 bins")
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    (
        images,
        detector_images,
        labels,
        source_paths,
        entries,
        truth_pion_counts,
        tank_branch,
        n_misaligned,
        geometry,
        response,
    ) = load_data(args)
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
    evaluation_files = evaluate_and_plot(
        model,
        images[test_indices],
        detector_images[test_indices],
        y_test,
        source_paths[test_indices],
        entries[test_indices],
        truth_pion_counts[test_indices],
        history_path,
        args.output,
        args.threshold,
    )
    print(
        "Test ROC AUC cross-check: "
        f"{evaluation_files['roc_auc_sklearn']:.5f}; "
        f"average precision: {evaluation_files['average_precision']:.5f}"
    )
    metadata = {
        "model_type": "annie_pion_ring_cnn",
        "model_variant": "A_PE_dual_view",
        "tensorflow_version": tf.__version__,
        "training_sources": [str(path) for path in args.root_files],
        "tree": args.tree,
        "geometry_file": args.geometry.name,
        "pmt_mask": args.pmt_mask,
        "pmt_count_included": geometry.included_count,
        "excluded_pmt_ids": geometry.excluded_ids,
        "training_pmt_response": response.mode,
        "pmt_response_scope": (
            "separate_branch_kind_0_hitPE_and_branch_kind_1_tankcluster"
            if response.payload_format == "final_pmt_tuning_parameters"
            else "shared_legacy_map_hitPE_and_hitPE_tankcluster"
        ),
        "pmt_response_calibration_file": (
            response.calibration_path.name if response.calibration_path else None
        ),
        "pmt_response_calibration_sha256": response.calibration_sha256,
        "pmt_response_payload_format": response.payload_format,
        "pmt_tune_variant": response.tune_variant,
        "pmt_tune_random_stream_version": response.random_stream_version,
        "pmt_response_mapped_pmt_count": response.mapped_pmt_count,
        "hitpe_branch": args.hitpe_branch,
        "hitid_branch": args.hitid_branch,
        "tankcluster_branch": tank_branch,
        "tankcluster_id_branch": args.tankcluster_id_branch,
        "image_height": args.image_height,
        "image_width": args.image_width,
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
        "projections": [
            "vertex_centered_equirectangular_with_azimuth_wrap",
            "annotation_free_unfolded_barrel_top_bottom",
        ],
        "label": "charged_pion_present" if args.charged_only else "any_pion_present",
        "truth_branches": ["truePiPlusCher", "truePiMinusCher", "truePi0"],
        "event_selection": "fit_individual_pmt",
        "event_selection_source": "Fit_indivdiualPMT_Gaussian_Convolution.cpp",
        "event_selection_expression": FIT_INDIVIDUAL_PMT_SELECTION_EXPRESSION,
        "event_selection_applied": not args.no_event_cuts,
        "bdt_preselection": not args.no_event_cuts,
        "threshold": args.threshold,
        "class_counts": {"no_pion": int(counts[0]), "pion": int(counts[1])},
        "misaligned_events_rejected": n_misaligned,
        "test_metrics": {name: float(value) for name, value in metrics.items()},
        "training_history": history_path.name,
        "evaluation": evaluation_files,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Saved {args.output}, {args.output.with_suffix('.json')}, "
        f"and {history_path}"
    )
    print("Evaluation outputs:")
    for name, value in evaluation_files.items():
        if isinstance(value, str) and value:
            print(f"  {name}: {value}")
    print("Pion operating points (score >= threshold):")
    for name in (
        "pion_metrics_at_configured_threshold",
        "pion_metrics_at_threshold_0p20",
    ):
        point = evaluation_files[name]
        print(
            f"  threshold={point['threshold']:.2f}: "
            f"efficiency={point['efficiency']:.5f}, "
            f"purity={point['purity']:.5f}, "
            f"efficiency*purity={point['efficiency_x_purity']:.5f}"
        )


if __name__ == "__main__":
    main()
