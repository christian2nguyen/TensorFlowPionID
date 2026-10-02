#!/usr/bin/env python3
"""Export an ANNIE ring Keras model as a TensorFlow SavedModel."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import tensorflow as tf


def _input_shapes(model: tf.keras.Model) -> Dict[str, Tuple[int, ...]]:
    shapes = {}
    for model_input in model.inputs:
        name = model_input.name.split(":", 1)[0]
        shape = tuple(int(dimension) for dimension in model_input.shape[1:])
        shapes[name] = shape
    return shapes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("keras_model", type=Path)
    parser.add_argument("saved_model_directory", type=Path)
    args = parser.parse_args()

    model = tf.keras.models.load_model(args.keras_model)
    shapes = _input_shapes(model)
    image_names = {"pmt_angular_image", "pmt_unfolded_image"}
    expected_names = image_names | ({"event_features"} if "event_features" in shapes else set())
    if set(shapes) != expected_names:
        raise ValueError(
            f"Unexpected model inputs {sorted(shapes)}; expected {sorted(expected_names)}"
        )

    if "event_features" in shapes:

        @tf.function(
            input_signature=[
                tf.TensorSpec(
                    [None, *shapes["pmt_angular_image"]],
                    tf.float32,
                    name="pmt_angular_image",
                ),
                tf.TensorSpec(
                    [None, *shapes["pmt_unfolded_image"]],
                    tf.float32,
                    name="pmt_unfolded_image",
                ),
                tf.TensorSpec(
                    [None, *shapes["event_features"]],
                    tf.float32,
                    name="event_features",
                ),
            ]
        )
        def serve(pmt_angular_image, pmt_unfolded_image, event_features):
            score = model(
                {
                    "pmt_angular_image": pmt_angular_image,
                    "pmt_unfolded_image": pmt_unfolded_image,
                    "event_features": event_features,
                },
                training=False,
            )
            return {"pion_score": score}

    else:

        @tf.function(
            input_signature=[
                tf.TensorSpec(
                    [None, *shapes["pmt_angular_image"]],
                    tf.float32,
                    name="pmt_angular_image",
                ),
                tf.TensorSpec(
                    [None, *shapes["pmt_unfolded_image"]],
                    tf.float32,
                    name="pmt_unfolded_image",
                ),
            ]
        )
        def serve(pmt_angular_image, pmt_unfolded_image):
            score = model(
                {
                    "pmt_angular_image": pmt_angular_image,
                    "pmt_unfolded_image": pmt_unfolded_image,
                },
                training=False,
            )
            return {"pion_score": score}

    args.saved_model_directory.parent.mkdir(parents=True, exist_ok=True)
    tf.saved_model.save(
        model,
        str(args.saved_model_directory),
        signatures={"serving_default": serve},
    )
    loaded = tf.saved_model.load(str(args.saved_model_directory))
    signature = loaded.signatures["serving_default"]
    print(f"Saved TensorFlow model to {args.saved_model_directory}")
    print(f"Inputs: {signature.structured_input_signature}")
    print(f"Outputs: {signature.structured_outputs}")


if __name__ == "__main__":
    main()
