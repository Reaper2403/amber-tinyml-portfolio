#!/usr/bin/env python3
"""Build Keras/TFLite exports for a Stage 1 checkpoint transfer bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf


def build_model(num_classes: int) -> tf.keras.Model:
    inp = tf.keras.Input(shape=(150, 6), name="imu_input")
    x = tf.keras.layers.Conv1D(32, 5, padding="same", use_bias=False, name="stem_conv")(inp)
    x = tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.1, name="stem_bn")(x)
    x = tf.keras.layers.ReLU(name="stem_relu")(x)

    def ds_block(x: tf.Tensor, out_channels: int, kernel_size: int, name: str) -> tf.Tensor:
        x = tf.keras.layers.SeparableConv1D(
            out_channels,
            kernel_size,
            padding="same",
            depth_multiplier=1,
            use_bias=False,
            name=f"{name}_sep",
        )(x)
        x = tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.1, name=f"{name}_bn")(x)
        x = tf.keras.layers.ReLU(name=f"{name}_relu")(x)
        return x

    x = ds_block(x, 64, 7, "b0")
    x = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, padding="valid", name="pool0")(x)
    x = ds_block(x, 96, 5, "b1")
    x = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, padding="valid", name="pool1")(x)
    x = ds_block(x, 128, 5, "b2")
    x = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, padding="valid", name="pool2")(x)
    x = ds_block(x, 160, 3, "b3")
    x = tf.keras.layers.GlobalAveragePooling1D(name="gap")(x)
    x = tf.keras.layers.Dense(128, activation="relu", name="fc0")(x)
    out = tf.keras.layers.Dense(num_classes, name="class_logits")(x)
    return tf.keras.Model(inp, out)


def load_weights(model: tf.keras.Model, weights: np.lib.npyio.NpzFile) -> None:
    model(np.zeros((1, 150, 6), dtype=np.float32), training=False)

    model.get_layer("stem_conv").set_weights([np.transpose(weights["stem.0.weight"], (2, 1, 0))])
    model.get_layer("stem_bn").set_weights(
        [
            weights["stem.1.weight"],
            weights["stem.1.bias"],
            weights["stem.1.running_mean"],
            weights["stem.1.running_var"],
        ]
    )

    block_map = [("b0", 0), ("b1", 2), ("b2", 4), ("b3", 6)]
    for keras_name, idx in block_map:
        sep = model.get_layer(f"{keras_name}_sep")
        depthwise = np.transpose(weights[f"features.{idx}.depthwise.weight"], (2, 0, 1))
        pointwise = np.transpose(weights[f"features.{idx}.pointwise.weight"], (2, 1, 0))
        sep.set_weights([depthwise, pointwise])
        bn = model.get_layer(f"{keras_name}_bn")
        bn.set_weights(
            [
                weights[f"features.{idx}.bn.weight"],
                weights[f"features.{idx}.bn.bias"],
                weights[f"features.{idx}.bn.running_mean"],
                weights[f"features.{idx}.bn.running_var"],
            ]
        )

    model.get_layer("fc0").set_weights([weights["head.1.weight"].T, weights["head.1.bias"]])
    model.get_layer("class_logits").set_weights([weights["head.4.weight"].T, weights["head.4.bias"]])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path, required=True, help="Directory produced by export_stage1_weights_and_sample.py")
    parser.add_argument("--representative-cache", type=Path, required=True, help="Path to train_windows.npz for int8 representative windows")
    parser.add_argument("--sample-tolerance", type=float, default=1e-4, help="Allowed max abs diff between PyTorch and Keras logits")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weights_path = args.bundle_dir / "keras_transfer_weights.npz"
    sample_path = args.bundle_dir / "reference_sample.npz"
    metadata_path = args.bundle_dir / "export_metadata.json"
    weights = np.load(weights_path)
    sample = np.load(sample_path)
    metadata = json.loads(metadata_path.read_text())

    num_classes = int(metadata["num_active_classes"])
    model = build_model(num_classes)
    load_weights(model, weights)

    x = sample["x"].astype(np.float32)
    pt_logits = sample["logits"].astype(np.float32)
    tf_logits = model(x, training=False).numpy()
    max_abs_diff = float(np.max(np.abs(pt_logits - tf_logits)))
    print(f"[tf] max abs diff vs PyTorch: {max_abs_diff:.8f}")
    if max_abs_diff > args.sample_tolerance:
        raise RuntimeError(
            f"Keras transfer diverged from PyTorch. max_abs_diff={max_abs_diff:.8f} "
            f"> tolerance={args.sample_tolerance:.8f}"
        )

    saved_model_dir = args.bundle_dir / "keras_saved_model"
    float32_tflite = args.bundle_dir / "keras_model_float32.tflite"
    int8_tflite = args.bundle_dir / "keras_model_int8.tflite"

    model.export(str(saved_model_dir))

    float_converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    float_model = float_converter.convert()
    float32_tflite.write_bytes(float_model)

    train_cache = np.load(args.representative_cache)
    rep_cache = [train_cache["x"][i : i + 1].astype(np.float32) for i in range(min(128, len(train_cache["x"])))]

    def representative_dataset():
        for window in rep_cache:
            yield [window]

    int8_converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    int8_converter.optimizations = [tf.lite.Optimize.DEFAULT]
    int8_converter.representative_dataset = representative_dataset
    int8_converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    int8_converter.inference_input_type = tf.int8
    int8_converter.inference_output_type = tf.int8
    int8_model = int8_converter.convert()
    int8_tflite.write_bytes(int8_model)

    float_interp = tf.lite.Interpreter(model_path=str(float32_tflite), experimental_delegates=[])
    float_interp.allocate_tensors()
    in_detail = float_interp.get_input_details()[0]
    out_detail = float_interp.get_output_details()[0]
    float_interp.set_tensor(in_detail["index"], x)
    float_interp.invoke()
    float_logits = float_interp.get_tensor(out_detail["index"])
    float_tflite_diff = float(np.max(np.abs(pt_logits - float_logits)))

    int8_interp = tf.lite.Interpreter(model_path=str(int8_tflite), experimental_delegates=[])
    int8_interp.allocate_tensors()
    in_detail = int8_interp.get_input_details()[0]
    out_detail = int8_interp.get_output_details()[0]
    input_scale, input_zero_point = in_detail["quantization"]
    q_input = np.round(x / input_scale + input_zero_point).astype(np.int32)
    q_input = np.clip(q_input, -128, 127).astype(np.int8)
    int8_interp.set_tensor(in_detail["index"], q_input)
    int8_interp.invoke()
    q_logits = int8_interp.get_tensor(out_detail["index"])
    output_scale, output_zero_point = out_detail["quantization"]
    q_logits = (q_logits.astype(np.int32) - output_zero_point) * output_scale
    int8_tflite_diff = float(np.max(np.abs(pt_logits - q_logits)))

    report = {
        "num_active_classes": num_classes,
        "active_labels": metadata["active_labels"],
        "pytorch_vs_keras_max_abs_diff": max_abs_diff,
        "pytorch_vs_float32_tflite_max_abs_diff": float_tflite_diff,
        "pytorch_vs_int8_tflite_max_abs_diff": int8_tflite_diff,
        "float32_tflite": str(float32_tflite),
        "int8_tflite": str(int8_tflite),
        "saved_model_dir": str(saved_model_dir),
    }
    report_path = args.bundle_dir / "validation_report.json"
    report_path.write_text(json.dumps(report, indent=2))

    print(f"[tf] wrote float32 tflite: {float32_tflite}")
    print(f"[tf] wrote int8 tflite: {int8_tflite}")
    print(f"[tf] wrote validation report: {report_path}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
