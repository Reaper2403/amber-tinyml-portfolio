#!/usr/bin/env python3
"""Build float32/int8 TFLite exports from an audio DS-CNN PyTorch checkpoint archive."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import re
import zipfile
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf


_DTYPE_MAP = {
    "FloatStorage": np.float32,
    "DoubleStorage": np.float64,
    "HalfStorage": np.float16,
    "LongStorage": np.int64,
    "IntStorage": np.int32,
    "ShortStorage": np.int16,
    "CharStorage": np.int8,
    "ByteStorage": np.uint8,
    "BoolStorage": np.bool_,
}


class _FakeStorageType:
    def __init__(self, name: str, dtype: np.dtype):
        self.__name__ = name
        self.dtype = dtype


@dataclass
class _StorageView:
    raw: bytes
    dtype: np.dtype


def _rebuild_tensor(storage: _StorageView, storage_offset, size, stride, *_args):
    size = tuple(int(v) for v in size)
    stride = tuple(int(v) for v in stride)
    offset_bytes = int(storage_offset) * np.dtype(storage.dtype).itemsize
    strides_bytes = tuple(int(v) * np.dtype(storage.dtype).itemsize for v in stride)
    arr = np.ndarray(
        shape=size,
        dtype=storage.dtype,
        buffer=storage.raw,
        offset=offset_bytes,
        strides=strides_bytes,
    )
    return np.array(arr, copy=True)


class _CheckpointUnpickler(pickle.Unpickler):
    def __init__(self, file_obj, zf: zipfile.ZipFile, archive_root: str):
        super().__init__(file_obj)
        self._zf = zf
        self._archive_root = archive_root

    def find_class(self, module: str, name: str):
        if module == "collections" and name == "OrderedDict":
            return OrderedDict
        if module == "torch._utils" and name in {"_rebuild_tensor_v2", "_rebuild_tensor_v3"}:
            return _rebuild_tensor
        if module == "torch._tensor" and name == "_rebuild_from_type_v2":
            def _rebuild_from_type_v2(func, new_type, args, state):
                return func(*args)

            return _rebuild_from_type_v2
        if module == "torch" and name.endswith("Storage"):
            if name not in _DTYPE_MAP:
                raise ValueError(f"Unsupported storage type: {name}")
            return _FakeStorageType(name, np.dtype(_DTYPE_MAP[name]))
        raise pickle.UnpicklingError(f"Unsupported pickle class: {module}.{name}")

    def persistent_load(self, pid):
        if not isinstance(pid, tuple) or len(pid) != 5 or pid[0] != "storage":
            raise pickle.UnpicklingError(f"Unsupported persistent id: {pid}")
        _, storage_type, key, _location, _size = pid
        raw = self._zf.read(f"{self._archive_root}/data/{key}")
        return _StorageView(raw=raw, dtype=np.dtype(storage_type.dtype))


def load_pytorch_checkpoint(path: Path) -> dict:
    with zipfile.ZipFile(path) as zf:
        data_names = [name for name in zf.namelist() if name.endswith("/data.pkl")]
        if len(data_names) != 1:
            raise RuntimeError(f"Expected exactly one data.pkl in {path}, found {data_names}")
        data_name = data_names[0]
        archive_root = data_name.rsplit("/", 1)[0]
        with zf.open(data_name, "r") as f:
            obj = _CheckpointUnpickler(f, zf, archive_root).load()
    return obj


def build_model(input_shape: tuple[int, int, int], num_classes: int, model_cfg: dict) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=input_shape, name="logmel_input")
    x = tf.keras.layers.ZeroPadding2D(((2, 2), (2, 2)), name="stem_pad")(inputs)
    x = tf.keras.layers.Conv2D(
        model_cfg["stem_filters"],
        kernel_size=(5, 5),
        strides=(2, 2),
        padding="valid",
        use_bias=False,
        name="stem_conv",
    )(x)
    x = tf.keras.layers.BatchNormalization(epsilon=1e-5, name="stem_bn")(x)
    x = tf.keras.layers.ReLU(name="stem_relu")(x)

    for idx, filters in enumerate(model_cfg["block_filters"]):
        strides = (2, 2) if idx in (1, 3) else (1, 1)
        x = tf.keras.layers.ZeroPadding2D(((1, 1), (1, 1)), name=f"block{idx + 1}_pad")(x)
        x = tf.keras.layers.DepthwiseConv2D(
            kernel_size=(3, 3),
            strides=strides,
            padding="valid",
            use_bias=False,
            name=f"block{idx + 1}_dw",
        )(x)
        x = tf.keras.layers.Conv2D(
            filters,
            kernel_size=(1, 1),
            padding="valid",
            use_bias=False,
            name=f"block{idx + 1}_pw",
        )(x)
        x = tf.keras.layers.BatchNormalization(epsilon=1e-5, name=f"block{idx + 1}_bn")(x)
        x = tf.keras.layers.ReLU(name=f"block{idx + 1}_relu")(x)
        x = tf.keras.layers.Dropout(model_cfg["dropout"], name=f"block{idx + 1}_drop")(x)

    x = tf.keras.layers.GlobalAveragePooling2D(name="global_pool")(x)
    x = tf.keras.layers.Dropout(0.20, name="head_drop")(x)
    outputs = tf.keras.layers.Dense(num_classes, name="classifier")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="audio_ds_cnn")


def _set_bn_weights(layer: tf.keras.layers.BatchNormalization, prefix: str, state: dict) -> None:
    layer.set_weights(
        [
            state[f"{prefix}.weight"],
            state[f"{prefix}.bias"],
            state[f"{prefix}.running_mean"],
            state[f"{prefix}.running_var"],
        ]
    )


def load_weights(model: tf.keras.Model, state: dict) -> None:
    sample_input = np.zeros((1,) + tuple(model.input_shape[1:]), dtype=np.float32)
    model(sample_input, training=False)

    model.get_layer("stem_conv").set_weights(
        [np.transpose(state["stem.0.weight"], (2, 3, 1, 0))]
    )
    _set_bn_weights(model.get_layer("stem_bn"), "stem.1", state)

    for idx, _filters in enumerate(model.get_config()["layers"]):
        pass

    for block_idx in range(4):
        base = f"blocks.{block_idx}.net"
        model.get_layer(f"block{block_idx + 1}_dw").set_weights(
            [np.transpose(state[f"{base}.0.weight"], (2, 3, 0, 1))]
        )
        model.get_layer(f"block{block_idx + 1}_pw").set_weights(
            [np.transpose(state[f"{base}.1.weight"], (2, 3, 1, 0))]
        )
        _set_bn_weights(model.get_layer(f"block{block_idx + 1}_bn"), f"{base}.2", state)

    model.get_layer("classifier").set_weights(
        [state["head.3.weight"].T, state["head.3.bias"]]
    )


def sanitize_name(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z_]+", "_", value).strip("_")


def resolve_dataset_dir(base: Path, target_dir_name: str) -> Path:
    matches = sorted([p for p in base.rglob(target_dir_name) if p.is_dir()])
    if not matches:
        raise FileNotFoundError(f"Could not find {target_dir_name} under {base}")
    return matches[0]


def load_manifest_paths(root: Path, split: str) -> list[str]:
    paths: list[str] = []
    with open(root / "manifests" / f"{split}.csv", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            paths.append(str(root / row["wav_path"]))
    return paths


def compute_frame_count(sample_count: int, fft_length: int, hop_length: int) -> int:
    pad = fft_length // 2
    return 1 + math.floor((sample_count + 2 * pad - fft_length) / hop_length)


def build_frontend(frontend: dict):
    sample_rate = int(frontend["sample_rate_hz"])
    sample_count = int(sample_rate * frontend["clip_duration_ms"] / 1000)
    frame_length = int(sample_rate * frontend["frame_length_ms"] / 1000)
    frame_step = int(sample_rate * frontend["frame_step_ms"] / 1000)
    fft_length = int(frontend["fft_length"])
    num_mel_bins = int(frontend["num_mel_bins"])
    lower_edge_hz = float(frontend["lower_edge_hz"])
    upper_edge_hz = float(frontend["upper_edge_hz"])
    db_min = float(frontend["log_clip_min_db"])
    db_max = float(frontend["log_clip_max_db"])
    frame_count = compute_frame_count(sample_count, fft_length, frame_step)

    mel_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=num_mel_bins,
        num_spectrogram_bins=fft_length // 2 + 1,
        sample_rate=sample_rate,
        lower_edge_hertz=lower_edge_hz,
        upper_edge_hertz=upper_edge_hz,
    )

    def waveform_to_logmel(audio_1d: tf.Tensor) -> tf.Tensor:
        audio_1d = tf.cast(audio_1d, tf.float32)
        audio_1d = audio_1d[:sample_count]
        audio_1d = tf.pad(audio_1d, [[0, tf.maximum(0, sample_count - tf.shape(audio_1d)[0])]])
        pad = fft_length // 2
        audio_1d = tf.pad(audio_1d, [[pad, pad]], mode="REFLECT")
        stft = tf.signal.stft(
            audio_1d,
            frame_length=frame_length,
            frame_step=frame_step,
            fft_length=fft_length,
            window_fn=tf.signal.hann_window,
            pad_end=False,
        )
        power = tf.math.square(tf.abs(stft))
        mel = tf.matmul(power, mel_matrix)
        mel_db = 10.0 * tf.math.log(tf.maximum(mel, 1e-10)) / tf.math.log(10.0)
        mel_db = tf.clip_by_value(mel_db, db_min, db_max)
        mel_db = (mel_db - db_min) / (db_max - db_min)
        mel_db = tf.transpose(mel_db, perm=[1, 0])
        mel_db = tf.ensure_shape(mel_db, [num_mel_bins, frame_count])
        return tf.expand_dims(mel_db, axis=-1)

    input_shape = (num_mel_bins, frame_count, 1)
    return waveform_to_logmel, input_shape, sample_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--public-root", type=Path, required=True)
    parser.add_argument("--device-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage-name", required=True)
    parser.add_argument("--rep-samples", type=int, default=192)
    parser.add_argument("--sample-tolerance", type=float, default=1e-4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    ckpt = load_pytorch_checkpoint(args.checkpoint)
    state = ckpt["model_state_dict"]
    label_names = ckpt["label_names"]
    frontend = ckpt["frontend_config"]
    model_cfg = ckpt["model_cfg"]

    waveform_to_logmel, input_shape, _sample_count = build_frontend(frontend)
    model = build_model(input_shape=input_shape, num_classes=len(label_names), model_cfg=model_cfg)
    load_weights(model, state)

    public_train_paths = load_manifest_paths(args.public_root, "train")
    device_train_paths = load_manifest_paths(args.device_root, "train")
    rep_paths = public_train_paths + device_train_paths
    rng = np.random.default_rng(42)
    if len(rep_paths) > args.rep_samples:
        indices = rng.choice(len(rep_paths), size=args.rep_samples, replace=False)
        rep_paths = [rep_paths[int(i)] for i in indices]

    def load_feature(path_str: str) -> np.ndarray:
        audio_bytes = tf.io.read_file(path_str)
        wav, _ = tf.audio.decode_wav(audio_bytes, desired_channels=1)
        wav = tf.squeeze(wav, axis=-1)
        feat = waveform_to_logmel(wav)
        return feat.numpy().astype(np.float32)

    rep_features = [load_feature(p) for p in rep_paths]
    sample_feature = rep_features[0][None, ...]

    keras_logits = model(sample_feature, training=False).numpy()

    saved_model_dir = args.output_dir / "saved_model"
    model.export(str(saved_model_dir))

    float32_tflite = args.output_dir / f"{sanitize_name(args.stage_name)}_float32.tflite"
    int8_tflite = args.output_dir / f"{sanitize_name(args.stage_name)}_int8.tflite"

    float_converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    float_model = float_converter.convert()
    float32_tflite.write_bytes(float_model)

    def representative_dataset():
        for feat in rep_features:
            yield [feat[None, ...]]

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
    float_interp.set_tensor(in_detail["index"], sample_feature.astype(np.float32))
    float_interp.invoke()
    float_logits = float_interp.get_tensor(out_detail["index"])
    float_diff = float(np.max(np.abs(keras_logits - float_logits)))

    int8_interp = tf.lite.Interpreter(model_path=str(int8_tflite), experimental_delegates=[])
    int8_interp.allocate_tensors()
    in_detail = int8_interp.get_input_details()[0]
    out_detail = int8_interp.get_output_details()[0]
    input_scale, input_zero_point = in_detail["quantization"]
    q_input = np.round(sample_feature / input_scale + input_zero_point).astype(np.int32)
    q_input = np.clip(q_input, -128, 127).astype(np.int8)
    int8_interp.set_tensor(in_detail["index"], q_input)
    int8_interp.invoke()
    q_logits = int8_interp.get_tensor(out_detail["index"])
    output_scale, output_zero_point = out_detail["quantization"]
    int8_logits = (q_logits.astype(np.int32) - output_zero_point) * output_scale
    int8_diff = float(np.max(np.abs(keras_logits - int8_logits)))

    metadata = {
        "stage_name": args.stage_name,
        "checkpoint": str(args.checkpoint),
        "label_names": label_names,
        "frontend_config": frontend,
        "model_cfg": model_cfg,
        "input_shape": input_shape,
        "float32_tflite": str(float32_tflite),
        "int8_tflite": str(int8_tflite),
        "keras_vs_float32_tflite_max_abs_diff": float_diff,
        "keras_vs_int8_tflite_max_abs_diff": int8_diff,
    }
    metadata_path = args.output_dir / "conversion_report.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))

    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
