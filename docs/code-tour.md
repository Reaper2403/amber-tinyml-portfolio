# Selected code tour

[← Portfolio](../README.md)

These are selected implementation artifacts, with origin and transformations recorded in [the source manifest](../evidence/source_manifest.json). The original project trees and datasets are intentionally not required merely to review the work.

| Area | Start with | What to inspect |
|---|---|---|
| Motion network | [src/imu/model.py](../src/imu/model.py) | Depthwise/pointwise blocks, temporal pooling, configurable output head; instantiate `Stage1ImuNet(num_classes=6)` for this artifact's vocabulary |
| Motion training | [src/imu/training.py](../src/imu/training.py) | Class weighting, balanced sampling, validation macro F1 and best-checkpoint selection |
| Audio network | [src/audio/model.py](../src/audio/model.py) | DS-CNN definitions extracted from the training notebook generator; recorded configuration supplied as a module constant |
| IMU export | [src/export/build_stage1_keras_tflite.py](../src/export/build_stage1_keras_tflite.py) | Weight transfer, fail-fast reference parity, full-integer conversion and output dequantization |
| Audio export | [src/export/build_audio_context_tflite.py](../src/export/build_audio_context_tflite.py) | Corrected padding/normalization and dedicated audio conversion path |
| Scheduler | [src/firmware/main.cpp](../src/firmware/main.cpp) | Both inference engines, audio burst task and combined status payload |
| Sensor/runtime | [src/firmware/imu_inference_engine.cpp](../src/firmware/imu_inference_engine.cpp) | Sensor units, rolling buffer, quantization, operator resolver and latency instrumentation |
| Audio/runtime | [src/firmware/audio_context_engine.cpp](../src/firmware/audio_context_engine.cpp) | I²S capture, spectral frontend, int8 inference and majority aggregation |
| BLE transport | [src/firmware/amber_ble.c](../src/firmware/amber_ble.c) | Bounded queue, age/depth-triggered discovery and paced notification draining |
| Temporal evaluation | [src/research/temporal_validation.py](../src/research/temporal_validation.py) | Train-only imputation/scaling, refit block selection and bootstrap helpers |

The firmware files are source for inspection, **not a complete build target**. Board drivers, the patched Arduino component, ESP-IDF configuration and generated headers are not bundled. The exporters additionally require original checkpoints and representative datasets, which are outside this public snapshot. See [what can be reproduced](reproduce.md).
