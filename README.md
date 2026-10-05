# Amber · TinyML from training to a wearable

**Ashutosh Chatterjee · M.Sc. Data Science, Hamburg University of Technology**

Two compact neural networks turn wrist motion and ambient audio into activity and context labels on an **ESP32-S3**. This portfolio brings together the model code, integer deployment artifacts, export pipeline, and selected firmware from the Amber prototype, with a separate example of temporal evaluation from the later research archive.

**Start here:** [Architecture](docs/architecture.md) · [Results and evidence](docs/results.md) · [Code tour](docs/code-tour.md) · [Research methods](docs/research-methods.md)

![Amber architecture: separate motion and audio inference paths on an ESP32-S3, followed by buffered Bluetooth summaries](assets/architecture.svg)

## What this work demonstrates

- **The full deployment loop:** sensor capture → training and device adaptation → checked PyTorch/Keras conversion → int8 TFLite → TensorFlow Lite Micro on ESP32-S3.
- **Resource-aware model design:** depthwise-separable CNNs, integer inputs and outputs, selected runtime operators, PSRAM-backed tensor arenas, and scheduled audio acquisition.
- **Debugging across ML and firmware:** a deployment failure traced to convolution padding and BatchNorm epsilon, followed by conversion fixes, frontend scaling checks, and runtime operator registration.
- **Evaluation beyond a headline score:** public/device split comparisons, validation/test gaps, simple baselines, temporal refitting, and explicit limits on what the evidence supports.

## Two models, one device

| | Wrist activity | Ambient audio |
|---|---|---|
| Input | 150 × 6 IMU window; nominal 50 Hz | 2 s mono audio at 16 kHz; 40 × 101 log-mel features |
| Network | Depthwise-separable 1D CNN | Depthwise-separable 2D CNN |
| Trained outputs | Rest, typing, phone interaction, drink, walk/transition, other hand activity | Meeting-like, music, environmental noise, speech |
| Included int8 artifact | **100,736 bytes** | **31,656 bytes** |
| Recorded standalone `Invoke()` time | **≈30 ms**, versus ≈460 ms float32 | **≈84 ms** |

Artifact sizes are verified from the included model bytes. Timing figures are historical prototype observations, not new benchmarks or end-to-end decision latency. [Evidence and qualifications →](docs/results.md)

The combined firmware continuously services the IMU and starts **five 2-second audio windows on a 60-second schedule**. It publishes the motion prediction and majority audio result, including confidence and audio age, through a buffered BLE queue. The two classifiers retain separate interpreters; the combined output is not a learned multimodal fusion model.

## Selected results

| Experiment | Split and metric | Result |
|---|---|---:|
| IMU public pretraining | Best validation macro F1 | 0.6708 |
| IMU office fine-tuning | Best validation macro F1 | **0.7457** |
| Audio public pretraining | Public test macro F1 | **0.9356** |
| Audio device fine-tuning | Device validation / test macro F1 | **0.8723 / 0.6697** |
| Audio device fine-tuning | Public test macro F1 after adaptation | 0.8638 |

The audio adaptation results expose a real generalization gap. The device test result and public regression are retained alongside the stronger validation result. These are historical training/evaluation metrics, not accuracy measured on the exact included int8 artifacts. [Machine-readable evidence →](evidence/)

## Inspect the work in five minutes

1. [System architecture](docs/architecture.md): sensing, concurrency, memory and BLE behavior.
2. [Conversion debugging case study](docs/export-debugging.md): why a trained model failed after conversion and what changed.
3. [Selected source](docs/code-tour.md): model architectures, training utilities, exporters and combined firmware.
4. [Research evaluation case study](docs/research-methods.md): examples selected from the latest `amber-thesis` archive, pinned to its inspected commit.
5. [Reproduce the portfolio checks](docs/reproduce.md): verify artifact hashes, run a host inference smoke check, and inspect temporal-split helpers using synthetic data.

## Further embedded work

[**KnockKey TinyML**](https://github.com/Reaper2403/KnockKey-TinyML) is a separate public project on the Arduino Nano 33 BLE Sense: temporal tap gestures, full-integer inference, a reproducible training/export workflow, and explicit comparison with counting and timing-rule baselines. Its reported noisy joined-stream benchmark is 88.3% for the int8 model versus 84.4% for filtered timing rules; it reports 204 µs median on-board inference across 40 measurements. The benchmark is a constructed labeled stream, not a prospective field trial. See that repository for the artifacts and protocol.

## Scope of this release

This is a **curated technical portfolio**, not a complete buildable firmware distribution or a full research-data release. The model binaries and selected source are real project artifacts. The diagrams, navigation and verification tools were prepared for this public snapshot on 5 October 2026. The firmware combines local working-tree files that are newer than the original audio-only upstream entry point. No fresh hardware deployment is claimed.

Original private repositories remain private. [Source provenance](evidence/source_manifest.json) records origin paths, transformations and SHA-256 hashes. [Reproduction scope](docs/reproduce.md) states what can run from this repository.
