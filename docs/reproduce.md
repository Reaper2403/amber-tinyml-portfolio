# Reproduce the portfolio checks

[← Portfolio](../README.md)

## 1. Verify the public snapshot (Python standard library)

From the repository root:

```bash
python3 tools/verify_portfolio.py
```

This checks source hashes, embedded model sizes/identifiers, JSON and Python syntax, local Markdown/image targets and absence of accidental local home paths or common credential patterns. It does not retrain a model or prove its accuracy.

## 2. Invoke both included models on a host

Optional check: use Python 3.11 and TensorFlow 2.19.1. Dependency setup did not complete during curation, so this invocation check is provided but **not reported as run**:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install tensorflow==2.19.1
python tools/smoke_tflite.py
```

The script checks tensor shapes, integer dtypes, class counts and finite outputs across deterministic synthetic inputs. It prints a JSON report. Synthetic inputs test artifact/runtime compatibility only; no class prediction should be interpreted as meaningful. Host execution speed is not an ESP32 benchmark.

## 3. Inspect and exercise temporal validation

```bash
python -m pip install numpy pandas
python tools/check_temporal_validation.py
```

This uses synthetic trajectories to check chronological split boundaries, disjoint training blocks, and train-only preprocessing. It does not rerun the original GLOBEM study.

## Source and artifact boundaries

- The included `.tflite` files are extracted directly from the selected firmware byte arrays. [Manifest](../models/manifest.json)
- `src/imu/model.py` is configurable; the deployed vocabulary requires six classes. `src/audio/model.py` is extracted from the training notebook builder with its recorded configuration.
- Export scripts need original checkpoints, transfer bundles and representative datasets. They are included for inspection, not as a promise that private training runs are reproducible from this snapshot.
- Firmware files require ESP-IDF 5.5.4, the project's patched Arduino-as-component setup, ES8311 board support, build configuration, and generated model headers. Those dependencies are not included as a complete buildable project.
- The selected firmware also includes an alternative audio-pretrain header in its source. That fallback weight file is not included; only the active device-fine-tuned audio model and active IMU model are published.
- No original training data, raw audio, phone database, participant-level research data, or private repository history is included.

To generate a review copy of a C/C++ model header, use the retained utility with the appropriate symbol:

```bash
python src/export/generate_model_header.py models/imu_office_int8.tflite imu_model.h --symbol stage1_classifier_keras_int8_tflite
python src/export/generate_model_header.py models/audio_device_int8.tflite audio_model.h --symbol audio_context_stage2_finetune_int8_tflite
```

These outputs alone do not complete the firmware dependency setup.
