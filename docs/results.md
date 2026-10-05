# Results and evidence

[← Portfolio](../README.md)

[Selected original device notes](../evidence/recorded-device-observations.md) provide the historical observations behind the timing and deployment statements.

The table below separates artifact facts, recorded observations and historical evaluation results. It does not combine them into a single deployment-accuracy claim.

| Claim | Evidence | Interpretation |
|---|---|---|
| IMU model: 100,736 bytes | [Model manifest](../models/manifest.json) and included binary | Exact extracted byte count; excludes runtime RAM |
| Audio model: 31,656 bytes | [Model manifest](../models/manifest.json) and included binary | Corrected local deployment artifact; earlier notes mention a smaller 30,320-byte export |
| IMU ≈460 → ≈30 ms | [Historical observations](../evidence/historical_observations.json) | Approximate standalone float32/int8 `Invoke()` observations; about 15× faster, not a controlled cross-platform benchmark |
| Audio ≈84 ms | [Historical observations](../evidence/historical_observations.json) | Standalone `Invoke()` only; excludes 2 s capture and frontend |
| IMU validation F1 0.6708 → 0.7457 | [Historical observations](../evidence/historical_observations.json) | Public pretraining versus office-fine-tuned checkpoints; not prospective multi-user evaluation |
| Audio public test F1 0.9356 | [Stage 1 metrics](../evidence/audio_stage1_metrics.json) | Public-data pretrained checkpoint evaluation |
| Audio device validation/test F1 0.8723/0.6697 | [Stage 2 metrics](../evidence/audio_stage2_metrics.json) | Device adaptation helps validation but leaves a substantial test gap |
| Audio public test F1 after adaptation 0.8638 | [Stage 2 metrics](../evidence/audio_stage2_metrics.json) | Regression relative to the public checkpoint; retained rather than hidden |

The audio study used AMI and MUSAN-derived public clips and a smaller watch-collected adaptation set. Project notes record 84,099 public canonical clips and 780 device clips (390 train, 195 validation, 195 test). Those counts describe clips, not independent people or recording sessions. The selected metrics alone cannot establish session-level independence or eliminate source/recording leakage; a new study should audit those splits explicitly.

For IMU training, the pipeline supported watchHAR, WISDM, Shoaib/UTwente and local office-seed inputs. The source model has a configurable class count; six active classes were used for this deployed artifact. The historical improvement is evidence of an adaptation workflow, not proof of generalization to arbitrary wearers.

## Conversion evidence

The included audio conversion report records maximum logit differences of approximately **5.72 × 10⁻⁶** for Keras versus float32 TFLite and **1.0624** for Keras versus int8 TFLite. These are sample-level logit comparisons, not accuracy changes. The checked report is preferred over older rounded numbers in development notes. [Conversion report](../evidence/audio_conversion_report.json)

The historical IMU exporter refuses to continue when its reference PyTorch/Keras logit difference exceeds the requested tolerance (default 1e-4), then checks both exported forms. This is an inspectable conversion gate, not a substitute for full test-set evaluation. [Exporter](../src/export/build_stage1_keras_tflite.py)

## What is checked in this release

The local verification report documents exactly which artifact, syntax, link and host-inference checks were run during curation. Synthetic host inputs establish that model tensors can be allocated and invoked; they do not establish accuracy or MCU timing. See [reproduction instructions](reproduce.md) and [verification report](../evidence/verification.json).

No fresh model training, original-data reevaluation, hardware flashing, combined-load benchmark, power measurement or battery-life test was performed for this portfolio.
