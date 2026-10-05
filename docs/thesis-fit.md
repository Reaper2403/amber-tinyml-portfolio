# Relevance to second-life battery TinyML

[← Portfolio](../README.md)

The [NCPS topic listing](https://www.tuhh.de/ncps/theses), checked on 5 October 2026, describes embedded battery-state prediction, an open-data/open-platform baseline, and possible extensions involving cross-chemistry models, uncertainty, or cells with unknown history. The following are proposed connections, not completed battery research.

| Thesis work | Relevant experience in this portfolio | What must still be learned or established |
|---|---|---|
| Reproduce a compact ML baseline | Data preparation, training, checkpoint/export paths and reference-output checks | Battery datasets, state definitions, baseline literature and domain-specific preprocessing |
| Deploy on a constrained platform | Two int8 models, TFLM, operator registration, embedded buffers and task scheduling | Selected battery platform, its memory/latency budget and target measurement interface |
| Compare quality and resource use | Float/int8 conversion checks and historical MCU timing | A controlled battery-task evaluation of predictive error, quantization loss, peak RAM, latency and energy |
| Study generalization | Public-to-device adaptation and visible validation/test gaps | Splits by cell, usage history, temperature and chemistry, as appropriate to the agreed question |
| Assess scientific contribution | Simple baselines, temporal refitting and sensitivity controls | Battery-appropriate uncertainty/calibration methods and a tractable, supervisor-agreed research scope |

A plausible starting discussion is to reproduce a published compact SoC or SoH baseline on the agreed open dataset, verify the train-to-device conversion, and benchmark error and resource use on the target board. One research extension could then be selected on the basis of the supervisor's priorities and available data.

Existing experience is strongest in machine learning, data engineering and embedded inference. Battery modelling, electrochemistry and battery-specific estimation methods remain areas to build. This portfolio supports that technical starting point without claiming the domain work has already been done.
