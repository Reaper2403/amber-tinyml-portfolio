# Selected methods from the later research archive

[← Portfolio](../README.md)

The latest inspected `Reaper2403/amber-thesis` revision is **`14fb5b2bdf671a1b0b3926dca9d7f49ad33db480`** (repository last pushed 9 August 2026). It studies passive-sensing validity and temporal forecasting. It is separate from the ESP32 deployment work; its results do not validate a wearable or a battery estimator.

Only the methods most relevant to a new ML research project are selected here.

## 1. Compare with a baseline that can explain the result

The later archive tested whether a complex passive-sensing model added information beyond prior labels and personal history. Stronger controls changed the interpretation of several initially promising findings. The useful habit is to ask what a simple persistence, calendar or history-only baseline already explains before claiming a learned advantage.

For battery research, the analogous baselines and split units must be chosen with the supervisor and the battery literature. Randomly splitting nearby measurements from the same cell would answer a different question from testing transfer to an unseen cell or chemistry.

## 2. Refit across time

The selected source includes two temporal designs:

- **Expanding refit:** early training uses the first 65% of a trajectory; the later model refits through the first test block before predicting the final block.
- **Disjoint refit:** early training/test use 0–35%/35–50%; later training/test use 50–85%/85–100%.

`RidgeRegressor.fit()` estimates imputation and standardization parameters from the training features and applies those stored values during prediction. These helpers illustrate the mechanics of the split and preprocessing; they are not a complete leakage audit. [Selected unchanged definitions](../src/research/temporal_validation.py)

## 3. Check what a metric really measures

The recorded analysis found a strong relationship between a forecastability score and behavioral variability (r ≈ −0.912). Controlling variability and entropy reduced the original split-half correlation from approximately 0.412 to 0.230. The result therefore did not justify the earlier interpretation of an independent stable trait. [Selected aggregate results](../evidence/research_selected_results.json)

The later revision went further: controls based on **training-period** variability/entropy preserved cross-domain profile contrasts, but neither scalar measure exceeded the project's 0.30 stability gate in every refit regime. The interpretation was narrowed to a post-hoc profile of where a pooled history model transfers better than a pooled calendar comparator, requiring replication. This qualification supersedes stronger wording in early research notes.

## 4. Keep provenance and negative results visible

The inspected archive has a 40-row report ledger, explicitly marking four rows as documentation-only because their runners/outputs are absent. It also acknowledges that Git history cannot independently verify the claimed timing of early analysis gates. Neither the record count nor internal “referee” terminology is evidence of external peer review or publication.

For this portfolio, the contribution is the inspectable method and willingness to revise a claim. No participant-level data, identifiers, private review conversations, or clinical conclusions are republished.

Source documents inspected at the pinned commit: `thesis_workspace/appendix/experiment_ledger.md`, `thesis_workspace/appendix/reproducibility.md`, `thesis_workspace/docs/12_referee_response_and_robustness.md`, and `thesis_workspace/docs/15_second_referee_revision.md`. The extracted code and numerical subset are self-contained here; original private links are not required for the code tour.
