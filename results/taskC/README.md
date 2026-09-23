# results/taskC — Multimodal Late Fusion

Task C combines the **frozen** Task A chest-radiograph model with the **frozen** Task B
clinical model. Neither component was retrained, refitted or re-thresholded for Task C, and
the cohort and split are the ones Task A and Task B already used.

## Frozen specification

| | |
|---|---|
| Primary strategy | weighted_average_probability |
| Formula | `p_fused = w * p_clinical + (1 - w) * p_cxr` |
| Clinical component | Task B LR, condition `C=1.0_cw=balanced`, selected by best Training cross-validated ROC-AUC among the Task B clinical models; fixed before any fusion was computed |
| CXR component | Task A `lr3e-4_aug_b`, seed 42, epoch 9 |
| Weights | clinical **0.75**, CXR **0.25** |
| Weight search | Validation ROC-AUC, on Validation only, grid step 0.05 |
| Threshold | 0.32592158196659415 — Validation Youden maximum, tie-break lowest; frozen for Test, recomputed on Test: False |
| Frozen file SHA256 | `2abebc28d4e4e87ef8c86465e337c27e8c42b4e1c6bc9ac1dcda7da840436bea` |

## Test result (primary)

ROC-AUC **0.944886** (95% CI 0.892422–0.983042),
PR-AUC 0.780516, Brier 0.078034,
ECE 0.128711 (10 equal_width bins),
n = 128, deaths = 17.

Validation ROC-AUC of the fused model was 0.920509.

## Directory layout

| Directory | What is in it |
|---|---|
| `preflight/` | The READ-ONLY audit of what Task A and Task B actually persisted. It is what constrained Task C to a one-parameter combiner: no Training-set predictions exist for either component, so nothing with more parameters could be fitted honestly. |
| `inputs/` | SHA256 of every Task A and Task B file consumed, and the patient-by-patient ID alignment proof. |
| `fusion/` | The Validation weight search, the weight-performance curve, and the secondary strategies. |
| `validation/` | Validation predictions, metrics, calibration, ablation and modality importance. |
| `modeling/` | `final_selection_taskC.json` — the frozen specification. |
| `qc/` | Pre-Test QC (run before the Test set was opened) and post-Test QC, plus the Test access log. |
| `evaluation/` | Test predictions, official metrics, curve points, calibration, modality importance, and the exploratory secondary arms. |
| `comparison/` | Human-guided vs AI-agent methodological comparison tables. |

## How to read these numbers

- **The secondary strategies did not compete.** Logit-space fusion and logistic stacking were
  fixed as non-competing secondary arms before the Validation search. They are reported in
  `fusion/` and `evaluation/secondary_exploratory_test.json` and were never eligible to become
  the primary model, whatever they scored.
- **A weight is not a contribution.** The CXR weight of 0.25 does not mean
  the radiograph supplies 25% of the performance. Use
  `evaluation/modality_importance_test.csv` for that question, and read it as post hoc and
  descriptive.
- **The weight curve is flat.** See `fusion/weight_search_primary.csv`: nearby weights perform
  similarly on Validation, so the selected weight should not be over-interpreted as an
  estimated quantity.
- **A lack of statistical significance is not evidence of equivalence.** With
  17 Test events, the confidence intervals here admit differences that would
  matter clinically. See `results/comparison/delong_results.csv`.

## Not included

Model checkpoints and preprocessed images are not published. Task C adds no new fitted weights
beyond the two fusion coefficients recorded in `modeling/final_selection_taskC.json`, so that
file plus the Task A and Task B artefacts is sufficient to reproduce every Task C number from
the saved component predictions.
