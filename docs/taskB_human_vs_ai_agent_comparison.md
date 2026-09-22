# Task B Human-guided vs AI-Agent Comparison

*Post hoc descriptive comparison of two clinical tabular pipelines built for the same research
question, on the same fixed cohort and the same fixed split.*

---

## Purpose

Two pipelines were built for Task B (in-hospital mortality from the admission clinical table):

- a **human-guided** pipeline, developed with the researcher and faculty feedback in the loop;
- an **autonomous AI-agent** pipeline, which designed its own variable set, preprocessing,
  search spaces and selection rules.

This document compares, transparently, **which variables each selected, which preprocessing
each adopted, which hyperparameters each chose, and how differently they performed** on
Validation and Test.

**This is not a contest.** No statistical test of superiority was run, and none of these numbers
was used to change any specification. In particular, **discrimination (ROC-AUC) alone is not
used to rank the pipelines** — calibration differs substantially between them, and a model with
a higher AUROC can be the worse-calibrated one.

The AI-agent pipeline was **not reverse-engineered towards the human-guided answers**: the
agent's candidate set, selection rule and search spaces were fixed before the human-guided final
model was consulted, and the human-guided Test results were only read while writing this
document, after the agent's Test evaluation had been completed and frozen.

---

## Cohort and evaluation conditions

| | Human-guided | AI-Agent |
|---|---|---|
| Cohort | 1,277 patients (fixed) | 1,277 patients (fixed) |
| Split | Train 1,021 / Validation 128 / Test 128 | identical |
| Deaths | 135 / 17 / 17 | identical |
| Test patients | the same 128 | the same 128 |
| Prediction time point | T0 = `visit_start_datetime` | identical |

The Test subject sets were checked and are **identical** (128 patients, 17 deaths, labels
agreeing for all 128).

**AUROC confidence intervals.** The human-guided pipeline used 10,000 bootstrap resamples with
seed 42; the AI-agent pipeline used 2,000 with seed 12345. The confidence intervals were
generated using different bootstrap settings and therefore were not used for a standardized
numerical comparison of uncertainty between the two pipelines. The point estimates are
comparable.

**ECE definitions differed and were harmonised.** The two pipelines did not compute ECE the same
way, so their reported values are not on one scale. Both the original values and a harmonised
recomputation are given below; see *Calibration: two ECE definitions*.

---

## Variable selection

Both pipelines audited all 131 clinical columns and both used **backward elimination by AIC on
the Training set only**, with the original clinical variable as the selection unit. They reached
different solutions.

| | Human-guided | AI-Agent |
|---|---|---|
| Candidate raw variables | 14 | 13 |
| Selection method | backward AIC | backward AIC, then a parameter-budget constraint (≤13 coefficients, from 135 Training deaths / EPV 10) |
| Stability assessment | not assessed | 500 bootstrap resamples of Training; retention rate reported per variable (reported, not used to select) |
| Final raw variables | 9 | 10 |
| Final input features | 12 | 12 (including `lymph_missing`) |

### Variable-by-variable

| Variable | Human candidate | Human final | Agent candidate | Agent final | Comment |
|---|:--:|:--:|:--:|:--:|---|
| age (3 bands) | ✓ | ✓ | ✓ | ✓ | both; 2 dummies |
| sex | ✓ | ✓ | ✓ | ✓ | both |
| SpO2 | ✓ | ✓ | ✓ | ✓ | both |
| lymphocyte count | ✓ | ✓ | ✓ | ✓ | both |
| eGFR | ✓ | ✓ | ✓ | ✓ | both |
| CRP | ✓ | ✓ | ✓ | ✓ | both |
| D-dimer | ✓ | ✓ | ✓ | ✓ | both — but see the source discrepancy below |
| lactate | ✓ | ✓ | ✓ | ✓ | both |
| **heart failure (hf)** | ✓ | **✓** | ✗ | ✗ | **human-guided only.** 3 dummies (HFpEF, HFrEF, Missing) |
| **SBP** | ✗ | ✗ | ✓ | **✓** | **AI-agent only.** haemodynamics; from CURB-65 / qSOFA / NEWS2 |
| **troponin (detectable)** | ✓ | ✗ | ✓ | **✓** | **AI-agent only in the final model.** human-guided dropped it at AIC step 3; the agent recoded it to detectable/undetectable and retained it |
| **lymph_missing** | ✗ | ✗ | ✓ | **✓** | **AI-agent only.** a missingness indicator, not a measurement |
| respiratory rate | ✓ | ✗ | ✓ | ✗ | candidate in both, in neither final model |
| BUN | ✓ | ✗ | ✓ | ✗ | dropped by AIC in both |
| CKD | ✓ | ✗ | ✗ | ✗ | human candidate; dropped at AIC step 5 |
| CAD | ✓ | ✗ | ✗ | ✗ | human candidate; dropped at AIC step 1 |
| comorbidity count (0/1/2+/NotAbstracted) | ✗ | ✗ | ✓ | ✗ | agent candidate; removed by the budget constraint (AIC cost +0.33) |

**Differences in one line.** Human-guided only: **heart failure**. AI-agent only: **SBP**,
**troponin detectable**, **lymph_missing**.

The two final sets share 8 of their clinical variables and differ in the ninth/tenth: the
human-guided model carries a chronic cardiac comorbidity, the agent model carries haemodynamics
plus a marker of myocardial injury plus a missingness indicator.

### ⚠ Discrepancy between human-guided source artefacts

The human-guided variable-selection artefacts and the fitted model **do not agree**:

- `01_変数選定/TaskB_BackwardAIC_最終選択変数.csv` and `03_前処理/TaskB_LR_最終12列_前処理仕様.json`
  record the final set as age, sex, hf, SpO2, **respiratory rate**, lymphocyte, eGFR, CRP,
  lactate (final AIC 524.98), with **D-dimer removed at step 2**;
- `04_Modeling/TaskB_LR_bestmodel_係数_OR.csv` — the coefficients of the model that was actually
  evaluated — contains **D-dimer** and **no respiratory rate**.

The evaluated model is the one in the coefficient file, and it matches the design the researcher
states. **This document uses the fitted set and reports the discrepancy rather than merging the
two.** It is recorded in `results/taskB/comparison/source_discrepancies.csv`.

---

## Preprocessing

| Item | Human-guided | AI-Agent | Key difference | Possible methodological implication |
|---|---|---|---|---|
| Abnormal values | not documented in the available artefacts | 7 physiologically impossible values set to missing (RR 67/88/95, HR 6/16, BMI 11.95/92.8); patients never dropped; no blanket outlier removal | explicit and recorded only in the agent pipeline | a step that is not documented cannot be reproduced or audited. This is a documentation difference; it does not establish that the analyses differ |
| Continuous imputation | Train median for all | per variable: Train mean when \|skew\| < 0.5, otherwise Train median | the agent picks the statistic per variable from the Training distribution | the median is robust for skewed variables, the mean more efficient for symmetric ones. Most of these variables are skewed, so the practical difference is small |
| Missing indicator | none for continuous variables | `lymph_missing` only, by criteria fixed before selection was run | the agent adds exactly one | its adjusted coefficient was near zero (OR 1.01, p = 0.96), so it largely occupies a coefficient. Arms A/B/C quantify the effect |
| Transformation | log1p on lymphocyte, CRP, lactate | log1p where Training skew ≥ +1.0 (RR, BUN, CRP, lymphocyte, D-dimer, lactate) | the agent states a rule and restricts it to right skew | a log transform worsens left skew; SpO2 (skew −2.10) is deliberately left untransformed in the agent pipeline |
| Scaling | StandardScaler fitted on Train | StandardScaler on Train for LR and MLP; trees receive raw values | the agent does not scale the tree inputs | trees are invariant to monotone transforms, so the tree fit is unaffected |
| Categorical encoding | one-hot; reference age `[18,59]` / sex FEMALE / hf No | one-hot with drop_first; reference age `[18,59]` / sex FEMALE / troponin undetectable | same scheme, different variables encoded | both yield odds ratios against a low-risk reference |
| Sex missing | Train mode (MALE); no Missing category | Train mode (MALE); no missing indicator, by explicit decision | same handling; the agent records the reason | 19 Training patients lack sex and 18 of them died. An indicator would encode that association, which may reflect a nonclinical missingness mechanism |
| Heart failure missing | explicit `hf_Missing` level, used as a predictor | heart failure not in the final model | only the human-guided model carries a missingness level as a predictor | `hf_Missing` acts as a missing indicator for one variable while the others have none — an asymmetry worth noting |
| Troponin | candidate, removed by backward AIC; treated as continuous | recoded to detectable / undetectable and retained | different representation of the same measurement | 78.2% of Training values sit at the assay floor (0.01), so a continuous scale assumes precision the assay does not provide |
| Preprocessing fit dataset | Train only | Train only | none | neither pipeline lets Validation or Test influence preprocessing parameters |

---

## Calibration: two ECE definitions

The two pipelines computed the expected calibration error with **different binning**, verified
by reading the code rather than inferring it from the numbers.

| | Human-guided | AI-Agent |
|---|---|---|
| Bins | **10** | **5** |
| Scheme | **equal-width** over [0, 1] | **equal-count** (quantiles of the predicted probability) |
| Code | `Notebook/TaskB_03_ClinicalModeling.ipynb`, cell 10:<br>`def expected_calibration_error(y_true, y_prob, n_bins=10)`<br>`bins = np.linspace(0.0, 1.0, n_bins + 1)` | `scripts/27_taskB_select_and_freeze.py`, `scripts/30_taskB_test_evaluation.py`, `calibration_table()`:<br>`edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))`, `n_bins = 5` |
| Aggregation | Σ (bin share) × \|mean predicted − observed rate\| | identical |

Only the binning differs; the aggregation is the same. An ECE is therefore meaningless without
its bin count and scheme, and the originally reported values are **not directly comparable**.

### Harmonised ECE (10 equal-width bins, matching the current manuscript)

Recomputed for all six models from the saved patient-level Test predictions, with **one
function and one bin definition** (`scripts/34_taskB_harmonized_ece.py`). As a validity check,
each pipeline's own definition applied to its own saved predictions **reproduced its reported
value for all six models**, confirming that the definitions above were read correctly.

| Model | Pipeline | Original ECE (own definition) | Own definition | **Harmonised ECE** (10 equal-width) |
|---|---|---:|---|---:|
| LR | Human-guided | 0.041302 | 10 equal-width | **0.041302** |
| LR | AI-Agent | 0.161595 | 5 equal-count | **0.161595** |
| XGBoost | Human-guided | 0.090264 | 10 equal-width | **0.090264** |
| XGBoost | AI-Agent | **0.022064** | 5 equal-count | **0.074899** |
| MLP | Human-guided | 0.308812 | 10 equal-width | **0.308812** |
| MLP | AI-Agent | 0.296720 | 5 equal-count | **0.300495** |

The human-guided values are unchanged because that pipeline already used 10 equal-width bins.

**This changes one reading materially.** Under its own 5-quantile definition the AI-agent
XGBoost appeared far better calibrated than the human-guided one (0.0221 vs 0.0903, a factor of
about four). Under the common definition the gap narrows to **0.0749 vs 0.0903** — still lower,
but by a small margin rather than a large one. The original figure was partly an artefact of
using coarser, data-adaptive bins.

Two smaller points:

- The AI-agent **LR** value is identical under both definitions (0.161595). That is not a
  coincidence of rounding: this model over-predicts in *every* bin, and when the sign of
  (predicted − observed) never changes, the weighted sum telescopes to
  `mean(predicted) − prevalence` = 0.2944 − 0.1328, independently of how the bins are drawn.
  It is the only one of the six models with that property.
- The **MLP** values move only slightly (AI-agent 0.2967 → 0.3005), and both MLPs remain poorly
  calibrated at roughly 0.30 under the common definition.

**Scope.** This recomputation is a post hoc descriptive comparison. It does **not** modify the
official Test metrics of either pipeline — those remain as each pipeline reported them — and it
was not used for model selection, retraining, or any threshold change. Outputs:
`results/taskB/comparison/harmonized_ece.csv` and `harmonized_ece_bins.csv` (per-bin detail).

---

## Logistic Regression

| | Human-guided | AI-Agent |
|---|---|---|
| Penalty | L2 | L2 |
| C | 1.0 | 1.0 |
| class_weight | **None** | **balanced** |
| Solver | liblinear | lbfgs |
| Selected by | Validation ROC-AUC, 8 conditions | Training-internal repeated 5-fold × 3 CV, 8 conditions |
| Threshold (Validation Youden) | 0.128797 | 0.496979 |

| Split | Metric | Human-guided | AI-Agent |
|---|---|---:|---:|
| Validation | ROC-AUC | 0.896131 | 0.895072 |
| Validation | PR-AUC | 0.600921 | 0.634584 |
| Validation | Brier | 0.079211 | 0.116927 |
| Validation | ECE | 0.075881 | 0.163569 |
| **Test** | **ROC-AUC** | **0.941176** | **0.946476** |
| Test | PR-AUC | 0.800251 | 0.770898 |
| Test | Brier | 0.057177 | 0.097765 |
| Test | ECE (own definition) | 0.041302 | 0.161595 |
| **Test** | **ECE (harmonised, 10 equal-width)** | **0.041302** | **0.161595** |

**Reading.** Discrimination is close on both splits. The AI-agent point estimate for Test ROC-AUC
is marginally higher (0.9465 vs 0.9412, a difference of 0.0053); the human-guided PR-AUC point
estimate is higher. The calibration-related metrics — Brier and ECE — are **better for the
human-guided model** on both Validation and Test, and this holds under the harmonised ECE
definition as well. **No statistical test was performed on this difference, so it does not
support a claim that either model is superior.** The most likely mechanical contributor to the
calibration gap is `class_weight="balanced"`, which inflates predicted probabilities: the
AI-agent LR predicts a mean risk of 0.294 against an observed prevalence of 0.133, and
over-predicts in every calibration bin. This is a plausible explanation, not a demonstrated
cause.

---

## XGBoost

| | Human-guided | AI-Agent |
|---|---|---|
| max_depth | 3 | 3 |
| learning_rate | **0.10** | **0.03** |
| min_child_weight | **1** | **10** |
| subsample / colsample_bytree | 0.8 / 0.8 | 0.8 / 0.8 |
| scale_pos_weight | 1.0 | 1.0 |
| reg_alpha / reg_lambda | 0 / 1 | — / 1.0 |
| Trees | 25 (best iteration 24, early stopping on Validation) | 300 fixed, no early stopping |
| Selected by | Validation ROC-AUC | Training-internal CV |
| Threshold | 0.131138 | 0.129078 |

| Split | Metric | Human-guided | AI-Agent |
|---|---|---:|---:|
| Validation | ROC-AUC | 0.885533 | 0.880763 |
| Validation | PR-AUC | 0.539380 | 0.549404 |
| Validation | Brier | 0.085551 | 0.085876 |
| Validation | ECE | 0.063749 | 0.039872 |
| **Test** | **ROC-AUC** | **0.935877** | **0.928988** |
| Test | PR-AUC | 0.718188 | 0.657762 |
| Test | Brier | 0.067800 | 0.073210 |
| Test | ECE (own definition) | 0.090264 | 0.022064 |
| **Test** | **ECE (harmonised, 10 equal-width)** | **0.090264** | **0.074899** |

**Reading.** The AI-agent chose a **lower learning rate and a much higher min_child_weight**,
i.e. a more constrained tree — a reasonable response to 135 events, though the search space
itself differed. On this Test set the human-guided discrimination point estimates are somewhat
higher (ROC-AUC +0.0069, PR-AUC +0.0604), while the **AI-agent model showed a lower ECE in this
Test set**.

The size of that ECE difference depends on the definition. Under the values each pipeline
originally reported it looks like 0.0221 vs 0.0903 — but those came from different binnings.
Under the common 10-equal-width definition it is **0.0749 vs 0.0903**, a much smaller margin.
This is a single-dataset observation under one binning choice; it does not establish that
either configuration calibrates better in general.

---

## MLP

| | Human-guided | AI-Agent |
|---|---|---|
| Architecture | 12 → 64 → 32 → 1 | 12 → 32 → 16 → 1 |
| Dropout | 0.2 | 0.3 |
| Optimiser | AdamW | AdamW |
| Learning rate | 1e-4 | 3e-4 |
| Weight decay | 1e-4 | 1e-3 |
| Batch size | 64 | 32 |
| pos_weight | 1.0 | 6.563 |
| Best epoch | 24 | 15 |
| Selected by | Validation ROC-AUC | Training-internal CV |
| Threshold | 0.494785 | 0.554057 |

| Split | Metric | Human-guided | AI-Agent |
|---|---|---:|---:|
| Validation | ROC-AUC | 0.918389 | 0.864865 |
| Validation | PR-AUC | 0.717544 | 0.647555 |
| Validation | Brier | 0.167252 | 0.168613 |
| Validation | ECE | 0.304689 | 0.295272 |
| **Test** | **ROC-AUC** | **0.879703** | **0.924218** |
| Test | PR-AUC | 0.688972 | 0.747650 |
| Test | Brier | 0.168616 | 0.160305 |
| Test | ECE (own definition) | 0.308812 | 0.296720 |
| **Test** | **ECE (harmonised, 10 equal-width)** | **0.308812** | **0.300495** |

**Reading.** On the Test set the **AI-agent MLP point estimates were better on all four metrics**
(ROC-AUC +0.0445, PR-AUC +0.0587, Brier −0.0083, ECE −0.0121 as originally reported, or −0.0083
under the harmonised definition). This is stated as a fact about these numbers. Both MLPs remain
poorly calibrated in absolute terms, at roughly 0.30 under the common definition.

**The improvement cannot be attributed to the AI agent as such.** The two MLPs differ in *every*
respect that could matter: input variables, network width, dropout, learning rate, weight decay,
batch size, class weighting, and the data used to select the configuration. No component was
varied in isolation, so the contribution of any single difference is unidentifiable from this
comparison.

One pattern is worth recording because it is a property of the *procedure* rather than of the
result: the human-guided MLP scored **higher on Validation (0.9184) than on Test (0.8797)**,
while the AI-agent MLP scored **lower on Validation (0.8649) than on Test (0.9242)**. The
human-guided pipeline selected the MLP configuration on Validation (17 events); the AI-agent
pipeline selected on Training-internal cross-validation and left Validation as a held-out check.
Selecting on a 17-event set is expected to make that set's estimate optimistic. This is
consistent with the pattern but is not proof of it — with 17 events, a swing of this size can
also arise by chance.

---

## Test performance comparison

| Model | Pipeline | ROC-AUC | 95% CI | PR-AUC | Brier | ECE (as reported) | **ECE (harmonised)** |
|---|---|---:|---|---:|---:|---:|---:|
| LR | Human-guided | 0.941176 | 0.887652–0.983042 | 0.800251 | 0.057177 | 0.041302 | **0.041302** |
| LR | AI-Agent | 0.946476 | 0.895601–0.984632 | 0.770898 | 0.097765 | 0.161595 | **0.161595** |
| XGBoost | Human-guided | 0.935877 | 0.876524–0.980392 | 0.718188 | 0.067800 | 0.090264 | **0.090264** |
| XGBoost | AI-Agent | 0.928988 | 0.873847–0.970866 | 0.657762 | 0.073210 | 0.022064 | **0.074899** |
| MLP | Human-guided | 0.879703 | 0.767886–0.962374 | 0.688972 | 0.168616 | 0.308812 | **0.308812** |
| MLP | AI-Agent | 0.924218 | 0.859565–0.977213 | 0.747650 | 0.160305 | 0.296720 | **0.300495** |

Test n = 128, deaths = 17, for every row. "ECE (as reported)" uses each pipeline's own binning
(human-guided 10 equal-width, AI-agent 5 equal-count) and the two columns are therefore **not**
on one scale; "ECE (harmonised)" applies 10 equal-width bins to every model.

**On the confidence intervals:** the human-guided intervals come from 10,000 stratified
bootstrap resamples (seed 42) and the AI-agent intervals from 2,000 (seed 12345). The confidence
intervals were generated using different bootstrap settings and therefore were not used for a
standardized numerical comparison of uncertainty between the two pipelines. All six intervals
overlap substantially.

> ⚠ A second human-guided artefact, `04_全モデル比較・統計解析/全5モデル_最終Test評価比較.csv`,
> reports the same point estimates, PR-AUC, Brier and ECE but slightly different CI bounds
> (LR 0.887122–0.981452, XGBoost 0.875980–0.979332, MLP 0.767886–0.960784). Two bootstrap runs
> with different draws. The Task B artefact is used here; nothing is averaged or invented.

---

## Interpretation

- Starting from the **same cohort, the same split and the same selection method** (backward AIC
  on Training), the two pipelines **reached different solutions** in both variable selection and
  hyperparameter selection. Eight clinical variables are shared; heart failure is unique to the
  human-guided model, and SBP, troponin-detectable and `lymph_missing` are unique to the
  AI-agent model.
- **LR and XGBoost Test discrimination fell in broadly the same range** (ROC-AUC 0.929–0.946,
  with overlapping confidence intervals).
- **For the MLP, the AI-agent Test point estimates were better on all four metrics.** Because
  inputs, architecture, class weighting and optimisation all differ, this cannot be attributed
  to the agent itself.
- **Calibration-related metrics varied by model and did not track discrimination.** Under the
  harmonised definition the human-guided LR had the best ECE (0.041) with the second-best AUROC;
  the AI-agent XGBoost had the second-best ECE (0.075) with the fourth-best AUROC; both MLPs sat
  near 0.30 despite AUROC above 0.87. **ROC-AUC alone was not sufficient to characterise these
  models.**
- **The two pipelines measured calibration differently**, which by itself changed one apparent
  conclusion: the AI-agent XGBoost's ECE advantage over the human-guided XGBoost shrank from
  about fourfold to a small margin once both were computed with the same bins. Reporting an ECE
  without its bin count and scheme is not sufficient for a cross-pipeline comparison.
- The AI-agent pipeline was **not reverse-engineered towards the human-guided final answers**.
- **No retuning was performed after seeing any Test result**, in either direction.
- This is an **exploratory methodological comparison**. It demonstrates neither superiority nor
  equivalence of either pipeline.

---

## Limitations

1. **No statistical test was performed.** Every difference quoted here is a difference in point
   estimates on one Test set of 128 patients with 17 deaths. Differences of the size seen here
   are well within what 17 events can produce by chance.
2. **Many factors differ simultaneously** between the two pipelines — variables, preprocessing,
   search spaces, the data used for selection, and the software versions. No factor was varied
   in isolation, so no difference in performance can be attributed to any single design choice.
3. **The confidence intervals were generated using different bootstrap settings** (10,000
   resamples / seed 42 vs 2,000 / seed 12345) and therefore were not used for a standardized
   numerical comparison of uncertainty between the two pipelines.
4. **The harmonised ECE is itself one binning choice.** Ten equal-width bins over [0, 1] was
   adopted to match the current manuscript. With 128 patients, the upper bins hold very few
   people (2–10 for most models), so individual bin estimates are unstable and the harmonised
   ECE carries real uncertainty that is not quantified here.
4. **The human-guided abnormal-value handling is not documented** in the artefacts available
   here, so that row of the preprocessing comparison is incomplete rather than empty.
5. **The human-guided variable-selection artefacts disagree with the fitted model** (respiratory
   rate vs D-dimer). The comparison uses the fitted model and reports the conflict.
6. **Both pipelines share the same cohort limitations**, including the presence of emergency
   department encounters that were discharged (322 of 1,277, 2 deaths) and the absence of a
   consciousness-level variable. See `results/taskB/README.md`.
7. **This comparison reads Test results for both pipelines.** It is therefore post hoc by
   construction and is reported as such; it was written only after both pipelines' Test
   evaluations were complete and frozen.

---

## Reproducibility

| Artefact | Path |
|---|---|
| This document | `docs/taskB_human_vs_ai_agent_comparison.md` |
| Design comparison (machine-readable) | `results/taskB/comparison/human_vs_ai_agent_model_spec.csv` |
| Performance comparison (machine-readable) | `results/taskB/comparison/human_vs_ai_agent_performance.csv` |
| Source conflicts found | `results/taskB/comparison/source_discrepancies.csv` |
| Harmonised ECE (all six models) | `results/taskB/comparison/harmonized_ece.csv` |
| Harmonised ECE, per bin | `results/taskB/comparison/harmonized_ece_bins.csv` |
| Harmonised ECE, definitions and evidence | `results/taskB/comparison/harmonized_ece_meta.json` |
| Generators | `scripts/33_taskB_human_vs_agent_comparison.py`, `scripts/34_taskB_harmonized_ece.py` |
| Shared ECE implementation | `src/covid_mortality/evaluation/metrics.py` (`calibration_bins`, `expected_calibration_error`, with explicit `n_bins` and `scheme`) |
| AI-agent frozen specification | `results/taskB/modeling/final_selection_taskB.json` |
| AI-agent Test metrics | `results/taskB/evaluation/test_metrics_table.csv` |
| AI-agent patient-level Test predictions | `results/taskB/evaluation/test_predictions_taskB.csv` |

Every value in the CSVs carries a `source` field naming the artefact it came from. Human-guided
values are read from the research artefacts under
`気合のCOVID19/02_TaskB_臨床データ/` and `04_全モデル比較・統計解析/`; those files are the
master copies and were opened read-only.

### Paired comparison — possible, not performed

The human-guided pipeline saved patient-level Test predictions for all three models
(`TaskB_{LR,XGBoost,MLP}_Test患者別予測.csv`, 128 rows each, columns `Subject ID`, `true_label`,
`prob`, `pred`). These were checked against the AI-agent file: **identical subject sets, 17
deaths in both, labels agreeing for all 128 patients** (row order differs, so a merge on
`Subject ID` is required).

A paired DeLong comparison of human-guided vs AI-agent is therefore **technically possible but
has not been run**. If it is run, it must be treated as a **separate exploratory methodological
analysis**, distinct from this study's confirmatory DeLong plan (Late Fusion vs single-modality
models, two families, Holm within each — see `results/comparison/delong_plan.json`), and it must
not be added to either confirmatory family.

---

## Conclusion

Two pipelines addressing the same question on the same patients, using the same selection
method, arrived at **different variable sets and different hyperparameters**, and produced Test
discrimination in a broadly similar range for LR and XGBoost, with better AI-agent point
estimates for the MLP. **Calibration differed in ways that ROC-AUC did not reveal**, which is
the most transferable observation here.

This is a descriptive, exploratory comparison. It does not establish that either pipeline is
better, and it was not used to change any specification in either pipeline.
