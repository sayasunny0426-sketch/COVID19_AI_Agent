# Task C Human-guided vs AI-Agent Comparison

*Exploratory methodological comparison of two independently designed multimodal fusion
pipelines built for the same research question, on the same fixed cohort and the same fixed
split.*

---

## 1. Purpose

Two pipelines produced a multimodal model for Task C:

- a **human-guided** pipeline, developed with the researcher and faculty feedback in the loop;
- an **autonomous AI-agent** pipeline, which chose its own components, fusion strategy, search
  space and selection rules.

This document compares **what each designed, how each behaved on Validation and on Test, how
each calibrated, what the paired DeLong comparisons showed, how each modality contributed, and
whether the two reached the same substantive conclusion.**

## 2. Scope and non-superiority statement

**This is not a test of superiority or inferiority.** No statistical comparison between the two
pipelines was performed, and none of the numbers here supports a causal or generalisable claim
that either approach is better. The two pipelines used **different CXR models and different
clinical variable sets**, so any performance difference confounds the pipeline with its
components.

Throughout, the two are described as **independently designed pipelines** that **converged on a
similar fusion structure**, **produced similar Test discrimination**, **differed in calibration
and component-level behaviour**, and **reached a similar substantive conclusion**.

The AI-agent pipeline was **not reverse-engineered towards the human-guided answers**: its
candidate strategies, component-selection rule, weight grid and selection metric were fixed in
`results/taskC/preflight/preflight_audit.json` before any fusion was computed, and the
human-guided fusion artefacts were read only while writing this document, after the AI-agent
Test evaluation was complete and frozen. That the agent arrived at w = 0.75 against the
human-guided 0.72 is an outcome of that independent search, not a target.

## 3. Cohort and split

| | Human-guided | AI-Agent |
|---|---|---|
| Patients | 1,277 (fixed) | 1,277 (fixed) |
| Split | Train 1,021 / Validation 128 / Test 128 | identical |
| Deaths | 135 / 17 / 17 | identical |
| Prediction time point | T0 = `visit_start_datetime` | identical |
| Test patients | the same 128 | the same 128 |

Subject sets and labels were checked and agree for all 128 Test patients.

## 4. Component models

### CXR component — **the largest difference between the two pipelines**

| | Human-guided | AI-Agent |
|---|---|---|
| Architecture | ResNet18 | ResNet18 |
| Configuration | human-guided Task A model | condition `lr3e-4_aug_b`, seed 42, best epoch 9 |
| Retrained for Task C | no | no |
| **Validation ROC-AUC** | **0.848967** | **0.871754** |
| **Test ROC-AUC** | **0.907260** | **0.834658** |

The two CXR models are **not the same model**. The AI-agent model scored higher on Validation
and markedly lower on Test (−0.0726). This single difference propagates into every comparison
that involves the image modality and is a property of the Task A models, not of the fusion step.

### Clinical component — **both pipelines chose Logistic Regression**

| | Human-guided | AI-Agent |
|---|---|---|
| Model | Clinical Logistic Regression | Clinical Logistic Regression |
| Selection basis | Validation ROC-AUC among 8 LR conditions | best **Training cross-validated** ROC-AUC among LR / XGBoost / MLP, fixed before fusion |
| Settings | L2, C=1.0, class_weight=None, solver=liblinear | L2, C=1.0, **class_weight=balanced**, solver=lbfgs |
| Final variables | 9 raw / 12 features: age, sex, **heart failure**, SpO2, lymphocyte, eGFR, CRP, D-dimer, lactate | 10 raw / 12 features: age, sex, SpO2, **SBP**, CRP, lymphocyte, D-dimer, lactate, eGFR, **troponin detectable** (+ `lymph_missing`) |
| Validation ROC-AUC | 0.896131 | 0.895072 |
| Test ROC-AUC | 0.941176 | 0.946476 |

Eight clinical variables are shared. The agent selected its component on 135 cross-validation
events rather than 17 Validation events, which also avoids spending the Validation set twice.

## 5. Fusion design

| | Human-guided | AI-Agent |
|---|---|---|
| Level | decision-level Late Fusion | decision-level Late Fusion |
| Space | probability-space weighted average | probability-space weighted average |
| Equation | `p = w·p_clinical + (1−w)·p_cxr` | identical |
| **Clinical weight** | **0.72** | **0.75** |
| **CXR weight** | **0.28** | **0.25** |
| Search space | w ∈ [0,1], step **0.01** (101 points) | w ∈ [0,1], step **0.05** (21 points) |
| Selection metric | Validation ROC-AUC maximum | Validation ROC-AUC maximum |
| Tie-break | PR-AUC, then Brier | PR-AUC, then Brier, then the w closest to 0.5 |
| Data used for weight | Validation only | Validation only |
| Strategy fixed before comparison | not recorded in the available artefacts | yes — probability-space averaging fixed before any search; logit-space fusion and logistic stacking held as non-competing secondary arms |
| Test locked during design | yes | yes |
| Early fusion | not used | considered and **explicitly rejected** (524 inputs against 135 Training events; no valid out-of-fold embeddings without retraining the frozen CXR model) |

The finer human-guided grid did not buy resolution the data can support: in the agent's search,
**6 of 21 weights sat within 0.005 ROC-AUC of the best**, so the weight is not identified to a
single point by 17 events in either pipeline.

## 6. Validation results

| Model | Pipeline | ROC-AUC | PR-AUC | Brier | ECE (10 eq-width) |
|---|---|---:|---:|---:|---:|
| CXR alone | Human-guided | 0.848967 | 0.560622 | 0.126565 | not recorded |
| CXR alone | AI-Agent | 0.871754 | 0.562283 | 0.096479 | 0.085898 |
| Clinical LR alone | Human-guided | 0.896131 | 0.600921 | 0.079211 | not recorded |
| Clinical LR alone | AI-Agent | 0.895072 | 0.634584 | 0.116927 | 0.168053 |
| **Late Fusion** | Human-guided | **0.912560** | 0.654359 | 0.075352 | not recorded |
| **Late Fusion** | AI-Agent | **0.920509** | 0.717351 | 0.089724 | 0.131177 |

**In both pipelines the fused model outscored both of its own components on Validation.** The
human-guided Validation weight-search artefact stores ROC-AUC, PR-AUC and Brier only, so its
Validation ECE is left empty rather than inferred.

## 7. Test performance

| Model | Pipeline | ROC-AUC | 95% CI (as reported) | PR-AUC | Brier | ECE (10 eq-width) |
|---|---|---:|---|---:|---:|---:|
| **Late Fusion** | Human-guided | **0.944356** | 0.884473–0.985162 | 0.797135 | 0.059210 | 0.064514 |
| **Late Fusion** | AI-Agent | **0.944886** | 0.892422–0.983042 | 0.780516 | 0.078034 | 0.128711 |
| Clinical LR | Human-guided | 0.941176 | 0.887122–0.981452 | 0.800251 | 0.057177 | 0.041302 |
| Clinical LR | AI-Agent | 0.946476 | 0.895601–0.984632 | 0.770898 | 0.097765 | 0.161595 |
| CXR ResNet18 | Human-guided | 0.907260 | 0.835718–0.962904 | 0.622926 | 0.101732 | 0.152336 |
| CXR ResNet18 | AI-Agent | 0.834658 | 0.725500–0.923200 | 0.457562 | 0.105808 | 0.097227 |

Test n = 128, deaths = 17 for every row.

**The two Late Fusion ROC-AUC point estimates differ by 0.00053** (0.944356 vs 0.944886) — a
difference far smaller than the width of either confidence interval.

### Confidence interval definitions

The human-guided pipeline used **10,000 bootstrap resamples, stratified, seed 42**; the AI-agent
pipeline used **2,000, stratified, seed 12345**. The confidence intervals were generated using
different bootstrap settings and therefore were not used for a standardized numerical comparison
of uncertainty between the two pipelines.

A harmonised recomputation from the saved patient-level Test predictions, using one setting
(2,000 resamples, stratified, seed 12345) for both, is in
`results/taskC/comparison/harmonized_ci.csv`. **It does not overwrite either pipeline's official
metrics.**

| Model | Pipeline | ROC-AUC | Harmonised 95% CI |
|---|---|---:|---|
| Late Fusion | Human-guided | 0.944356 | 0.881823–0.986751 |
| Late Fusion | AI-Agent | 0.944886 | 0.892422–0.983042 |
| Clinical LR | Human-guided | 0.941176 | 0.885533–0.981452 |
| Clinical LR | AI-Agent | 0.946476 | 0.895601–0.984632 |
| CXR ResNet18 | Human-guided | 0.907260 | 0.840964–0.964507 |
| CXR ResNet18 | AI-Agent | 0.834658 | 0.725477–0.923172 |

## 8. Calibration

**The ECE definitions are identical**, verified by reading both implementations:

- human-guided: `Notebook/TaskC_01_LateFusion_Validation構築.ipynb`,
  `expected_calibration_error(y_true, y_prob, n_bins=10)` with
  `bin_edges = np.linspace(0.0, 1.0, n_bins + 1)` — **10 equal-width bins over [0,1]**;
- AI-agent: `src/covid_mortality/evaluation/metrics.py`,
  `expected_calibration_error(y, p, n_bins=10, scheme="equal_width")` — the same.

Unlike the Task B comparison, **no harmonisation of ECE is needed** and the values below are
directly comparable.

| | Human-guided | AI-Agent |
|---|---:|---:|
| Late Fusion Brier | **0.059210** | 0.078034 |
| Late Fusion ECE | **0.064514** | 0.128711 |

**The human-guided Late Fusion is better calibrated on this Test set on both metrics.** The
difference must not be attributed to the pipeline as such: the agent's clinical component uses
`class_weight="balanced"`, which inflates predicted probabilities (its Clinical LR alone has
ECE 0.161595 against the human-guided 0.041302), and that miscalibration is carried into the
fused probability by the weighted average. This is a component-level explanation, not a
demonstrated cause.

## 9. Paired DeLong comparisons

Both pipelines used the same two-family structure with Holm correction applied separately
within each family.

### Family A — Late Fusion vs the clinical models (3 comparisons, Holm within A)

| Pipeline | Comparison | ΔAUC | unadjusted p | Holm p | Significant |
|---|---|---:|---:|---:|---|
| Human-guided | LF vs Clinical LR | +0.003180 | 0.811397 | 0.949950 | No |
| Human-guided | LF vs Clinical XGBoost | +0.008479 | 0.474975 | 0.949950 | No |
| Human-guided | LF vs Clinical MLP | +0.064653 | 0.020613 | 0.061839 | No |
| AI-Agent | LF vs Clinical LR | **−0.001590** | 0.769705 | 0.769705 | No |
| AI-Agent | LF vs Clinical XGBoost | +0.015898 | 0.227127 | 0.539785 | No |
| AI-Agent | LF vs Clinical MLP | +0.020668 | 0.179928 | 0.539785 | No |

### Family B — Late Fusion vs the reference models (2 comparisons, separate Holm within B)

| Pipeline | Comparison | ΔAUC | unadjusted p | Holm p | Significant |
|---|---|---:|---:|---:|---|
| Human-guided | LF vs Clinical LR | +0.003180 | 0.811397 | 0.811397 | No |
| Human-guided | LF vs CXR | +0.037096 | 0.045901 | **0.091803** | **No** |
| AI-Agent | LF vs Clinical LR | −0.001590 | 0.769705 | 0.769705 | No |
| AI-Agent | LF vs CXR | +0.110228 | 0.002079 | **0.004158** | **Yes** |

### Methodological clarification: two adjusted p-values for one comparison

`Late Fusion vs Clinical LR` belongs to **both** families, in both pipelines. It therefore has
**one unadjusted p-value and two Holm-adjusted p-values**, one per family:

| Pipeline | unadjusted p | Holm p within Family A | Holm p within Family B |
|---|---:|---:|---:|
| Human-guided | 0.811397 | 0.949950 | 0.811397 |
| AI-Agent | 0.769705 | 0.769705 | 0.769705 |

**This is not an inconsistency between sources.** It follows necessarily from the pre-specified
design of correcting within each family separately: Family A contains three comparisons and
Family B two, so the same raw p-value is multiplied by different step-down factors. Both
pipelines show the same pattern. The `family` column is required in every DeLong output
precisely so that the two adjusted values are never read as competing estimates of one quantity.

### On the one comparison where the pipelines differ in significance

The AI-agent's Late Fusion vs CXR comparison is significant after Holm correction and the
human-guided one is not.

**The larger Late Fusion–CXR difference in the AI-Agent pipeline largely reflects the lower CXR-only ROC-AUC in that pipeline and should not be interpreted as evidence of greater incremental value from adding CXR information.**

The CXR-only Test ROC-AUC is 0.834658 in the AI-agent pipeline against 0.907260 in the
human-guided one. A wider gap to a weaker image baseline is a statement about the baseline, not
about what CXR contributes.

**A lack of statistical significance is not evidence of equivalence.** With 17 events, every
confidence interval here admits differences that would matter clinically.

## 10. Modality contribution

Permutation importance on Test, reported as ROC-AUC decrease when one modality's probabilities
are permuted.

| Pipeline | Modality | Baseline AUROC | Mean ΔAUC | SD |
|---|---|---:|---:|---:|
| Human-guided | clinical | 0.944356 | **0.169060** | 0.034147 |
| Human-guided | CXR | 0.944356 | **0.044672** | 0.019843 |
| AI-Agent | clinical | 0.944886 | **0.350539** | 0.058264 |
| AI-Agent | CXR | 0.944886 | **0.003551** | 0.006233 |

**Both pipelines agree on the direction: the clinical modality contributes far more than CXR.**
They differ in degree — in the AI-agent model, permuting the CXR probabilities changes ROC-AUC
almost not at all (0.0036), consistent with its weaker Test CXR component.

**Weight is not importance.** A CXR weight of 0.25 does **not** mean CXR supplies 25% of the
performance. The weight is a coefficient in a linear combination of probabilities; the
contribution depends on how much the two components disagree and on how informative each is.
Both pipelines show CXR contributing less than its weight would naively suggest.

## 11. Secondary / exploratory strategies

The AI-agent pipeline ran two additional strategies as **secondary, non-competing** arms:

| Strategy | AI-Agent Validation ROC-AUC | AI-Agent Test ROC-AUC |
|---|---:|---:|
| **Primary: probability-space weighted average** | 0.920509 | **0.944886** |
| logit-space weighted average | 0.929518 | 0.932697 |
| logistic stacking (3 parameters on 17 Validation events) | 0.926338 | 0.935877 |

Both secondary strategies scored **higher than the primary on Validation and lower on Test**.
The primary strategy was fixed before the search and was not promoted on the basis of either
result. In short: **more flexible Validation-performing strategies did not generalize better to
the Test set.** This is one dataset with 17 Test events; it does not establish that simpler
combiners are generally preferable.

The human-guided pipeline did not report secondary fusion strategies in the artefacts available
here.

## 12. Similarities between the pipelines

All confirmed from artefacts:

1. **Both selected Clinical Logistic Regression** as the clinical component.
2. **Both adopted decision-level Late Fusion**, in probability space, with the same equation.
3. **Both weighted clinical information above CXR**, and by a similar amount — **0.72 vs 0.75**.
4. **Both selected the weight on Validation only**, by maximum Validation ROC-AUC with PR-AUC
   and Brier as tie-breaks.
5. **Both produced almost identical Late Fusion Test ROC-AUC** — 0.944356 vs 0.944886.
6. **In neither pipeline did Late Fusion significantly outperform Clinical LR.**
7. **Both found the clinical modality more influential than CXR** in permutation importance.
8. Both kept the Test set locked during design and both reused frozen component models.

## 13. Differences between the pipelines

1. **CXR component Test performance differs markedly**: 0.907260 vs 0.834658.
2. **Clinical LR differs** in variables (heart failure vs SBP / troponin / `lymph_missing`) and
   in class weighting, giving Test 0.941176 vs 0.946476.
3. **Calibration differs**: Late Fusion ECE 0.064514 vs 0.128711, Brier 0.059210 vs 0.078034.
4. **CXR permutation importance is an order of magnitude smaller** in the AI-agent model
   (0.0036 vs 0.0447).
5. **Late Fusion vs CXR is significant after Holm in the AI-agent pipeline and not in the
   human-guided one.** The larger Late Fusion–CXR difference in the AI-Agent pipeline largely
   reflects the lower CXR-only ROC-AUC in that pipeline and should not be interpreted as
   evidence of greater incremental value from adding CXR information.
6. **Component selection basis differs**: Validation (17 events) vs Training cross-validation
   (135 events).
7. **Pre-registration of the fusion strategy** is documented only in the AI-agent pipeline.

Because of these differences, **the two pipelines are not a replication of one another.** They
are two independent designs that happened to converge on a similar structure.

## 14. Interpretation

- Two independently designed pipelines **converged on the same fusion structure**: Clinical LR
  plus a frozen CXR ResNet18, combined at the decision level by a probability-space weighted
  average with clinical weighted above image (0.72 and 0.75).
- **Late Fusion Test discrimination was nearly identical** (0.944356 vs 0.944886), despite
  different CXR models, different clinical variables and different preprocessing.
- **Neither pipeline demonstrated a statistically significant improvement of Late Fusion over
  Clinical Logistic Regression** (human-guided ΔAUC +0.003180, Holm p 0.811–0.950; AI-agent
  ΔAUC −0.001590, Holm p 0.769705).
- **Calibration differed** in ways ROC-AUC did not reveal, and tracked the clinical component's
  own calibration rather than the fusion step.
- **Both pipelines found the clinical modality dominant**, and in the AI-agent model the
  incremental contribution of CXR on Test was very small.

Taken together: **Clinical LR is a strong model. Adding CXR information did not demonstrate a
statistically significant improvement over Clinical LR in Test ROC-AUC in either pipeline. CXR
may contain complementary information — both pipelines found fusion better than either component
on Validation — but its incremental predictive value beyond clinical information appears limited
in this cohort.**

## 15. Limitations

1. **No statistical comparison between the two pipelines was performed.** Every difference quoted
   is a difference in point estimates on one Test set of 128 patients with 17 deaths.
2. **Component models differ**, so no performance difference can be attributed to the pipeline
   design rather than to its parts. No factor was varied in isolation.
3. **The originally reported confidence intervals used different bootstrap settings**; the
   harmonised recomputation addresses only the resampling, not the underlying sample size.
4. **The weight is not identified to a single point** by 17 Validation events in either pipeline.
5. **Both pipelines share the cohort's limitations**, including emergency department encounters
   that were discharged (322 of 1,277, 2 deaths) and the absence of a consciousness-level
   variable — see `results/taskB/README.md`.
6. **This document reads Test results for both pipelines** and is therefore post hoc by
   construction. It was written only after both Test evaluations were complete and frozen.
7. **The human-guided Validation ECE and threshold provenance for the fusion model are not
   recorded** in the artefacts available here, so those cells are empty rather than inferred.

## 16. Source discrepancies

**Two** discrepancies were found, recorded in
`results/taskC/comparison/source_discrepancies.csv`. Neither was resolved by guessing.

1. **Clinical LR confidence interval differs between authoritative artefacts.** The Task C and
   five-model artefacts give 0.887122–0.981452; the Task B artefact gives 0.887652–0.983042.
   Point estimates, PR-AUC, Brier and ECE agree, so this is two bootstrap runs with different
   draws. The Task C artefact is used here and the difference is reported rather than averaged.
2. **Human-guided Validation ECE for the fusion model is not stored, so it cannot be checked.**
   The weight-search artefact holds ROC-AUC, PR-AUC and Brier only, with no ECE column for any
   weight. The corresponding cells in this document are left empty; no value is inferred.

**Not a discrepancy.** The overlapping comparison carrying two Holm-adjusted p-values is a
consequence of the pre-specified two-family design, not a conflict between sources. It is
explained in §9 under *Methodological clarification* and is deliberately **not** listed here.

## 17. Reproducibility

| Artefact | Path |
|---|---|
| This document | `docs/taskC_human_vs_ai_agent_comparison.md` |
| Design comparison | `results/taskC/comparison/human_vs_ai_agent_model_spec.csv` |
| Performance comparison | `results/taskC/comparison/human_vs_ai_agent_performance.csv` |
| DeLong comparison | `results/taskC/comparison/human_vs_ai_agent_delong.csv` |
| Modality importance | `results/taskC/comparison/human_vs_ai_agent_modality_importance.csv` |
| Harmonised CI | `results/taskC/comparison/harmonized_ci.csv` |
| Source conflicts | `results/taskC/comparison/source_discrepancies.csv` |
| Provenance and definitions | `results/taskC/comparison/comparison_meta.json` |
| Generator | `scripts/43_taskC_human_vs_agent_comparison.py` |
| AI-agent frozen specification | `results/taskC/modeling/final_selection_taskC.json` |

Every row of the CSVs carries a `source` field naming the artefact it came from. Human-guided
values were read from `気合のCOVID19/03_TaskC_マルチモーダル/` and
`04_全モデル比較・統計解析/`, opened read-only. **No official metric of either pipeline was
modified.**

### Paired pipeline comparison — possible, not performed

Both pipelines saved patient-level Late Fusion Test predictions
(`TaskC_LateFusion_Test患者別予測結果.csv` and
`results/comparison/test_predictions_all_models.csv`). These were checked: **identical subject
sets, 17 deaths in both, labels agreeing for all 128 patients, and the same row order.**

A paired DeLong comparison of human-guided Late Fusion against AI-agent Late Fusion is therefore
**technically possible but has not been run.** If run, it must be treated as a **separate
exploratory methodological analysis**, entirely outside this study's confirmatory DeLong
families, and must not be added to Family A or Family B.

## 18. Overall conclusion

The independently designed AI-agent pipeline converged on a multimodal fusion structure similar
to the human-guided pipeline, including selection of Clinical Logistic Regression,
decision-level Late Fusion, and greater weighting of clinical than CXR information.

Despite differences in component models and preprocessing, the two pipelines produced nearly
identical Late Fusion Test ROC-AUC point estimates (0.944356 and 0.944886).

Neither pipeline demonstrated a statistically significant improvement of Late Fusion over
Clinical Logistic Regression.

These findings suggest that clinical information provided most of the predictive signal in this
cohort, while the incremental value of CXR information was limited.

The pipelines differed in calibration and in component-level behaviour, and they are not a
replication of one another. This comparison is descriptive and exploratory; it establishes
neither superiority nor equivalence of either approach.
