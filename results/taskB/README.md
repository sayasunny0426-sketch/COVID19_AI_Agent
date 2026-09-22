# results/taskB — Clinical table data model

Task B predicts in-hospital mortality from the clinical table available at admission (T0 =
`visit_start_datetime`), using the same fixed cohort and the same fixed split as Task A, so
that the three tasks can later be compared on the same patients with a paired DeLong test.

**Test status.** Test predictions and Test performance metrics have not been generated. The
Test set had previously been accessed for limited QC purposes, but no Test information was
used for preprocessing fitting, variable selection, hyperparameter tuning, checkpoint
selection, or model selection. See `qc/test_access_log.json`.

---

## Cohort description

The analysis population is the **fixed cohort of 1,277 patients** (Train 1,021 / Validation
128 / Test 128; deaths 135 / 17 / 17), identical to Task A and Task C.

| Encounter type (`visit_concept_name`) | n | deaths | death rate |
|---|---:|---:|---:|
| Inpatient Visit | 955 | 167 | 17.5% |
| Emergency Room Visit | 322 | 2 | 0.6% |
| **Total** | **1,277** | **169** | **13.2%** |

**The cohort includes emergency department encounters that were discharged.** In the Training
set these are 262 of 1,021 patients, with 1 death (0.4%) and a median length of stay of 1 day
(212 of 262 stayed a single day). Inpatient-coded encounters in the Training set are 759
patients with 134 deaths (17.7%).

This matters for interpretation and is stated again under Limitations: for an encounter that
was evaluated in the emergency department and sent home, in-hospital death is close to
structurally impossible, so part of any model's discrimination on this cohort reflects the
admit-or-discharge decision rather than physiological severity.

Missingness follows the same structure. Laboratory values are missing far more often in
emergency department encounters than in inpatient ones (CRP 72.9% vs 4.9%; D-dimer 78.2% vs
13.8%; urea and creatinine 55.0% vs 0.4%), while vital signs are present for almost everyone
(0.2–2.8% missing). 143 Training patients have no laboratory panel at all, and 142 of those
143 are emergency department encounters.

---

## Analysis specification (frozen before the Test set is opened)

| Item | Value |
|---|---|
| Candidate variables | 13, selected from clinical knowledge, prior literature and the 4C Mortality Score before any outcome association was examined |
| Parameter budget | Training deaths 135 / EPV 10 = **13 coefficients** |
| Variable selection | **backward elimination by AIC on the Training set only**, subject to the budget. The unit is the original clinical variable; all dummies of a categorical variable move together. **Validation and Test were not used for variable selection.** |
| Final model inputs | 10 clinical variables / **12 coefficients** (events per parameter 11.25) |
| Missing indicator | `lymph` only, by criteria fixed before selection was run |
| Hyperparameters | repeated stratified 5-fold × 3 cross-validation **inside the Training set** |
| Model family comparison and threshold | Validation |
| Test | final evaluation only, once, after explicit approval |

Full specification: `modeling/final_selection_taskB.json`.

---

## Directory layout

| Path | Contents |
|---|---|
| `preprocessing/` | variable audit with a keep/drop reason for all 131 columns, candidate table with evidence, distribution summaries and figures, missingness mechanism, missing-indicator decision, frozen preprocessing spec |
| `variable_selection/` | backward-AIC history, bootstrap stability (500 resamples), final variables, coefficients, missing-indicator rule sensitivity (arms A/B/C) |
| `modeling/` | search tables per model family, Validation comparison, Validation predictions, calibration, frozen specification |
| `qc/` | pre-Test QC report, Test access log |
| `evaluation/` | **empty.** Will hold `test_predictions_taskB.csv` and the Test metrics after approval |
| `feature_importance/` | **empty.** Permutation importance is computed on Validation after the models are frozen |
| `secondary_exploratory_unused/` | population-definition tables for an inpatient-only analysis that was **not performed**. Not part of any result |

---

## Limitations

1. **The cohort mixes two structurally different encounter types.** Emergency department
   encounters that were discharged (322 of 1,277; 2 deaths) cannot realistically experience
   in-hospital death. Discrimination measured on this cohort therefore partly reflects the
   admit-or-discharge triage decision rather than illness severity alone. An inpatient-only
   analysis was considered and deliberately not performed, to keep one patient set across Task
   A, B and C (decision_log D-081); it is recorded as a future analysis candidate.
2. **Missingness is driven by the care pathway.** Laboratory missingness is largely determined
   by whether a panel was ordered, which in turn tracks the encounter type. Across the whole
   Training set, missingness of every laboratory variable is associated with *lower* mortality
   (odds ratios 0.04–0.43); within inpatient encounters the association reverses or disappears.
   Only the lymphocyte count survived the pre-specified criteria for a missing indicator.
3. **Sex missingness was strongly associated with mortality and may reflect a nonclinical
   missingness mechanism.** 19 Training patients have no recorded sex and 18 of them died
   (94.7%, vs 11.7% when sex is recorded). The cause has not been confirmed. No missing
   indicator was created for sex; the value is imputed with the Training mode. This caveat also
   applies to Task A and Task C if clinical variables are used there.
4. **Consciousness level is not available.** The 4C Mortality Score, qSOFA, CURB-65 and NEWS2
   all include an altered-mentation item, and none can be reproduced in full from this dataset.
5. **Age is available only in three bands** (`[18,59]`, `(59,74]`, `(74,90]`, with ages above 90
   folded into the top band). Continuous age does not exist in the source file, so the age
   cut-points of established scores cannot be reproduced.
6. **SpO2 carries no information on supplemental oxygen**, so the same value can mean different
   things clinically. No correction is possible with these data.
7. **Troponin is censored at the assay floor.** 78.2% of Training values sit at 0.01, so the
   variable is used as detectable / undetectable rather than as a continuous measurement.
8. **The Validation set contains 17 deaths and the Test set 17.** Confidence intervals are wide
   and the power to separate models is limited. Hyperparameters were therefore selected by
   cross-validation inside the Training set rather than on Validation, but the family
   comparison and the threshold still rest on 17 events.
9. **Variable selection is a fitted step.** Backward AIC was run on the same Training data used
   to fit the models, so the selected set carries selection optimism. Bootstrap stability is
   reported (500 resamples) to quantify how reproducible the selection is, but it does not
   remove the optimism.
10. **The unit of the events-per-variable guideline is unresolved** (original variables vs
    estimated parameters; open question Q-F1). The stricter reading, estimated parameters, was
    used.
