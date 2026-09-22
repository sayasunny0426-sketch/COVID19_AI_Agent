# results/comparison

Holds the cross-task comparison that is assembled **after Task C**.

| File | Status |
|---|---|
| `delong_plan.json` | **written.** The authoritative paired-DeLong plan: two comparison families, each Holm-corrected separately |
| `test_predictions_all_models.csv` | not yet created. Will merge Task A, Task B and Task C predictions on `subject_id` |
| `delong_results.json` / `.csv` | not yet created. **No DeLong test has been run.** |

## Comparison families (fixed before the Test evaluation; record corrected 2026-09-23)

**Family A — Late Fusion vs the clinical tabular models** (3 comparisons, Holm within Family A)
- Late Fusion vs Clinical Logistic Regression
- Late Fusion vs Clinical XGBoost
- Late Fusion vs Clinical MLP

**Family B — Late Fusion vs the single-modality reference models** (2 comparisons, Holm within Family B)
- Late Fusion vs Clinical Logistic Regression
- Late Fusion vs CXR ResNet18

`Late Fusion vs Clinical Logistic Regression` appears in both families and therefore carries
**two Holm-adjusted p-values**, one per family. The unadjusted p-value is the same in both.
Always report the family next to the adjusted p-value.

Comparisons among the single-modality models themselves are exploratory and reported with
unadjusted p-values, labelled as such.

## Record correction

`decision_log` D-080 (2026-09-22) recorded a single family of four comparisons. That was an
error in the record; the agreed design is the two families above. Corrected in D-083 and
change_log CL-012. The frozen Task B specification file still contains the earlier text and was
deliberately left unedited, because its SHA256 is the evidence that the Task B Test evaluation
used exactly the specification that was frozen. This file supersedes that key.
