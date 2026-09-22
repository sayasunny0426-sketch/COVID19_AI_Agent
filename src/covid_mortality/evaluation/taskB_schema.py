"""Patient-level prediction schema shared by Task A, Task B and Task C.

The final comparison in this study is a paired DeLong test over the same Test patients, so
every task has to emit predictions that can be merged on one key, one row per patient, with
identical id typing. This module fixes that contract before the Test set is opened, and
provides the validator that both the Task B Test script and the later fusion step run.

Key facts the contract depends on:
  * the join key is `subject_id`, a string such as "A001942" -- the same column name and the
    same values Task A wrote in results/taskA/evaluation/test_predictions_primary.csv;
  * patient order is the order of the fixed split manifest, filtered to the split, and is
    recorded in the run metadata so a later merge can be verified rather than assumed;
  * probabilities are floats in [0, 1]; a missing prediction is an error, never a blank.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

ID_COLUMN = "subject_id"
LABEL_COLUMN = "true_label"

# Task B's own file. Written only by the approved Test evaluation script.
TASKB_TEST_PREDICTIONS = "results/taskB/evaluation/test_predictions_taskB.csv"
TASKB_REQUIRED = [ID_COLUMN, LABEL_COLUMN,
                  "prob_clinical_lr", "prob_clinical_xgboost", "prob_clinical_mlp"]
TASKB_OPTIONAL = ["split", "prob_clinical_xgboost_native",
                  "pred_clinical_lr", "pred_clinical_xgboost", "pred_clinical_mlp",
                  "model_version", "run_id"]

# The combined file assembled after Task C. Source of truth for the final comparison.
COMBINED_PREDICTIONS = "results/comparison/test_predictions_all_models.csv"
COMBINED_REQUIRED = [ID_COLUMN, LABEL_COLUMN,
                     "prob_clinical_lr", "prob_clinical_xgboost", "prob_clinical_mlp",
                     "prob_cxr", "prob_late_fusion"]

# Pre-specified comparison family and correction for the final paired DeLong analysis.
# Fixed here, before any Test performance is known, so it cannot be chosen to suit a result.
DELONG_PLAN = {
    "test": "paired DeLong (DeLong 1988; Sun & Xu 2014 fast algorithm), midranks for ties",
    "implementation": "src/covid_mortality/evaluation/delong.py (numpy only, shared with Task A)",
    "pairing": ("the same Test patients for every model; predictions are merged on subject_id "
                "and the row order is verified before testing. Two-sample AUC comparisons are "
                "not used."),
    # Two families, each corrected separately. Corrected on 2026-09-23: an earlier record
    # (decision_log D-080) described a single family of four comparisons, which did not match
    # the agreed design. See D-083 and results/comparison/delong_plan.json.
    "families": {
        "A_late_fusion_vs_clinical_models": {
            "comparisons": ["late_fusion vs clinical_lr",
                            "late_fusion vs clinical_xgboost",
                            "late_fusion vs clinical_mlp"],
            "n_comparisons": 3,
            "correction": "Holm, within Family A only",
            "question": "does multimodal Late Fusion improve on each clinical tabular model"},
        "B_late_fusion_vs_reference_models": {
            "comparisons": ["late_fusion vs clinical_lr", "late_fusion vs cxr"],
            "n_comparisons": 2,
            "correction": "Holm, within Family B only",
            "question": "does multimodal Late Fusion improve on the two single-modality "
                        "reference models (the clinical reference and the imaging model)"},
    },
    "note_on_overlap": ("`late_fusion vs clinical_lr` belongs to both families, so it carries "
                        "two Holm-adjusted p-values, one per family. Report the family "
                        "alongside every adjusted p-value so the two are not confused; the "
                        "unadjusted p-value is the same in both."),
    "exploratory": ("comparisons among the single-modality models themselves are exploratory, "
                    "reported with unadjusted p-values and labelled as such"),
    "multiplicity_correction": "Holm, applied separately within each family",
    "outputs_per_comparison": ["family", "auc_a", "auc_b", "auc_difference",
                               "se_of_difference", "z", "p_value_unadjusted",
                               "p_value_holm_within_family", "ci95_low", "ci95_high"],
    "output_files": ["results/comparison/delong_results.json",
                     "results/comparison/delong_results.csv"],
    "plan_file": "results/comparison/delong_plan.json",
    "fixed_before_test": True,
    "record_correction": ("The family structure recorded in decision_log D-080 on 2026-09-22 "
                          "was wrong: it described one family of four comparisons. The agreed "
                          "design is the two families above. Corrected 2026-09-23 (D-083). "
                          "The correction changes which comparisons share a Holm correction; "
                          "it does not change any model, prediction or Test metric, and no "
                          "DeLong test has been run."),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def patient_order(split_manifest: pd.DataFrame, split: str) -> list[str]:
    """The fixed row order for a split: manifest order, filtered. Recorded, not assumed."""
    sub = split_manifest[split_manifest["split"] == split]
    return [str(x) for x in sub["Subject ID"]]


def validate(df: pd.DataFrame, required: list[str], expected_ids: list[str] | None = None,
             expected_labels: np.ndarray | None = None) -> dict:
    """Raise on any breach of the contract; return a summary for the QC report."""
    problems = []
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        problems.append(f"missing required columns: {missing_cols}")

    if ID_COLUMN in df.columns:
        ids = df[ID_COLUMN].astype(str)
        if ids.duplicated().any():
            dup = ids[ids.duplicated()].unique().tolist()
            problems.append(f"duplicate {ID_COLUMN}: {dup[:5]}")
        if ids.isna().any() or (ids == "").any():
            problems.append(f"blank {ID_COLUMN} values")
        if not ids.map(type).eq(str).all():
            problems.append(f"{ID_COLUMN} is not string-typed")
        if expected_ids is not None:
            if list(ids) != list(expected_ids):
                if set(ids) == set(expected_ids):
                    problems.append("subject order differs from the recorded patient order")
                else:
                    problems.append("subject set differs from the expected split membership")

    prob_cols = [c for c in df.columns if c.startswith("prob_")]
    for c in prob_cols:
        v = pd.to_numeric(df[c], errors="coerce")
        if v.isna().any():
            problems.append(f"{c}: {int(v.isna().sum())} missing or non-numeric predictions")
        elif not ((v >= 0) & (v <= 1)).all():
            problems.append(f"{c}: values outside [0, 1]")

    if LABEL_COLUMN in df.columns:
        lab = pd.to_numeric(df[LABEL_COLUMN], errors="coerce")
        if not lab.isin([0, 1]).all():
            problems.append(f"{LABEL_COLUMN} is not 0/1")
        if expected_labels is not None and not np.array_equal(lab.values, expected_labels):
            problems.append("true_label does not match the fixed split")

    if problems:
        raise ValueError("prediction file violates the shared schema:\n  - "
                         + "\n  - ".join(problems))
    return {"n_rows": len(df), "n_unique_ids": int(df[ID_COLUMN].nunique()),
            "probability_columns": prob_cols,
            "n_events": int(pd.to_numeric(df[LABEL_COLUMN]).sum()),
            "id_dtype": "str", "schema_ok": True}
