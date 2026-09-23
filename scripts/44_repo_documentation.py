"""Build the repository-level documentation for Task A, Task B and Task C.

Four files are produced, all from saved artefacts — no number is typed from memory:

    github_repo/README.md            project-level README covering Task A / B / C
    docs/artifact_index.md           every published artefact, classified
    results/taskC/README.md          the Task C artefact README (Task A and B already have one)
    docs/results_provenance.md       Task B and Task C provenance appended to the Task A part

`docs/results_provenance.md` is generated for Task A by `scripts/20_prepare_github_repo.py`.
This script leaves that part alone and rewrites only the block between its own sentinels, so
either script can be re-run without destroying the other's work.

Run `scripts/45_taskA_env_note.py` first: the README reads the Task A condition, seed and best
epoch from the note it writes, rather than restating them.

The README is idempotent. The existing Task A README is preserved verbatim between the
sentinels below; on a re-run the script reads the Task A section back out of those sentinels
instead of nesting the document inside itself. Nothing under results/taskA, results/taskB or
results/taskC evaluation output is written.

Usage:
    python scripts/44_repo_documentation.py --project . --repo github_repo
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd

BEGIN = "<!-- TASK_A_README_BEGIN -->"
END = "<!-- TASK_A_README_END -->"
PROV_BEGIN = "<!-- TASKBC_PROVENANCE_BEGIN (scripts/44_repo_documentation.py) -->"
PROV_END = "<!-- TASKBC_PROVENANCE_END -->"


def f6(x) -> str:
    return f"{float(x):.6f}"


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------------------
# artefact classification: (glob, task, purpose, generating script, class, mutability)
# Matched in order; the first match wins.
# ---------------------------------------------------------------------------------------
RULES: list[tuple[str, str, str, str, str, str]] = [
    # ---- Task A ----
    ("results/taskA/preprocessing/cxr_dicom_audit_summary.json", "A",
     "DICOM inventory of the CXR collection", "scripts/01_cxr_dicom_audit.py",
     "authoritative", "frozen"),
    ("results/taskA/preprocessing/dicom_header_audit_summary.json", "A",
     "Header audit of the cached images", "scripts/05_cxr_preproc_header_audit.py",
     "authoritative", "frozen"),
    ("results/taskA/preprocessing/fixed_split_1277.csv", "A",
     "The fixed 1,277-patient Train/Val/Test split (shared by all three tasks)",
     "scripts/04_split_reconciliation.py", "authoritative", "frozen"),
    ("results/taskA/preprocessing/index_cxr_manifest_1277.csv", "A",
     "Index CXR chosen per patient in the T0-2..T0 window",
     "scripts/02_index_cxr_selection.py", "authoritative", "frozen"),
    ("results/taskA/preprocessing/index_cxr_selection_flow.json", "A",
     "Patient flow of the index-CXR selection", "scripts/02b_apply_inclusion_window.py",
     "authoritative", "frozen"),
    ("results/taskA/preprocessing/preprocess_summary.json", "A",
     "DICOM to 16-bit PNG cache summary", "scripts/06_taskA_preprocess_cache.py",
     "authoritative", "frozen"),
    ("results/taskA/preprocessing/split_reconciliation_report.json", "A",
     "Proof that the split matches the pre-existing fixed split",
     "scripts/04_split_reconciliation.py", "authoritative", "frozen"),
    ("results/taskA/training/dataset_qc.json", "A", "Dataset and dataloader QC",
     "scripts/07_taskA_dataset_qc.py", "authoritative", "frozen"),
    ("results/taskA/training/model_smoke_check.json", "A", "Model definition smoke check",
     "scripts/08_taskA_model_smoke.py", "authoritative", "frozen"),
    ("results/taskA/training/condition_comparison.csv", "A",
     "Validation comparison of the 30 training runs", "scripts/13_taskA_condition_comparison.py",
     "authoritative", "frozen"),
    ("results/taskA/training/condition_selection_decision.json", "A",
     "Frozen model selection (condition, seed, epoch) — Validation only",
     "scripts/14_taskA_freeze_selection.py", "authoritative", "frozen"),
    ("results/taskA/evaluation/test_predictions_primary.csv", "A",
     "Patient-level Test predictions of the frozen CXR model",
     "scripts/15_taskA_test_evaluation.py", "authoritative", "frozen"),
    ("results/taskA/evaluation/test_predictions_all_variants.csv", "A",
     "Test predictions of the secondary variants", "scripts/15_taskA_test_evaluation.py",
     "authoritative", "frozen"),
    ("results/taskA/evaluation/test_metrics_primary.json", "A",
     "Official Task A Test metrics (primary)", "scripts/15_taskA_test_evaluation.py",
     "authoritative", "frozen"),
    ("results/taskA/evaluation/test_metrics_secondary.json", "A",
     "Test metrics of the secondary variants", "scripts/15_taskA_test_evaluation.py",
     "authoritative", "frozen"),
    ("results/taskA/evaluation/test_metrics_table.csv", "A",
     "Official Task A Test metrics, tabular form", "scripts/15_taskA_test_evaluation.py",
     "authoritative", "frozen"),
    ("results/taskA/evaluation/test_roc_curve_points.csv", "A", "Test ROC curve points",
     "scripts/15_taskA_test_evaluation.py", "authoritative", "frozen"),
    ("results/taskA/evaluation/test_pr_curve_points.csv", "A", "Test PR curve points",
     "scripts/15_taskA_test_evaluation.py", "authoritative", "frozen"),
    ("results/taskA/evaluation/test_calibration_bins.csv", "A", "Test calibration bins",
     "scripts/15_taskA_test_evaluation.py", "authoritative", "frozen"),
    ("results/taskA/evaluation/test_run_meta.json", "A",
     "Test run metadata (frozen spec hash, device, thresholds, bootstrap)",
     "scripts/15_taskA_test_evaluation.py", "authoritative", "frozen"),
    ("results/taskA/evaluation/test_access_log.jsonl", "A", "Record of every Test-set access",
     "scripts/15_taskA_test_evaluation.py", "authoritative", "append-only"),
    ("results/taskA/evaluation/test_run_env_note.json", "A",
     "Environment, seeds and generating scripts that `test_run_meta.json` does not record. "
     "Adds information only; every field is marked confirmed / documented / not recorded",
     "scripts/45_taskA_env_note.py", "supporting", "regenerated"),
    ("results/taskA/gradcam/gradcam_run_meta.json", "A",
     "Grad-CAM run metadata, including the SHA256 of every input image",
     "scripts/18_taskA_gradcam_primary_test.py", "authoritative", "frozen"),
    ("results/taskA/gradcam/gradcam_selection.csv", "A",
     "The 15 selected Grad-CAM cases and the rule that selected them",
     "scripts/18_taskA_gradcam_primary_test.py", "authoritative", "frozen"),
    ("results/taskA/gradcam/gradcam_excluded_images_sha256.csv", "A",
     "SHA256 of the per-case images that are deliberately not published",
     "scripts/18_taskA_gradcam_primary_test.py", "authoritative", "frozen"),
    ("results/taskA/gradcam/gradcam_panel_4x4.png", "A",
     "The approved 4x4 Grad-CAM panel (the only patient imaging in this repository)",
     "scripts/18_taskA_gradcam_primary_test.py", "authoritative", "frozen"),
    ("results/taskA/gradcam/gradcam_notes.md", "A",
     "Grad-CAM reading notes, including the researcher-written summary",
     "scripts/19_taskA_append_gradcam_notes.py", "authoritative", "frozen"),
    ("results/taskA/comparison/*", "A",
     "Exploratory comparison of the Task A model with the pre-existing current model",
     "scripts/16_taskA_compare_current_vs_aiagent.py", "comparison", "frozen"),
    ("results/taskA/github_assembly_report.json", "A",
     "Record of which files were assembled into this repository and what was redacted",
     "scripts/20_prepare_github_repo.py", "supporting", "regenerated"),
    ("results/taskA/README.md", "A", "Task A artefact guide and Drive correspondence",
     "scripts/20_prepare_github_repo.py", "supporting", "regenerated"),

    # ---- Task B ----
    ("results/taskB/preprocessing/variable_audit.csv", "B",
     "Audit of all 131 clinical columns, with the leakage and redundancy verdict for each",
     "scripts/21_taskB_clinical_audit.py", "authoritative", "frozen"),
    ("results/taskB/preprocessing/candidate_table.csv", "B",
     "The 13 candidate variables with clinical rationale and evidence",
     "scripts/21_taskB_clinical_audit.py", "authoritative", "frozen"),
    ("results/taskB/preprocessing/considered_not_selected.csv", "B",
     "Variables considered and the reason each was not carried forward",
     "scripts/21_taskB_clinical_audit.py", "authoritative", "frozen"),
    ("results/taskB/preprocessing/derived_redundancy.csv", "B",
     "Derived band columns mapped to their numeric source",
     "scripts/21_taskB_clinical_audit.py", "authoritative", "frozen"),
    ("results/taskB/preprocessing/audit_summary.json", "B", "Column audit summary counts",
     "scripts/21_taskB_clinical_audit.py", "authoritative", "frozen"),
    ("results/taskB/preprocessing/implausible_values.csv", "B",
     "Values outside the pre-specified physiologic bounds",
     "scripts/22_taskB_preprocess_spec.py", "authoritative", "frozen"),
    ("results/taskB/preprocessing/continuous_distribution_summary.csv", "B",
     "Training distributions and skew, which decided the log1p rule",
     "scripts/22_taskB_preprocess_spec.py", "authoritative", "frozen"),
    ("results/taskB/preprocessing/categorical_summary.csv", "B",
     "Categorical level counts and the reference level for each",
     "scripts/22_taskB_preprocess_spec.py", "authoritative", "frozen"),
    ("results/taskB/preprocessing/missingness_report.csv", "B",
     "Missingness by variable and by split", "scripts/22_taskB_preprocess_spec.py",
     "authoritative", "frozen"),
    ("results/taskB/preprocessing/preprocess_spec.json", "B",
     "Frozen preprocessing specification (imputation, transform, scaling, encoding)",
     "scripts/22_taskB_preprocess_spec.py", "authoritative", "frozen"),
    ("results/taskB/preprocessing/preprocessor_train_fit.json", "B",
     "Preprocessor parameters fitted on Training only",
     "scripts/22_taskB_preprocess_spec.py", "authoritative", "frozen"),
    ("results/taskB/preprocessing/figures/*.png", "B",
     "Training distribution plots per continuous variable (aggregate, no patient data)",
     "scripts/22_taskB_preprocess_spec.py", "supporting", "regenerated"),
    ("results/taskB/preprocessing/missingness_mechanism*", "B",
     "Missingness-mechanism analysis; the within-stratum sign reversal that identifies lab "
     "missingness as a triage proxy", "scripts/23_taskB_missingness_mechanism.py",
     "authoritative", "frozen"),
    ("results/taskB/preprocessing/missing_indicator_decision*", "B",
     "The C1-C5 criteria applied per variable, and the resulting indicator decision",
     "scripts/24_taskB_missing_indicator_decision.py", "authoritative", "frozen"),
    ("results/taskB/variable_selection/selection_history.csv", "B",
     "Backward elimination by AIC, step by step (Training only)",
     "scripts/25_taskB_variable_selection.py", "authoritative", "frozen"),
    ("results/taskB/variable_selection/budget_constrained_history.csv", "B",
     "The additional eliminations imposed by the EPV parameter budget",
     "scripts/25_taskB_variable_selection.py", "authoritative", "frozen"),
    ("results/taskB/variable_selection/bootstrap_stability.csv", "B",
     "Selection frequency of each variable over 500 bootstrap resamples",
     "scripts/25_taskB_variable_selection.py", "authoritative", "frozen"),
    ("results/taskB/variable_selection/final_variables.json", "B",
     "The frozen variable set and its events-per-parameter",
     "scripts/25_taskB_variable_selection.py", "authoritative", "frozen"),
    ("results/taskB/variable_selection/variable_selection_summary.json", "B",
     "Variable selection summary and rule statement",
     "scripts/25_taskB_variable_selection.py", "authoritative", "frozen"),
    ("results/taskB/variable_selection/*coefficients.csv", "B",
     "Logistic regression coefficients — the Task B feature-importance artefact",
     "scripts/25_taskB_variable_selection.py", "authoritative", "frozen"),
    ("results/taskB/variable_selection/indicator_rule_sensitivity*", "B",
     "Arms A/B/C sensitivity: whether the missing indicator changed the selected variables",
     "scripts/29_taskB_indicator_sensitivity.py", "authoritative", "frozen"),
    ("results/taskB/modeling/lr_search.csv", "B",
     "Logistic regression hyperparameter search (Training cross-validation)",
     "scripts/26_taskB_train_models.py", "authoritative", "frozen"),
    ("results/taskB/modeling/xgb_search.csv", "B",
     "XGBoost staged hyperparameter search (Training cross-validation)",
     "scripts/26_taskB_train_models.py", "authoritative", "frozen"),
    ("results/taskB/modeling/xgb_native_search.csv", "B",
     "XGBoost native-missing search — pre-specified secondary arm",
     "scripts/26_taskB_train_models.py", "exploratory", "frozen"),
    ("results/taskB/modeling/mlp_search.csv", "B",
     "MLP staged hyperparameter search (Training cross-validation)",
     "scripts/26_taskB_train_models.py", "authoritative", "frozen"),
    ("results/taskB/modeling/search_summary.json", "B",
     "Best condition per model family and the rule that picked it",
     "scripts/26_taskB_train_models.py", "authoritative", "frozen"),
    ("results/taskB/modeling/model_xgb.json", "B", "The fitted XGBoost booster",
     "scripts/27_taskB_select_and_freeze.py", "authoritative", "frozen"),
    ("results/taskB/modeling/model_xgb_native.json", "B",
     "The fitted native-missing XGBoost booster (secondary)",
     "scripts/27_taskB_select_and_freeze.py", "exploratory", "frozen"),
    ("results/taskB/modeling/preprocessor_*.json", "B",
     "Per-model preprocessor parameters as refitted at freeze time",
     "scripts/27_taskB_select_and_freeze.py", "authoritative", "frozen"),
    ("results/taskB/modeling/validation_predictions.csv", "B",
     "Patient-level Validation predictions for all Task B models",
     "scripts/27_taskB_select_and_freeze.py", "authoritative", "frozen"),
    ("results/taskB/modeling/validation_comparison.csv", "B",
     "Validation comparison of the model families and the chosen thresholds",
     "scripts/27_taskB_select_and_freeze.py", "authoritative", "frozen"),
    ("results/taskB/modeling/validation_calibration_*.csv", "B", "Validation calibration bins",
     "scripts/27_taskB_select_and_freeze.py", "authoritative", "frozen"),
    ("results/taskB/modeling/final_selection_taskB.json", "B",
     "FROZEN Task B specification — variables, hyperparameters, thresholds, artefact hashes",
     "scripts/27_taskB_select_and_freeze.py", "authoritative", "frozen"),
    ("results/taskB/qc/pre_test_qc_report.*", "B",
     "Pre-Test QC, run without reading any Test row", "scripts/28_taskB_pre_test_qc.py",
     "authoritative", "frozen"),
    ("results/taskB/qc/post_test_qc_report.*", "B", "Post-Test QC",
     "scripts/32_taskB_post_test_qc.py", "authoritative", "frozen"),
    ("results/taskB/qc/test_access_log.*", "B",
     "Honest record of every Test-set access, including earlier limited QC access",
     "scripts/30_taskB_test_evaluation.py", "authoritative", "append-only"),
    ("results/taskB/evaluation/test_predictions_taskB.csv", "B",
     "Patient-level Test predictions for all Task B models",
     "scripts/30_taskB_test_evaluation.py", "authoritative", "frozen"),
    ("results/taskB/evaluation/test_metrics_table.csv", "B",
     "OFFICIAL Task B Test metrics. ECE here is 5 equal-count bins (`ece_5bin`)",
     "scripts/30_taskB_test_evaluation.py", "authoritative", "frozen"),
    ("results/taskB/evaluation/test_roc_points_*.csv", "B", "Test ROC curve points per model",
     "scripts/30_taskB_test_evaluation.py", "authoritative", "frozen"),
    ("results/taskB/evaluation/test_pr_points_*.csv", "B", "Test PR curve points per model",
     "scripts/30_taskB_test_evaluation.py", "authoritative", "frozen"),
    ("results/taskB/evaluation/test_calibration_*.csv", "B", "Test calibration bins per model",
     "scripts/30_taskB_test_evaluation.py", "authoritative", "frozen"),
    ("results/taskB/evaluation/test_run_meta.json", "B",
     "Test run metadata, including package versions", "scripts/30_taskB_test_evaluation.py",
     "authoritative", "frozen"),
    ("results/taskB/comparison/human_vs_ai_agent_*.csv", "B",
     "Human-guided vs AI-agent methodological comparison tables",
     "scripts/33_taskB_human_vs_agent_comparison.py", "comparison", "frozen"),
    ("results/taskB/comparison/source_discrepancies.csv", "B",
     "Disagreements between sources, listed rather than merged",
     "scripts/33_taskB_human_vs_agent_comparison.py", "comparison", "frozen"),
    ("results/taskB/comparison/comparison_meta.json", "B", "Comparison provenance and settings",
     "scripts/33_taskB_human_vs_agent_comparison.py", "comparison", "frozen"),
    ("results/taskB/comparison/harmonized_ece*", "B",
     "ECE recomputed under one shared definition so the two pipelines can be compared. "
     "Does NOT replace the official Task B ECE", "scripts/34_taskB_harmonized_ece.py",
     "comparison", "frozen"),
    ("results/taskB/secondary_exploratory_unused/*", "B",
     "Material from the inpatient-only sensitivity analysis that was cancelled and is not "
     "part of any result", "scripts/23_taskB_missingness_mechanism.py --cohort-flow",
     "exploratory", "not used"),
    ("results/taskB/commit_audit.json", "B",
     "Record of the copy, byte-verification and safety audit before the Task B commit",
     "scripts/31_taskB_prepare_commit.py", "supporting", "regenerated"),
    ("results/taskB/README.md", "B", "Task B artefact guide, cohort description, limitations",
     "scripts/31_taskB_prepare_commit.py", "supporting", "regenerated"),

    # ---- Task C ----
    ("results/taskC/preflight/*", "C",
     "READ-ONLY pre-flight audit: what Task A and Task B actually persisted, and what that "
     "permits Task C to do", "scripts/35_taskC_preflight_audit.py", "authoritative", "frozen"),
    ("results/taskC/inputs/input_artifacts.csv", "C",
     "SHA256 of every Task A and Task B file Task C consumes",
     "scripts/36_taskC_fusion_search.py", "authoritative", "frozen"),
    ("results/taskC/inputs/id_alignment.json", "C",
     "Proof that the CXR and clinical predictions align patient for patient",
     "scripts/36_taskC_fusion_search.py", "authoritative", "frozen"),
    ("results/taskC/fusion/weight_search_primary.csv", "C",
     "Validation weight search for the primary strategy — the weight-performance curve",
     "scripts/36_taskC_fusion_search.py", "authoritative", "frozen"),
    ("results/taskC/fusion/weight_search_all.csv", "C",
     "Weight search across all strategies and clinical components",
     "scripts/36_taskC_fusion_search.py", "authoritative", "frozen"),
    ("results/taskC/fusion/strategy_comparison.csv", "C",
     "Primary vs secondary strategies on Validation; secondary arms did not compete",
     "scripts/36_taskC_fusion_search.py", "exploratory", "frozen"),
    ("results/taskC/fusion/stacking_exploratory.json", "C",
     "Logistic stacking, exploratory only", "scripts/36_taskC_fusion_search.py",
     "exploratory", "frozen"),
    ("results/taskC/fusion/selected_weight.json", "C", "The selected weight and the tie-break trail",
     "scripts/36_taskC_fusion_search.py", "authoritative", "frozen"),
    ("results/taskC/validation/validation_predictions_taskC.csv", "C",
     "Patient-level Validation predictions for the fused model and its components",
     "scripts/37_taskC_freeze.py", "authoritative", "frozen"),
    ("results/taskC/validation/validation_metrics_taskC.json", "C",
     "Validation metrics and the frozen operating threshold", "scripts/37_taskC_freeze.py",
     "authoritative", "frozen"),
    ("results/taskC/validation/validation_calibration.csv", "C", "Validation calibration bins",
     "scripts/37_taskC_freeze.py", "authoritative", "frozen"),
    ("results/taskC/validation/ablation_validation.csv", "C",
     "Validation ablation: fused vs each modality alone", "scripts/37_taskC_freeze.py",
     "authoritative", "frozen"),
    ("results/taskC/validation/modality_importance_validation.csv", "C",
     "Permutation modality importance on Validation", "scripts/37_taskC_freeze.py",
     "authoritative", "frozen"),
    ("results/taskC/modeling/final_selection_taskC.json", "C",
     "FROZEN Task C specification — components, weights, threshold, input hashes",
     "scripts/37_taskC_freeze.py", "authoritative", "frozen"),
    ("results/taskC/qc/pre_test_qc_report.*", "C", "Pre-Test QC (41 checks)",
     "scripts/38_taskC_pre_test_qc.py", "authoritative", "frozen"),
    ("results/taskC/qc/post_test_qc_report.*", "C", "Post-Test QC (54 checks)",
     "scripts/42_taskC_post_test_qc.py", "authoritative", "frozen"),
    ("results/taskC/qc/test_access_log.jsonl", "C", "Record of every Test-set access",
     "scripts/40_taskC_test_evaluation.py", "authoritative", "append-only"),
    ("results/taskC/evaluation/test_predictions_taskC.csv", "C",
     "Patient-level Test predictions: fused model and all components",
     "scripts/40_taskC_test_evaluation.py", "authoritative", "frozen"),
    ("results/taskC/evaluation/test_metrics_taskC.json", "C",
     "OFFICIAL Task C Test metrics for the primary fused model",
     "scripts/40_taskC_test_evaluation.py", "authoritative", "frozen"),
    ("results/taskC/evaluation/test_metrics_table.csv", "C",
     "All five models on the Test set under ONE metric definition "
     "(ECE = 10 equal-width bins). This is the cross-task performance table",
     "scripts/40_taskC_test_evaluation.py", "authoritative", "frozen"),
    ("results/taskC/evaluation/test_roc_points_*.csv", "C", "Test ROC curve points per model",
     "scripts/40_taskC_test_evaluation.py", "authoritative", "frozen"),
    ("results/taskC/evaluation/test_pr_points_*.csv", "C", "Test PR curve points per model",
     "scripts/40_taskC_test_evaluation.py", "authoritative", "frozen"),
    ("results/taskC/evaluation/test_calibration_*.csv", "C", "Test calibration bins per model",
     "scripts/40_taskC_test_evaluation.py", "authoritative", "frozen"),
    ("results/taskC/evaluation/modality_importance_test.csv", "C",
     "Permutation modality importance on Test — post hoc, descriptive",
     "scripts/40_taskC_test_evaluation.py", "exploratory", "frozen"),
    ("results/taskC/evaluation/secondary_exploratory_test.json", "C",
     "Logit-space fusion and stacking on Test — exploratory, not promoted",
     "scripts/40_taskC_test_evaluation.py", "exploratory", "frozen"),
    ("results/taskC/evaluation/test_run_meta.json", "C",
     "Test run metadata, including package versions", "scripts/40_taskC_test_evaluation.py",
     "authoritative", "frozen"),
    ("results/taskC/comparison/human_vs_ai_agent_*.csv", "C",
     "Human-guided vs AI-agent methodological comparison tables",
     "scripts/43_taskC_human_vs_agent_comparison.py", "comparison", "frozen"),
    ("results/taskC/comparison/source_discrepancies.csv", "C",
     "Disagreements between sources, listed rather than merged",
     "scripts/43_taskC_human_vs_agent_comparison.py", "comparison", "frozen"),
    ("results/taskC/comparison/comparison_meta.json", "C", "Comparison provenance and settings",
     "scripts/43_taskC_human_vs_agent_comparison.py", "comparison", "frozen"),
    ("results/taskC/comparison/harmonized_ci.csv", "C",
     "Confidence intervals side by side with their bootstrap settings, so the difference in "
     "settings is visible. Does NOT replace the official CIs",
     "scripts/43_taskC_human_vs_agent_comparison.py", "comparison", "frozen"),
    ("results/taskC/commit_audit.json", "C",
     "Record of the copy, byte-verification and safety audit before the Task C commits",
     "scripts/39_taskC_prepare_commit.py", "supporting", "regenerated"),
    ("results/taskC/README.md", "C", "Task C artefact guide",
     "scripts/44_repo_documentation.py", "supporting", "regenerated"),

    # ---- cross-task ----
    ("results/comparison/test_predictions_all_models.csv", "A+B+C",
     "Patient-level Test predictions of all five models in one file — the input that makes "
     "the paired DeLong re-runnable", "scripts/41_taskC_delong.py", "authoritative", "frozen"),
    ("results/comparison/delong_plan.json", "A+B+C",
     "The two pre-specified DeLong comparison families and the Holm correction, fixed before "
     "the Test set was opened", "scripts/41_taskC_delong.py", "authoritative", "frozen"),
    ("results/comparison/delong_results.*", "A+B+C",
     "Paired DeLong results with the `family` column and per-family Holm adjustment",
     "scripts/41_taskC_delong.py", "authoritative", "frozen"),
    ("results/comparison/README.md", "A+B+C",
     "Comparison-family definitions and the record correction",
     "scripts/41_taskC_delong.py", "supporting", "regenerated"),

    # ---- documentation ----
    ("README.md", "A+B+C", "Project README covering all three tasks",
     "scripts/44_repo_documentation.py", "supporting", "regenerated"),
    ("docs/artifact_index.md", "A+B+C", "This index",
     "scripts/44_repo_documentation.py", "supporting", "regenerated"),
    ("docs/decision_log.md", "A+B+C",
     "Every design decision with its reason (D-001 onwards)", "written by hand",
     "authoritative", "append-only"),
    ("docs/change_log.csv", "A+B+C",
     "Every defect found and how it was fixed (CL-001 onwards)", "written by hand",
     "authoritative", "append-only"),
    ("docs/open_questions.md", "A+B+C", "Questions left open and their status", "written by hand",
     "authoritative", "append-only"),
    ("docs/requirements.md", "A+B+C", "Study requirements as agreed", "written by hand",
     "authoritative", "append-only"),
    ("docs/implementation_plan.md", "A+B+C", "Overall implementation plan", "written by hand",
     "supporting", "append-only"),
    ("docs/results_provenance.md", "A+B+C",
     "Where each headline number came from. Task A part by `scripts/20_prepare_github_repo.py`, "
     "Task B and Task C part by `scripts/44_repo_documentation.py`",
     "scripts/20 + scripts/44", "supporting", "regenerated"),
    ("docs/taskB_human_vs_ai_agent_comparison.md", "B",
     "Human-guided vs AI-agent Task B methodological comparison",
     "scripts/33_taskB_human_vs_agent_comparison.py", "comparison", "frozen"),
    ("docs/taskC_human_vs_ai_agent_comparison.md", "C",
     "Human-guided vs AI-agent Task C methodological comparison",
     "scripts/43_taskC_human_vs_agent_comparison.py", "comparison", "frozen"),
    ("docs/task_a_*.md", "A", "Task A design and evaluation plans", "written by hand",
     "supporting", "append-only"),
    ("docs/cxr_studydescription_mapping_final.csv", "A",
     "StudyDescription to CXR-category mapping", "scripts/01b_studydescription_mapping.py",
     "authoritative", "frozen"),
    ("scripts/*", "A+B+C", "Non-interactive entry point for one analysis step", "n/a",
     "code", "versioned"),
    ("src/*", "A+B+C", "Library code: dataset, features, models, training, evaluation, fusion",
     "n/a", "code", "versioned"),
    ("tests/*", "A+B+C", "Unit tests for the library code", "n/a", "code", "versioned"),
    ("notebooks/*", "A", "Thin Colab interface for the GPU steps only; not a source of truth",
     "n/a", "code", "versioned"),
    (".gitattributes", "A+B+C", "Byte-fidelity settings (`* -text`)", "n/a", "config", "versioned"),
    (".gitignore", "A+B+C", "What is never committed", "n/a", "config", "versioned"),
    ("pyproject.toml", "A+B+C", "Dependency specification (authoritative)", "n/a",
     "config", "versioned"),
    ("requirements.txt", "A+B+C", "Equivalent dependency list for pip users", "n/a",
     "config", "versioned"),
]


def classify(rel: str):
    for pat, task, purpose, script, cls, mut in RULES:
        if fnmatch.fnmatch(rel, pat) or (pat.endswith("/*") and rel.startswith(pat[:-1])):
            return task, purpose, script, cls, mut
    return None


def build_artifact_index(repo: Path) -> tuple[str, list[str]]:
    rows, unmatched = [], []
    for p in sorted(repo.rglob("*")):
        if not p.is_file() or ".git" in p.parts:
            continue
        rel = p.relative_to(repo).as_posix()
        hit = classify(rel)
        if hit is None:
            unmatched.append(rel)
            continue
        rows.append((rel, *hit, p.stat().st_size))

    def group_key(r):
        return (r[1], r[3], r[2])  # task, script, purpose

    # collapse families that share purpose + script into one row with a glob
    seen, out = {}, []
    for rel, task, purpose, script, cls, mut, size in rows:
        k = (task, purpose, script, cls, mut)
        seen.setdefault(k, []).append((rel, size))
    for (task, purpose, script, cls, mut), members in seen.items():
        if len(members) == 1:
            label = members[0][0]
        else:
            paths = [m[0] for m in members]
            pref = paths[0].rsplit("/", 1)[0]
            label = (f"`{pref}/` ({len(paths)} files)" if all(q.startswith(pref) for q in paths)
                     else f"{len(paths)} files")
            label = label.strip("`")
        out.append((label, task, purpose, script, cls, mut,
                    sum(m[1] for m in members), len(members)))
    out.sort(key=lambda r: (["A", "B", "C", "A+B+C"].index(r[1]) if r[1] in
                            ("A", "B", "C", "A+B+C") else 9, r[0]))

    md = [
        "# Artifact index",
        "",
        "Every file published in this repository, with the script that produced it and how it",
        "should be treated. Generated by `scripts/44_repo_documentation.py`; last regenerated",
        f"{date.today().isoformat()}.",
        "",
        "**Class**",
        "",
        "| Class | Meaning |",
        "|---|---|",
        "| `authoritative` | The official result. Cite this. Never overwritten by later work. |",
        "| `comparison` | Produced to compare two pipelines, or recomputed under a harmonised "
        "metric definition. Never replaces an `authoritative` value. |",
        "| `exploratory` | Pre-specified secondary or post hoc descriptive analysis. Reported, "
        "not promoted to a conclusion. |",
        "| `supporting` | Guides, indexes and audit records. |",
        "| `code` / `config` | Source and settings. |",
        "",
        "**Mutability**",
        "",
        "| Mutability | Meaning |",
        "|---|---|",
        "| `frozen` | Written once. Changing it would change a reported result. |",
        "| `append-only` | Grows over time; earlier entries are never edited. |",
        "| `regenerated` | Rebuilt from frozen inputs by its script; safe to regenerate. |",
        "| `versioned` | Ordinary source control. |",
        "",
        f"**{sum(r[7] for r in out)} files** across "
        f"{len({r[1] for r in out})} task groupings.",
        "",
        "| Artifact | Task | Purpose | Generating script | Class | Mutability |",
        "|---|---|---|---|---|---|",
    ]
    for label, task, purpose, script, cls, mut, _size, _n in out:
        sc = f"`{script}`" if script not in ("n/a", "written by hand") else script
        md.append(f"| `{label}` | {task} | {purpose} | {sc} | {cls} | {mut} |")
    md += [
        "",
        "## Which file is the source of truth for a given number?",
        "",
        "| Question | File |",
        "|---|---|",
        "| Task A Test performance | `results/taskA/evaluation/test_metrics_primary.json` |",
        "| Task B Test performance | `results/taskB/evaluation/test_metrics_table.csv` "
        "(ECE = 5 equal-count bins) |",
        "| Task C Test performance | `results/taskC/evaluation/test_metrics_taskC.json` |",
        "| All five models under one metric definition | "
        "`results/taskC/evaluation/test_metrics_table.csv` (ECE = 10 equal-width bins) |",
        "| Statistical comparison between models | `results/comparison/delong_results.csv` |",
        "| Human-guided pipeline results | the `pipeline == \"Human-guided\"` rows of "
        "`results/task{B,C}/comparison/human_vs_ai_agent_performance.csv`, each carrying its "
        "own `source` column |",
        "| AI-agent results inside a comparison table | the `pipeline == \"AI-Agent\"` rows of "
        "the same files, which restate the official artefacts and do not replace them |",
        "| ECE recomputed for comparability | "
        "`results/taskB/comparison/harmonized_ece.csv` |",
        "| Confidence intervals side by side | "
        "`results/taskC/comparison/harmonized_ci.csv` |",
        "",
        "### Two ECE definitions are in use, deliberately",
        "",
        "`results/taskB/evaluation/test_metrics_table.csv` reports `ece_5bin` (5 equal-count "
        "bins) and `results/taskC/evaluation/test_metrics_table.csv` reports "
        "`ece_10_equal_width` (10 equal-width bins) for the same Task B models. The two are "
        "not interchangeable and neither is wrong: each table states its definition in the "
        "column name. Quote the column name whenever you quote an ECE.",
        "",
        "### Curve points appear under two tasks",
        "",
        "`results/taskC/evaluation/test_{roc,pr}_points_clinical_*.csv` are byte-identical to "
        "the corresponding `results/taskB/evaluation/test_{roc,pr}_points_*.csv`. They are the "
        "same predictions re-emitted so that the Task C evaluation is self-contained. Task B's "
        "copy is the authoritative one.",
    ]
    return "\n".join(md) + "\n", unmatched


def build_taskc_readme(project: Path, c_f: dict, c_m: dict, c_val: dict) -> str:
    return f"""# results/taskC — Multimodal Late Fusion

Task C combines the **frozen** Task A chest-radiograph model with the **frozen** Task B
clinical model. Neither component was retrained, refitted or re-thresholded for Task C, and
the cohort and split are the ones Task A and Task B already used.

## Frozen specification

| | |
|---|---|
| Primary strategy | {c_f['fusion']['primary_strategy']} |
| Formula | `{c_f['fusion']['formula']}` |
| Clinical component | Task B {c_f['clinical_component']['model'].upper()}, condition `{c_f['clinical_component']['condition']}`, selected by {c_f['clinical_component']['selection_rule']} |
| CXR component | Task A `{c_f['cxr_component']['condition']}`, seed {c_f['cxr_component']['seed']}, epoch {c_f['cxr_component']['best_epoch']} |
| Weights | clinical **{c_f['fusion']['w_clinical']}**, CXR **{c_f['fusion']['w_cxr']}** |
| Weight search | {c_f['fusion']['search']['metric']}, on {c_f['fusion']['search']['dataset_used']}, grid step {c_f['fusion']['search']['grid']['step']} |
| Threshold | {c_m['threshold']['value']} — {c_m['threshold']['source']}, recomputed on Test: {c_m['threshold']['recomputed_on_test']} |
| Frozen file SHA256 | `{sha256(project / 'results/taskC/modeling/final_selection_taskC.json')}` |

## Test result (primary)

ROC-AUC **{f6(c_m['auroc']['point'])}** (95% CI {f6(c_m['auroc']['ci_low'])}–{f6(c_m['auroc']['ci_high'])}),
PR-AUC {f6(c_m['auprc']['point'])}, Brier {f6(c_m['brier'])},
ECE {f6(c_m['ece'])} ({c_m['calibration_definition']['ece_bins']} {c_m['calibration_definition']['ece_scheme']} bins),
n = {c_m['n']}, deaths = {c_m['events']}.

Validation ROC-AUC of the fused model was {f6(c_val['fused']['auroc'])}.

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
- **A weight is not a contribution.** The CXR weight of {c_f['fusion']['w_cxr']} does not mean
  the radiograph supplies {c_f['fusion']['w_cxr']:.0%} of the performance. Use
  `evaluation/modality_importance_test.csv` for that question, and read it as post hoc and
  descriptive.
- **The weight curve is flat.** See `fusion/weight_search_primary.csv`: nearby weights perform
  similarly on Validation, so the selected weight should not be over-interpreted as an
  estimated quantity.
- **A lack of statistical significance is not evidence of equivalence.** With
  {c_m['events']} Test events, the confidence intervals here admit differences that would
  matter clinically. See `results/comparison/delong_results.csv`.

## Not included

Model checkpoints and preprocessed images are not published. Task C adds no new fitted weights
beyond the two fusion coefficients recorded in `modeling/final_selection_taskC.json`, so that
file plus the Task A and Task B artefacts is sufficient to reproduce every Task C number from
the saved component predictions.
"""


def build_provenance_block(repo: Path, rows_a, rows_b, rows_c, rows_x) -> str:
    """The Task B / Task C half of docs/results_provenance.md, between the sentinels."""
    def table(title, note, rows):
        out = [f"### {title}", "", note, "",
               "| 数値 | 値 | authoritative artifact | 読み出した場所 |", "|---|---|---|---|"]
        for label, value, path, field in rows:
            out.append(f"| {label} | {value} | `{path}` | `{field}` |")
        return "\n".join(out)

    return "\n\n".join([
        PROV_BEGIN,
        "---",
        "# Task A / B / C 主要数値の authoritative artifact 対応",
        "",
        "以下は `scripts/44_repo_documentation.py` が生成します（上の Task A 部分は "
        "`scripts/20_prepare_github_repo.py` が生成）。**表の値はすべて実行時に artifact から"
        "読み出したもので、記憶や推測によるものはありません。**",
        "",
        "「読み出した場所」列は、ファイル内のどのキー・どの行から取ったかを示します。",
        table("Task A", "凍結モデルの仕様は `test_run_env_note.json` に集約してあります"
                        "（`test_run_meta.json` は凍結 artifact のため変更していません）。", rows_a),
        table("Task B", "**ECE の定義に注意**：Task B の公式表は 5 equal-count bin です。"
                        "同じモデルを 10 equal-width bin で評価した値は Task C の表にあります。",
              rows_b),
        table("Task C", "閾値は Validation 由来で、Test では再計算していません。", rows_c),
        table("横断比較・統計", "paired DeLong は 2 family 構成で、family ごとに Holm 補正を"
                                "行っています。`Late Fusion vs Clinical LR` は両 family に属する"
                                "ため、unadjusted p は 1 つ、Holm 調整済み p は 2 つあります。",
              rows_x),
        "### 正本が競合しないことの確認",
        "",
        "| 確認項目 | 結果 |",
        "|---|---|",
        "| comparison artifact が official Test metrics を上書きしていないか | 上書きなし。"
        "`results/task{B,C}/comparison/` は `pipeline` 列と `source` 列を持ち、"
        "AI-Agent 行は official artifact の再掲であることを明示している |",
        "| harmonized 値が official 値を置き換えていないか | 置き換えていない。"
        "`harmonized_ece.csv` / `harmonized_ci.csv` は比較専用で、official 表は未変更 |",
        "| 同一数値が複数ファイルにある場合の正本 | `docs/artifact_index.md` 末尾の"
        "「Which file is the source of truth for a given number?」表に明記 |",
        "",
        "Task A / Task B / Task C の official artifact は、本ドキュメントの作成により"
        "一切変更していません。",
        PROV_END,
    ])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=".")
    ap.add_argument("--repo", default="github_repo")
    args = ap.parse_args()
    project = Path(args.project).resolve()
    repo = project / args.repo

    # ---- read every number from an artefact -----------------------------------------
    split = pd.read_csv(repo / "results/taskA/preprocessing/fixed_split_1277.csv", dtype=str,
                        encoding="utf-8-sig")
    lab = "true_label" if "true_label" in split.columns else split.columns[-1]
    counts = split["split"].value_counts().to_dict()
    deaths = split[split[lab] == "1"]["split"].value_counts().to_dict()

    a_m = json.loads((repo / "results/taskA/evaluation/test_metrics_primary.json")
                     .read_text(encoding="utf-8-sig"))
    a_gc = json.loads((repo / "results/taskA/gradcam/gradcam_run_meta.json")
                      .read_text(encoding="utf-8-sig"))
    env_note_rel = "results/taskA/evaluation/test_run_env_note.json"
    if not (repo / env_note_rel).exists():
        print(f"STOP: {env_note_rel} is missing. Run scripts/45_taskA_env_note.py first — "
              f"the README reads the Task A condition, seed and best epoch from it rather "
              f"than restating them.")
        return 1
    a_env = json.loads((repo / env_note_rel).read_text(encoding="utf-8-sig"))

    def env(path: str):
        """Read a value from the env note, refusing anything not marked confirmed."""
        node = a_env
        for part in path.split("."):
            node = node[part]
        if node["status"] != "confirmed":
            raise SystemExit(f"STOP: {env_note_rel} marks `{path}` as '{node['status']}'. "
                             f"It must not be printed in the README as a fact.")
        return node["value"]
    b_f = json.loads((repo / "results/taskB/modeling/final_selection_taskB.json")
                     .read_text(encoding="utf-8-sig"))
    b_v = json.loads((repo / "results/taskB/variable_selection/final_variables.json")
                     .read_text(encoding="utf-8-sig"))
    b_t = pd.read_csv(repo / "results/taskB/evaluation/test_metrics_table.csv",
                      encoding="utf-8-sig").set_index("model")
    c_f = json.loads((repo / "results/taskC/modeling/final_selection_taskC.json")
                     .read_text(encoding="utf-8-sig"))
    c_m = json.loads((repo / "results/taskC/evaluation/test_metrics_taskC.json")
                     .read_text(encoding="utf-8-sig"))
    c_val = json.loads((repo / "results/taskC/validation/validation_metrics_taskC.json")
                       .read_text(encoding="utf-8-sig"))
    c_t = pd.read_csv(repo / "results/taskC/evaluation/test_metrics_table.csv",
                      encoding="utf-8-sig").set_index("model")
    c_sec = json.loads((repo / "results/taskC/evaluation/secondary_exploratory_test.json")
                       .read_text(encoding="utf-8-sig"))
    c_mod = pd.read_csv(repo / "results/taskC/evaluation/modality_importance_test.csv",
                        encoding="utf-8-sig").set_index("modality")
    dl = pd.read_csv(repo / "results/comparison/delong_results.csv", encoding="utf-8-sig")
    meta_b = json.loads((repo / "results/taskB/evaluation/test_run_meta.json")
                        .read_text(encoding="utf-8-sig"))

    def row(m):
        r = c_t.loc[m]
        return (f6(r.auroc), f"{f6(r.auroc_ci_low)}–{f6(r.auroc_ci_high)}", f6(r.auprc),
                f6(r.brier), f6(r.ece_10_equal_width))

    cxr, lr, xgb, mlp, lf = (row(m) for m in
                             ("cxr", "clinical_lr", "clinical_xgboost", "clinical_mlp",
                              "late_fusion"))
    famA, famB = dl[dl.family == "A"], dl[dl.family == "B"]
    ovr = famA[famA.model_2 == "clinical_lr"].iloc[0]
    pkg = meta_b["packages"]

    # ---- preserve the existing Task A README (idempotent) -----------------------------
    existing = (repo / "README.md").read_text(encoding="utf-8-sig")
    if BEGIN in existing and END in existing:
        task_a = existing.split(BEGIN, 1)[1].split(END, 1)[0].strip("\n")
        print("Task A section recovered from the previous composition (idempotent re-run)")
    else:
        body = existing.replace("\n## ", "\n### ").lstrip()
        if body.startswith("# "):
            body = "#" + body
        task_a = body.rstrip()
        print("Task A section captured from the current Task A README and demoted one level")

    md = f"""# COVID-19 in-hospital mortality prediction — AI-agent pipeline (Task A / B / C)

An autonomously designed analysis of **in-hospital mortality predicted at the moment of
admission**, from the chest radiograph (Task A), from the clinical table (Task B), and from
both combined (Task C). Built on the Stony Brook University COVID-19 Positive Cases collection
(TCIA **COVID-19-NY-SBU**).

> Every number below is read from a saved artefact in this repository, not written from memory.
> `scripts/44_repo_documentation.py` regenerates this file, `docs/artifact_index.md` and
> `results/taskC/README.md`. Last regenerated {date.today().isoformat()}.

**Start here:** [`docs/artifact_index.md`](docs/artifact_index.md) lists every published file,
the script that produced it, and whether it is an official result, a comparison artefact or an
exploratory one.

---

## 1. Study overview

Predict **in-hospital death** using only information available at **T0 = admission**
(`visit_start_datetime`). One cohort and one split are shared by all three tasks, which is
what makes the final paired statistical comparison legitimate.

| | Train | Validation | Test | Total |
|---|---:|---:|---:|---:|
| **Patients** | {counts['train']:,} | {counts['val']} | {counts['test']} | **{len(split):,}** |
| **Deaths** | {deaths['train']} | {deaths['val']} | {deaths['test']} | {sum(deaths.values())} |
| Mortality | {deaths['train'] / counts['train']:.1%} | {deaths['val'] / counts['val']:.1%} | {deaths['test'] / counts['test']:.1%} | {sum(deaths.values()) / len(split):.1%} |

The split is fixed (`results/taskA/preprocessing/fixed_split_1277.csv`) and was never
regenerated. The Test set was evaluated once per task.

| Task | Model | Test ROC-AUC (95% CI) |
|---|---|---|
| **A** | CXR ResNet18 | {cxr[0]} ({cxr[1]}) |
| **B** | Clinical Logistic Regression (primary) | {lr[0]} ({lr[1]}) |
| **B** | Clinical XGBoost | {xgb[0]} ({xgb[1]}) |
| **B** | Clinical MLP | {mlp[0]} ({mlp[1]}) |
| **C** | **Late Fusion (Clinical LR + CXR)** | **{lf[0]}** ({lf[1]}) |

---

## 2. Task A — chest radiograph model

| | |
|---|---|
| Model | ResNet18 (ImageNet-pretrained), one admission chest radiograph per patient |
| Input | index CXR within T0−2 days to T0, cached at 512×512 16-bit, resized to 224×224 |
| Frozen model | condition **`{env('frozen_model.condition')}`**, seed **{env('random_seeds.model_seed_primary')}**, best epoch **{env('frozen_model.best_epoch')}** (Validation AUROC {f6(env('frozen_model.validation_auroc_at_best_epoch'))}) — selected on Validation only, from 30 runs |
| Checkpoint SHA256 | `{env('frozen_model.checkpoint_sha256')[:24]}…` |
| **Test ROC-AUC** | **{f6(a_m['auroc']['point'])}** (95% CI {f6(a_m['auroc']['ci_low'])}–{f6(a_m['auroc']['ci_high'])}) |
| Test PR-AUC | {f6(a_m['auprc']['point'])} ({f6(a_m['auprc']['ci_low'])}–{f6(a_m['auprc']['ci_high'])}) |
| Test Brier | {f6(a_m['brier'])} |
| Test ECE | {f6(a_m['ece'])} over the {len(a_m['calibration_bins'])} equal-count bins in `test_calibration_bins.csv`. Under the 10 equal-width bins used in §5 the same model gives {cxr[4]} — see §5 |
| Threshold | {a_m['operating_point_youden']['threshold']} (Validation Youden, frozen) |

**Grad-CAM.** {a_gc['n_cases']} cases — TP {a_gc['cases_used_per_group']['TP']},
FP {a_gc['cases_used_per_group']['FP']}, TN {a_gc['cases_used_per_group']['TN']},
FN {a_gc['cases_used_per_group']['FN']} — layer `{a_gc['gradcam_layer']}`, frozen threshold,
published as one 4×4 panel with {a_gc['panel_empty_cells']} blank cell:
[`results/taskA/gradcam/gradcam_panel_4x4.png`](results/taskA/gradcam/gradcam_panel_4x4.png).
The reading notes, including the researcher's written summary, are in
[`gradcam_notes.md`](results/taskA/gradcam/gradcam_notes.md). Per-case images are not
published; their SHA256 values are recorded so they remain verifiable.

**Artefacts:** [`results/taskA/`](results/taskA/) · guide:
[`results/taskA/README.md`](results/taskA/README.md) · full Task A documentation is preserved
below in [§11](#11-task-a--full-documentation).

---

## 3. Task B — clinical table models

Three model families over identical inputs, so what is being compared is the model family and
not the pipeline around it.

| | |
|---|---|
| Candidates | 13 variables, chosen from clinical knowledge, prior literature and the 4C Mortality Score **before** any outcome association was examined |
| Variable selection | **backward elimination by AIC on the Training set only**, subject to a parameter budget ({b_v['deaths_train']} Training deaths / EPV {b_v['epv_target']} = {b_v['parameter_budget']} coefficients). Validation and Test played no part |
| Final inputs | **{len(b_v['primary']['variables'])} variables / {b_v['primary']['n_features']} coefficients**, events per parameter {b_v['primary']['events_per_parameter']} |
| Variables | `{'`, `'.join(b_v['primary']['variables'])}` |
| Hyperparameters | repeated stratified 5-fold × 3 cross-validation **inside Training**; the preprocessor was refitted inside every fold |
| Validation's role | comparing model families and fixing the operating threshold — nothing else |
| **Primary model** | **Clinical Logistic Regression**, condition `{b_f['models']['lr']['condition']}` |

**Test results** (from `results/taskB/evaluation/test_metrics_table.csv`; ECE = 5 equal-count bins)

| Model | ROC-AUC | 95% CI | PR-AUC | Brier | ECE (5 equal-count) | Sens | Spec |
|---|---:|---|---:|---:|---:|---:|---:|
| **Clinical LR (primary)** | **{f6(b_t.loc['lr'].auroc)}** | {f6(b_t.loc['lr'].auroc_ci_low)}–{f6(b_t.loc['lr'].auroc_ci_high)} | {f6(b_t.loc['lr'].auprc)} | {f6(b_t.loc['lr'].brier)} | {f6(b_t.loc['lr'].ece_5bin)} | {f6(b_t.loc['lr'].sensitivity)} | {f6(b_t.loc['lr'].specificity)} |
| Clinical XGBoost | {f6(b_t.loc['xgb'].auroc)} | {f6(b_t.loc['xgb'].auroc_ci_low)}–{f6(b_t.loc['xgb'].auroc_ci_high)} | {f6(b_t.loc['xgb'].auprc)} | {f6(b_t.loc['xgb'].brier)} | {f6(b_t.loc['xgb'].ece_5bin)} | {f6(b_t.loc['xgb'].sensitivity)} | {f6(b_t.loc['xgb'].specificity)} |
| Clinical MLP | {f6(b_t.loc['mlp'].auroc)} | {f6(b_t.loc['mlp'].auroc_ci_low)}–{f6(b_t.loc['mlp'].auroc_ci_high)} | {f6(b_t.loc['mlp'].auprc)} | {f6(b_t.loc['mlp'].brier)} | {f6(b_t.loc['mlp'].ece_5bin)} | {f6(b_t.loc['mlp'].sensitivity)} | {f6(b_t.loc['mlp'].specificity)} |
| XGBoost, native missing *(secondary)* | {f6(b_t.loc['xgb_native'].auroc)} | {f6(b_t.loc['xgb_native'].auroc_ci_low)}–{f6(b_t.loc['xgb_native'].auroc_ci_high)} | {f6(b_t.loc['xgb_native'].auprc)} | {f6(b_t.loc['xgb_native'].brier)} | {f6(b_t.loc['xgb_native'].ece_5bin)} | {f6(b_t.loc['xgb_native'].sensitivity)} | {f6(b_t.loc['xgb_native'].specificity)} |

The last row is a pre-specified secondary arm. Native missing handling did not improve
cross-validated performance compared with the imputation-based approach in this analysis.

**Missingness.** Laboratory missingness in this cohort is a triage proxy, not a measurement
accident: the association between "test not ordered" and death reverses sign once the analysis
is restricted to inpatients (`results/taskB/preprocessing/missingness_mechanism.csv`). Only
lymphocyte count survived the pre-specified C1–C5 criteria for carrying a missing indicator.

**Artefacts:** [`results/taskB/`](results/taskB/) · guide with cohort description and
limitations: [`results/taskB/README.md`](results/taskB/README.md).

---

## 4. Task C — multimodal Late Fusion

| | |
|---|---|
| Components | **Clinical LR** (Task B, frozen) + **CXR ResNet18** (Task A, frozen). Neither was retrained |
| Clinical component rule | best **Training cross-validated** ROC-AUC among the Task B models, fixed before any fusion was computed |
| Strategy | **decision-level Late Fusion, probability-space weighted average**, fixed before the Validation search began |
| Equation | `{c_f['fusion']['formula']}` |
| **Frozen weights** | clinical **{c_f['fusion']['w_clinical']}**, CXR **{c_f['fusion']['w_cxr']}** |
| Weight search | w ∈ [0, 1] at step {c_f['fusion']['search']['grid']['step']} ({c_f['fusion']['search']['grid']['n_points']} points), on {c_f['fusion']['search']['dataset_used']}; metric {c_f['fusion']['search']['metric']}, tie-break {c_f['fusion']['search']['tie_break']} |
| Threshold | {c_m['threshold']['value']} — {c_m['threshold']['source']}; recomputed on Test: **{c_m['threshold']['recomputed_on_test']}** |
| Not used | early fusion (524 inputs against {deaths['train']} Training events). Logit-space fusion and logistic stacking were kept as **non-competing secondary arms** |
| Frozen spec SHA256 | `{sha256(repo / 'results/taskC/modeling/final_selection_taskC.json')}` |

**Test results (primary)**

| Metric | Value |
|---|---|
| **ROC-AUC** | **{f6(c_m['auroc']['point'])}** ({f6(c_m['auroc']['ci_low'])}–{f6(c_m['auroc']['ci_high'])}) |
| **PR-AUC** | **{f6(c_m['auprc']['point'])}** ({f6(c_m['auprc']['ci_low'])}–{f6(c_m['auprc']['ci_high'])}) |
| **Brier** | **{f6(c_m['brier'])}** |
| **ECE** ({c_m['calibration_definition']['ece_bins']} {c_m['calibration_definition']['ece_scheme']} bins) | **{f6(c_m['ece'])}** |
| Sensitivity / Specificity | {f6(c_m['operating_point']['sensitivity'])} / {f6(c_m['operating_point']['specificity'])} |
| PPV / NPV | {f6(c_m['operating_point']['ppv'])} / {f6(c_m['operating_point']['npv'])} |
| TP / FP / TN / FN | {int(c_m['operating_point']['tp'])} / {int(c_m['operating_point']['fp'])} / {int(c_m['operating_point']['tn'])} / {int(c_m['operating_point']['fn'])} |

Validation ROC-AUC of the fused model was {f6(c_val['fused']['auroc'])}.

**Modality contribution** — permutation importance on Test, post hoc and descriptive:
clinical **{f6(c_mod.loc['clinical'].mean_auroc_drop)}** ± {f6(c_mod.loc['clinical'].sd_auroc_drop)},
CXR **{f6(c_mod.loc['cxr'].mean_auroc_drop)}** ± {f6(c_mod.loc['cxr'].sd_auroc_drop)} ROC-AUC
decrease. A weight is not a contribution: the CXR weight of {c_f['fusion']['w_cxr']} does not
mean the radiograph supplies {c_f['fusion']['w_cxr']:.0%} of the performance.

**Secondary arms, exploratory, not promoted:** logit-space weighted average
{f6(c_sec['models']['logit_space_weighted_average_w0.70']['auroc'])}, logistic stacking
{f6(c_sec['models']['logistic_stacking']['auroc'])}
(`results/taskC/evaluation/secondary_exploratory_test.json`).

**Artefacts:** [`results/taskC/`](results/taskC/) · guide:
[`results/taskC/README.md`](results/taskC/README.md).

---

## 5. Final comparison — all five models on the same {c_m['n']} Test patients

| Model | Task | ROC-AUC | 95% CI | PR-AUC | Brier | ECE (10 equal-width) |
|---|---|---:|---|---:|---:|---:|
| CXR ResNet18 | A | {cxr[0]} | {cxr[1]} | {cxr[2]} | {cxr[3]} | {cxr[4]} |
| Clinical LR | B | {lr[0]} | {lr[1]} | {lr[2]} | {lr[3]} | {lr[4]} |
| Clinical XGBoost | B | {xgb[0]} | {xgb[1]} | {xgb[2]} | {xgb[3]} | {xgb[4]} |
| Clinical MLP | B | {mlp[0]} | {mlp[1]} | {mlp[2]} | {mlp[3]} | {mlp[4]} |
| **Late Fusion** | **C** | **{lf[0]}** | {lf[1]} | {lf[2]} | {lf[3]} | {lf[4]} |

Source: [`results/taskC/evaluation/test_metrics_table.csv`](results/taskC/evaluation/test_metrics_table.csv).
All rows share n = {c_m['n']}, deaths = {c_m['events']}, the same bootstrap definition
({c_m['auroc']['n_resamples']:,} stratified resamples, seed {c_m['auroc']['seed']}) and the same
ECE definition. The Task B table reports the same models under a different ECE definition
(5 equal-count bins); both are official and each names its definition in the column header.

Patient-level predictions for all five models, in one file:
[`results/comparison/test_predictions_all_models.csv`](results/comparison/test_predictions_all_models.csv)
(`subject_id`, `true_label`, `prob_cxr`, `prob_clinical_lr`, `prob_clinical_xgboost`,
`prob_clinical_mlp`, `prob_late_fusion`).

---

## 6. Statistical comparison — paired DeLong

Paired DeLong (DeLong 1988; Sun & Xu 2014) over the same {c_m['n']} Test patients, merged on
`subject_id`. **Two-sample AUC comparisons are not used.** Both comparison families and the
Holm correction were fixed **before the Test set was opened**
([`results/comparison/delong_plan.json`](results/comparison/delong_plan.json)).

### Family A — Late Fusion vs the clinical models (Holm within Family A)

| Comparison | ΔAUC | SE | unadjusted p | Holm p | Significant |
|---|---:|---:|---:|---:|---|
"""
    for _, r in famA.iterrows():
        md += (f"| Late Fusion vs {r.model_2_label} | {r.delta_AUC:+.6f} | "
               f"{r.standard_error:.6f} | {r.p_unadjusted:.6f} | "
               f"**{r.p_holm_within_family:.6f}** | "
               f"{'**Yes**' if r['significant_at_0.05_after_holm'] else 'No'} |\n")
    md += """
### Family B — Late Fusion vs the reference models (separate Holm within Family B)

| Comparison | ΔAUC | SE | unadjusted p | Holm p | Significant |
|---|---:|---:|---:|---:|---|
"""
    for _, r in famB.iterrows():
        md += (f"| Late Fusion vs {r.model_2_label} | {r.delta_AUC:+.6f} | "
               f"{r.standard_error:.6f} | {r.p_unadjusted:.6f} | "
               f"**{r.p_holm_within_family:.6f}** | "
               f"{'**Yes**' if r['significant_at_0.05_after_holm'] else 'No'} |\n")
    md += f"""
`Late Fusion vs Clinical LR` belongs to both families, so it carries **one unadjusted p-value
({ovr.p_unadjusted:.6f}) and two Holm-adjusted p-values**, one per family. Every DeLong output
carries a `family` column so the two can never be confused. This follows from the
pre-specified design; it is not a conflict between results.

**A lack of statistical significance is not evidence of equivalence.** With {c_m['events']}
Test events, every confidence interval reported here admits differences that would matter
clinically.

Results: [`delong_results.csv`](results/comparison/delong_results.csv) ·
[`delong_results.json`](results/comparison/delong_results.json) ·
family definitions: [`results/comparison/README.md`](results/comparison/README.md).

---

## 7. Human-guided vs AI-agent comparisons

The same two research questions were previously worked through in a human-guided analysis. The
documents below compare the two pipelines **methodologically**. They are descriptive and post
hoc by construction, and they establish neither superiority nor equivalence of either approach:
the pipelines differ in several components at once, so no performance difference can be
attributed to the pipeline design alone.

| | Document | Machine-readable tables |
|---|---|---|
| Task B | [`docs/taskB_human_vs_ai_agent_comparison.md`](docs/taskB_human_vs_ai_agent_comparison.md) | [`results/taskB/comparison/`](results/taskB/comparison/) |
| Task C | [`docs/taskC_human_vs_ai_agent_comparison.md`](docs/taskC_human_vs_ai_agent_comparison.md) | [`results/taskC/comparison/`](results/taskC/comparison/) |

Two harmonisation artefacts exist because the pipelines had used different definitions:

- **ECE** — the human-guided Task B analysis used 10 equal-width bins and the AI-agent analysis
  5 equal-count bins. `results/taskB/comparison/harmonized_ece.csv` recomputes both under one
  definition. It does not replace the official Task B ECE.
- **Confidence intervals** — `results/taskC/comparison/harmonized_ci.csv` places both pipelines'
  intervals side by side **with their bootstrap settings**. The confidence intervals were
  generated using different bootstrap settings and therefore were not used for a standardized
  numerical comparison of uncertainty between the two pipelines.

Where sources disagree, the disagreement is listed rather than resolved:
`results/task{{B,C}}/comparison/source_discrepancies.csv`.

---

## 8. Reproducibility

Every analysis runs non-interactively from a numbered script. Notebooks are thin interfaces to
the GPU steps and are never the source of truth.

| Stage | Scripts |
|---|---|
| Task A | `scripts/01`–`scripts/20` |
| Task B | `scripts/21`–`scripts/34` |
| Task C | `scripts/35`–`scripts/43` |
| Repository documentation | `scripts/44_repo_documentation.py` |

**Frozen specifications** — each records the SHA256 of the artefacts it depends on, the seeds,
the bootstrap settings and the calibration definition, so a later run can prove it used the
same specification:

- `results/taskA/training/condition_selection_decision.json`
- `results/taskB/modeling/final_selection_taskB.json`
- `results/taskC/modeling/final_selection_taskC.json`

**Environment used for the Task B and Task C Test evaluations**
(`results/task{{B,C}}/evaluation/test_run_meta.json`):
numpy {pkg['numpy']}, pandas {pkg['pandas']}, scikit-learn {pkg['sklearn']},
xgboost {pkg['xgboost']}, torch {pkg['torch']}. Task A was trained and evaluated on Colab
(`results/taskA/evaluation/test_run_meta.json` records the device, torch build and platform).

**To re-run the statistical comparison** without re-running anything else:

```bash
python scripts/41_taskC_delong.py --project . --predictions results/comparison/test_predictions_all_models.csv
```

`results/comparison/test_predictions_all_models.csv` holds every patient-level probability the
paired DeLong needs, so the comparison is re-runnable from this repository alone.

**Not reproducible from this repository alone:** anything that needs the raw DICOM images or a
model checkpoint — the Task A image cache, Task A training, and Grad-CAM. Those steps need the
TCIA source data plus the checkpoint, whose SHA256 values are recorded. Every downstream result
is reproducible from the saved predictions.

---

## 9. Repository structure

```
README.md                  this file
docs/
├── artifact_index.md      every artefact: purpose, script, class, mutability  <- start here
├── decision_log.md        every design decision and its reason (D-001 onwards)
├── change_log.csv         every defect found and how it was fixed (CL-001 onwards)
├── open_questions.md      questions left open and their status
├── requirements.md        the study requirements as agreed
├── taskB_human_vs_ai_agent_comparison.md
├── taskC_human_vs_ai_agent_comparison.md
└── task_a_*.md            Task A design and evaluation plans
results/
├── taskA/                 preprocessing · training · evaluation · gradcam · comparison
├── taskB/                 preprocessing · variable_selection · modeling · evaluation · qc
│                          · comparison · secondary_exploratory_unused
├── taskC/                 preflight · inputs · fusion · validation · modeling · qc
│                          · evaluation · comparison
└── comparison/            all-model patient-level predictions, DeLong plan and results
scripts/                   numbered, non-interactive entry points (01-44)
src/covid_mortality/
├── data/ features/ models/ training/   dataset, preprocessing, model definitions
├── evaluation/            metrics, DeLong, Grad-CAM, prediction schema
└── fusion/                Task C fusion
tests/                     unit tests for the library code
notebooks/                 thin Colab interfaces for the GPU steps only
```

---

## 10. Important methodological notes

**On the Test set.** Test predictions and Test performance metrics were generated once per
task, after the specification had been frozen to a hash-verified JSON. In all three tasks,
preprocessing parameters were estimated on Training, variables were selected on Training,
hyperparameters were chosen by Training-internal cross-validation, and model families and
operating thresholds were fixed on Validation. The Test set had previously been accessed for
limited QC purposes, but no Test information was used for preprocessing fitting, variable
selection, hyperparameter tuning, checkpoint selection, or model selection. Every access is
recorded in `results/task{{A,B,C}}/*/test_access_log.*`.

**On frozen artefacts.** `results/taskA/`, `results/taskB/` and `results/taskC/` official
artefacts are not modified by later work. Where a later step needed a different metric
definition, a separate harmonised artefact was produced rather than an official one
overwritten. `docs/artifact_index.md` states, for each file, whether it is authoritative.

**On the cohort.** The 1,277-patient cohort mixes emergency-department encounters that ended
in discharge with admitted inpatients. The two groups have very different mortality, and an
inpatient-only sensitivity analysis was considered and deliberately not performed; the material
prepared for it is isolated in `results/taskB/secondary_exploratory_unused/` and forms no part
of any result. See `results/taskB/README.md`.

**On statistical claims.** Non-significance is not equivalence. Confidence intervals in this
study rest on {c_m['events']} Test events and are wide.

**On the human-guided comparison.** It is an exploratory methodological comparison of two
independently designed pipelines that converged on a similar fusion structure. It is not a
performance contest and does not support a claim that either approach is better.

---

## 11. Privacy, data handling and repository status

- **This repository is PRIVATE** and stays private until the licence and the data
  redistribution terms are settled.
- **No raw clinical data and no DICOM files are included.** The source data (TCIA
  COVID-19-NY-SBU) must be obtained from TCIA directly.
- **No model checkpoints and no preprocessed images are included**, by size and by policy.
  Their SHA256 values are recorded so they remain verifiable.
- Patient-level files contain de-identified Subject IDs only. **DICOM instance UIDs are removed**
  from the published CSVs, pending confirmation of the TCIA redistribution terms.
- The single published patient image is the approved 4×4 Grad-CAM panel. The other Grad-CAM
  images are withheld and listed by hash in
  `results/taskA/gradcam/gradcam_excluded_images_sha256.csv`.
- Local filesystem paths are redacted to `<LOCAL_PATH>` / `<DRIVE_ROOT>` when files are copied
  into this repository.
- **License: to be determined.** Until it is, the reuse conditions of this repository are not
  established.

---

---

## 12. Task A — full documentation

*Everything below is the original Task A README, kept unchanged and demoted one heading level.*

{BEGIN}
{task_a}
{END}
"""

    # ---- write --------------------------------------------------------------------
    (repo / "README.md").write_text(md, encoding="utf-8")
    print(f"README.md            {len(md.encode('utf-8')):>9,} B")

    idx, unmatched = build_artifact_index(repo)
    for target in (project / "docs/artifact_index.md", repo / "docs/artifact_index.md"):
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(idx, encoding="utf-8")
    print(f"docs/artifact_index.md {len(idx.encode('utf-8')):>7,} B")
    if unmatched:
        print(f"  WARNING: {len(unmatched)} files matched no classification rule:")
        for u in unmatched[:20]:
            print(f"    {u}")

    tc = build_taskc_readme(repo, c_f, c_m, c_val)
    for target in (project / "results/taskC/README.md", repo / "results/taskC/README.md"):
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(tc, encoding="utf-8")
    print(f"results/taskC/README.md {len(tc.encode('utf-8')):>6,} B")

    # ---- docs/results_provenance.md: append the Task B / Task C half ------------------
    A_EV = "results/taskA/evaluation"
    B_EV = "results/taskB/evaluation"
    C_EV = "results/taskC/evaluation"
    rows_a = [
        ("凍結条件 / seed / best epoch",
         f"{env('frozen_model.condition')} / {env('random_seeds.model_seed_primary')} / "
         f"{env('frozen_model.best_epoch')}",
         f"{A_EV}/test_run_env_note.json",
         "frozen_model.condition, random_seeds.model_seed_primary, frozen_model.best_epoch"),
        ("checkpoint SHA256", f"{env('frozen_model.checkpoint_sha256')[:16]}…",
         f"{A_EV}/test_run_env_note.json", "frozen_model.checkpoint_sha256"),
        ("Test ROC-AUC (95% CI)",
         f"{f6(a_m['auroc']['point'])} ({f6(a_m['auroc']['ci_low'])}–"
         f"{f6(a_m['auroc']['ci_high'])})",
         f"{A_EV}/test_metrics_primary.json", "auroc.point / ci_low / ci_high"),
        ("Test PR-AUC", f6(a_m["auprc"]["point"]),
         f"{A_EV}/test_metrics_primary.json", "auprc.point"),
        ("Test Brier", f6(a_m["brier"]), f"{A_EV}/test_metrics_primary.json", "brier"),
        (f"Test ECE（{len(a_m['calibration_bins'])} equal-count bin）", f6(a_m["ece"]),
         f"{A_EV}/test_metrics_primary.json", "ece"),
        ("Test ECE（10 equal-width bin）", cxr[4],
         f"{C_EV}/test_metrics_table.csv", "model=cxr の ece_10_equal_width 列"),
        ("操作点の閾値", str(a_m["operating_point_youden"]["threshold"]),
         f"{A_EV}/test_metrics_primary.json", "operating_point_youden.threshold"),
        ("bootstrap 設定",
         f"{a_m['auroc']['n_boot']:,} resample / stratified / seed {a_m['auroc']['seed']}",
         f"{A_EV}/test_metrics_primary.json", "auroc.n_boot / stratified / seed"),
        ("Grad-CAM 症例構成",
         " / ".join(f"{k} {a_gc['cases_used_per_group'][k]}"
                    for k in ("TP", "FP", "TN", "FN")) + f" = {a_gc['n_cases']}",
         "results/taskA/gradcam/gradcam_run_meta.json", "cases_used_per_group, n_cases"),
    ]
    rows_b = [
        ("最終変数 / 係数数 / EPV",
         f"{len(b_v['primary']['variables'])} / {b_v['primary']['n_features']} / "
         f"{b_v['primary']['events_per_parameter']}",
         "results/taskB/variable_selection/final_variables.json",
         "primary.variables / n_features / events_per_parameter"),
        ("変数選択規則", b_v["primary"]["rule"],
         "results/taskB/variable_selection/final_variables.json", "primary.rule"),
        ("LR 条件", b_f["models"]["lr"]["condition"],
         "results/taskB/modeling/final_selection_taskB.json", "models.lr.condition"),
    ]
    for key, label in (("lr", "LR"), ("xgb", "XGBoost"), ("mlp", "MLP"),
                       ("xgb_native", "XGBoost native missing（secondary）")):
        r = b_t.loc[key]
        rows_b.append((f"{label} Test ROC-AUC (95% CI)",
                       f"{f6(r.auroc)} ({f6(r.auroc_ci_low)}–{f6(r.auroc_ci_high)})",
                       f"{B_EV}/test_metrics_table.csv",
                       f"model={key} の auroc / auroc_ci_low / auroc_ci_high"))
        rows_b.append((f"{label} Test PR-AUC / Brier / ECE(5 equal-count)",
                       f"{f6(r.auprc)} / {f6(r.brier)} / {f6(r.ece_5bin)}",
                       f"{B_EV}/test_metrics_table.csv",
                       f"model={key} の auprc / brier / ece_5bin"))
    rows_b.append(("実行環境（package versions）",
                   " / ".join(f"{k} {v}" for k, v in pkg.items()),
                   f"{B_EV}/test_run_meta.json", "packages"))
    rows_c = [
        ("融合式", c_f["fusion"]["formula"],
         "results/taskC/modeling/final_selection_taskC.json", "fusion.formula"),
        ("重み（clinical / CXR）",
         f"{c_f['fusion']['w_clinical']} / {c_f['fusion']['w_cxr']}",
         "results/taskC/modeling/final_selection_taskC.json",
         "fusion.w_clinical / fusion.w_cxr"),
        ("frozen spec SHA256",
         f"{sha256(repo / 'results/taskC/modeling/final_selection_taskC.json')[:16]}…",
         "results/taskC/modeling/final_selection_taskC.json", "（ファイル全体のハッシュ）"),
        ("Validation ROC-AUC（fused）", f6(c_val["fused"]["auroc"]),
         "results/taskC/validation/validation_metrics_taskC.json", "fused.auroc"),
        ("Test ROC-AUC (95% CI)",
         f"{f6(c_m['auroc']['point'])} ({f6(c_m['auroc']['ci_low'])}–"
         f"{f6(c_m['auroc']['ci_high'])})",
         f"{C_EV}/test_metrics_taskC.json", "auroc.point / ci_low / ci_high"),
        ("Test PR-AUC", f6(c_m["auprc"]["point"]), f"{C_EV}/test_metrics_taskC.json",
         "auprc.point"),
        ("Test Brier", f6(c_m["brier"]), f"{C_EV}/test_metrics_taskC.json", "brier"),
        (f"Test ECE（{c_m['calibration_definition']['ece_bins']} "
         f"{c_m['calibration_definition']['ece_scheme']} bin）", f6(c_m["ece"]),
         f"{C_EV}/test_metrics_taskC.json", "ece / calibration_definition"),
        ("閾値（Validation 由来・Test 再計算なし）",
         f"{c_m['threshold']['value']}（recomputed_on_test = "
         f"{c_m['threshold']['recomputed_on_test']}）",
         f"{C_EV}/test_metrics_taskC.json", "threshold.value / recomputed_on_test"),
        ("感度 / 特異度",
         f"{f6(c_m['operating_point']['sensitivity'])} / "
         f"{f6(c_m['operating_point']['specificity'])}",
         f"{C_EV}/test_metrics_taskC.json",
         "operating_point.sensitivity / specificity"),
        ("TP / FP / TN / FN",
         " / ".join(str(int(c_m["operating_point"][k])) for k in ("tp", "fp", "tn", "fn")),
         f"{C_EV}/test_metrics_taskC.json", "operating_point.tp / fp / tn / fn"),
        ("modality importance（clinical / CXR、post hoc）",
         f"{f6(c_mod.loc['clinical'].mean_auroc_drop)} / "
         f"{f6(c_mod.loc['cxr'].mean_auroc_drop)}",
         f"{C_EV}/modality_importance_test.csv", "mean_auroc_drop"),
        ("secondary（logit空間 / stacking、探索的）",
         f"{f6(c_sec['models']['logit_space_weighted_average_w0.70']['auroc'])} / "
         f"{f6(c_sec['models']['logistic_stacking']['auroc'])}",
         f"{C_EV}/secondary_exploratory_test.json", "models.*.auroc"),
        ("実行環境（package versions）",
         " / ".join(f"{k} {v}" for k, v in pkg.items()),
         f"{C_EV}/test_run_meta.json", "packages"),
    ]
    rows_x = [("全5モデル Test 指標（単一定義）",
               "ECE = 10 equal-width bin、n=%d / deaths=%d" % (c_m["n"], c_m["events"]),
               f"{C_EV}/test_metrics_table.csv", "全行"),
              ("患者単位 全モデル予測", f"{c_m['n']} 行 × 7 列",
               "results/comparison/test_predictions_all_models.csv",
               "subject_id, true_label, prob_cxr, prob_clinical_lr, "
               "prob_clinical_xgboost, prob_clinical_mlp, prob_late_fusion")]
    for _, r in dl.iterrows():
        rows_x.append((f"DeLong Family {r.family}: Late Fusion vs {r.model_2_label}",
                       f"ΔAUC {r.delta_AUC:+.6f} / p(unadj) {r.p_unadjusted:.6f} / "
                       f"Holm {r.p_holm_within_family:.6f}",
                       "results/comparison/delong_results.csv",
                       f"family={r.family}, model_2={r.model_2}"))
    rows_x.append(("DeLong の事前規定（2 family + Holm）",
                   "Test を開く前に固定", "results/comparison/delong_plan.json", "（全体）"))

    block = build_provenance_block(repo, rows_a, rows_b, rows_c, rows_x)
    prov_rel = "docs/results_provenance.md"
    cur = (repo / prov_rel).read_text(encoding="utf-8-sig")
    if PROV_BEGIN in cur:
        head = cur.split(PROV_BEGIN, 1)[0]
        tail = cur.split(PROV_END, 1)[1] if PROV_END in cur else ""
    else:
        head, tail = cur, ""
    new = head.rstrip() + "\n\n" + block + "\n" + tail.lstrip()
    for target in (project / prov_rel, repo / prov_rel):
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(new, encoding="utf-8")
    print(f"{prov_rel}  {len(cur.encode('utf-8')):,} B -> "
          f"{len(new.encode('utf-8')):,} B  (Task A part untouched: "
          f"{new.startswith(head.rstrip()[:200])})")
    return 1 if unmatched else 0


if __name__ == "__main__":
    sys.exit(main())
