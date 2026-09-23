"""Task C step 3: freeze the complete specification before the Test set is opened.

Everything the Test evaluation will need is written into one machine-readable file, together
with the SHA256 of every input it depends on, so a later run can prove it used the same
specification. Nothing here is decided; it only records what steps 1-2 already fixed.

Output:
    results/taskC/modeling/final_selection_taskC.json

Usage:
    python scripts/37_taskC_freeze.py --project . --taskA-runs <...>
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from covid_mortality.evaluation import taskB_schema as sch  # noqa: E402
from covid_mortality.fusion import taskC_fusion as tc  # noqa: E402

ECE_BINS, ECE_SCHEME = 10, "equal_width"
BOOTSTRAP = {"n_resamples": 2000, "stratified": True, "seed": 12345}
SEED = 42

TASKC_TEST_PREDICTIONS = "results/taskC/evaluation/test_predictions_taskC.csv"
COMBINED = "results/comparison/test_predictions_all_models.csv"
COMBINED_REQUIRED = ["subject_id", "true_label", "prob_cxr", "prob_clinical_lr",
                     "prob_clinical_xgboost", "prob_clinical_mlp", "prob_late_fusion"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=".")
    ap.add_argument("--taskA-runs", required=True)
    ap.add_argument("--taskA-eval", default="github_repo/results/taskA/evaluation")
    args = ap.parse_args()
    project, runs = Path(args.project).resolve(), Path(args.taskA_runs)
    a_eval = project / args.taskA_eval
    out = project / "results/taskC/modeling"
    out.mkdir(parents=True, exist_ok=True)

    sel = json.loads((project / "results/taskC/fusion/selected_weight.json")
                     .read_text(encoding="utf-8"))
    vm = json.loads((project / "results/taskC/validation/validation_metrics_taskC.json")
                    .read_text(encoding="utf-8"))
    taskB = json.loads((project / "results/taskB/modeling/final_selection_taskB.json")
                       .read_text(encoding="utf-8"))
    split = pd.read_csv(project / "data/splits/COVID19_固定患者split_1277.csv",
                        dtype=str, encoding="utf-8-sig")

    inputs = {
        "taskA_frozen_selection": runs / "final_selection.json",
        "taskA_val_predictions": runs / "lr3e-4_aug_b/seed42/val_predictions.csv",
        "taskA_test_predictions": a_eval / "test_predictions_primary.csv",
        "taskB_frozen_selection": project / "results/taskB/modeling/final_selection_taskB.json",
        "taskB_val_predictions": project / "results/taskB/modeling/validation_predictions.csv",
        "taskB_test_predictions": project / "results/taskB/evaluation/test_predictions_taskB.csv",
        "split_manifest": project / "data/splits/COVID19_固定患者split_1277.csv",
        "taskC_selected_weight": project / "results/taskC/fusion/selected_weight.json",
        "taskC_validation_metrics": project / "results/taskC/validation/validation_metrics_taskC.json",
    }
    src = {k: {"path": str(p).replace(str(project) + "\\", ""), "sha256": tc.sha256(p)}
           for k, p in inputs.items() if p.exists()}
    missing = [k for k, p in inputs.items() if not p.exists()]
    if missing:
        print(f"STOP: missing inputs {missing}")
        return 1

    frozen = {
        "frozen": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "script": Path(__file__).name, "seed": SEED,
        "task": "Task C -- multimodal Late Fusion of the frozen Task A CXR model and a frozen "
                "Task B clinical model",
        "cohort": {"n": 1277, "train": 1021, "val": 128, "test": 128,
                   "deaths": {"train": 135, "val": 17, "test": 17},
                   "split": "identical to Task A and Task B"},

        "cxr_component": {
            "source": "Task A frozen primary model",
            "condition": "lr3e-4_aug_b", "seed": 42, "best_epoch": 9,
            "retrained_for_taskC": False,
            "used_via": "the predictions Task A already saved; the checkpoint is not reloaded",
            "validation_auroc": vm["components"]["cxr"]["auroc"]},

        "clinical_component": {
            "source": "Task B frozen model", "model": sel["clinical_component"],
            "condition": taskB["models"][sel["clinical_component"]]["condition"],
            "artefact": taskB["models"][sel["clinical_component"]]["artefact"],
            "artefact_sha256": taskB["models"][sel["clinical_component"]]["artefact_sha256"],
            "retrained_for_taskC": False,
            "selection_rule": "best Training cross-validated ROC-AUC among the Task B clinical "
                              "models; fixed before any fusion was computed",
            "dataset_used_for_selection": "Training cross-validation",
            "validation_auroc": vm["components"][sel["clinical_component"]]["auroc"]},

        "fusion": {
            "primary_strategy": sel["primary_strategy"],
            "formula": "p_fused = w * p_clinical + (1 - w) * p_cxr",
            "w_clinical": sel["w_clinical"], "w_cxr": sel["w_cxr"],
            "search": sel["selection_rule"],
            "n_rows_tied": sel["n_rows_tied_on_all_metrics"],
            "weight_curve_flatness": sel["weight_curve_flatness"],
            "strategy_fixed_before_search": True,
            "secondary_not_competing": [
                "logit-space weighted average", "logistic stacking",
                "probability-space weighted average with the XGBoost or MLP component"],
            "note": ("secondary strategies scored higher on Validation than the primary one. "
                     "The primary strategy was fixed before the search and is not changed on "
                     "that basis; the secondary results are reported, not adopted.")},

        "threshold": {"primary": vm["thresholds"]["youden"],
                      "rule": vm["thresholds"]["rule"],
                      "secondary_exploratory": vm["thresholds"]["sensitivity80"],
                      "recomputed_on_test": False},

        "calibration_definition": {
            "brier": "standard Brier score",
            "ece_bins": ECE_BINS, "ece_scheme": ECE_SCHEME,
            "statement": f"ECE with {ECE_BINS} equal-width bins over [0,1]",
            "implementation": "src/covid_mortality/evaluation/metrics.py :: "
                              "expected_calibration_error(y, p, n_bins, scheme)",
            "same_on_validation_and_test": True},

        "bootstrap": BOOTSTRAP,
        "bootstrap_note": ("identical to Task A and Task B. The human-guided pipeline used "
                           "10,000 resamples with seed 42; any cross-pipeline CI comparison "
                           "needs a separate harmonised artefact"),

        "validation_results": {"fused": vm["fused"], "components": vm["components"],
                               "operating_point_youden": vm["operating_point_youden"]},

        "modality_importance_plan": {
            "method": "modality-level permutation importance",
            "n_repeats": 30, "seed": SEED, "metric": "ROC-AUC decrease",
            "validation": "done (results/taskC/validation/modality_importance_validation.csv)",
            "test": "post hoc descriptive only; may not change any specification"},

        "prediction_schema": {
            "taskC_file": TASKC_TEST_PREDICTIONS,
            "primary_fusion_column": "prob_late_fusion",
            "combined_file": COMBINED,
            "combined_required_columns": COMBINED_REQUIRED,
            "join_key": "subject_id", "id_dtype": "str",
            "patient_order": "fixed split manifest order, filtered to split == 'test'",
            "patient_order_sha256": tc.sha256(inputs["split_manifest"]),
            "constraints": ["128 patients", "unique subject_id", "no missing probability",
                            "probabilities in [0,1]", "labels verified against the fixed split",
                            "deterministic merge on subject_id"]},

        "delong_plan": json.loads((project / "results/comparison/delong_plan.json")
                                  .read_text(encoding="utf-8"))["families"],
        "delong_plan_file": "results/comparison/delong_plan.json",

        "input_sha256": src,
        "test_status": ("Test predictions and Test performance metrics have not been generated "
                        "for Task C. The Task A and Task B Test files exist and were hashed, but "
                        "no Task C design decision used them."),
        "test_predictions_generated": False,
    }
    p = out / "final_selection_taskC.json"
    p.write_text(json.dumps(frozen, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"frozen -> {p}")
    print(f"SHA256 {tc.sha256(p)}")
    print(f"\nprimary: {frozen['fusion']['primary_strategy']}, clinical = "
          f"{frozen['clinical_component']['model'].upper()}, "
          f"w_clinical = {frozen['fusion']['w_clinical']}, "
          f"w_cxr = {frozen['fusion']['w_cxr']}")
    print(f"threshold {frozen['threshold']['primary']:.6f} (Validation Youden, frozen)")
    print(f"Validation fused AUROC {frozen['validation_results']['fused']['auroc']:.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
