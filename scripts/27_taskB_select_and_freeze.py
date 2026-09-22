"""Task B step 7: refit the winning condition per family, score Validation, freeze everything.

What each split does here (decision_log D-071):
  Training    refits the winning condition of each family; the preprocessor is fitted here.
  Validation  compares the families, fixes the classification threshold, and nothing else.
  Test        not read. This script must run to completion before Test is ever opened.

The frozen JSON records the feature list, the hyperparameters, the thresholds and the SHA256
of every saved artefact, so the Test script can verify that nothing changed in between.

Usage:
    python scripts/27_taskB_select_and_freeze.py --project .
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from covid_mortality.evaluation import taskB_schema as sch  # noqa: E402
from covid_mortality.evaluation.metrics import (average_precision, binary_rates, brier_score,  # noqa: E402
                                                roc_auc, threshold_at_min_sensitivity,
                                                youden_threshold)

CALIBRATION_BINS = 5


def calibration_table(y: np.ndarray, p: np.ndarray, n_bins: int = CALIBRATION_BINS) -> pd.DataFrame:
    """Equal-count (quantile) bins, identical to the Task A implementation in scripts/15."""
    edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    idx = np.digitize(p, edges[1:-1], right=True)
    rows = []
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        rows.append({"bin": b, "n": int(m.sum()), "mean_predicted": float(p[m].mean()),
                     "observed_rate": float(y[m].mean()), "events": int(y[m].sum())})
    return pd.DataFrame(rows)


def expected_calibration_error(cal: pd.DataFrame, n: int) -> float:
    return float((cal.n / n * (cal.mean_predicted - cal.observed_rate).abs()).sum())
from covid_mortality.features import taskB_clinical as tb  # noqa: E402
from covid_mortality.features.taskB_preprocess import TaskBPreprocessor  # noqa: E402
from covid_mortality.training import taskB_models as tm  # noqa: E402

warnings.filterwarnings("ignore")
SEED = 42
FAMILIES = ["lr", "xgb", "mlp"]
SECONDARY = ["xgb_native"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def best_row(modeling: Path, name: str) -> pd.Series:
    df = pd.read_csv(modeling / f"{name}_search.csv", encoding="utf-8-sig")
    return df.sort_values("cv_mean_auroc", ascending=False).iloc[0]


def refit(name: str, row: pd.Series, train: pd.DataFrame, feats: list[str], out: Path):
    """Refit on the whole Training set and return (predict_fn, artefact_path, params)."""
    native = name.endswith("native")
    use_feats = [f for f in feats if not f.endswith("_missing")] if native else feats
    pre = TaskBPreprocessor(native_missing=native, scale=not native).fit(train)
    Xtr = pre.transform(train)[use_feats].to_numpy(float)
    y = train.y.values
    pre.save(out / f"preprocessor_{name}.json")

    if name == "lr":
        cw = None if row.class_weight in ("None", "nan") else row.class_weight
        params = {"C": float(row.C), "class_weight": cw, "penalty": "l2", "solver": "lbfgs"}
        _, make = tm.logistic_factory(params["C"], cw)
        model = make().fit(Xtr, y)
        import joblib
        path = out / "model_lr.joblib"
        joblib.dump(model, path)
        return (lambda X: model.predict_proba(X)[:, 1]), path, params, pre, use_feats

    if name.startswith("xgb"):
        params = {k: (int(row[k]) if k in ("max_depth", "min_child_weight") else float(row[k]))
                  for k in ("max_depth", "learning_rate", "min_child_weight", "subsample",
                            "colsample_bytree", "scale_pos_weight", "reg_lambda")}
        _, make = tm.xgb_factory(params, int(json.loads(
            (out / "search_summary.json").read_text(encoding="utf-8"))["grids"]["xgb_n_estimators"]))
        model = make().fit(Xtr, y, verbose=False)
        path = out / f"model_{name}.json"
        model.get_booster().save_model(str(path))
        return (lambda X: model.predict_proba(X)[:, 1]), path, params, pre, use_feats

    # mlp
    params = {"hidden": eval(row.hidden), "dropout": float(row.dropout), "lr": float(row.lr),
              "weight_decay": float(row.weight_decay), "batch_size": int(row.batch_size),
              "pos_weight": float(row.pos_weight)}
    _, trainer = tm.mlp_factory(**params)
    _, model, best_epoch = trainer(Xtr, y, Xtr)
    import torch
    path = out / "model_mlp.pth"
    torch.save(model.state_dict(), path)
    params["best_epoch"] = int(best_epoch)

    def predict(X):
        model.eval()
        with torch.no_grad():
            return torch.sigmoid(model(torch.tensor(X, dtype=torch.float32))).squeeze(1).numpy()

    return predict, path, params, pre, use_feats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=".")
    args = ap.parse_args()
    project = Path(args.project).resolve()
    modeling = project / "results/taskB/modeling"
    df = tb.load_development(project)   # Training + Validation only
    train, val = tb.training_only(df), df[df.split == "val"].copy()
    feats = json.loads((project / "results/taskB/variable_selection/final_variables.json")
                       .read_text(encoding="utf-8"))["primary"]["features"]
    print(f"Training {len(train)} (deaths {int(train.y.sum())}), "
          f"Validation {len(val)} (deaths {int(val.y.sum())}). Test not read.\n")

    yv = val.y.values
    rows, frozen, preds = [], {}, {"subject_id": val[tb.ID_COLUMN].values, "true_label": yv}
    for name in FAMILIES + SECONDARY:
        row = best_row(modeling, name)
        predict, path, params, pre, use_feats = refit(name, row, train, feats, modeling)
        pv = predict(pre.transform(val)[use_feats].to_numpy(float))
        preds[f"prob_{name}"] = pv
        thr_y = youden_threshold(yv, pv, tie_break="lowest")
        thr_s = threshold_at_min_sensitivity(yv, pv, 0.80)
        rates = binary_rates(yv, pv, thr_y)
        cal = calibration_table(yv, pv)
        cal.assign(model=name).to_csv(modeling / f"validation_calibration_{name}.csv",
                                      index=False, encoding="utf-8-sig")
        ece = expected_calibration_error(cal, len(yv))
        rows.append({"model": name, "role": "primary" if name in FAMILIES else "secondary",
                     "condition": row.condition,
                     "cv_mean_auroc": float(row.cv_mean_auroc),
                     "cv_sd_auroc": float(row.cv_sd_auroc),
                     "val_auroc": round(float(roc_auc(yv, pv)), 6),
                     "val_auprc": round(float(average_precision(yv, pv)), 6),
                     "val_brier": round(float(brier_score(yv, pv)), 6),
                     "val_ece_5bin": round(float(ece), 6),
                     "threshold_youden": round(float(thr_y), 9),
                     "threshold_sens80": round(float(thr_s), 9),
                     "val_sensitivity_at_youden": round(rates["sensitivity"], 4),
                     "val_specificity_at_youden": round(rates["specificity"], 4),
                     "val_ppv_at_youden": round(rates["ppv"], 4),
                     "val_npv_at_youden": round(rates["npv"], 4),
                     "n_features": len(use_feats)})
        frozen[name] = {"condition": row.condition, "params": params,
                        "features": use_feats, "n_features": len(use_feats),
                        "artefact": path.name, "artefact_sha256": sha256(path),
                        "preprocessor": f"preprocessor_{name}.json",
                        "preprocessor_sha256": sha256(modeling / f"preprocessor_{name}.json"),
                        "threshold_youden": float(thr_y), "threshold_sens80": float(thr_s),
                        "cv_mean_auroc": float(row.cv_mean_auroc),
                        "val_auroc": float(roc_auc(yv, pv))}
        print(f"{name:11s} CV {row.cv_mean_auroc:.4f}  Validation AUROC "
              f"{roc_auc(yv, pv):.4f}  AUPRC {average_precision(yv, pv):.4f}  "
              f"Brier {brier_score(yv, pv):.4f}  thr {thr_y:.6f}")

    res = pd.DataFrame(rows)
    res.to_csv(modeling / "validation_comparison.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(preds).to_csv(modeling / "validation_predictions.csv", index=False,
                               encoding="utf-8-sig")
    print("\n=== Validation comparison ===")
    print(res.to_string(index=False))

    primary = res[res.role == "primary"].sort_values("val_auroc", ascending=False).iloc[0]
    print(f"\nbest primary family on Validation: {primary.model} "
          f"(AUROC {primary.val_auroc:.4f})")

    # ---- plans fixed before the Test set is opened ------------------------------------
    importance_plan = {
        "primary_method": "permutation importance",
        "primary_rationale": ("the only model-agnostic measure that puts a linear model, a tree "
                              "ensemble and a neural network on the same scale, so the three "
                              "families can be compared fairly"),
        "dataset": "Validation (the Test set is not used for interpretability)",
        "metric": "decrease in ROC-AUC when one feature is permuted",
        "n_repeats": 30, "seed": SEED,
        "secondary_per_family": {
            "lr": "standardised coefficients and odds ratios with 95% CI",
            "xgb": "native gain-based importance",
            "mlp": "none beyond permutation importance"},
        "not_used": {"SHAP": "values are not comparable across model families in one scale, "
                             "so it cannot serve the primary cross-model comparison"},
        "caveats": [
            "Feature importance is not causal importance.",
            "Correlated predictors share importance, so a low value does not mean a variable "
            "is clinically unimportant. In this feature set bun/egfr and crp/ddimer/lymph are "
            "correlated.",
            "Permutation importance is computed on 128 Validation patients with 17 events, so "
            "individual values carry wide uncertainty.",
        ],
        "output": "results/taskB/feature_importance/permutation_importance_validation.csv",
    }
    prediction_schema = {
        "file": sch.TASKB_TEST_PREDICTIONS,
        "required_columns": sch.TASKB_REQUIRED,
        "optional_columns": sch.TASKB_OPTIONAL,
        "join_key": sch.ID_COLUMN, "id_dtype": "str",
        "one_row_per_patient": True,
        "patient_order": "fixed split manifest order, filtered to split == 'test'",
        "patient_order_sha256": hashlib.sha256(
            "\n".join(sch.patient_order(tb.split_manifest(project), "test")).encode()).hexdigest(),
        "constraints": ["no duplicate subject_id", "no missing prediction",
                        "probabilities in [0,1]", "true_label verified against the fixed split"],
        "merge_target": sch.COMBINED_PREDICTIONS,
        "merge_partners": ["results/taskA/evaluation/test_predictions_primary.csv (prob_cxr)",
                           "Task C Late Fusion (prob_late_fusion)"],
        "status": "fixed before the Test evaluation; the file does not exist yet",
    }

    final = {
        "frozen": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "script": Path(__file__).name, "seed": SEED,
        "cohort": {"train": len(train), "val": len(val),
                   "deaths_train": int(train.y.sum()), "deaths_val": int(val.y.sum())},
        "features_primary": feats, "n_features_primary": len(feats),
        "selection_rule": ("hyperparameters by repeated stratified CV inside Training; "
                           "family comparison and threshold on Validation; Test unused"),
        "threshold_rule": ("primary: Validation Youden maximum, tie-break lowest threshold. "
                           "secondary (exploratory): highest threshold with Validation "
                           "sensitivity >= 0.80. Neither is recomputed on Test."),
        "models": frozen,
        "best_primary_family_on_validation": primary.model,
        "report_all_three": True,
        "candidate_variables": [v.name for v in tb.CANDIDATES],
        "missing_indicator_rule": {
            "adopted": ["lymph"],
            "wording": "pre-specified before variable selection in this Task B run",
            "why_kept_in_primary": [
                "the rule was fixed before variable selection was run and is applied consistently",
                "the specification is not changed on the basis of Validation or Test performance",
                "the size of its effect is made explicit by the arm A/B/C sensitivity analysis"],
            "not_because": "it produced a higher Validation AUROC",
            "sensitivity": "results/taskB/variable_selection/indicator_rule_sensitivity.csv"},
        "variable_selection": {
            "method": "backward elimination by AIC, subject to the parameter budget",
            "unit": "original clinical variable (all dummies of a categorical variable move together)",
            "dataset": "Training set only (n=1021, 135 deaths)",
            "validation_used": False, "test_used": False,
            "statement": ("Variable selection was performed by backward elimination by AIC on "
                          "the Training set only. Neither the Validation set nor the Test set "
                          "was used for variable selection."),
            "stability": "500 bootstrap resamples of Training; reported, not used to select"},
        "analysis_population": {
            "primary": "the fixed cohort of 1,277 patients, identical to Task A and Task C",
            "rationale": ("keeping one patient set across Task A, B and C is what makes the "
                          "paired DeLong comparison and the Late Fusion analysis possible"),
            "composition_note": ("the cohort mixes inpatient-coded encounters with emergency "
                                 "department encounters that were discharged; this is stated in "
                                 "the cohort description and the limitations"),
            "inpatient_only_sensitivity": "not performed (decision_log D-081)"},
        "secondary_analyses": {
            "xgb_native": ("pre-specified secondary model. Native missing handling did not "
                           "improve cross-validated performance compared with the "
                           "imputation-based approach in this analysis.")},
        "future_analysis_candidates": [
            {"analysis": "inpatient-only analysis restricted to visit_concept_name == "
                         "'Inpatient Visit' (955 of 1,277 patients)",
             "purpose": "separate discrimination of illness severity from reproduction of the "
                        "admit-or-discharge triage decision",
             "status": "not performed; population-definition tables only, kept unused in "
                       "results/taskB/secondary_exploratory_unused/",
             "reference": "decision_log D-081"}],
        "feature_importance_plan": importance_plan,
        "prediction_schema": prediction_schema,
        "delong_plan": sch.DELONG_PLAN,
        "test_status": ("Test predictions and Test performance metrics have not been generated. "
                        "The Test set had previously been accessed for limited QC purposes, but "
                        "no Test information was used for preprocessing fitting, variable "
                        "selection, hyperparameter tuning, checkpoint selection, or model "
                        "selection."),
        "test_access_log": "results/taskB/qc/test_access_log.json",
        "test_predictions_generated": False,
    }
    p = modeling / "final_selection_taskB.json"
    p.write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nfrozen -> {p}\nSHA256 {sha256(p)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
