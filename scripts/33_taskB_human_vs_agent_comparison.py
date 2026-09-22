"""Task B: assemble the human-guided vs AI-agent comparison tables.

Descriptive only. Nothing is refit, no variable, hyperparameter, threshold or model choice is
revisited, and no statistical test is run. Every number is copied from an authoritative result
artefact and carries the file it came from.

Human-guided authoritative sources (Google Drive, 気合のCOVID19/02_TaskB_臨床データ):
    05_Evaluation/TaskB_3モデル_最終Test評価比較.csv      Test metrics for all three models
    04_Modeling/TaskB_LR_8条件_Validation比較.csv          LR Validation
    04_Modeling/TaskB_XGBoost_Stage2_16条件比較.csv        XGBoost Validation
    04_Modeling/TaskB_MLP_限定再評価_4条件比較.csv          MLP Validation
    05_Evaluation/TaskB_{LR,XGBoost,MLP}_最終Test評価結果.csv   hyperparameters as evaluated
    04_Modeling/TaskB_LR_bestmodel_係数_OR.csv             the variables actually fitted

AI-agent sources (this repository):
    results/taskB/modeling/final_selection_taskB.json      frozen specification
    results/taskB/modeling/validation_comparison.csv       Validation
    results/taskB/evaluation/test_metrics_table.csv        Test

Usage:
    python scripts/33_taskB_human_vs_agent_comparison.py --project . \
        --drive-root "<DRIVE_ROOT>"
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

H_TEST = "02_TaskB_臨床データ/05_Evaluation/TaskB_3モデル_最終Test評価比較.csv"
H_LR_VAL = "02_TaskB_臨床データ/04_Modeling/TaskB_LR_8条件_Validation比較.csv"
H_XGB_VAL = "02_TaskB_臨床データ/04_Modeling/TaskB_XGBoost_Stage2_16条件比較.csv"
H_MLP_VAL = "02_TaskB_臨床データ/04_Modeling/TaskB_MLP_限定再評価_4条件比較.csv"
H_LR_COEF = "02_TaskB_臨床データ/04_Modeling/TaskB_LR_bestmodel_係数_OR.csv"
H_FIVE = "04_全モデル比較・統計解析/全5モデル_最終Test評価比較.csv"
H_NAMES = {"LR": "TaskB_LogisticRegression", "XGBoost": "TaskB_XGBoost", "MLP": "TaskB_MLP"}
A_NAMES = {"LR": "lr", "XGBoost": "xgb", "MLP": "mlp"}


def spec_rows() -> list[dict]:
    """Design comparison. 'human' and 'agent' are quoted from the artefacts listed above."""
    S_H = "Human-guided final result artifact"
    S_A = "AI-Agent frozen Task B result"
    R = lambda c, i, h, a, k, im, sh=S_H, sa=S_A: {  # noqa: E731
        "category": c, "item": i, "human_guided": h, "ai_agent": a,
        "key_difference": k, "possible_methodological_implication": im,
        "source_human": sh, "source_ai_agent": sa}
    return [
        R("cohort", "patients", "1,277 (fixed)", "1,277 (fixed)", "none",
          "the two pipelines are directly comparable on the same patients"),
        R("cohort", "split", "Train 1,021 / Val 128 / Test 128", "Train 1,021 / Val 128 / Test 128",
          "none", "same split, so Validation and Test estimates refer to the same patients"),
        R("cohort", "deaths", "135 / 17 / 17", "135 / 17 / 17", "none", "identical event counts"),
        R("variables", "columns audited", "131", "131", "none",
          "both started from the whole clinical table"),
        R("variables", "candidate raw variables", "14", "13", "one fewer candidate in the agent set",
          "a smaller candidate set means fewer chances for selection to overfit, but also fewer "
          "clinical domains offered to the model"),
        R("variables", "final raw variables", "9", "10", "+1",
          "both end near the events-per-parameter budget"),
        R("variables", "final input features", "12", "12 (including lymph_missing)", "same count, different content",
          "the coefficient budget is the same; what occupies it differs"),
        R("variables", "final variable list",
          "age; sex; heart failure; SpO2; lymphocyte; eGFR; CRP; D-dimer; lactate",
          "age; sex; SpO2; SBP; CRP; lymphocyte; D-dimer; lactate; eGFR; troponin detectable "
          "(+ lymph_missing)",
          "human-guided only: heart failure. AI-agent only: SBP, troponin detectable, lymph_missing",
          "the two pipelines cover different clinical domains: the human-guided set carries a "
          "chronic cardiac comorbidity, the agent set carries haemodynamics, myocardial injury "
          "and a missingness marker"),
        R("variables", "selection method", "backward elimination by AIC (Train only)",
          "backward elimination by AIC (Train only), then a parameter-budget constraint",
          "the agent adds an explicit coefficient budget on top of AIC",
          "AIC alone can exceed an events-per-parameter target; the budget makes that limit binding"),
        R("variables", "selection stability", "not assessed",
          "500 bootstrap resamples of Training; retention rate reported per variable",
          "stability quantified only in the agent pipeline",
          "reported, not used to select; it shows how reproducible the chosen set is"),
        R("variables", "selection unit", "original clinical variable (dummies as a group)",
          "original clinical variable (dummies as a group)", "none",
          "both answer 'do we use age?' rather than 'do we use one age dummy?'"),
        R("preprocessing", "abnormal-value handling",
          "not documented in the available artefacts",
          "7 physiologically impossible values set to missing (RR 67/88/95, HR 6/16, "
          "BMI 11.95/92.8); patients never dropped; no blanket outlier removal",
          "explicit and recorded in the agent pipeline, not found for the human-guided one",
          "an undocumented step cannot be reproduced or audited; this is a documentation "
          "difference, not necessarily an analytic one"),
        R("preprocessing", "continuous imputation", "Train median for all continuous variables",
          "per variable: Train mean when |skew| < 0.5, otherwise Train median",
          "the agent chooses the statistic per variable from the Training distribution",
          "the median is robust for skewed variables; the mean is more efficient for symmetric "
          "ones. The practical difference is small when most variables are skewed"),
        R("preprocessing", "missing indicator", "none for continuous variables",
          "lymph_missing only, by criteria fixed before selection was run",
          "the agent adds exactly one indicator",
          "its adjusted coefficient was near zero, so it mainly occupies a coefficient; an "
          "arm A/B/C sensitivity analysis quantifies the effect"),
        R("preprocessing", "transformation", "log1p on lymphocyte, CRP, lactate",
          "log1p on variables with Training skew >= +1.0 (RR, BUN, CRP, lymphocyte, D-dimer, lactate)",
          "the agent applies a stated rule and restricts it to right skew",
          "applying log to a left-skewed variable worsens it; SpO2 (skew -2.10) is left untransformed "
          "in the agent pipeline"),
        R("preprocessing", "scaling", "StandardScaler fitted on Train",
          "StandardScaler fitted on Train for LR and MLP; trees use raw values",
          "the agent does not scale inputs to the tree model",
          "trees are invariant to monotone transforms, so this does not change the tree fit"),
        R("preprocessing", "categorical encoding",
          "one-hot, reference age [18,59] / sex FEMALE / hf No",
          "one-hot, drop_first, reference age [18,59] / sex FEMALE / troponin undetectable",
          "same scheme, different variables encoded",
          "both give odds ratios relative to a low-risk reference"),
        R("preprocessing", "sex missing", "imputed with the Train mode (MALE); no Missing category",
          "imputed with the Train mode (MALE); no missing indicator, by explicit decision",
          "same handling; the agent records why",
          "19 Training patients lack sex and 18 of them died; an indicator would encode that "
          "association, which may reflect a nonclinical missingness mechanism"),
        R("preprocessing", "heart failure missing", "explicit Missing category (hf_Missing)",
          "heart failure not in the final model",
          "only the human-guided model carries a missingness level as a predictor",
          "hf_Missing behaves as a missing indicator for one variable while other variables have none"),
        R("preprocessing", "troponin", "candidate, removed by backward AIC; treated as continuous",
          "treated as detectable / undetectable and retained",
          "different representation of the same measurement",
          "78.2% of Training values sit at the assay floor 0.01, so a continuous scale assumes "
          "precision the assay does not provide"),
        R("preprocessing", "fit dataset", "Train only", "Train only", "none",
          "neither pipeline lets Validation or Test influence preprocessing parameters"),
        R("hyperparameters", "LR", "L2, C=1.0, class_weight=None, solver=liblinear",
          "L2, C=1.0, class_weight=balanced, solver=lbfgs",
          "class weighting: none vs balanced",
          "class weighting shifts predicted probabilities upward, which changes calibration and "
          "the operating threshold without necessarily changing the ranking"),
        R("hyperparameters", "LR selection", "Validation ROC-AUC over 8 conditions",
          "Training-internal repeated 5-fold x 3 cross-validation over 8 conditions",
          "selection data: Validation (17 events) vs Training CV (135 events)",
          "selecting on 17 events is noisy and makes the Validation estimate optimistic for the "
          "chosen condition"),
        R("hyperparameters", "XGBoost",
          "depth 3, lr 0.10, min_child_weight 1, subsample 0.8, colsample 0.8, "
          "scale_pos_weight 1.0, reg_alpha 0, reg_lambda 1, 25 trees (best iteration 24)",
          "depth 3, lr 0.03, min_child_weight 10, subsample 0.8, colsample 0.8, "
          "scale_pos_weight 1.0, reg_lambda 1.0, n_estimators 300",
          "the agent chose a lower learning rate and a much higher min_child_weight",
          "a higher min_child_weight forces more events per leaf, which constrains the tree in a "
          "135-event problem"),
        R("hyperparameters", "XGBoost early stopping",
          "early stopping on the Validation set", "no early stopping; fixed 300 trees at lr 0.03, "
          "condition chosen by Training-internal CV",
          "the agent does not use Validation to stop training",
          "using Validation to stop, to choose among conditions and to set the threshold makes "
          "the Validation estimate optimistic three times over"),
        R("hyperparameters", "MLP architecture", "12 -> 64 -> 32 -> 1, dropout 0.2",
          "12 -> 32 -> 16 -> 1, dropout 0.3",
          "the agent used a smaller network with more dropout",
          "12 inputs and 135 events give little support for a wide network"),
        R("hyperparameters", "MLP optimisation",
          "AdamW, lr 1e-4, weight decay 1e-4, batch 64, pos_weight 1.0, best epoch 24",
          "AdamW, lr 3e-4, weight decay 1e-3, batch 32, pos_weight 6.563, best epoch 15",
          "the agent used class weighting, a higher learning rate and stronger weight decay",
          "pos_weight 6.563 raises predicted probabilities, which moves the operating threshold "
          "from 0.49 to 0.55 and affects calibration"),
        R("evaluation", "bootstrap for the AUROC CI", "10,000 resamples, seed 42",
          "2,000 resamples, seed 12345",
          "different resample count and seed",
          "the confidence intervals were generated using different bootstrap settings and "
          "therefore were not used for a standardized numerical comparison of uncertainty "
          "between the two pipelines; the point estimates are comparable"),
        R("evaluation", "ECE definition",
          "10 equal-width bins over [0,1] (Notebook/TaskB_03_ClinicalModeling.ipynb cell 10: "
          "bins = np.linspace(0.0, 1.0, n_bins + 1), n_bins=10)",
          "5 equal-count (quantile) bins (scripts/27 and scripts/30, calibration_table(): "
          "edges = np.quantile(p, np.linspace(0, 1, n_bins + 1)), n_bins=5)",
          "different bin count AND different binning scheme; the aggregation is identical",
          "the reported ECE values are not on one scale. A harmonised recomputation with 10 "
          "equal-width bins for all six models is in harmonized_ece.csv; it changes the "
          "AI-agent XGBoost value from 0.022064 to 0.074899 and leaves the human-guided values "
          "unchanged"),
        R("evaluation", "threshold rule", "Validation Youden maximum",
          "Validation Youden maximum, tie-break lowest, frozen before Test",
          "same rule; the agent froze the value and verified it was not recomputed on Test",
          "threshold-dependent metrics depend on this value"),
        R("evaluation", "feature importance",
          "permutation importance on Validation, 30 repeats",
          "permutation importance on Validation, 30 repeats (planned; primary), plus LR "
          "coefficients and XGBoost gain",
          "same primary method",
          "keeps the three model families on one scale"),
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=".")
    ap.add_argument("--drive-root", required=True)
    args = ap.parse_args()
    project, drive = Path(args.project).resolve(), Path(args.drive_root)
    out = project / "results/taskB/comparison"
    out.mkdir(parents=True, exist_ok=True)

    # ---- specification table -------------------------------------------------------
    spec = pd.DataFrame(spec_rows())
    spec.to_csv(out / "human_vs_ai_agent_model_spec.csv", index=False, encoding="utf-8-sig")
    print(f"model spec rows: {len(spec)}")

    # ---- performance table ---------------------------------------------------------
    h_test = pd.read_csv(drive / H_TEST, encoding="utf-8-sig").set_index("model")
    h_five = pd.read_csv(drive / H_FIVE, encoding="utf-8-sig").set_index("model")
    h_val = {
        "LR": pd.read_csv(drive / H_LR_VAL, encoding="utf-8-sig")
              .sort_values("val_roc_auc", ascending=False).iloc[0],
        "XGBoost": pd.read_csv(drive / H_XGB_VAL, encoding="utf-8-sig")
                   .sort_values("val_roc_auc", ascending=False).iloc[0],
        "MLP": pd.read_csv(drive / H_MLP_VAL, encoding="utf-8-sig")
               .sort_values("val_roc_auc", ascending=False).iloc[0],
    }
    a_test = pd.read_csv(project / "results/taskB/evaluation/test_metrics_table.csv",
                         encoding="utf-8-sig").set_index("model")
    a_val = pd.read_csv(project / "results/taskB/modeling/validation_comparison.csv",
                        encoding="utf-8-sig").set_index("model")

    rows = []
    add = lambda m, p, s, k, v, src: rows.append(  # noqa: E731
        {"model": m, "pipeline": p, "split": s, "metric": k,
         "value": (None if v is None else round(float(v), 6)), "source": src})
    for m in ("LR", "XGBoost", "MLP"):
        hn, an = H_NAMES[m], A_NAMES[m]
        h, a = h_test.loc[hn], a_test.loc[an]
        for k, hc, ac in [("roc_auc", "roc_auc", "auroc"),
                          ("roc_auc_ci_low", "roc_auc_ci_lower", "auroc_ci_low"),
                          ("roc_auc_ci_high", "roc_auc_ci_upper", "auroc_ci_high"),
                          ("pr_auc", "pr_auc", "auprc"), ("brier", "brier_score", "brier"),
                          ("ece", "ece", "ece_5bin"), ("threshold", "threshold", "threshold"),
                          ("sensitivity", "sensitivity", "sensitivity"),
                          ("specificity", "specificity", "specificity")]:
            add(m, "Human-guided", "Test", k, h[hc], f"Human-guided final result artifact: {H_TEST}")
            add(m, "AI-Agent", "Test", k, a[ac],
                "AI-Agent final Test evaluation: results/taskB/evaluation/test_metrics_table.csv")
        hv = h_val[m]
        # harmonised Test ECE, recomputed under one definition by scripts/34
        harm_path = project / "results/taskB/comparison/harmonized_ece.csv"
        if harm_path.exists():
            h_ece = pd.read_csv(harm_path, encoding="utf-8-sig")
            for pipe in ("Human-guided", "AI-Agent"):
                r = h_ece[(h_ece.model == m) & (h_ece.pipeline == pipe)]
                if len(r):
                    add(m, pipe, "Test", "ece_harmonized_10_equal_width",
                        r.iloc[0].harmonized_ece,
                        "Harmonized post hoc recomputation: "
                        "results/taskB/comparison/harmonized_ece.csv")
        for k, hc, ac in [("roc_auc", "val_roc_auc", "val_auroc"),
                          ("pr_auc", "val_pr_auc", "val_auprc"),
                          ("brier", "val_brier", "val_brier"), ("ece", "val_ece", "val_ece_5bin")]:
            add(m, "Human-guided", "Validation", k, hv[hc],
                f"Human-guided final result artifact: {[H_LR_VAL, H_XGB_VAL, H_MLP_VAL][('LR','XGBoost','MLP').index(m)]}")
            add(m, "AI-Agent", "Validation", k, a_val.loc[an][ac],
                "AI-Agent frozen Task B result: results/taskB/modeling/validation_comparison.csv")
    perf = pd.DataFrame(rows)
    perf.to_csv(out / "human_vs_ai_agent_performance.csv", index=False, encoding="utf-8-sig")
    print(f"performance rows: {len(perf)}")

    # ---- contradictions found between authoritative artefacts ------------------------
    coef = pd.read_csv(drive / H_LR_COEF, encoding="utf-8-sig")
    fitted = set(coef.feature)
    issues = [{
        "issue": "human-guided variable-selection artefacts disagree with the fitted model",
        "detail": ("01_変数選定/TaskB_BackwardAIC_最終選択変数.csv and 03_前処理/"
                   "TaskB_LR_最終12列_前処理仕様.json list the final set as age, sex, hf, spo2, "
                   "**rr**, lymph, egfr, crp, lactate (final AIC 524.98) and record ddimer as "
                   "removed at step 2. The fitted model in 04_Modeling/TaskB_LR_bestmodel_係数_OR.csv "
                   "contains **ddimer** and no rr."),
        "resolution": ("the evaluated model is the one in the coefficient file, which matches the "
                       "design the researcher states. The comparison uses that set and reports "
                       "the discrepancy rather than merging the two."),
        "fitted_features": ", ".join(sorted(fitted)),
    }, {
        "issue": "two human-guided artefacts give different AUROC confidence intervals",
        "detail": (f"{H_TEST} gives LR 0.887652-0.983042, XGBoost 0.876524-0.980392, "
                   f"MLP 0.767886-0.962374; {H_FIVE} gives LR 0.887122-0.981452, "
                   f"XGBoost 0.875980-0.979332, MLP 0.767886-0.960784. Point estimates, PR-AUC, "
                   f"Brier and ECE are identical in both."),
        "resolution": ("two bootstrap runs with different draws. The Task B artefact is used and "
                       "the difference is reported; no value is averaged or invented."),
        "fitted_features": "",
    }]
    pd.DataFrame(issues).to_csv(out / "source_discrepancies.csv", index=False, encoding="utf-8-sig")

    meta = {"generated": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
            "script": Path(__file__).name,
            "nature": "post hoc descriptive comparison; no refit, no retuning, no statistical test",
            "human_guided_sources": [H_TEST, H_LR_VAL, H_XGB_VAL, H_MLP_VAL, H_LR_COEF, H_FIVE],
            "ai_agent_sources": ["results/taskB/modeling/final_selection_taskB.json",
                                 "results/taskB/modeling/validation_comparison.csv",
                                 "results/taskB/evaluation/test_metrics_table.csv"],
            "discrepancies_found": len(issues),
            "paired_delong_run": False}
    (out / "comparison_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                              encoding="utf-8")
    print(f"discrepancies recorded: {len(issues)}")
    print(f"wrote {len(list(out.glob('*')))} files to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
