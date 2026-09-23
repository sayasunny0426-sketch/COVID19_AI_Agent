"""Task C: assemble the human-guided vs AI-agent methodological comparison tables.

Exploratory and descriptive. Nothing is refit, no specification is revisited, and no new
statistical test is run: the DeLong numbers on both sides are read from artefacts that already
exist. Every value carries the file it came from.

Human-guided authoritative sources (Google Drive, 気合のCOVID19):
    03_TaskC_マルチモーダル/TaskC_LateFusion_最終Test評価比較.csv
    03_TaskC_マルチモーダル/TaskC_LateFusion_Validation重み探索結果.csv
    03_TaskC_マルチモーダル/TaskC_LateFusion_Validation最終重み候補.csv
    03_TaskC_マルチモーダル/TaskC_LateFusion_DeLong_Holm比較.csv          (Family B shape)
    03_TaskC_マルチモーダル/TaskC_LateFusion_ModalityPermutationImportance.csv
    03_TaskC_マルチモーダル/TaskC_LateFusion_Test患者別予測結果.csv
    04_全モデル比較・統計解析/LateFusion_vs_Clinical3モデル_DeLong_Holm.csv (Family A shape)
    04_全モデル比較・統計解析/全5モデル_最終Test評価比較.csv
    Notebook/TaskC_01_LateFusion_Validation構築.ipynb                     (ECE / bootstrap code)

AI-agent sources (this repository):
    results/taskC/modeling/final_selection_taskC.json
    results/taskC/fusion/{selected_weight.json, weight_search_primary.csv}
    results/taskC/validation/validation_metrics_taskC.json
    results/taskC/evaluation/{test_metrics_table.csv, modality_importance_test.csv,
                              secondary_exploratory_test.json}
    results/comparison/delong_results.csv

Usage:
    python scripts/43_taskC_human_vs_agent_comparison.py --project . \
        --drive-root "<DRIVE_ROOT>"
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from covid_mortality.evaluation.metrics import (average_precision, bootstrap_ci,  # noqa: E402
                                                brier_score, expected_calibration_error, roc_auc)
from covid_mortality.fusion import taskC_fusion as tc  # noqa: E402

H = "03_TaskC_マルチモーダル"
HC = "04_全モデル比較・統計解析"
S_H = "Human-guided final result artifact"
S_A = "AI-Agent frozen Task C result"
ECE_BINS, ECE_SCHEME = 10, "equal_width"
HARMONISED = {"n_resamples": 2000, "stratified": True, "seed": 12345}


def spec_rows(hw, aw, a_frozen) -> list[dict]:
    R = lambda c, i, h, a, k, im: {  # noqa: E731
        "category": c, "item": i, "human_guided": h, "ai_agent": a,
        "key_difference": k, "possible_methodological_implication": im,
        "source_human": S_H, "source_ai_agent": S_A}
    return [
        R("cohort", "patients", "1,277 (fixed)", "1,277 (fixed)", "none",
          "the two pipelines are directly comparable on the same patients"),
        R("cohort", "split", "Train 1,021 / Val 128 / Test 128", "identical", "none",
          "Validation and Test estimates refer to the same patients"),
        R("cohort", "deaths", "135 / 17 / 17", "identical", "none", "identical event counts"),
        R("cohort", "prediction time point", "T0 = visit_start_datetime", "identical", "none",
          "both predict from information available at admission"),

        R("CXR component", "model", "ResNet18 (human-guided Task A)",
          "ResNet18, condition lr3e-4_aug_b, seed 42, best epoch 9",
          "different training configuration and checkpoint",
          "the two CXR models are not the same model; their performance differs substantially, "
          "which propagates into every fusion comparison"),
        R("CXR component", "Validation ROC-AUC", "0.848967", "0.871754",
          "the AI-agent CXR model scored higher on Validation",
          "Validation and Test order differently for these two CXR models"),
        R("CXR component", "Test ROC-AUC", "0.907260", "0.834658",
          "the human-guided CXR model scored markedly higher on Test (+0.0726)",
          "this single difference drives most of the divergence in the Late Fusion vs CXR "
          "comparison; it is a property of the Task A models, not of the fusion step"),
        R("CXR component", "retrained for Task C", "no", "no", "none",
          "both reuse a frozen image model"),

        R("clinical component", "model", "Clinical Logistic Regression",
          "Clinical Logistic Regression", "**none -- both converged on LR**",
          "the two pipelines chose the same model family independently"),
        R("clinical component", "selection basis", "Validation ROC-AUC among 8 LR conditions",
          "best Training cross-validated ROC-AUC among LR / XGBoost / MLP, fixed before fusion",
          "the agent chose the component on 135 CV events rather than 17 Validation events",
          "selecting on 17 events is noisier; the agent's rule also avoids spending Validation "
          "twice"),
        R("clinical component", "LR settings",
          "L2, C=1.0, class_weight=None, solver=liblinear",
          "L2, C=1.0, class_weight=balanced, solver=lbfgs",
          "class weighting: none vs balanced",
          "class weighting inflates predicted probabilities, which changes calibration and the "
          "operating threshold without necessarily changing the ranking"),
        R("clinical component", "final variables",
          "9 raw variables / 12 features: age, sex, heart failure, SpO2, lymphocyte, eGFR, CRP, "
          "D-dimer, lactate",
          "10 raw variables / 12 features: age, sex, SpO2, SBP, CRP, lymphocyte, D-dimer, "
          "lactate, eGFR, troponin detectable (+ lymph_missing)",
          "human-guided only: heart failure. AI-agent only: SBP, troponin detectable, "
          "lymph_missing",
          "different clinical domains are represented; 8 variables are shared"),
        R("clinical component", "Validation ROC-AUC", "0.896131", "0.895072",
          "essentially the same", "the two LR models discriminate almost identically on "
          "Validation despite different inputs"),
        R("clinical component", "Test ROC-AUC", "0.941176", "0.946476",
          "the AI-agent LR point estimate is marginally higher (+0.0053)",
          "no statistical test was run on this difference"),

        R("fusion", "level", "decision-level Late Fusion", "decision-level Late Fusion",
          "**none**", "both rejected early fusion in favour of combining probabilities"),
        R("fusion", "equation", "p = w x p_clinical + (1 - w) x p_cxr", "identical",
          "**none -- both used probability-space weighted averaging**",
          "the same functional form was reached independently"),
        R("fusion", "clinical weight", f"{hw['selected_w_clinical']}",
          f"{aw['w_clinical']}", "0.72 vs 0.75, a difference of 0.03",
          "both weight clinical information above CXR, in a similar proportion"),
        R("fusion", "CXR weight", f"{hw['selected_w_cxr']}", f"{aw['w_cxr']}",
          "0.28 vs 0.25", "see above"),
        R("fusion", "weight search space", "w in [0,1], step 0.01 (101 points)",
          "w in [0,1], step 0.05 (21 points)",
          "the human-guided grid is five times finer",
          "with 17 Validation events a finer grid does not add resolution; the agent recorded "
          "that 6 of its 21 weights sit within 0.005 AUROC of the best"),
        R("fusion", "selection metric", "Validation ROC-AUC maximum", "identical", "none",
          "same primary criterion"),
        R("fusion", "tie-break", "Validation PR-AUC, then Brier",
          "Validation PR-AUC, then Brier, then the w closest to 0.5",
          "the agent adds a third, deterministic tie-break",
          "makes the outcome reproducible when the first three metrics tie"),
        R("fusion", "data used for weight selection", "Validation only", "Validation only",
          "none", "neither used Test to choose the weight"),
        R("fusion", "strategy fixed before comparison",
          "not recorded in the available artefacts",
          "yes: probability-space weighted averaging was fixed before any Validation search, "
          "with logit-space fusion and logistic stacking as non-competing secondary arms",
          "the agent pre-registered the strategy; the human-guided pipeline's ordering is not "
          "documented in the artefacts available here",
          "pre-registration removes the option of switching strategy after seeing Validation "
          "results; it is a documentation difference here, not evidence about what was done"),
        R("fusion", "Test locked during design", "yes", "yes", "none",
          "both froze the specification before the Test set was opened"),

        R("evaluation", "ECE definition", "10 equal-width bins over [0,1]",
          "10 equal-width bins over [0,1]",
          "**none -- the definitions are identical**",
          "unlike the Task B comparison, no harmonisation of ECE is needed here; the values are "
          "directly comparable"),
        R("evaluation", "bootstrap for the AUROC CI", "10,000 resamples, stratified, seed 42",
          f"{HARMONISED['n_resamples']} resamples, stratified, seed {HARMONISED['seed']}",
          "different resample count and seed",
          "the confidence intervals were generated using different bootstrap settings and "
          "therefore were not used for a standardized numerical comparison of uncertainty "
          "between the two pipelines; a harmonised recomputation is provided separately"),
        R("evaluation", "threshold rule",
          "not recorded for the fusion model in the available artefacts",
          "Validation Youden maximum, tie-break lowest, frozen and not recomputed on Test",
          "the agent's threshold provenance is explicit",
          "threshold-dependent metrics depend on this value"),
        R("evaluation", "modality importance",
          "permutation importance on Test, with percentile bounds",
          "permutation importance on Validation and on Test, 30 repeats, seed 42, Test labelled "
          "post hoc descriptive",
          "the agent reports both splits and marks the Test one as non-decisional",
          "modality importance on Test cannot be used to change a specification"),
        R("evaluation", "DeLong families",
          "two files matching the same two-family shape",
          "two pre-registered families, Holm within each, family column required",
          "the agent fixed the family structure in a plan file before the Test set was opened",
          "the overlapping comparison carries two adjusted p-values in both pipelines"),
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=".")
    ap.add_argument("--drive-root", required=True)
    args = ap.parse_args()
    project, drive = Path(args.project).resolve(), Path(args.drive_root)
    out = project / "results/taskC/comparison"
    out.mkdir(parents=True, exist_ok=True)

    # ---- read both sides -------------------------------------------------------------
    h_test = pd.read_csv(drive / H / "TaskC_LateFusion_最終Test評価比較.csv",
                         encoding="utf-8-sig").set_index("model")
    h_five = pd.read_csv(drive / HC / "全5モデル_最終Test評価比較.csv",
                         encoding="utf-8-sig").set_index("model")
    h_w = pd.read_csv(drive / H / "TaskC_LateFusion_Validation重み探索結果.csv",
                      encoding="utf-8-sig")
    h_sel = pd.read_csv(drive / H / "TaskC_LateFusion_Validation最終重み候補.csv",
                        encoding="utf-8-sig").iloc[0]
    h_famB = pd.read_csv(drive / H / "TaskC_LateFusion_DeLong_Holm比較.csv", encoding="utf-8-sig")
    h_famA = pd.read_csv(drive / HC / "LateFusion_vs_Clinical3モデル_DeLong_Holm.csv",
                         encoding="utf-8-sig")
    h_mod = pd.read_csv(drive / H / "TaskC_LateFusion_ModalityPermutationImportance.csv",
                        encoding="utf-8-sig")
    h_pred = pd.read_csv(drive / H / "TaskC_LateFusion_Test患者別予測結果.csv",
                         dtype={"Subject ID": str}, encoding="utf-8-sig")

    a_frozen = json.loads((project / "results/taskC/modeling/final_selection_taskC.json")
                          .read_text(encoding="utf-8"))
    a_sel = json.loads((project / "results/taskC/fusion/selected_weight.json")
                       .read_text(encoding="utf-8"))
    a_val = json.loads((project / "results/taskC/validation/validation_metrics_taskC.json")
                       .read_text(encoding="utf-8"))
    a_test = pd.read_csv(project / "results/taskC/evaluation/test_metrics_table.csv",
                         encoding="utf-8-sig").set_index("model")
    a_mod = pd.read_csv(project / "results/taskC/evaluation/modality_importance_test.csv",
                        encoding="utf-8-sig")
    a_sec = json.loads((project / "results/taskC/evaluation/secondary_exploratory_test.json")
                       .read_text(encoding="utf-8"))
    a_dl = pd.read_csv(project / "results/comparison/delong_results.csv", encoding="utf-8-sig")
    a_pred = pd.read_csv(project / "results/comparison/test_predictions_all_models.csv",
                         dtype={"subject_id": str}, encoding="utf-8-sig")

    # ---- 1. design table --------------------------------------------------------------
    spec = pd.DataFrame(spec_rows(h_sel, a_sel, a_frozen))
    spec.to_csv(out / "human_vs_ai_agent_model_spec.csv", index=False, encoding="utf-8-sig")
    print(f"model spec rows: {len(spec)}")

    # ---- 2. performance table ----------------------------------------------------------
    rows = []
    add = lambda m, p, s, k, v, src: rows.append(  # noqa: E731
        {"model": m, "pipeline": p, "split": s, "metric": k,
         "value": None if v is None else round(float(v), 6), "source": src})
    hs = f"{S_H}: {H}/TaskC_LateFusion_最終Test評価比較.csv"
    as_ = f"{S_A}: results/taskC/evaluation/test_metrics_table.csv"
    for m, hname, aname in [("Late Fusion", "Late Fusion", "late_fusion"),
                            ("Clinical LR", "Clinical", "clinical_lr"),
                            ("CXR ResNet18", "CXR", "cxr")]:
        hr, ar = h_test.loc[hname], a_test.loc[aname]
        for k, hc, ac in [("roc_auc", "roc_auc", "auroc"),
                          ("roc_auc_ci_low", "ci95_lower", "auroc_ci_low"),
                          ("roc_auc_ci_high", "ci95_upper", "auroc_ci_high"),
                          ("pr_auc", "pr_auc", "auprc"), ("brier", "brier_score", "brier"),
                          ("ece_10_equal_width", "ece", "ece_10_equal_width")]:
            add(m, "Human-guided", "Test", k, hr[hc], hs)
            add(m, "AI-Agent", "Test", k, ar[ac], as_)
    hvs = f"{S_H}: {H}/TaskC_LateFusion_Validation重み探索結果.csv"
    avs = f"{S_A}: results/taskC/validation/validation_metrics_taskC.json"
    for m, wsel, akey in [("Late Fusion", float(h_sel.selected_w_clinical), "fused"),
                          ("Clinical LR", 1.0, "lr"), ("CXR ResNet18", 0.0, "cxr")]:
        hr = h_w[np.isclose(h_w.w_clinical, wsel)].iloc[0]
        av = a_val["fused"] if akey == "fused" else a_val["components"][akey]
        for k, hc in [("roc_auc", "roc_auc"), ("pr_auc", "pr_auc"), ("brier", "brier_score")]:
            add(m, "Human-guided", "Validation", k, hr[hc], hvs)
            add(m, "AI-Agent", "Validation", k,
                av["auroc"] if k == "roc_auc" else av["auprc"] if k == "pr_auc" else av["brier"],
                avs)
        add(m, "Human-guided", "Validation", "ece_10_equal_width", None,
            f"{S_H}: not recorded in the weight-search artefact")
        add(m, "AI-Agent", "Validation", "ece_10_equal_width", av["ece"], avs)

    # harmonised CI, computed from the saved patient-level predictions of both pipelines
    y = a_pred.true_label.to_numpy(int)
    hp = h_pred.set_index("Subject ID").loc[a_pred.subject_id].reset_index()
    harm = []
    for m, hcol, acol in [("Late Fusion", "latefusion_prob", "prob_late_fusion"),
                          ("Clinical LR", "clinical_prob", "prob_clinical_lr"),
                          ("CXR ResNet18", "cxr_prob", "prob_cxr")]:
        for pipe, p in (("Human-guided", hp[hcol].to_numpy(float)),
                        ("AI-Agent", a_pred[acol].to_numpy(float))):
            ci = bootstrap_ci(y, p, roc_auc, n_boot=HARMONISED["n_resamples"],
                              stratified=HARMONISED["stratified"], seed=HARMONISED["seed"])
            harm.append({"model": m, "pipeline": pipe, "auroc": round(ci["point"], 6),
                         "ci_low": round(ci["ci_low"], 6), "ci_high": round(ci["ci_high"], 6),
                         **HARMONISED})
            add(m, pipe, "Test", "roc_auc_ci_low_harmonized", ci["ci_low"],
                "Harmonized recomputation from the saved patient-level Test predictions")
            add(m, pipe, "Test", "roc_auc_ci_high_harmonized", ci["ci_high"],
                "Harmonized recomputation from the saved patient-level Test predictions")
    pd.DataFrame(harm).to_csv(out / "harmonized_ci.csv", index=False, encoding="utf-8-sig")
    perf = pd.DataFrame(rows)
    perf.to_csv(out / "human_vs_ai_agent_performance.csv", index=False, encoding="utf-8-sig")
    print(f"performance rows: {len(perf)}")

    # ---- 3. DeLong table -----------------------------------------------------------------
    dl = []
    for _, r in h_famA.iterrows():
        dl.append({"pipeline": "Human-guided", "family": "A",
                   "comparison": r.comparison, "auc_1": r.auc_1, "auc_2": r.auc_2,
                   "delta_auc": r.auc_difference, "standard_error": r.standard_error,
                   "z": r.z, "p_unadjusted": r.p_value, "p_holm": r.p_holm,
                   "significant_after_holm": bool(r.significant_holm),
                   "source": f"{S_H}: {HC}/LateFusion_vs_Clinical3モデル_DeLong_Holm.csv"})
    for _, r in h_famB.iterrows():
        dl.append({"pipeline": "Human-guided", "family": "B",
                   "comparison": r.comparison, "auc_1": r.auc_1, "auc_2": r.auc_2,
                   "delta_auc": r.auc_difference, "standard_error": r.standard_error,
                   "z": r.z, "p_unadjusted": r.p_value, "p_holm": r.p_holm,
                   "significant_after_holm": bool(r.significant_holm),
                   "source": f"{S_H}: {H}/TaskC_LateFusion_DeLong_Holm比較.csv"})
    for _, r in a_dl.iterrows():
        dl.append({"pipeline": "AI-Agent", "family": r.family,
                   "comparison": f"{r.model_1_label} vs {r.model_2_label}",
                   "auc_1": r.AUC_1, "auc_2": r.AUC_2, "delta_auc": r.delta_AUC,
                   "standard_error": r.standard_error, "z": r.z,
                   "p_unadjusted": r.p_unadjusted, "p_holm": r.p_holm_within_family,
                   "significant_after_holm": bool(r["significant_at_0.05_after_holm"]),
                   "source": f"{S_A}: results/comparison/delong_results.csv"})
    pd.DataFrame(dl).to_csv(out / "human_vs_ai_agent_delong.csv", index=False,
                            encoding="utf-8-sig")
    print(f"delong rows: {len(dl)}")

    # ---- 4. modality importance ------------------------------------------------------------
    mi = []
    for _, r in h_mod.iterrows():
        mi.append({"pipeline": "Human-guided", "modality": r.modality.lower(),
                   "baseline_auroc": r.baseline_auc, "mean_auroc_drop": r.auc_drop,
                   "sd_auroc_drop": r.sd_permuted_auc, "dataset": "Test",
                   "source": f"{S_H}: {H}/TaskC_LateFusion_ModalityPermutationImportance.csv"})
    for _, r in a_mod.iterrows():
        mi.append({"pipeline": "AI-Agent", "modality": r.modality,
                   "baseline_auroc": r.baseline_auroc, "mean_auroc_drop": r.mean_auroc_drop,
                   "sd_auroc_drop": r.sd_auroc_drop, "dataset": "Test",
                   "source": f"{S_A}: results/taskC/evaluation/modality_importance_test.csv"})
    pd.DataFrame(mi).to_csv(out / "human_vs_ai_agent_modality_importance.csv", index=False,
                            encoding="utf-8-sig")

    # ---- 5. source discrepancies -------------------------------------------------------------
    disc = []
    lr_a = float(h_test.loc["Clinical"].ci95_lower)
    lr_b = 0.887652   # the Task B artefact value, recorded during the Task B comparison
    if abs(lr_a - lr_b) > 1e-9:
        disc.append({
            "issue": "human-guided Clinical LR AUROC confidence interval differs between artefacts",
            "detail": (f"{H}/TaskC_LateFusion_最終Test評価比較.csv and {HC}/全5モデル_最終Test評価比較.csv "
                       f"give {lr_a:.6f}-{float(h_test.loc['Clinical'].ci95_upper):.6f}; "
                       f"02_TaskB_臨床データ/05_Evaluation/TaskB_3モデル_最終Test評価比較.csv gives "
                       f"0.887652-0.983042. Point estimates, PR-AUC, Brier and ECE agree."),
            "sources": "Task C artefact, five-model artefact, Task B artefact",
            "which_source_was_used": "the Task C artefact, for the Task C comparison",
            "rationale": ("two bootstrap runs with different draws; nothing is averaged or "
                          "invented, and the difference is reported")})
    # NOTE: the overlapping comparison (Late Fusion vs Clinical LR) carrying two different
    # Holm-adjusted p-values is NOT a source discrepancy. It follows necessarily from the
    # pre-specified design of correcting within each family separately, and both pipelines show
    # it. It belongs in the DeLong section of the comparison document as a methodological
    # clarification, not here.
    if not any("Validation ECE" in d["issue"] for d in disc):
        disc.append({
            "issue": "human-guided Validation ECE for the fusion model is not recorded",
            "detail": (f"{H}/TaskC_LateFusion_Validation重み探索結果.csv holds ROC-AUC, PR-AUC and "
                       f"Brier only; no ECE column exists for any weight."),
            "sources": "human-guided Validation weight-search artefact",
            "which_source_was_used": "none; the cell is left empty rather than filled",
            "rationale": "no value is inferred for a metric that was not stored"})
    pd.DataFrame(disc).to_csv(out / "source_discrepancies.csv", index=False,
                              encoding="utf-8-sig")
    print(f"source discrepancies recorded: {len(disc)}")

    # ---- 6. meta ---------------------------------------------------------------------------
    feasible = (set(h_pred["Subject ID"]) == set(a_pred.subject_id)
                and int(h_pred.true_label.sum()) == int(a_pred.true_label.sum()))
    meta = {
        "generated": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "script": Path(__file__).name,
        "nature": ("exploratory methodological comparison; descriptive only. No refit, no "
                   "retuning, no new statistical test, and no claim of superiority."),
        "ece_definitions": {"human_guided": "10 equal-width bins over [0,1] "
                                            "(Notebook/TaskC_01_LateFusion_Validation構築.ipynb)",
                            "ai_agent": "10 equal-width bins over [0,1] "
                                        "(src/covid_mortality/evaluation/metrics.py)",
                            "identical": True,
                            "harmonisation_needed": False},
        "bootstrap_definitions": {"human_guided": {"n_resamples": 10000, "stratified": True,
                                                   "seed": 42},
                                  "ai_agent": HARMONISED,
                                  "identical": False,
                                  "harmonised_artefact": "harmonized_ci.csv"},
        "paired_pipeline_comparison": {
            "technically_feasible": bool(feasible),
            "evidence": ("both pipelines saved patient-level Late Fusion Test predictions for "
                         "the same 128 patients with 17 deaths and agreeing labels"),
            "executed": False,
            "if_executed": ("must be treated as a separate exploratory methodological analysis, "
                            "entirely outside the study's confirmatory DeLong families")},
        "human_guided_sources": [f"{H}/TaskC_LateFusion_最終Test評価比較.csv",
                                 f"{H}/TaskC_LateFusion_Validation重み探索結果.csv",
                                 f"{H}/TaskC_LateFusion_Validation最終重み候補.csv",
                                 f"{H}/TaskC_LateFusion_DeLong_Holm比較.csv",
                                 f"{H}/TaskC_LateFusion_ModalityPermutationImportance.csv",
                                 f"{H}/TaskC_LateFusion_Test患者別予測結果.csv",
                                 f"{HC}/LateFusion_vs_Clinical3モデル_DeLong_Holm.csv",
                                 f"{HC}/全5モデル_最終Test評価比較.csv",
                                 "Notebook/TaskC_01_LateFusion_Validation構築.ipynb"],
        "ai_agent_sources": ["results/taskC/modeling/final_selection_taskC.json",
                             "results/taskC/fusion/selected_weight.json",
                             "results/taskC/validation/validation_metrics_taskC.json",
                             "results/taskC/evaluation/test_metrics_table.csv",
                             "results/taskC/evaluation/modality_importance_test.csv",
                             "results/taskC/evaluation/secondary_exploratory_test.json",
                             "results/comparison/delong_results.csv"],
        "discrepancies_found": len(disc),
        "official_metrics_overwritten": False,
    }
    (out / "comparison_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                              encoding="utf-8")
    print(f"paired pipeline comparison technically feasible: {feasible} (not executed)")
    print(f"wrote {len(list(out.glob('*')))} files to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
