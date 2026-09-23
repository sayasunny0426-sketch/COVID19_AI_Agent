"""Task C step 2: choose the fusion weight on Validation, and record everything around it.

The primary strategy and the clinical component were fixed before this ran (decision_log D-085,
D-086): probability-space weighted average, clinical component = LR. This script therefore has
exactly one thing to decide -- the weight w -- and it decides it on the Validation set by
ROC-AUC, tie-broken by PR-AUC, then Brier, then proximity to 0.5.

Everything else it computes is secondary or exploratory and is saved for transparency without
competing for the primary selection:
    * the same probability-space weighted average with the XGBoost and MLP clinical components
    * logit-space weighted averaging
    * logistic stacking (3 parameters on 17 Validation events)
    * ablation and modality-level permutation importance

Test is not read.

Outputs (results/taskC/fusion/ and results/taskC/validation/):
    weight_search_primary.csv      every w for the primary strategy: the weight curve
    weight_search_all.csv          every w for every strategy x clinical component
    strategy_comparison.csv        best row per strategy, primary flagged
    selected_weight.json           the chosen w and the rule that chose it
    stacking_exploratory.json      the logistic stack coefficients, clearly labelled
    validation_predictions_taskC.csv   patient-level Validation predictions of the fused model
    validation_metrics_taskC.json  metrics of the fused model and of each component
    validation_calibration.csv     calibration bins of the fused model
    ablation_validation.csv        each component alone vs fused
    modality_importance_validation.csv   permutation importance at modality level

Usage:
    python scripts/36_taskC_fusion_search.py --project . --taskA-runs <...>
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
from covid_mortality.evaluation.metrics import (average_precision, binary_rates,  # noqa: E402
                                                brier_score, calibration_bins,
                                                expected_calibration_error, roc_auc,
                                                threshold_at_min_sensitivity, youden_threshold)
from covid_mortality.fusion import taskC_fusion as tc  # noqa: E402

ECE_BINS, ECE_SCHEME = 10, "equal_width"
WEIGHT_STEP = 0.05
PRIMARY_CLINICAL, PRIMARY_SPACE = "lr", "probability"
SEED = 42
PERM_REPEATS = 30


def metrics(y, p) -> dict:
    return {"auroc": float(roc_auc(y, p)), "auprc": float(average_precision(y, p)),
            "brier": float(brier_score(y, p)),
            "ece": float(expected_calibration_error(y, p, n_bins=ECE_BINS, scheme=ECE_SCHEME))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=".")
    ap.add_argument("--taskA-runs", required=True)
    args = ap.parse_args()
    project, runs = Path(args.project).resolve(), Path(args.taskA_runs)
    fus = project / "results/taskC/fusion"
    vald = project / "results/taskC/validation"
    fus.mkdir(parents=True, exist_ok=True)
    vald.mkdir(parents=True, exist_ok=True)

    split = pd.read_csv(project / "data/splits/COVID19_固定患者split_1277.csv",
                        dtype=str, encoding="utf-8-sig")
    val = tc.load_split("val", runs / "lr3e-4_aug_b/seed42/val_predictions.csv",
                        project / "results/taskB/modeling/validation_predictions.csv",
                        split, tc.CLINICAL_COLUMNS)
    y, p_cxr = val.y, val.cxr()
    print(f"Validation: {len(y)} patients, {int(y.sum())} deaths. Test is not read.")

    grid = tc.weight_grid(WEIGHT_STEP)
    rows = []
    for clin in ("lr", "xgb", "mlp"):
        p_clin = val.clinical(clin)
        for space in ("probability", "logit"):
            for w in grid:
                m = metrics(y, tc.fuse(p_clin, p_cxr, float(w), space))
                rows.append({"strategy": f"weighted_average_{space}", "space": space,
                             "clinical_component": clin, "w_clinical": float(w),
                             "w_cxr": round(1.0 - float(w), 10),
                             "role": ("PRIMARY" if (clin == PRIMARY_CLINICAL
                                                    and space == PRIMARY_SPACE)
                                      else "secondary"),
                             **{f"val_{k}": round(v, 6) for k, v in m.items()}})
    allw = pd.DataFrame(rows)
    allw.to_csv(fus / "weight_search_all.csv", index=False, encoding="utf-8-sig")

    prim = allw[allw.role == "PRIMARY"].reset_index(drop=True)
    prim.to_csv(fus / "weight_search_primary.csv", index=False, encoding="utf-8-sig")
    print(f"\n=== primary weight curve: {PRIMARY_SPACE}-space, clinical = "
          f"{PRIMARY_CLINICAL.upper()} ===")
    print(prim[["w_clinical", "w_cxr", "val_auroc", "val_auprc", "val_brier",
                "val_ece"]].to_string(index=False))

    # ---- the pre-registered selection rule -------------------------------------------
    ranked = prim.sort_values(
        by=["val_auroc", "val_auprc", "val_brier"],
        ascending=[False, False, True]).copy()
    top = ranked[np.isclose(ranked.val_auroc, ranked.val_auroc.iloc[0])]
    top = top[np.isclose(top.val_auprc, top.val_auprc.iloc[0])]
    top = top[np.isclose(top.val_brier, top.val_brier.iloc[0])]
    chosen = top.iloc[(top.w_clinical - 0.5).abs().argsort()].iloc[0]
    w_star = float(chosen.w_clinical)
    tie_size = len(top)
    print(f"\nselected w (clinical) = {w_star:.2f}, CXR weight {1 - w_star:.2f}"
          f"  [{tie_size} row(s) tied on all three metrics]")

    # how flat is the curve? the honest answer to "is 17 events enough to pick a weight?"
    near = prim[prim.val_auroc >= chosen.val_auroc - 0.005]
    flat = {"best_auroc": float(chosen.val_auroc),
            "n_weights_within_0.005_auroc": int(len(near)),
            "w_range_within_0.005_auroc": [float(near.w_clinical.min()),
                                           float(near.w_clinical.max())],
            "auroc_at_w0_cxr_only": float(prim[prim.w_clinical == 0].val_auroc.iloc[0]),
            "auroc_at_w1_clinical_only": float(prim[prim.w_clinical == 1].val_auroc.iloc[0]),
            "auroc_at_w0.5_simple_average": float(prim[np.isclose(prim.w_clinical, 0.5)]
                                                  .val_auroc.iloc[0])}
    print(f"weights within 0.005 AUROC of the best: {flat['n_weights_within_0.005_auroc']} "
          f"(w from {flat['w_range_within_0.005_auroc'][0]:.2f} to "
          f"{flat['w_range_within_0.005_auroc'][1]:.2f})")

    (fus / "selected_weight.json").write_text(json.dumps({
        "generated": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "script": Path(__file__).name,
        "primary_strategy": f"weighted_average_{PRIMARY_SPACE}",
        "clinical_component": PRIMARY_CLINICAL,
        "w_clinical": w_star, "w_cxr": round(1 - w_star, 10),
        "selection_rule": {"metric": "Validation ROC-AUC",
                           "tie_break": ["Validation PR-AUC", "Validation Brier",
                                         "w closest to 0.5"],
                           "grid": {"low": 0.0, "high": 1.0, "step": WEIGHT_STEP,
                                    "n_points": len(grid)},
                           "dataset_used": "Validation only", "test_used": False},
        "n_rows_tied_on_all_metrics": tie_size,
        "validation_metrics_at_selected_w": {k: float(chosen[f"val_{k}"])
                                             for k in ("auroc", "auprc", "brier", "ece")},
        "weight_curve_flatness": flat,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- strategy comparison (secondary strategies never override the primary) --------
    best_rows = []
    for (strategy, clin), g in allw.groupby(["strategy", "clinical_component"]):
        r = g.sort_values(["val_auroc", "val_auprc", "val_brier"],
                          ascending=[False, False, True]).iloc[0]
        best_rows.append({"strategy": strategy, "clinical_component": clin,
                          "role": r.role, "best_w_clinical": float(r.w_clinical),
                          **{k: float(r[k]) for k in
                             ("val_auroc", "val_auprc", "val_brier", "val_ece")}})

    # logistic stacking, exploratory: 3 parameters on 17 events
    stack_info = {}
    for clin in ("lr", "xgb", "mlp"):
        model, predict = tc.fit_logistic_stack(val.clinical(clin), p_cxr, y, seed=SEED)
        p_stack = predict(val.clinical(clin), p_cxr)
        m = metrics(y, p_stack)
        best_rows.append({"strategy": "logistic_stacking", "clinical_component": clin,
                          "role": "exploratory", "best_w_clinical": None,
                          **{f"val_{k}": v for k, v in m.items()}})
        stack_info[clin] = {"intercept": float(model.intercept_[0]),
                            "coef_logit_clinical": float(model.coef_[0][0]),
                            "coef_logit_cxr": float(model.coef_[0][1]),
                            "validation_metrics": m,
                            "caveat": ("3 parameters fitted on 128 Validation patients with 17 "
                                       "events, on the same data used to report the metric; "
                                       "these numbers are optimistic and exploratory only")}
    (fus / "stacking_exploratory.json").write_text(json.dumps({
        "role": "exploratory; never a candidate for the primary strategy",
        "fitted_on": "Validation (the only data with both components' predictions)",
        "models": stack_info}, ensure_ascii=False, indent=2), encoding="utf-8")

    comp = pd.DataFrame(best_rows).sort_values(["role", "val_auroc"],
                                               ascending=[True, False])
    comp.to_csv(fus / "strategy_comparison.csv", index=False, encoding="utf-8-sig")
    print("\n=== strategy comparison (primary is fixed; the rest are secondary) ===")
    print(comp.to_string(index=False))

    # ---- the fused model on Validation -------------------------------------------------
    p_fused = tc.fuse(val.clinical(PRIMARY_CLINICAL), p_cxr, w_star, PRIMARY_SPACE)
    thr_y = float(youden_threshold(y, p_fused, tie_break="lowest"))
    thr_s = float(threshold_at_min_sensitivity(y, p_fused, 0.80))
    rates = binary_rates(y, p_fused, thr_y)
    vm = metrics(y, p_fused)
    out_pred = val.frame.copy()
    out_pred["prob_late_fusion"] = p_fused
    out_pred["pred_late_fusion"] = (p_fused >= thr_y).astype(int)
    out_pred.to_csv(vald / "validation_predictions_taskC.csv", index=False,
                    encoding="utf-8-sig")
    pd.DataFrame(calibration_bins(y, p_fused, n_bins=ECE_BINS, scheme=ECE_SCHEME)).to_csv(
        vald / "validation_calibration.csv", index=False, encoding="utf-8-sig")

    # ---- ablation and modality importance (Validation, descriptive) --------------------
    abl = [{"model": "late_fusion (primary)", "w_clinical": w_star, **vm},
           {"model": "clinical LR alone", "w_clinical": 1.0,
            **metrics(y, val.clinical("lr"))},
           {"model": "CXR alone", "w_clinical": 0.0, **metrics(y, p_cxr)},
           {"model": "simple average (w=0.5)", "w_clinical": 0.5,
            **metrics(y, tc.fuse(val.clinical(PRIMARY_CLINICAL), p_cxr, 0.5, PRIMARY_SPACE))}]
    pd.DataFrame(abl).to_csv(vald / "ablation_validation.csv", index=False, encoding="utf-8-sig")
    print("\n=== ablation on Validation ===")
    print(pd.DataFrame(abl).round(6).to_string(index=False))

    rng = np.random.default_rng(SEED)
    imp = []
    for mod, arr in (("clinical", val.clinical(PRIMARY_CLINICAL)), ("cxr", p_cxr)):
        drops = []
        for _ in range(PERM_REPEATS):
            shuffled = rng.permutation(arr)
            pf = (tc.fuse(shuffled, p_cxr, w_star, PRIMARY_SPACE) if mod == "clinical"
                  else tc.fuse(val.clinical(PRIMARY_CLINICAL), shuffled, w_star, PRIMARY_SPACE))
            drops.append(vm["auroc"] - roc_auc(y, pf))
        imp.append({"modality": mod, "mean_auroc_drop": float(np.mean(drops)),
                    "sd_auroc_drop": float(np.std(drops, ddof=1)),
                    "n_repeats": PERM_REPEATS, "seed": SEED, "dataset": "Validation",
                    "baseline_auroc": vm["auroc"]})
    pd.DataFrame(imp).to_csv(vald / "modality_importance_validation.csv", index=False,
                             encoding="utf-8-sig")
    print("\n=== modality permutation importance on Validation ===")
    print(pd.DataFrame(imp).round(6).to_string(index=False))

    (vald / "validation_metrics_taskC.json").write_text(json.dumps({
        "generated": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "n": len(y), "deaths": int(y.sum()),
        "primary_strategy": f"weighted_average_{PRIMARY_SPACE}",
        "clinical_component": PRIMARY_CLINICAL, "w_clinical": w_star,
        "fused": vm,
        "components": {"cxr": metrics(y, p_cxr),
                       **{c: metrics(y, val.clinical(c)) for c in ("lr", "xgb", "mlp")}},
        "thresholds": {"youden": thr_y, "sensitivity80": thr_s,
                       "rule": "Validation Youden maximum, tie-break lowest; frozen for Test"},
        "operating_point_youden": {k: float(v) for k, v in rates.items()},
        "calibration_definition": {"ece_bins": ECE_BINS, "ece_scheme": ECE_SCHEME},
        "test_used": False,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nfused model on Validation: AUROC {vm['auroc']:.6f}  AUPRC {vm['auprc']:.6f}  "
          f"Brier {vm['brier']:.6f}  ECE {vm['ece']:.6f}")
    print(f"threshold (Validation Youden) {thr_y:.6f}")
    print(f"\nwrote {len(list(fus.glob('*')))} files to {fus}")
    print(f"wrote {len(list(vald.glob('*')))} files to {vald}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
