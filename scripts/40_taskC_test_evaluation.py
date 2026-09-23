"""Task C final Test evaluation of the frozen Late Fusion model.

Opens the Test set. Refuses unless the frozen specification exists, every input still hashes to
what was frozen, and --approved is passed. Nothing is fitted here: the fusion weight, the
clinical component, the threshold and the calibration definition all come from the frozen file.

Outputs (results/taskC/evaluation/ and results/comparison/):
    test_predictions_taskC.csv          patient-level, primary fusion column prob_late_fusion
    test_metrics_taskC.json             primary metrics with bootstrap CIs
    test_metrics_table.csv              primary plus every component, one table
    test_roc_points_*.csv               ROC coordinates per model
    test_pr_points_*.csv                PR coordinates per model
    test_calibration_*.csv              calibration bins per model
    modality_importance_test.csv        post hoc descriptive, does not change any specification
    secondary_exploratory_test.json     logit-space fusion and stacking, clearly separated
    test_run_meta.json                  environment, hashes, seeds, command
    test_access_log.jsonl               append-only record that the Test set was opened
    ../comparison/test_predictions_all_models.csv   the merged file the DeLong step consumes

Usage:
    python scripts/40_taskC_test_evaluation.py --project . --taskA-runs <...> --approved
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from covid_mortality.evaluation.metrics import (average_precision, binary_rates,  # noqa: E402
                                                bootstrap_ci, brier_score, calibration_bins,
                                                expected_calibration_error, roc_auc, roc_curve)
from covid_mortality.fusion import taskC_fusion as tc  # noqa: E402

warnings.filterwarnings("ignore")
PRIMARY_COL = "prob_late_fusion"
COMBINED_COLUMNS = ["subject_id", "true_label", "prob_cxr", "prob_clinical_lr",
                    "prob_clinical_xgboost", "prob_clinical_mlp", PRIMARY_COL]


def pr_points(y, p) -> pd.DataFrame:
    rec, pre = [], []
    for t in np.unique(p)[::-1]:
        pred = p >= t
        tp = float((pred & (y == 1)).sum())
        pre.append(tp / max(pred.sum(), 1))
        rec.append(tp / max((y == 1).sum(), 1))
    return pd.DataFrame({"recall": rec, "precision": pre})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=".")
    ap.add_argument("--taskA-runs", required=True)
    ap.add_argument("--taskA-eval", default="github_repo/results/taskA/evaluation")
    ap.add_argument("--approved", action="store_true")
    args = ap.parse_args()
    project, runs = Path(args.project).resolve(), Path(args.taskA_runs)
    a_eval = project / args.taskA_eval
    out = project / "results/taskC/evaluation"
    cmp_dir = project / "results/comparison"

    fp = project / "results/taskC/modeling/final_selection_taskC.json"
    if not fp.exists():
        print("STOP: the frozen Task C specification does not exist. Run scripts/37 first.")
        return 1
    frozen = json.loads(fp.read_text(encoding="utf-8"))

    for role, rec in frozen["input_sha256"].items():
        p = project / rec["path"]
        if not p.exists():
            p = Path(rec["path"])
        if not p.exists() or tc.sha256(p) != rec["sha256"]:
            print(f"STOP: input '{role}' is missing or has changed since the freeze")
            return 1
    print(f"frozen specification verified ({len(frozen['input_sha256'])} inputs, hashes match)")

    if not args.approved:
        print("\nSTOP: the Test set is not opened without --approved.")
        return 1
    if (out / "test_predictions_taskC.csv").exists():
        print("STOP: Task C Test predictions already exist. The Test set is evaluated once.")
        return 1
    out.mkdir(parents=True, exist_ok=True)
    cmp_dir.mkdir(parents=True, exist_ok=True)
    print("APPROVED RUN: opening the Test set.")

    split = pd.read_csv(project / "data/splits/COVID19_固定患者split_1277.csv",
                        dtype=str, encoding="utf-8-sig")
    test = tc.load_split("test", a_eval / "test_predictions_primary.csv",
                         project / "results/taskB/evaluation/test_predictions_taskB.csv",
                         split, tc.TEST_CLINICAL_COLUMNS)
    y = test.y
    print(f"test: {len(y)} patients, {int(y.sum())} deaths")

    # ---- the frozen primary model ---------------------------------------------------
    w = frozen["fusion"]["w_clinical"]
    clin_name = frozen["clinical_component"]["model"]
    thr = frozen["threshold"]["primary"]
    bins, scheme = (frozen["calibration_definition"]["ece_bins"],
                    frozen["calibration_definition"]["ece_scheme"])
    boot = frozen["bootstrap"]
    p_fused = tc.fuse(test.clinical(clin_name), test.cxr(), w, "probability")
    print(f"primary: p_fused = {w} * p_clinical_{clin_name} + {round(1 - w, 10)} * p_cxr")
    print(f"threshold {thr:.6f} (Validation Youden, frozen; not recomputed on Test)")

    models = {"late_fusion": p_fused, "cxr": test.cxr(),
              "clinical_lr": test.clinical("lr"), "clinical_xgboost": test.clinical("xgb"),
              "clinical_mlp": test.clinical("mlp")}
    rows = []
    for name, p in models.items():
        auc = bootstrap_ci(y, p, roc_auc, n_boot=boot["n_resamples"], seed=boot["seed"],
                           stratified=boot["stratified"])
        apr = bootstrap_ci(y, p, average_precision, n_boot=boot["n_resamples"],
                           seed=boot["seed"], stratified=boot["stratified"])
        cal = pd.DataFrame(calibration_bins(y, p, n_bins=bins, scheme=scheme))
        ece = float(expected_calibration_error(y, p, n_bins=bins, scheme=scheme))
        r = binary_rates(y, p, thr) if name == "late_fusion" else {}
        rows.append({"model": name,
                     "role": "PRIMARY" if name == "late_fusion" else "component (reference)",
                     "n": len(y), "events": int(y.sum()),
                     "auroc": round(auc["point"], 6), "auroc_ci_low": round(auc["ci_low"], 6),
                     "auroc_ci_high": round(auc["ci_high"], 6),
                     "auprc": round(apr["point"], 6), "auprc_ci_low": round(apr["ci_low"], 6),
                     "auprc_ci_high": round(apr["ci_high"], 6),
                     "brier": round(float(brier_score(y, p)), 6),
                     "ece_10_equal_width": round(ece, 6),
                     **{k: (round(float(v), 6) if isinstance(v, (int, float)) else v)
                        for k, v in r.items()}})
        cal.to_csv(out / f"test_calibration_{name}.csv", index=False, encoding="utf-8-sig")
        rc = roc_curve(y, p)
        pd.DataFrame({"fpr": rc["fpr"], "tpr": rc["tpr"],
                      "threshold": rc["thresholds"]}).to_csv(
            out / f"test_roc_points_{name}.csv", index=False, encoding="utf-8-sig")
        pr_points(y, p).to_csv(out / f"test_pr_points_{name}.csv", index=False,
                               encoding="utf-8-sig")
        print(f"  {name:16s} AUROC {auc['point']:.6f} ({auc['ci_low']:.4f}-{auc['ci_high']:.4f})"
              f"  AUPRC {apr['point']:.6f}  Brier {brier_score(y, p):.6f}  ECE {ece:.6f}")
    table = pd.DataFrame(rows)
    table.to_csv(out / "test_metrics_table.csv", index=False, encoding="utf-8-sig")

    # ---- patient-level outputs ------------------------------------------------------
    pf = test.frame.copy()
    pf = pf.rename(columns={"prob_lr": "prob_clinical_lr", "prob_xgb": "prob_clinical_xgboost",
                            "prob_mlp": "prob_clinical_mlp",
                            "prob_xgb_native": "prob_clinical_xgboost_native"})
    pf[PRIMARY_COL] = p_fused
    pf["pred_late_fusion"] = (p_fused >= thr).astype(int)
    pf.to_csv(out / "test_predictions_taskC.csv", index=False, encoding="utf-8-sig")
    combined = pf[COMBINED_COLUMNS].copy()
    combined.to_csv(cmp_dir / "test_predictions_all_models.csv", index=False,
                    encoding="utf-8-sig")
    print(f"\ncombined file: {len(combined)} patients, columns {list(combined.columns)}")

    # ---- modality importance on Test: post hoc descriptive only ----------------------
    plan = frozen["modality_importance_plan"]
    rng = np.random.default_rng(plan["seed"])
    base = float(roc_auc(y, p_fused))
    imp = []
    for mod, arr in (("clinical", test.clinical(clin_name)), ("cxr", test.cxr())):
        drops = []
        for _ in range(plan["n_repeats"]):
            sh = rng.permutation(arr)
            pfz = (tc.fuse(sh, test.cxr(), w, "probability") if mod == "clinical"
                   else tc.fuse(test.clinical(clin_name), sh, w, "probability"))
            drops.append(base - roc_auc(y, pfz))
        imp.append({"modality": mod, "mean_auroc_drop": round(float(np.mean(drops)), 6),
                    "sd_auroc_drop": round(float(np.std(drops, ddof=1)), 6),
                    "n_repeats": plan["n_repeats"], "seed": plan["seed"], "dataset": "Test",
                    "baseline_auroc": round(base, 6),
                    "role": "post hoc descriptive; does not change any specification"})
    pd.DataFrame(imp).to_csv(out / "modality_importance_test.csv", index=False,
                             encoding="utf-8-sig")
    print("\nmodality importance (Test, post hoc descriptive):")
    print(pd.DataFrame(imp)[["modality", "mean_auroc_drop", "sd_auroc_drop"]].to_string(index=False))

    # ---- secondary / exploratory, kept strictly separate -----------------------------
    val = tc.load_split("val", runs / "lr3e-4_aug_b/seed42/val_predictions.csv",
                        project / "results/taskB/modeling/validation_predictions.csv",
                        split, tc.CLINICAL_COLUMNS)
    sec = {"role": ("secondary / exploratory. NOT used for model selection and NOT promoted to "
                    "primary. The primary specification was frozen before the Test set was "
                    "opened and is unchanged."), "models": {}}
    p_logit = tc.fuse(test.clinical(clin_name), test.cxr(), 0.70, "logit")
    sec["models"]["logit_space_weighted_average_w0.70"] = {
        "note": "weight chosen on Validation within the secondary arm",
        "auroc": round(float(roc_auc(y, p_logit)), 6),
        "auprc": round(float(average_precision(y, p_logit)), 6),
        "brier": round(float(brier_score(y, p_logit)), 6),
        "ece_10_equal_width": round(float(expected_calibration_error(
            y, p_logit, n_bins=bins, scheme=scheme)), 6)}
    _, predict = tc.fit_logistic_stack(val.clinical(clin_name), val.cxr(), val.y)
    p_stack = predict(test.clinical(clin_name), test.cxr())
    sec["models"]["logistic_stacking"] = {
        "note": ("3 parameters fitted on the 128 Validation patients (17 events), then applied "
                 "to Test; exploratory"),
        "auroc": round(float(roc_auc(y, p_stack)), 6),
        "auprc": round(float(average_precision(y, p_stack)), 6),
        "brier": round(float(brier_score(y, p_stack)), 6),
        "ece_10_equal_width": round(float(expected_calibration_error(
            y, p_stack, n_bins=bins, scheme=scheme)), 6)}
    (out / "secondary_exploratory_test.json").write_text(
        json.dumps(sec, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nsecondary (exploratory, not promoted): "
          f"logit-space AUROC {sec['models']['logit_space_weighted_average_w0.70']['auroc']:.6f}, "
          f"stacking AUROC {sec['models']['logistic_stacking']['auroc']:.6f}")

    # ---- metrics json, run metadata, access log ---------------------------------------
    prim = table[table.model == "late_fusion"].iloc[0]
    (out / "test_metrics_taskC.json").write_text(json.dumps({
        "model": "Task C Late Fusion (primary)",
        "formula": f"p_fused = {w} * p_clinical_{clin_name} + {round(1 - w, 10)} * p_cxr",
        "n": len(y), "events": int(y.sum()),
        "auroc": {"point": float(prim.auroc), "ci_low": float(prim.auroc_ci_low),
                  "ci_high": float(prim.auroc_ci_high), **boot},
        "auprc": {"point": float(prim.auprc), "ci_low": float(prim.auprc_ci_low),
                  "ci_high": float(prim.auprc_ci_high), **boot},
        "brier": float(prim.brier), "ece": float(prim.ece_10_equal_width),
        "calibration_definition": frozen["calibration_definition"],
        "threshold": {"value": thr, "source": frozen["threshold"]["rule"],
                      "recomputed_on_test": False},
        "operating_point": {k: float(prim[k]) for k in
                            ("sensitivity", "specificity", "ppv", "npv", "tp", "tn", "fp", "fn")
                            if k in prim.index},
        "frozen_sha256": tc.sha256(fp),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    meta = {"finished": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
            "script": Path(__file__).name, "split_evaluated": "test",
            "frozen_sha256": tc.sha256(fp), "seed": plan["seed"], "bootstrap": boot,
            "python": platform.python_version(), "platform": platform.platform(),
            "packages": {m: __import__(m).__version__
                         for m in ("numpy", "pandas", "sklearn", "torch", "xgboost")},
            "inputs": frozen["input_sha256"],
            "outputs": sorted(p.name for p in out.glob("*")),
            "specification_changed": False}
    (out / "test_run_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                            encoding="utf-8")
    qc = project / "results/taskC/qc"
    qc.mkdir(parents=True, exist_ok=True)
    with open(qc / "test_access_log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"event": "test evaluation", **meta}, ensure_ascii=False) + "\n")
    print(f"\nwrote {len(list(out.glob('*')))} files to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
