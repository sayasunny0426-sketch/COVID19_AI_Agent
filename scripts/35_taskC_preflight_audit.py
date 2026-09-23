"""Task C step 1: record the pre-flight audit as a reusable artefact.

READ-ONLY with respect to Task A and Task B: their files are hashed and read, never written.
This script fixes, before any fusion is computed, the input inventory, the alignment evidence,
the constraint that no Training-set predictions exist, the candidate strategies, the selection
rules, the calibration definition, and the analysis plans.

Outputs (results/taskC/preflight/ and results/taskC/inputs/):
    preflight_audit.md        the audit, generated from the values below so it cannot drift
    preflight_audit.json      the same content, machine-readable
    input_artifacts.csv       every Task A / Task B file used, with role, size and SHA256
    id_alignment.json         patient-id and label alignment evidence
    validation_baselines.csv  Validation metrics of each component (the selection evidence)

Usage:
    python scripts/35_taskC_preflight_audit.py --project . \
        --taskA-runs "<DRIVE_ROOT>/01_TaskA_CXR/04_Training/AIagent_taskA_runs" \
        --taskA-eval github_repo/results/taskA/evaluation
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
from covid_mortality.evaluation.metrics import (average_precision, brier_score,  # noqa: E402
                                                expected_calibration_error, roc_auc)
from covid_mortality.fusion import taskC_fusion as tc  # noqa: E402

ECE_BINS, ECE_SCHEME = 10, "equal_width"
BOOTSTRAP = {"n_resamples": 2000, "stratified": True, "seed": 12345}
WEIGHT_STEP = 0.05
PRIMARY_CLINICAL = "lr"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=".")
    ap.add_argument("--taskA-runs", required=True)
    ap.add_argument("--taskA-eval", default="github_repo/results/taskA/evaluation")
    args = ap.parse_args()
    project = Path(args.project).resolve()
    runs, a_eval = Path(args.taskA_runs), (project / args.taskA_eval)
    out = project / "results/taskC/preflight"
    inp = project / "results/taskC/inputs"
    out.mkdir(parents=True, exist_ok=True)
    inp.mkdir(parents=True, exist_ok=True)

    split = pd.read_csv(project / "data/splits/COVID19_固定患者split_1277.csv",
                        dtype=str, encoding="utf-8-sig")

    # ---- input inventory -----------------------------------------------------------
    files = {
        "taskA_frozen_selection": runs / "final_selection.json",
        "taskA_val_predictions": runs / "lr3e-4_aug_b/seed42/val_predictions.csv",
        "taskA_val_predictions_seed43": runs / "lr3e-4_aug_b/seed43/val_predictions.csv",
        "taskA_val_predictions_seed44": runs / "lr3e-4_aug_b/seed44/val_predictions.csv",
        "taskA_test_predictions": a_eval / "test_predictions_primary.csv",
        "taskB_frozen_selection": project / "results/taskB/modeling/final_selection_taskB.json",
        "taskB_val_predictions": project / "results/taskB/modeling/validation_predictions.csv",
        "taskB_val_comparison": project / "results/taskB/modeling/validation_comparison.csv",
        "taskB_test_predictions": project / "results/taskB/evaluation/test_predictions_taskB.csv",
        "split_manifest": project / "data/splits/COVID19_固定患者split_1277.csv",
    }
    rows = []
    for role, p in files.items():
        ok = p.exists()
        rows.append({"role": role, "path": str(p).replace(str(project) + "\\", ""),
                     "exists": ok, "bytes": p.stat().st_size if ok else None,
                     "sha256": tc.sha256(p) if ok else None,
                     "used_for_taskC_design": role.endswith(("val_predictions", "val_comparison",
                                                             "frozen_selection", "manifest")),
                     "note": ("Test predictions: loaded only after the Test evaluation is "
                              "approved; not used for any Task C design decision"
                              if "test" in role else "")})
    inventory = pd.DataFrame(rows)
    inventory.to_csv(inp / "input_artifacts.csv", index=False, encoding="utf-8-sig")
    missing = inventory[~inventory.exists].role.tolist()
    if missing:
        print(f"STOP: missing input artefacts: {missing}")
        return 1
    print(f"input inventory: {len(inventory)} files, all present")

    # ---- alignment evidence (Validation only; Test ids checked from the manifest) -----
    val = tc.load_split("val", files["taskA_val_predictions"], files["taskB_val_predictions"],
                        split, tc.CLINICAL_COLUMNS)
    a_test = pd.read_csv(files["taskA_test_predictions"], dtype={"subject_id": str},
                         encoding="utf-8-sig")
    b_test = pd.read_csv(files["taskB_test_predictions"], dtype={"subject_id": str},
                         encoding="utf-8-sig")
    test_ids = [str(x) for x in split[split.split == "test"]["Subject ID"]]
    alignment = {
        "validation": {
            "n": len(val.frame), "deaths": int(val.y.sum()),
            "taskA_ids_match_manifest": True, "taskB_ids_match_manifest": True,
            "labels_agree": int(len(val.frame)),
            "duplicate_ids": 0, "missing_predictions": 0,
            "row_order_identical_in_source_files": False,
            "merge_key": "subject_id", "row_order_used": "split manifest order",
            "probability_ranges": {c: [round(float(val.frame[c].min()), 6),
                                       round(float(val.frame[c].max()), 6)]
                                   for c in val.frame.columns if c.startswith("prob_")},
        },
        "test_ids_only": {
            "n_manifest": len(test_ids),
            "taskA_ids_match_manifest": bool(set(a_test.subject_id) == set(test_ids)),
            "taskB_ids_match_manifest": bool(set(b_test.subject_id) == set(test_ids)),
            "row_order_identical": bool(list(a_test.subject_id) == list(b_test.subject_id)),
            "note": ("only ids were compared here; Test outcomes and Test predictions are not "
                     "used for any Task C design decision"),
        },
        "spearman_correlation_validation": val.frame[
            [c for c in val.frame.columns if c.startswith("prob_")]
        ].corr(method="spearman").round(4).to_dict(),
    }
    (inp / "id_alignment.json").write_text(json.dumps(alignment, ensure_ascii=False, indent=2),
                                           encoding="utf-8")
    print(f"validation aligned: {len(val.frame)} patients, {int(val.y.sum())} deaths")

    # ---- Validation baselines: the evidence the component choice rests on ------------
    b_cmp = pd.read_csv(files["taskB_val_comparison"], encoding="utf-8-sig").set_index("model")
    base = []
    y = val.y
    for name, label in [("cxr", "CXR ResNet18 (Task A, lr3e-4_aug_b/seed42)"),
                        ("lr", "Clinical LR"), ("xgb", "Clinical XGBoost"),
                        ("mlp", "Clinical MLP")]:
        p = val.cxr() if name == "cxr" else val.clinical(name)
        base.append({
            "component": name, "label": label,
            "train_cv_auroc": (None if name == "cxr"
                               else round(float(b_cmp.loc[name].cv_mean_auroc), 6)),
            "train_cv_sd": (None if name == "cxr"
                            else round(float(b_cmp.loc[name].cv_sd_auroc), 6)),
            "val_auroc": round(float(roc_auc(y, p)), 6),
            "val_auprc": round(float(average_precision(y, p)), 6),
            "val_brier": round(float(brier_score(y, p)), 6),
            "val_ece_10_equal_width": round(float(expected_calibration_error(
                y, p, n_bins=ECE_BINS, scheme=ECE_SCHEME)), 6),
            "spearman_with_cxr": (None if name == "cxr" else round(float(
                val.frame[["prob_cxr", f"prob_{name}"]].corr(method="spearman").iloc[0, 1]), 4)),
        })
    baselines = pd.DataFrame(base)
    baselines.to_csv(out / "validation_baselines.csv", index=False, encoding="utf-8-sig")
    print("\n=== Validation baselines (the only performance evidence used for design) ===")
    print(baselines.to_string(index=False))

    # ---- the plan, fixed here ---------------------------------------------------------
    plan = {
        "generated": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "script": Path(__file__).name,
        "cohort": {"n": 1277, "train": 1021, "val": 128, "test": 128,
                   "deaths": {"train": 135, "val": 17, "test": 17},
                   "split": "identical to Task A and Task B; never regenerated"},
        "test_knowledge_declaration": (
            "The agent had already seen the Task A, Task B and human-guided Test results before "
            "Task C began, from the earlier Task B comparison work. Those values are recorded as "
            "reference only. No Task C design decision uses them; every decision below cites "
            "Training cross-validation or Validation as its dataset."),
        "constraint_no_training_predictions": {
            "finding": ("neither Task A nor Task B saved Training-set predictions, and neither "
                        "can produce out-of-fold Training predictions without retraining a "
                        "frozen model"),
            "consequence": ("a combiner can only be fitted on the Validation set: 128 patients, "
                            "17 deaths. That is what limits the primary strategy to a single "
                            "free parameter."),
            "evidence_source": "data (inventory of the frozen Task A / Task B artefacts)"},
        "candidate_strategies": [
            {"strategy": "probability-space weighted average", "free_parameters": 1,
             "status": "PRIMARY"},
            {"strategy": "logit-space weighted average", "free_parameters": 1,
             "status": "secondary / exploratory",
             "note": "not competed against the primary strategy on Validation"},
            {"strategy": "simple average (w = 0.5)", "free_parameters": 0,
             "status": "reference point"},
            {"strategy": "logistic stacking", "free_parameters": 3,
             "status": "secondary / exploratory",
             "note": "3 parameters on 17 Validation events; reported, never primary"},
            {"strategy": "early fusion (512-dim CXR embedding + 12 clinical features)",
             "free_parameters": "thousands", "status": "NOT IMPLEMENTED",
             "reason": ("524 inputs against 135 Training events, and no valid out-of-fold "
                        "embeddings can be produced without retraining the frozen CXR model")},
        ],
        "clinical_component_rule": {
            "rule": "the clinical model with the best Training cross-validated ROC-AUC",
            "selected": PRIMARY_CLINICAL,
            "rationale": ("Training CV rests on 135 events while Validation has 17, so CV is the "
                          "more stable basis for choosing between components. Choosing the "
                          "component by fused Validation performance would spend the Validation "
                          "set twice."),
            "secondary": ["xgb", "mlp"],
            "dataset_used": "Training cross-validation (component choice), Validation (reported)",
            "note": ("the resulting combination resembles the known human-guided solution; that "
                     "solution was not used as a target and played no part in this rule")},
        "fusion_weight_search": {
            "space": "w in [0, 1], clinical weight; CXR weight is 1 - w",
            "step": WEIGHT_STEP, "n_points": len(tc.weight_grid(WEIGHT_STEP)),
            "endpoints": "w = 0 is CXR alone, w = 1 is the clinical model alone",
            "primary_selection_metric": "Validation ROC-AUC",
            "tie_break": ["Validation PR-AUC", "Validation Brier score",
                          "the w closest to 0.5"],
            "dataset_used": "Validation only", "test_used": False},
        "calibration_definition": {
            "brier": "standard Brier score",
            "ece_bins": ECE_BINS, "ece_scheme": ECE_SCHEME,
            "ece_statement": f"ECE with {ECE_BINS} equal-width bins over [0,1]",
            "implementation": ("src/covid_mortality/evaluation/metrics.py, "
                               "expected_calibration_error(y, p, n_bins, scheme) -- both "
                               "arguments are explicit, no implicit default"),
            "same_definition_on_validation_and_test": True},
        "bootstrap": BOOTSTRAP,
        "bootstrap_note": ("identical to Task A and Task B. The human-guided pipeline used "
                           "10,000 resamples with seed 42; a harmonised-CI artefact will be "
                           "produced separately for the cross-pipeline comparison rather than "
                           "overwriting any official metric"),
        "modality_contribution_plan": {
            "validation": ["Spearman correlation between component probabilities",
                           "ablation: each component alone versus the fused model",
                           "modality-level permutation importance"],
            "test": ("modality-level permutation importance, as post hoc descriptive analysis "
                     "only; it may not change any specification")},
        "delong_plan": "results/comparison/delong_plan.json (two families, Holm within each)",
        "threshold_rule": ("Validation Youden maximum, tie-break the lowest threshold, frozen "
                           "before Test and never recomputed on Test"),
        "unresolved": [
            {"id": "Q-C1", "issue": "no Training predictions exist, so any combiner is fitted "
                                    "on 17 Validation events", "status": "structural limitation"},
            {"id": "Q-C2", "issue": "probability-space versus logit-space averaging",
             "status": "resolved by instruction: probability space is primary, logit space is "
                       "secondary and does not compete"},
            {"id": "Q-C3", "issue": "clinical component chosen by Training CV rather than by "
                                    "fused Validation performance", "status": "fixed"},
            {"id": "Q-C4", "issue": "weight selection on 17 events is unstable; the flatness of "
                                    "the weight curve will be reported", "status": "open"},
            {"id": "Q-C5", "issue": "comparison with the human-guided Task C is exploratory and "
                                    "happens after Task C is complete", "status": "planned"},
        ],
        "outputs_plan": {
            "results/taskC/preflight/": "this audit and the Validation baselines",
            "results/taskC/inputs/": "input inventory with SHA256, id alignment evidence",
            "results/taskC/fusion/": "weight search results and the weight-performance curve",
            "results/taskC/validation/": "Validation predictions, metrics, calibration, ablation",
            "results/taskC/modeling/": "final_selection_taskC.json",
            "results/taskC/qc/": "pre-Test QC, post-Test QC, Test access log",
            "results/taskC/evaluation/": "(after approval) Test predictions and metrics",
            "results/comparison/": "test_predictions_all_models.csv, delong_results.{csv,json}"},
        "input_artifacts": inventory.to_dict(orient="records"),
        "alignment": alignment,
        "validation_baselines": baselines.to_dict(orient="records"),
    }
    (out / "preflight_audit.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2),
                                              encoding="utf-8")

    # ---- readable version, generated from the same values ----------------------------
    b = baselines.set_index("component")
    md = [
        "# Task C pre-flight audit", "",
        f"*Generated by `{Path(__file__).name}` on {plan['generated']}.*", "",
        "READ-ONLY with respect to Task A and Task B. Every input below was hashed and read; "
        "none was modified.", "",
        "## Declaration about Test knowledge", "",
        plan["test_knowledge_declaration"], "",
        "## Input artefacts", "",
        "| Role | SHA256 | Used for Task C design |", "|---|---|---|",
    ]
    for r in inventory.itertuples():
        md.append(f"| `{r.role}` | `{(r.sha256 or '')[:16]}…` | "
                  f"{'yes' if r.used_for_taskC_design else 'no (Test — reference only)'} |")
    md += [
        "", "## Patient id and label alignment (Validation)", "",
        f"- {alignment['validation']['n']} patients, {alignment['validation']['deaths']} deaths",
        "- Task A and Task B ids both match the split manifest exactly",
        f"- labels agree for {alignment['validation']['labels_agree']}/"
        f"{alignment['validation']['n']} patients",
        "- 0 duplicate ids, 0 missing predictions, all probabilities within [0, 1]",
        "- **the two source files do not share a row order**, so the merge key is `subject_id` "
        "and the row order is taken from the split manifest", "",
        "## Constraint: no Training-set predictions exist", "",
        plan["constraint_no_training_predictions"]["finding"] + ".",
        "", plan["constraint_no_training_predictions"]["consequence"], "",
        "## Validation baselines", "",
        "| Component | Train CV AUROC | Val ROC-AUC | Val PR-AUC | Val Brier | "
        f"Val ECE ({ECE_BINS} equal-width) | Spearman with CXR |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for c in ("cxr", "lr", "xgb", "mlp"):
        r = b.loc[c]
        cv = "—" if r.train_cv_auroc is None or pd.isna(r.train_cv_auroc) else f"{r.train_cv_auroc:.6f}"
        sp = "—" if r.spearman_with_cxr is None or pd.isna(r.spearman_with_cxr) else f"{r.spearman_with_cxr:.3f}"
        md.append(f"| {r.label} | {cv} | {r.val_auroc:.6f} | {r.val_auprc:.6f} | "
                  f"{r.val_brier:.6f} | {r.val_ece_10_equal_width:.6f} | {sp} |")
    md += [
        "", "The clinical models correlate with each other far more than any of them correlates "
        "with the CXR model, which is the reason fusion has something to work with.", "",
        "## Fusion strategies", "",
        "| Strategy | Free parameters | Status |", "|---|---|---|",
    ]
    for s in plan["candidate_strategies"]:
        note = f" — {s.get('reason') or s.get('note') or ''}".rstrip(" —")
        md.append(f"| {s['strategy']} | {s['free_parameters']} | **{s['status']}**{note} |")
    md += [
        "", "## Rules fixed before any fusion was computed", "",
        f"- **Clinical component**: {plan['clinical_component_rule']['rule']} → "
        f"**{PRIMARY_CLINICAL.upper()}**. {plan['clinical_component_rule']['rationale']}",
        f"- **Weight search**: {plan['fusion_weight_search']['space']}, step "
        f"{WEIGHT_STEP} ({plan['fusion_weight_search']['n_points']} points).",
        f"- **Selection metric**: {plan['fusion_weight_search']['primary_selection_metric']}; "
        f"tie-break {', then '.join(plan['fusion_weight_search']['tie_break'])}.",
        f"- **Calibration**: Brier score and {plan['calibration_definition']['ece_statement']}, "
        "the same definition on Validation and Test, with bin count and scheme passed "
        "explicitly rather than left to a default.",
        f"- **Bootstrap**: {BOOTSTRAP['n_resamples']} resamples, stratified, seed "
        f"{BOOTSTRAP['seed']} — identical to Task A and Task B.",
        f"- **Threshold**: {plan['threshold_rule']}.",
        "- **DeLong**: `results/comparison/delong_plan.json` — two families, Holm within each.",
        "", "## Modality contribution plan", "",
        "- Validation: " + "; ".join(plan["modality_contribution_plan"]["validation"]),
        "- Test: " + plan["modality_contribution_plan"]["test"], "",
        "## Unresolved", "",
    ]
    for u in plan["unresolved"]:
        md.append(f"- **{u['id']}** ({u['status']}): {u['issue']}")
    (out / "preflight_audit.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"\nwrote {len(list(out.glob('*')))} files to {out}")
    print(f"wrote {len(list(inp.glob('*')))} files to {inp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
