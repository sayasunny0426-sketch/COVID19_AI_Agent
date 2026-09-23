"""Paired DeLong comparisons over the combined Test predictions, with Holm within each family.

The comparison families and the correction were fixed before the Test set was opened
(results/comparison/delong_plan.json, decision_log D-080 / D-083). This script applies them and
does not choose anything.

    Family A   Late Fusion vs Clinical LR / Clinical XGBoost / Clinical MLP   (3, Holm)
    Family B   Late Fusion vs Clinical LR / CXR ResNet18                      (2, Holm)

`Late Fusion vs Clinical LR` belongs to both families and therefore carries two Holm-adjusted
p-values, one per family. The unadjusted p-value is the same in both, and `family` is a
required column so the two can never be confused.

All comparisons are paired: the same 128 Test patients, merged on subject_id. Two-sample AUC
comparisons are not used anywhere.

Outputs:
    results/comparison/delong_results.csv
    results/comparison/delong_results.json

Usage:
    python scripts/41_taskC_delong.py --project .
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
from covid_mortality.evaluation.delong import delong_roc_test  # noqa: E402
from covid_mortality.fusion import taskC_fusion as tc  # noqa: E402

COLUMN_OF = {"late_fusion": "prob_late_fusion", "clinical_lr": "prob_clinical_lr",
             "clinical_xgboost": "prob_clinical_xgboost", "clinical_mlp": "prob_clinical_mlp",
             "cxr": "prob_cxr"}
LABEL = {"late_fusion": "Late Fusion (Task C primary)", "clinical_lr": "Clinical LR",
         "clinical_xgboost": "Clinical XGBoost", "clinical_mlp": "Clinical MLP",
         "cxr": "CXR ResNet18"}


def holm(pvals: list[float]) -> list[float]:
    """Holm step-down adjusted p-values, monotone and capped at 1."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * pvals[idx]
        running = max(running, val)
        adj[idx] = min(running, 1.0)
    return [float(x) for x in adj]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=".")
    args = ap.parse_args()
    project = Path(args.project).resolve()
    cmp_dir = project / "results/comparison"
    pred_path = cmp_dir / "test_predictions_all_models.csv"
    plan_path = cmp_dir / "delong_plan.json"
    if not pred_path.exists():
        print("STOP: the combined Test prediction file does not exist. Run scripts/40 first.")
        return 1
    plan = json.loads(plan_path.read_text(encoding="utf-8"))

    df = pd.read_csv(pred_path, dtype={"subject_id": str}, encoding="utf-8-sig")
    y = df.true_label.to_numpy(int)
    print(f"paired comparisons over {len(df)} Test patients, {int(y.sum())} deaths")
    if df.subject_id.duplicated().any():
        print("STOP: duplicate subject_id in the combined file")
        return 1
    for c in COLUMN_OF.values():
        v = df[c].to_numpy(float)
        if np.isnan(v).any() or not ((v >= 0) & (v <= 1)).all():
            print(f"STOP: {c} has missing values or values outside [0,1]")
            return 1

    families = {
        "A": {"key": "A_late_fusion_vs_clinical_models",
              "pairs": [("late_fusion", "clinical_lr"),
                        ("late_fusion", "clinical_xgboost"),
                        ("late_fusion", "clinical_mlp")]},
        "B": {"key": "B_late_fusion_vs_reference_models",
              "pairs": [("late_fusion", "clinical_lr"), ("late_fusion", "cxr")]},
    }
    # the plan is the authority; verify this script matches it rather than assuming
    for fam, spec in families.items():
        planned = plan["families"][spec["key"]]["comparisons"]
        if len(planned) != len(spec["pairs"]):
            print(f"STOP: family {fam} has {len(spec['pairs'])} pairs but the plan has "
                  f"{len(planned)}")
            return 1
    print("family structure matches results/comparison/delong_plan.json")

    rows = []
    for fam, spec in families.items():
        res = []
        for a, b in spec["pairs"]:
            r = delong_roc_test(y, df[COLUMN_OF[a]].to_numpy(float),
                                df[COLUMN_OF[b]].to_numpy(float))
            res.append((a, b, r))
        adj = holm([r["p_value"] for _, _, r in res])
        for (a, b, r), pa in zip(res, adj):
            rows.append({
                "family": fam,
                "family_name": spec["key"],
                "model_1": a, "model_1_label": LABEL[a],
                "model_2": b, "model_2_label": LABEL[b],
                "n": len(y), "events": int(y.sum()),
                "AUC_1": round(float(r["auc_a"]), 6), "AUC_2": round(float(r["auc_b"]), 6),
                "delta_AUC": round(float(r["difference"]), 6),
                "standard_error": round(float(r["se"]), 6),
                "variance": round(float(r["se"]) ** 2, 8),
                "z": round(float(r["z"]), 6),
                "p_unadjusted": float(r["p_value"]),
                "p_holm_within_family": float(pa),
                "ci95_low": round(float(r["ci_low"]), 6),
                "ci95_high": round(float(r["ci_high"]), 6),
                "significant_at_0.05_after_holm": bool(pa < 0.05),
                "note": r.get("note", ""),
                "interpretation": (
                    f"{LABEL[a]} minus {LABEL[b]} = {r['difference']:+.4f} "
                    f"(95% CI {r['ci_low']:+.4f} to {r['ci_high']:+.4f}); "
                    f"Holm-adjusted p = {pa:.4f} within family {fam}. "
                    + ("The difference is statistically significant at the 0.05 level after "
                       "Holm correction." if pa < 0.05 else
                       "The difference is not statistically significant after Holm correction. "
                       "This is not evidence of equivalence: with 17 events the confidence "
                       "interval is wide and includes differences that would matter clinically.")),
            })
    out = pd.DataFrame(rows)
    out.to_csv(cmp_dir / "delong_results.csv", index=False, encoding="utf-8-sig")

    print("\n=== paired DeLong results ===")
    print(out[["family", "model_1", "model_2", "AUC_1", "AUC_2", "delta_AUC",
               "standard_error", "z", "p_unadjusted", "p_holm_within_family",
               "significant_at_0.05_after_holm"]].to_string(index=False))

    dup = out[out.duplicated(subset=["model_1", "model_2"], keep=False)]
    if len(dup):
        print("\nthe comparison present in both families, with its two adjusted p-values:")
        print(dup[["family", "model_1", "model_2", "p_unadjusted",
                   "p_holm_within_family"]].to_string(index=False))

    payload = {
        "generated": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "script": Path(__file__).name,
        "test": plan["test"], "implementation": plan["implementation"],
        "pairing": plan["pairing"],
        "families": {k: plan["families"][v["key"]] for k, v in families.items()},
        "multiplicity_correction": plan["multiplicity_correction"],
        "note_on_overlap": plan["note_on_overlap"],
        "n": len(y), "events": int(y.sum()),
        "input_file": str(pred_path.relative_to(project)).replace("\\", "/"),
        "input_sha256": tc.sha256(pred_path),
        "plan_sha256": tc.sha256(plan_path),
        "comparisons": rows,
        "caution": ("A lack of statistical significance must not be interpreted as equivalence. "
                    "With 128 Test patients and 17 deaths these comparisons have limited power."),
    }
    (cmp_dir / "delong_results.json").write_text(json.dumps(payload, ensure_ascii=False,
                                                            indent=2), encoding="utf-8")
    print(f"\nwrote delong_results.csv and delong_results.json to {cmp_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
