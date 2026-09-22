"""Task B step 9: what does the missing-indicator rule actually change?

The rule adopted in D-070 forces `lymph_missing` into the design matrix, so backward AIC never
gets the option of dropping it on its own. Its adjusted coefficient turned out to be ~0, and
because it occupies one of the 13 coefficients allowed by the events-per-parameter budget, the
comorbidity count -- a 4C Mortality Score component -- was the variable pushed out.

This script makes that trade-off visible by running the whole selection twice on Training and
scoring both on Validation:

    A  forced-in   the indicator is part of the lymph variable group (the rule as specified)
    B  selectable  the indicator is its own selection unit, so AIC may drop it
    C  none        no missing indicator at all

The purpose is transparency, not re-specification. The primary specification is NOT changed on
the basis of these numbers, and the Test set is not read. Note on wording: the rule was
pre-specified before variable selection was run in this Task B analysis; it was not registered
in advance of the study.

Usage:
    python scripts/29_taskB_indicator_sensitivity.py --project .
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from covid_mortality.evaluation.metrics import average_precision, brier_score, roc_auc  # noqa: E402
from covid_mortality.features import taskB_clinical as tb  # noqa: E402
from covid_mortality.features.taskB_preprocess import TaskBPreprocessor  # noqa: E402
from covid_mortality.training import taskB_models as tm  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module  # noqa: E402

vs = import_module("25_taskB_variable_selection")
warnings.filterwarnings("ignore")
ARMS = {
    "A_forced_in": "指示変数を lymph 変数群に含め、選択の対象外とする（D-070 の規則どおり）",
    "B_selectable": "指示変数を独立した選択単位とし、AIC が落とせるようにする",
    "C_no_indicator": "欠測指示変数を一切作らない",
}


def build(train, arm: str):
    pre = TaskBPreprocessor(missing_indicators=() if arm == "C_no_indicator" else ("lymph",))
    pre.fit(train)
    X = pre.transform(train)
    groups = pre.groups()
    if arm == "B_selectable" and "lymph_missing" in X.columns:
        groups = {k: [c for c in v if c != "lymph_missing"] for k, v in groups.items()}
        groups["lymph_missing"] = ["lymph_missing"]
    return pre, X, groups


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=".")
    args = ap.parse_args()
    project = Path(args.project).resolve()
    out = project / "results/taskB/variable_selection"
    out.mkdir(parents=True, exist_ok=True)

    dev = tb.load_development(project)
    train, val = tb.training_only(dev), dev[dev.split == "val"]
    y, yv = train.y.values, val.y.values
    budget = int(y.sum()) // 10
    print(f"Training {len(train)} (deaths {int(y.sum())}), Validation {len(val)} "
          f"(deaths {int(yv.sum())}). Test not read. Budget {budget} coefficients.\n")

    rows = []
    for arm, desc in ARMS.items():
        pre, X, groups = build(train, arm)
        selected, _ = vs.backward_aic(X, y, groups, verbose=False)
        # apply the same budget constraint as the primary pipeline
        while sum(len(groups[v]) for v in selected) > budget and len(selected) > 1:
            cur = vs.aic_of(X[[c for v in selected for c in groups[v]]], y)
            cost, victim = min((vs.aic_of(X[[c for u in selected if u != v
                                             for c in groups[u]]], y) - cur, v)
                               for v in selected)
            selected.remove(victim)
        cols = [c for v in selected for c in groups[v]]

        # Validation performance of the logistic model on that variable set, using the
        # condition chosen for the primary LR (C=1.0, class_weight balanced)
        fp, make = tm.logistic_factory(1.0, "balanced")
        cv = tm.cv_score(train, cols, fp)
        m = make().fit(X[cols].to_numpy(float), y)
        pv = m.predict_proba(pre.transform(val)[cols].to_numpy(float))[:, 1]
        rows.append({
            "arm": arm, "description": desc,
            "indicator_in_model": "lymph_missing" in cols,
            "n_variables": len([v for v in selected if v != "lymph_missing"]),
            "n_coefficients": len(cols),
            "selected_variables": ", ".join(v for v in selected if v != "lymph_missing"),
            "comorbidity_retained": "comorbidity" in selected,
            "cv_mean_auroc": round(cv.mean_auroc, 6), "cv_sd_auroc": round(cv.sd_auroc, 6),
            "val_auroc": round(float(roc_auc(yv, pv)), 6),
            "val_auprc": round(float(average_precision(yv, pv)), 6),
            "val_brier": round(float(brier_score(yv, pv)), 6),
        })
        print(f"[{arm}] {len(cols)} coefficients, indicator in model: "
              f"{'lymph_missing' in cols}, comorbidity retained: {'comorbidity' in selected}")
        print(f"    variables: {[v for v in selected if v != 'lymph_missing']}")
        print(f"    CV {cv.mean_auroc:.4f}+-{cv.sd_auroc:.4f}  "
              f"Validation AUROC {roc_auc(yv, pv):.4f}  AUPRC {average_precision(yv, pv):.4f}\n")

    res = pd.DataFrame(rows)
    res.to_csv(out / "indicator_rule_sensitivity.csv", index=False, encoding="utf-8-sig")
    print(res[["arm", "n_coefficients", "indicator_in_model", "comorbidity_retained",
               "cv_mean_auroc", "val_auroc", "val_auprc", "val_brier"]].to_string(index=False))

    (out / "indicator_rule_sensitivity.json").write_text(json.dumps({
        "generated": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "script": Path(__file__).name,
        "purpose": ("make the effect of the missing-indicator rule visible before the primary "
                    "specification is frozen; the specification is not changed by these numbers"),
        "wording": ("the rule was pre-specified before variable selection was run in this Task B "
                    "analysis, not registered in advance of the study"),
        "datasets_used": "Training (selection, CV) and Validation (reported performance)",
        "test_used": False,
        "arms": rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote indicator_rule_sensitivity.csv/.json to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
