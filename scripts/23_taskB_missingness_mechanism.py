"""Task B step 3: decide the missing-data handling from evidence, not from a blanket rule.

Faculty feedback §4 asks for missing indicators and XGBoost native missing handling to be
*evaluated*, not assumed. This script evaluates them on the Training set only:

  1. per variable: missing rate, death rate among missing vs observed, risk difference,
     odds ratio and Fisher exact p;
  2. the missing mechanism: how much of the missingness is explained by visit type
     (Emergency Room vs Inpatient), i.e. by the care pathway rather than by the patient;
  3. the decisive test -- does missingness still carry outcome information *within* the
     Inpatient stratum? If missingness predicts death only across strata and not within
     them, an indicator is a triage proxy, not a severity marker.

It also defines the inpatient sensitivity population precisely and counts it per split.

No model is fitted here and no Test outcome is used for any decision; split composition is
reported only to show the sensitivity analysis is feasible.

Usage:
    python scripts/23_taskB_missingness_mechanism.py --project .
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from covid_mortality.features import taskB_clinical as tb  # noqa: E402

INPATIENT = "Inpatient Visit"


def assoc(missing: np.ndarray, y: np.ndarray) -> dict:
    """Association between 'value is missing' and death, with an exact test."""
    a = int(((missing == 1) & (y == 1)).sum())   # missing, died
    b = int(((missing == 1) & (y == 0)).sum())   # missing, survived
    c = int(((missing == 0) & (y == 1)).sum())   # observed, died
    d = int(((missing == 0) & (y == 0)).sum())   # observed, survived
    r_miss = a / (a + b) if a + b else float("nan")
    r_obs = c / (c + d) if c + d else float("nan")
    try:
        or_, p = stats.fisher_exact([[a, b], [c, d]])
    except ValueError:
        or_, p = float("nan"), float("nan")
    return {"n_missing": a + b, "n_observed": c + d,
            "death_rate_missing": round(r_miss, 4), "death_rate_observed": round(r_obs, 4),
            "risk_difference": round(r_miss - r_obs, 4),
            "odds_ratio_missing_vs_observed": round(float(or_), 3) if or_ == or_ else None,
            "fisher_p": float(p)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=".")
    ap.add_argument("--cohort-flow", action="store_true",
                    help="also emit the cohort-level visit-type tables. This READS "
                         "Test metadata (visit type and split counts) and is off by "
                         "default; it was run once, with the researcher's agreement, "
                         "to define the inpatient sensitivity population.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    project = Path(args.project).resolve()
    out = Path(args.out) if args.out else project / "results/taskB/preprocessing"
    out.mkdir(parents=True, exist_ok=True)

    df = tb.load_development(project)   # Training + Validation only
    train = tb.training_only(df)
    y = train.y.values
    inpat = (train.visit_concept_name == INPATIENT).values
    print(f"Training only: {len(train)} patients, {int(y.sum())} deaths; "
          f"inpatient {int(inpat.sum())}, other {int((~inpat).sum())}")

    # ---- 1-3. per-variable missingness evidence -----------------------------------
    rows = []
    for v in tb.CANDIDATES:
        if v.kind == "derived":
            s = tb.comorbidity_count(train)
        else:
            s = train[v.column]
        m = s.isna().values.astype(int)
        if m.sum() == 0:
            continue
        overall = assoc(m, y)

        # how much of the missingness sits in the non-inpatient (ER) pathway
        share_er = float((m == 1)[~inpat].sum() / m.sum())
        miss_rate_inpat = float(m[inpat].mean())
        miss_rate_er = float(m[~inpat].mean())

        # the decisive test: within the Inpatient stratum only
        within = assoc(m[inpat], y[inpat])

        rows.append({
            "variable": v.name, "domain": v.domain,
            "missing_rate_train": round(float(m.mean()), 4),
            "missing_rate_inpatient": round(miss_rate_inpat, 4),
            "missing_rate_ER": round(miss_rate_er, 4),
            "share_of_missing_that_is_ER": round(share_er, 4),
            **{f"all_{k}": val for k, val in overall.items()},
            **{f"inpat_{k}": val for k, val in within.items()},
        })
    ev = pd.DataFrame(rows)
    ev["all_fisher_p"] = ev["all_fisher_p"].map(lambda p: f"{p:.3e}")
    ev["inpat_fisher_p"] = ev["inpat_fisher_p"].map(lambda p: f"{p:.3e}")
    ev.to_csv(out / "missingness_mechanism.csv", index=False, encoding="utf-8-sig")

    print("\n=== A. missingness vs death, whole Training set ===")
    print(ev[["variable", "missing_rate_train", "all_death_rate_missing",
              "all_death_rate_observed", "all_risk_difference",
              "all_odds_ratio_missing_vs_observed", "all_fisher_p"]].to_string(index=False))

    print("\n=== B. where does the missingness come from? ===")
    print(ev[["variable", "missing_rate_inpatient", "missing_rate_ER",
              "share_of_missing_that_is_ER"]].to_string(index=False))

    print("\n=== C. decisive test: the same association WITHIN the inpatient stratum ===")
    print(ev[["variable", "inpat_n_missing", "inpat_death_rate_missing",
              "inpat_death_rate_observed", "inpat_risk_difference",
              "inpat_odds_ratio_missing_vs_observed", "inpat_fisher_p"]].to_string(index=False))

    # ---- the shared "no lab panel ordered" pattern ---------------------------------
    labs = [v for v in tb.CANDIDATES
            if v.domain in ("腎（急性）", "炎症", "血液", "凝固", "組織灌流", "腎（慢性）", "心筋")]
    M = pd.DataFrame({v.name: train[v.column].isna().astype(int) for v in labs})
    nolab = (M.sum(axis=1) == len(labs)).values
    panel = {
        "n_lab_variables": len(labs),
        "patients_with_no_lab_at_all": int(nolab.sum()),
        "of_which_ER": int((nolab & ~inpat).sum()),
        "of_which_inpatient": int((nolab & inpat).sum()),
        "death_rate_no_lab": round(float(y[nolab].mean()), 4),
        "death_rate_any_lab": round(float(y[~nolab].mean()), 4),
        "min_pairwise_phi": round(float(M.corr().values[np.triu_indices(len(labs), 1)].min()), 3),
        "max_pairwise_phi": round(float(M.corr().values[np.triu_indices(len(labs), 1)].max()), 3),
        "within_inpatient": assoc(nolab[inpat].astype(int), y[inpat]),
    }
    print("\n=== D. the shared 'no lab panel ordered' pattern ===")
    print(json.dumps(panel, ensure_ascii=False, indent=2))

    # ---- inpatient sensitivity population, defined precisely -----------------------
    # This block needs the visit type of every split, so it reads Test metadata. It is off by
    # default and was run once, with the researcher's agreement, to define the population.
    if args.cohort_flow:
        print("\n=== E. sensitivity population definition ===")
        comp = (df.assign(is_death=df.y)
                  .groupby(["split", "visit_concept_name"])
                  .agg(n=("y", "size"), deaths=("y", "sum")).reset_index())
        comp["death_rate"] = (comp.deaths / comp.n).round(4)
        comp.to_csv(out / "visit_type_composition.csv", index=False, encoding="utf-8-sig")
        print(comp.to_string(index=False))
        tot = (df.groupby("visit_concept_name").agg(n=("y", "size"), deaths=("y", "sum")))
        tot["death_rate"] = (tot.deaths / tot.n).round(4)
        print("\nwhole fixed cohort (1,277) by visit type:")
        print(tot.to_string())

        los = pd.to_numeric(train["length_of_stay"], errors="coerce")
        print("\nlength of stay by visit type (Training; context only, never a predictor):")
        print(train.assign(los=los).groupby("visit_concept_name").los
              .describe()[["count", "mean", "50%", "max"]].round(2).to_string())
    else:
        print('\n[cohort-flow skipped: Test metadata not read. Re-run with --cohort-flow if the cohort-level tables are needed.]')

    summary = {
        "generated": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "script": Path(__file__).name,
        "evaluated_on": "training set only (n=1021, deaths=135)",
        "no_lab_panel_pattern": panel,
        "cohort_flow_emitted": bool(args.cohort_flow),
        "note": ("Training-only analysis. The cohort-level visit-type tables, which read Test "
                 "metadata, are emitted only with --cohort-flow."),
    }
    if args.cohort_flow:
        summary["cohort_by_visit_type"] = {k: {"n": int(r.n), "deaths": int(r.deaths),
                                               "death_rate": float(r.death_rate)}
                                           for k, r in tot.iterrows()}
        summary["split_by_visit_type"] = comp.to_dict(orient="records")
    (out / "missingness_mechanism_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote missingness_mechanism.csv and the summary to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
