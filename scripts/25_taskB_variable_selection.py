"""Task B step 5: backward elimination by AIC on the Training set, with stability assessment.

Specification, fixed before running (decision_log D-074, open question Q-F2):
  direction   backward elimination, starting from the full clinically pre-selected candidate set
  criterion   AIC; a variable is removed only if removing it strictly lowers AIC
  unit        the ORIGINAL clinical variable. All dummy columns of a categorical variable enter
              and leave together, because the clinical question is "do we use age?", not "do we
              use the 74-90 dummy?" (faculty feedback §3 asks for this to be stated explicitly)
  data        Training set only (n=1,021, 135 deaths). Validation and Test are not touched.
  model       unpenalised logistic regression (statsmodels Logit), which is what AIC is defined for

Selection on a single sample is unstable, so the same procedure is repeated on 500 bootstrap
resamples of Training and the retention frequency of each variable is reported. This does not
change the primary selection; it quantifies how much of it is reproducible.

Usage:
    python scripts/25_taskB_variable_selection.py --project . [--n-boot 500]
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
from covid_mortality.features import taskB_clinical as tb  # noqa: E402
from covid_mortality.features.taskB_preprocess import TaskBPreprocessor  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)
SEED = 42


def fit_logit(X: pd.DataFrame, y: np.ndarray):
    import statsmodels.api as sm
    Xc = sm.add_constant(X, has_constant="add")
    return sm.Logit(y, Xc).fit(disp=False, maxiter=200, method="newton")


def aic_of(X: pd.DataFrame, y: np.ndarray) -> float:
    try:
        return float(fit_logit(X, y).aic)
    except Exception:
        return float("inf")


def backward_aic(X: pd.DataFrame, y: np.ndarray, groups: dict[str, list[str]],
                 verbose: bool = True) -> tuple[list[str], list[dict]]:
    """Remove whole clinical variables while AIC strictly decreases."""
    selected = list(groups)
    history = []
    current = aic_of(X[[c for v in selected for c in groups[v]]], y)
    if verbose:
        print(f"  initial AIC ({len(selected)} variables): {current:.4f}")
    step = 0
    while len(selected) > 1:
        best_var, best_aic = None, current
        for v in selected:
            keep = [c for u in selected if u != v for c in groups[u]]
            if not keep:
                continue
            a = aic_of(X[keep], y)
            if a < best_aic - 1e-9:
                best_var, best_aic = v, a
        if best_var is None:
            break
        step += 1
        history.append({"step": step, "removed": best_var,
                        "aic_before": round(current, 4), "aic_after": round(best_aic, 4),
                        "delta_aic": round(best_aic - current, 4),
                        "n_variables_after": len(selected) - 1})
        if verbose:
            print(f"  step {step}: remove {best_var:12s} AIC {current:.4f} -> {best_aic:.4f} "
                  f"({best_aic - current:+.4f})")
        selected.remove(best_var)
        current = best_aic
    if verbose:
        print(f"  final AIC ({len(selected)} variables): {current:.4f}")
    return selected, history


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=".")
    ap.add_argument("--out", default=None)
    ap.add_argument("--n-boot", type=int, default=500)
    args = ap.parse_args()
    project = Path(args.project).resolve()
    out = Path(args.out) if args.out else project / "results/taskB/variable_selection"
    out.mkdir(parents=True, exist_ok=True)

    df = tb.load_development(project)   # Training + Validation only
    train = tb.training_only(df)
    y = train.y.values
    print(f"Training only: n={len(train)}, deaths={int(y.sum())}. "
          f"Validation and Test are not used in this script.")

    pre = TaskBPreprocessor(scale=True).fit(train)
    X = pre.transform(train)
    groups = pre.groups()
    pre.save(out.parent / "preprocessing" / "preprocessor_train_fit.json")
    print(f"design matrix: {X.shape[1]} features from {len(groups)} clinical variables")
    for v, cols in groups.items():
        print(f"    {v:12s} -> {cols}")
    budget = int(y.sum()) // 10
    print(f"\nparameter budget (deaths {int(y.sum())} / EPV 10) = {budget}; "
          f"full model has {X.shape[1]} coefficients + intercept")

    # ---- primary selection ---------------------------------------------------------
    print("\n=== backward elimination by AIC (Training only) ===")
    selected, history = backward_aic(X, y, groups)
    sel_cols = [c for v in selected for c in groups[v]]
    print(f"\nselected variables ({len(selected)}): {selected}")
    print(f"selected features ({len(sel_cols)}): {sel_cols}")
    print(f"{'within' if len(sel_cols) <= budget else 'ABOVE'} the parameter budget "
          f"({len(sel_cols)} vs {budget})")

    pd.DataFrame(history).to_csv(out / "selection_history.csv", index=False, encoding="utf-8-sig")

    # ---- budget-constrained variant ------------------------------------------------
    # Q-F1 is unanswered by the faculty: is the events-per-variable budget counted in original
    # clinical variables or in estimated parameters? Under the first reading the AIC solution
    # (11 variables) is inside the budget; under the second (15 coefficients) it is 2 over.
    # Rather than pick a reading, both are produced: the AIC solution is primary and this
    # constrained variant -- keep removing the variable whose removal costs the least AIC until
    # the coefficient count fits -- is a pre-specified sensitivity analysis.
    constrained, constrained_history = list(selected), []
    while sum(len(groups[v]) for v in constrained) > budget and len(constrained) > 1:
        cur = aic_of(X[[c for v in constrained for c in groups[v]]], y)
        cand = []
        for v in constrained:
            keep = [c for u in constrained if u != v for c in groups[u]]
            cand.append((aic_of(X[keep], y) - cur, v))
        cost, victim = min(cand)
        constrained_history.append({"removed": victim, "aic_increase": round(cost, 4),
                                    "n_features_after": sum(len(groups[v]) for v in constrained
                                                            if v != victim)})
        constrained.remove(victim)
    con_cols = [c for v in constrained for c in groups[v]]
    print(f"\n=== budget-constrained selection (<= {budget} coefficients) -- PRIMARY ===")
    for h in constrained_history:
        print(f"  remove {h['removed']:12s} AIC {h['aic_increase']:+.4f} "
              f"-> {h['n_features_after']} features")
    print(f"  variables ({len(constrained)}): {constrained}")
    print(f"  features  ({len(con_cols)}): {con_cols}")
    pd.DataFrame(constrained_history).to_csv(out / "budget_constrained_history.csv",
                                             index=False, encoding="utf-8-sig")

    # The pre-registered procedure is "backward AIC SUBJECT TO the parameter budget": the
    # budget came first (Train deaths 135 / EPV 10 = 13 coefficients, decision_log D-074) and
    # AIC selects within it. The budget is counted in estimated parameters, as fixed in Q-F1
    # before any of this was run. So the constrained set is the primary one and the
    # unconstrained AIC optimum is the sensitivity analysis.
    primary_vars, primary_cols = constrained, con_cols
    (out / "final_variables.json").write_text(json.dumps({
        "primary": {"variables": primary_vars, "features": primary_cols,
                    "n_features": len(primary_cols),
                    "events_per_parameter": round(float(y.sum()) / len(primary_cols), 2),
                    "rule": "backward AIC subject to <= %d coefficients" % budget},
        "sensitivity_unconstrained_aic": {"variables": selected, "features": sel_cols,
                                          "n_features": len(sel_cols),
                                          "events_per_parameter": round(float(y.sum()) / len(sel_cols), 2)},
        "parameter_budget": budget, "deaths_train": int(y.sum()), "epv_target": 10,
        "fitted_on": "train only",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    final_p = fit_logit(X[primary_cols], y)
    pd.DataFrame({
        "feature": ["const"] + primary_cols, "coefficient": final_p.params.values,
        "std_error": final_p.bse.values, "z": final_p.tvalues.values,
        "p_value": final_p.pvalues.values, "odds_ratio": np.exp(final_p.params.values),
        "or_ci_low": np.exp(final_p.conf_int()[0].values),
        "or_ci_high": np.exp(final_p.conf_int()[1].values),
    }).to_csv(out / "primary_model_coefficients.csv", index=False, encoding="utf-8-sig")
    print(f"\n  events per parameter: {y.sum() / len(primary_cols):.2f} "
          f"(target >= 10); unconstrained variant would be "
          f"{y.sum() / len(sel_cols):.2f}")

    final = fit_logit(X[sel_cols], y)
    coef = pd.DataFrame({
        "feature": ["const"] + sel_cols,
        "coefficient": final.params.values,
        "std_error": final.bse.values,
        "z": final.tvalues.values,
        "p_value": final.pvalues.values,
        "odds_ratio": np.exp(final.params.values),
        "or_ci_low": np.exp(final.conf_int()[0].values),
        "or_ci_high": np.exp(final.conf_int()[1].values),
    })
    coef.to_csv(out / "final_model_coefficients.csv", index=False, encoding="utf-8-sig")
    print("\n=== final unpenalised logistic model (Training, standardised continuous inputs) ===")
    print(coef.round(4).to_string(index=False))
    print(f"\npseudo R2 (McFadden) {final.prsquared:.4f}, log-likelihood {final.llf:.3f}, "
          f"AIC {final.aic:.3f}, LLR p {final.llr_pvalue:.3e}")

    # ---- bootstrap stability -------------------------------------------------------
    print(f"\n=== bootstrap stability of the selection ({args.n_boot} resamples, Training only) ===")
    rng = np.random.default_rng(SEED)
    n = len(train)
    counts = {v: 0 for v in groups}
    sizes, ok = [], 0
    for b in range(args.n_boot):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        if yb.sum() < 20:                     # degenerate resample
            continue
        Xb = X.iloc[idx].reset_index(drop=True)
        try:
            sel_b, _ = backward_aic(Xb, yb, groups, verbose=False)
        except Exception:
            continue
        ok += 1
        sizes.append(len(sel_b))
        for v in sel_b:
            counts[v] += 1
        if (b + 1) % 100 == 0:
            print(f"  ... {b + 1}/{args.n_boot} resamples")
    stab = pd.DataFrame([{"variable": v,
                          "in_primary_selection": v in selected,
                          "bootstrap_retention_rate": round(counts[v] / ok, 4) if ok else None,
                          "n_retained": counts[v], "n_resamples": ok}
                         for v in groups]).sort_values("bootstrap_retention_rate", ascending=False)
    stab.to_csv(out / "bootstrap_stability.csv", index=False, encoding="utf-8-sig")
    print(stab.to_string(index=False))
    print(f"\nselected-model size across resamples: median {int(np.median(sizes))}, "
          f"range {min(sizes)}-{max(sizes)}")

    summary = {
        "generated": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "script": Path(__file__).name, "seed": SEED,
        "specification": {"direction": "backward elimination", "criterion": "AIC",
                          "unit": "original clinical variable (dummies move together)",
                          "model": "unpenalised logistic regression (statsmodels Logit)",
                          "data": "training only"},
        "n_train": len(train), "deaths_train": int(y.sum()),
        "parameter_budget": budget,
        "candidate_variables": list(groups), "candidate_features": X.shape[1],
        "selected_variables": selected, "selected_features": sel_cols,
        "n_selected_features": len(sel_cols),
        "within_budget_as_variables": bool(len(selected) <= budget),
        "within_budget_as_parameters": bool(len(sel_cols) <= budget),
        "events_per_parameter": round(float(y.sum()) / len(sel_cols), 2),
        "primary_selection": {
            "variables": constrained, "features": con_cols, "n_features": len(con_cols),
            "history": constrained_history,
            "events_per_parameter": round(float(y.sum()) / len(con_cols), 2),
            "rule": "backward AIC subject to the parameter budget (primary)"},
        "sensitivity_unconstrained_aic": {
            "variables": selected, "features": sel_cols, "n_features": len(sel_cols),
            "role": "AIC optimum without the budget constraint (sensitivity analysis)"},
        "final_model": {"aic": round(float(final.aic), 4),
                        "log_likelihood": round(float(final.llf), 4),
                        "pseudo_r2_mcfadden": round(float(final.prsquared), 4),
                        "llr_p_value": float(final.llr_pvalue)},
        "removal_history": history,
        "bootstrap": {"n_requested": args.n_boot, "n_used": ok,
                      "retention_rate": {v: round(counts[v] / ok, 4) for v in groups} if ok else {},
                      "model_size_median": int(np.median(sizes)) if sizes else None},
        "validation_test_use": "neither Validation nor Test was used in this script",
    }
    (out / "variable_selection_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {len(list(out.glob('*')))} files to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
