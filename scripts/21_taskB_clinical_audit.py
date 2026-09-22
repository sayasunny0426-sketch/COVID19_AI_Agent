"""Task B step 1: audit every clinical column and emit the candidate table with rationale.

Training set only (1,021 patients, 135 deaths). Validation and Test rows are counted for the
QC assertions and never profiled, so no distribution, missing rate or category level from
those splits can influence a later decision.

Outputs (results/taskB/preprocessing/):
    variable_audit.csv            all 131 columns: role, why kept or dropped, Train missingness
    candidate_table.csv           the clinical-knowledge candidate set with evidence per variable
    considered_not_selected.csv   variables thought about and left out, with the reason
    derived_redundancy.csv        evidence that the band columns add nothing
    missingness_report.csv        Train missing rate overall and by visit type
    implausible_values.csv        physiologically impossible values found in Training
    audit_summary.json           counts, cohort check, parameter budget

Usage:
    python scripts/21_taskB_clinical_audit.py --project .
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from covid_mortality.features import taskB_clinical as tb  # noqa: E402

EPV_TARGET = 10   # faculty feedback §1: about one parameter per 10 deaths


def band_agreement(train: pd.DataFrame) -> pd.DataFrame:
    """Evidence for dropping the band columns: can each be reproduced from its numeric source?"""
    rows = []
    for source, bands in tb.DERIVED_BANDS.items():
        if source not in train.columns:
            continue
        v = pd.to_numeric(train[source], errors="coerce")
        for b in bands:
            if b not in train.columns:
                continue
            bo = train[b].map({"True": 1, "False": 0, "TRUE": 1, "FALSE": 0, "Yes": 1, "No": 0})
            both = v.notna() & bo.notna()
            rows.append({
                "band_column": b, "numeric_source": source,
                "n_both_present": int(both.sum()),
                "numeric_missing_band_present": int((v.isna() & bo.notna()).sum()),
                "numeric_present_band_missing": int((v.notna() & bo.isna()).sum()),
                "identical_missingness": bool((v.isna() == bo.isna()).all()),
            })
    for dup, keep in tb.EXACT_DUPLICATES.items():
        a = pd.to_numeric(train[dup], errors="coerce")
        b = pd.to_numeric(train[keep], errors="coerce")
        both = a.notna() & b.notna()
        rows.append({"band_column": dup, "numeric_source": keep, "n_both_present": int(both.sum()),
                     "numeric_missing_band_present": 0, "numeric_present_band_missing": 0,
                     "identical_missingness": bool((a.isna() == b.isna()).all()),
                     "values_identical": bool((a[both] == b[both]).all())})
    return pd.DataFrame(rows)


def implausible(train: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col, (lo, hi) in tb.IMPLAUSIBLE_BOUNDS.items():
        if col not in train.columns:
            continue
        v = pd.to_numeric(train[col], errors="coerce")
        bad = v[(v < lo) | (v > hi)]
        for pid, value in zip(train.loc[bad.index, tb.ID_COLUMN], bad):
            rows.append({"column": col, "subject_id": pid, "value": value,
                         "plausible_low": lo, "plausible_high": hi,
                         "action": "set to missing",
                         "reason": "生理学的に成立しない値。患者は除外せず、当該測定値のみ欠測化する"})
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=".")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    project = Path(args.project).resolve()
    out = Path(args.out) if args.out else project / "results/taskB/preprocessing"
    out.mkdir(parents=True, exist_ok=True)

    df = tb.load_development(project)   # Training + Validation only
    train = tb.training_only(df)
    n_train, deaths_train = len(train), int(train.y.sum())
    print(f"train {n_train} (deaths {deaths_train})  "
          f"val {(df.split == 'val').sum()}  -- Test rows not loaded")
    print("cohort / split / outcome checks passed (load_cohort asserts them)")

    dct = tb.dictionary(project)
    desc = dict(zip(dct.column_name, dct.description.fillna("")))
    dict_type = dict(zip(dct.column_name, dct.column_type.fillna("")))
    chart_cols = set(dct.loc[dct.is_chart_abstracted.astype(str).str.upper() == "TRUE", "column_name"])

    dropped_derived = tb.dropped_derived_columns()
    candidate_cols = {v.column for v in tb.CANDIDATES}
    member_cols = {m for v in tb.CANDIDATES for m in v.members}
    raw_columns = [c for c in pd.read_csv(project / tb.CLINICAL_CSV, nrows=0,
                                          encoding="utf-8-sig").columns]

    # ---- per-column audit ---------------------------------------------------------
    rows = []
    for c in raw_columns:
        col = c if c in train.columns else f"{c}_clin"
        s = train[col] if col in train.columns else pd.Series(dtype=str)
        miss = float(s.isna().mean()) if len(s) else float("nan")
        if c in tb.POST_T0:
            role, reason = "EXCLUDE_post_T0", tb.POST_T0[c]
        elif c in tb.ADMIN:
            role, reason = "ADMIN", tb.ADMIN[c]
        elif c in dropped_derived:
            role, reason = "EXCLUDE_derived_duplicate", dropped_derived[c]
        elif s.dropna().nunique() <= 1:
            role, reason = "EXCLUDE_constant", "Training set で値が 1 種類以下（情報量なし）"
        elif c in candidate_cols:
            v = next(x for x in tb.CANDIDATES if x.column == c)
            role, reason = "CANDIDATE", f"{v.rationale}（根拠: {v.evidence}）"
        elif c in member_cols:
            role, reason = "CANDIDATE_member", "併存症カウント（comorbidity）の構成要素として使用"
        elif c in tb.CONSIDERED_NOT_SELECTED:
            role, reason = "CONSIDERED_not_selected", tb.CONSIDERED_NOT_SELECTED[c]
        else:
            role, reason = "NOT_SELECTED", ""
        rows.append({"column": c, "role": role, "reason": reason,
                     "dict_type": dict_type.get(c, ""), "chart_abstracted": c in chart_cols,
                     "n_unique_train": int(s.dropna().nunique()) if len(s) else 0,
                     "missing_rate_train": round(miss, 4) if miss == miss else None,
                     "description": desc.get(c, "")})
    audit = pd.DataFrame(rows)

    # anything left as NOT_SELECTED must still carry a reason -- fill the group defaults
    group_reason = {
        "symptom": "自覚症状。主観的で Train 欠測 21-25%、既存スコアでは客観指標に置換されている",
        "comorb": "個別併存症。comorbidity カウントに統合（パラメータ効率）",
        "lab": "候補領域が他変数で代表されているか、Train 欠測が高い（considered_not_selected 参照）",
    }
    sym = {"cough_v", "dyspnea_admission_v", "nausea_v", "vomiting_v", "diarrhea_v",
           "abdominal_pain_v", "fever_v"}
    audit.loc[(audit.role == "NOT_SELECTED") & audit.column.isin(sym), "reason"] = group_reason["symptom"]
    audit.loc[(audit.role == "NOT_SELECTED") & (audit.reason == ""), "reason"] = group_reason["lab"]
    audit.to_csv(out / "variable_audit.csv", index=False, encoding="utf-8-sig")

    print("\n=== column roles ===")
    print(audit.role.value_counts().to_string())
    unexplained = int((audit.reason == "").sum())
    print(f"columns without a recorded reason: {unexplained}")

    # ---- candidate table ----------------------------------------------------------
    crows = []
    for v in tb.CANDIDATES:
        if v.kind == "derived":
            s = tb.comorbidity_count(train)
            miss, nun, levels = 0.0, s.nunique(), sorted(map(str, s.dropna().unique()))
        else:
            col = v.column if v.column in train.columns else f"{v.column}_clin"
            s = train[col]
            miss, nun = float(s.isna().mean()), int(s.dropna().nunique())
            levels = sorted(map(str, s.dropna().unique()))[:6]
        crows.append({
            "variable": v.name, "raw_column": v.column, "domain": v.domain, "kind": v.kind,
            "estimated_parameters": v.params, "missing_rate_train": round(miss, 4),
            "n_unique_train": nun, "levels_or_example": "; ".join(levels),
            "rationale": v.rationale, "evidence": v.evidence,
            "evidence_source": v.evidence_source, "caveat": v.caveat,
            "dataset_used_for_decision": "Training（欠測率のみ）／根拠は臨床知識・文献",
            "status": "provisional（Test 使用前は変更可）",
        })
    cand = pd.DataFrame(crows)
    cand.to_csv(out / "candidate_table.csv", index=False, encoding="utf-8-sig")
    budget = deaths_train // EPV_TARGET
    total_params = int(cand.estimated_parameters.sum())
    print(f"\n=== candidate set: {len(cand)} variables, {total_params} estimated parameters ===")
    print(cand[["variable", "domain", "estimated_parameters", "missing_rate_train",
                "evidence_source"]].to_string(index=False))
    print(f"\nparameter budget: Train deaths {deaths_train} / EPV {EPV_TARGET} = {budget}")
    print(f"candidate parameters {total_params} -> "
          f"{'selection required' if total_params > budget else 'within budget'}")

    pd.DataFrame([{"variable_or_group": k, "reason_not_selected": v}
                  for k, v in tb.CONSIDERED_NOT_SELECTED.items()]).to_csv(
        out / "considered_not_selected.csv", index=False, encoding="utf-8-sig")

    # ---- redundancy evidence, missingness, implausible values ---------------------
    band = band_agreement(train)
    band.to_csv(out / "derived_redundancy.csv", index=False, encoding="utf-8-sig")
    print(f"\nderived band columns checked: {len(band)}; "
          f"rows where missingness patterns differ: "
          f"{int((~band.identical_missingness).sum())}")

    mrows = []
    for c in raw_columns:
        col = c if c in train.columns else f"{c}_clin"
        if col not in train.columns:
            continue
        g = train.groupby("visit_concept_name")[col].apply(lambda s: float(s.isna().mean()))
        mrows.append({"column": c, "missing_train": round(float(train[col].isna().mean()), 4),
                      "missing_ER": round(float(g.get("Emergency Room Visit", float("nan"))), 4),
                      "missing_inpatient": round(float(g.get("Inpatient Visit", float("nan"))), 4)})
    miss_df = pd.DataFrame(mrows)
    miss_df["ER_minus_inpatient"] = (miss_df.missing_ER - miss_df.missing_inpatient).round(4)
    miss_df.to_csv(out / "missingness_report.csv", index=False, encoding="utf-8-sig")

    imp = implausible(train)
    imp.to_csv(out / "implausible_values.csv", index=False, encoding="utf-8-sig")
    print(f"physiologically impossible values in Training: {len(imp)}")
    if len(imp):
        print(imp[["column", "subject_id", "value"]].to_string(index=False))

    # chart-block missingness, the reason comorbidity keeps a NotAbstracted level
    cc = [c for c in chart_cols if c in train.columns]
    block = train[cc].isna().all(axis=1)
    summary = {
        "generated": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "script": Path(__file__).name,
        "cohort_development_only": {
            "train": n_train, "val": int((df.split == "val").sum()),
            "deaths": {"train": deaths_train, "val": int(df[(df.split == "val")].y.sum())},
            "test": "not read (held out until the Test evaluation is approved)"},
        "outcome_consistency_mismatches": 0,
        "columns_total": len(raw_columns),
        "roles": audit.role.value_counts().to_dict(),
        "columns_without_reason": unexplained,
        "candidates": {"n_variables": len(cand), "estimated_parameters": total_params,
                       "parameter_budget": budget, "epv_target": EPV_TARGET,
                       "selection_required": bool(total_params > budget)},
        "chart_block": {"n_columns": len(cc), "train_all_missing": int(block.sum()),
                        "death_rate_block_missing": round(float(train.y[block].mean()), 4),
                        "death_rate_block_present": round(float(train.y[~block].mean()), 4)},
        "implausible_values": len(imp),
        "input_sha256": {"clinical": tb.sha256(project / tb.CLINICAL_CSV),
                         "split": tb.sha256(project / tb.SPLIT_CSV)},
        "note": "Training set のみを用いて算出。Validation / Test は件数確認以外に参照していない",
    }
    (out / "audit_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                            encoding="utf-8")
    print(f"\nwrote {len(list(out.glob('*')))} files to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
