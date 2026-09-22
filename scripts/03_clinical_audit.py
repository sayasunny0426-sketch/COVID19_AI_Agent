"""Clinical data audit.

PRELIMINARY use (D-035): the 2026-09-18 run was on all 1,384 patients before the CXR-based
cohort and split were fixed; its outputs must not drive final selection or preprocessing.
The formal re-audit must run on the Training set of the fixed cohort only.

Label-blind structural audit of the TCIA COVID-19-NY-SBU clinical file and the
NBIA series metadata file. It does NOT build the cohort, split the data, impute,
scale or fit anything.

Leakage rule: the split manifest is not available yet, so this script never
computes distribution statistics (mean / median / quantiles / skewness) on the
full data -- those drive imputation choices and must come from the Training set
only. It reports only counts, missingness, value sets, min/max and counts
outside pre-specified plausibility bounds (data-quality checks).
The outcome column is used only for its own value counts and for a cross-tab
with visit type (outcome-definition audit), never against predictors.

Usage:
    python scripts/03_clinical_audit.py --root .
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd

MISSING_TOKENS = {"NA", ""}
CLINICAL_FILE = "data/raw/clinical/患者データファイル.csv"
METADATA_FILE = "data/raw/clinical/metadata.csv"

# Continuous source columns (short name -> column in the clinical file).
CONT = {
    "temp": "8331-1_Oral temperature",
    "spo2": "59408-5_Oxygen saturation in Arterial blood by Pulse oximetry",
    "rr": "9279-1_Respiratory rate",
    "hr": "76282-3_Heart rate.beat-to-beat by EKG",
    "sbp": "8480-6_Systolic blood pressure",
    "map": "76536-2_Mean blood pressure by Noninvasive",
    "wbc": "33256-9_Leukocytes [#/volume] corrected for nucleated erythrocytes in Blood by Automated count",
    "neut": "751-8_Neutrophils [#/volume] in Blood by Automated count",
    "lymph": "731-0_Lymphocytes [#/volume] in Blood by Automated count",
    "na": "2951-2_Sodium [Moles/volume] in Serum or Plasma",
    "ast": "1920-8_Aspartate aminotransferase [Enzymatic activity/volume] in Serum or Plasma",
    "alt": "1744-2_Alanine aminotransferase [Enzymatic activity/volume] in Serum or Plasma by No addition of P-5'-P",
    "ck": "2157-6_Creatine kinase [Enzymatic activity/volume] in Serum or Plasma",
    "lactate": "2524-7_Lactate [Moles/volume] in Serum or Plasma",
    "trop": "6598-7_Troponin T.cardiac [Mass/volume] in Serum or Plasma",
    "ntprobnp": "33762-6_Natriuretic peptide.B prohormone N-Terminal [Mass/volume] in Serum or Plasma",
    "pct": "75241-0_Procalcitonin [Mass/volume] in Serum or Plasma by Immunoassay",
    "ddimer": "48058-2_Fibrin D-dimer DDU [Mass/volume] in Platelet poor plasma by Immunoassay",
    "ferritin": "2276-4_Ferritin [Mass/volume] in Serum or Plasma",
    "crp": "1988-5_C reactive protein [Mass/volume] in Serum or Plasma",
    "a1c": "4548-4_Hemoglobin A1c/Hemoglobin.total in Blood",
    "bmi": "39156-5_Body mass index (BMI) [Ratio]",
    "na_dup": "2951-2_Sodium [Moles/volume] in Serum or Plasma.1",
    "k": "2823-3_Potassium [Moles/volume] in Serum or Plasma",
    "cl": "2075-0_Chloride [Moles/volume] in Serum or Plasma",
    "hco3": "1963-8_Bicarbonate [Moles/volume] in Serum or Plasma",
    "bun": "3094-0_Urea nitrogen [Mass/volume] in Serum or Plasma",
    "cr": "2160-0_Creatinine [Mass/volume] in Serum or Plasma",
    "egfr": "62238-1_Glomerular filtration rate/1.73 sq M.predicted [Volume Rate/Area] in Serum, Plasma or Blood by Creatinine-based formula (CKD-EPI)",
    "ph": "33254-4_pH of Arterial blood adjusted to patient's actual temperature",
    "esr": "30341-2_Erythrocyte sedimentation rate",
    "glucose": "2345-7_Glucose [Mass/volume] in Serum or Plasma",
    "ldl": "13457-7_Cholesterol in LDL [Mass/volume] in Serum or Plasma by calculation",
    "vldl": "13458-5_Cholesterol in VLDL [Mass/volume] in Serum or Plasma by calculation",
    "tg": "2571-8_Triglyceride [Mass/volume] in Serum or Plasma",
    "hdl": "2085-9_Cholesterol in HDL [Mass/volume] in Serum or Plasma",
    "days_prior_sx": "days_prior_sx",
    "vent_days": "invasive_vent_days",
    "los": "length_of_stay",
}

# PROVISIONAL physiological hard bounds (Claude proposal, not yet approved).
# Values outside are *candidates* for source verification, never auto-removed.
# Units are assumed from typical US reporting and must be confirmed.
BOUNDS = {
    "temp": (25, 45, "degC"),
    "spo2": (30, 100, "%"),
    "rr": (4, 80, "/min"),
    "hr": (20, 300, "/min"),
    "sbp": (40, 300, "mmHg"),
    "map": (20, 250, "mmHg"),
    "wbc": (0.1, 500, "10^3/uL (assumed)"),
    "neut": (0, 500, "10^3/uL (assumed)"),
    "lymph": (0, 500, "10^3/uL (assumed)"),
    "na": (90, 200, "mmol/L"),
    "k": (1.0, 12.0, "mmol/L"),
    "cl": (50, 160, "mmol/L"),
    "hco3": (2, 60, "mmol/L"),
    "bun": (0.5, 300, "mg/dL (assumed)"),
    "cr": (0.1, 30, "mg/dL (assumed)"),
    "egfr": (0, 250, "mL/min/1.73m2"),
    "ph": (6.5, 8.0, "-"),
    "glucose": (10, 3000, "mg/dL (assumed)"),
    "bmi": (10, 100, "kg/m2"),
    "a1c": (3, 25, "%"),
    "ast": (1, 50000, "U/L"),
    "alt": (1, 50000, "U/L"),
    "ck": (1, 500000, "U/L"),
    "lactate": (0.1, 40, "mmol/L"),
    "esr": (0, 200, "mm/h"),
    "days_prior_sx": (0, 365, "days"),
}

# Binary threshold columns and the continuous column they appear to be derived from.
DERIVED = [
    ("BMI.over30", "bmi", "gt", 30), ("BMI.over35", "bmi", "gt", 35),
    ("temperature.over38", "temp", "gt", 38), ("pulseOx.under90", "spo2", "lt", 90),
    ("Respiration.over24", "rr", "gt", 24), ("HeartRate.over100", "hr", "gt", 100),
    ("Lymphocytes.under1k", "lymph", "lt", 1), ("Aspartate.over40", "ast", "gt", 40),
    ("Alanine.over60", "alt", "gt", 60),
    ("A1C.over6.5", "a1c", "gt", 6.5), ("A1C.under6.5", "a1c", "lt", 6.5),
    ("A1C.6.6to7.9", "a1c", "between", (6.6, 7.9)), ("A1C.8to9.9", "a1c", "between", (8, 9.9)),
    ("A1C.over10", "a1c", "gt", 10),
    ("Sodium.above145", "na", "gt", 145), ("Sodium.between135and145", "na", "between", (135, 145)),
    ("Sodium.below135", "na", "lt", 135),
    ("Potassium.above5.2", "k", "gt", 5.2), ("Potassium.between3.5and5.2", "k", "between", (3.5, 5.2)),
    ("Potassium.below3.5", "k", "lt", 3.5),
    ("Chloride.above107", "cl", "gt", 107), ("Chloride.between96and107", "cl", "between", (96, 107)),
    ("Chloride.below96", "cl", "lt", 96),
    ("Bicarbonate.above31", "hco3", "gt", 31), ("Bicarbonate.between21and31", "hco3", "between", (21, 31)),
    ("Bicarbonate.below21", "hco3", "lt", 21),
    ("Blood_Urea_Nitrogen.above20", "bun", "gt", 20), ("Blood_Urea_Nitrogen.between5and20", "bun", "between", (5, 20)),
    ("Blood_Urea_Nitrogen.below5", "bun", "lt", 5),
    ("Creatinine.above1.2", "cr", "gt", 1.2), ("Creatinine.between0.5and1.2", "cr", "between", (0.5, 1.2)),
    ("Creatinine.below0.5", "cr", "lt", 0.5),
    ("eGFR.above60", "egfr", "gt", 60), ("eGFR.between30and60", "egfr", "between", (30, 60)),
    ("eGFR.below30", "egfr", "lt", 30),
    ("blood_pH.above7.45", "ph", "gt", 7.45), ("blood_pH.between7.35and7.45", "ph", "between", (7.35, 7.45)),
    ("blood_pH.below7.35", "ph", "lt", 7.35),
    ("Troponin.above0.01", "trop", "gt", 0.01),
    ("D_dimer.above3000", "ddimer", "gt", 3000), ("D_dimer.between500and3000", "ddimer", "between", (500, 3000)),
    ("D_dimer.below500", "ddimer", "lt", 500),
    ("ESR.above30", "esr", "gt", 30),
    ("SBP.below120", "sbp", "lt", 120), ("SBP.between120and139", "sbp", "between", (120, 139)),
    ("SBP.above139", "sbp", "gt", 139),
    ("MAP.below65", "map", "lt", 65), ("MAP.between65and90", "map", "between", (65, 90)),
    ("MAP.above90", "map", "gt", 90),
    ("procalcitonin.below0.25", "pct", "lt", 0.25),
    ("procalcitonin.between0.25and0.5", "pct", "between", (0.25, 0.5)),
    ("procalcitonin.above0.5", "pct", "gt", 0.5),
    ("ferritin.above1k", "ferritin", "gt", 1000),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def is_missing(s: pd.Series) -> pd.Series:
    return s.isin(MISSING_TOKENS)


def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.where(~is_missing(s)), errors="coerce")


def column_profile(df: pd.DataFrame, typed: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in df.columns:
        s = df[c]
        miss = is_missing(s)
        nonmiss = s[~miss]
        num = pd.to_numeric(nonmiss, errors="coerce")
        n_non_numeric = int(num.isna().sum())
        values = sorted(nonmiss.unique().tolist())
        rows.append({
            "column": c,
            "pandas_dtype": str(typed[c].dtype),
            "n_missing": int(miss.sum()),
            "missing_rate": round(float(miss.mean()), 4),
            "n_unique_nonmissing": int(nonmiss.nunique()),
            "all_numeric": n_non_numeric == 0 and len(nonmiss) > 0,
            "value_set_if_<=10": "|".join(values) if len(values) <= 10 else "",
            "non_numeric_tokens_if_mostly_numeric": "|".join(sorted(nonmiss[num.isna()].unique().tolist())[:10])
            if 0 < n_non_numeric < len(nonmiss) else "",
        })
    return pd.DataFrame(rows)


def predicate(v: pd.Series, op: str, thr, variant: str) -> pd.Series:
    if op == "gt":
        return v > thr if variant == "strict" else v >= thr
    if op == "lt":
        return v < thr if variant == "strict" else v <= thr
    lo, hi = thr
    return (v >= lo) & (v <= hi) if variant == "strict" else (v > lo) & (v < hi)


def derived_consistency(clin: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col, src, op, thr in DERIVED:
        b = clin[col]
        v = to_num(clin[CONT[src]])
        b_miss, v_miss = is_missing(b), v.isna()
        both = ~b_miss & ~v_miss
        bb = b[both].map({"TRUE": True, "FALSE": False})
        res = {"binary_column": col, "source_column": CONT[src], "rule": f"{op} {thr}",
               "n_both_present": int(both.sum()),
               "n_binary_missing_only": int((b_miss & ~v_miss).sum()),
               "n_source_missing_only": int((~b_miss & v_miss).sum())}
        for variant in ("strict", "inclusive"):
            res[f"mismatch_{variant}"] = int((predicate(v[both], op, thr, variant) != bb).sum())
        rows.append(res)
    return pd.DataFrame(rows)


def numeric_ranges(clin: pd.DataFrame) -> pd.DataFrame:
    """Min/max, bound violations and value heaping at min/max (possible LOD/caps).
    No mean/median/quantiles on purpose (see module docstring)."""
    rows = []
    for key, col in CONT.items():
        v = to_num(clin[col]).dropna()
        if v.empty:
            continue
        lo, hi, unit = BOUNDS.get(key, (None, None, ""))
        rows.append({
            "key": key, "column": col, "n_present": len(v),
            "min": v.min(), "max": v.max(),
            "n_at_min": int((v == v.min()).sum()), "n_at_max": int((v == v.max()).sum()),
            "provisional_lower": lo, "provisional_upper": hi, "assumed_unit": unit,
            "n_below_lower": int((v < lo).sum()) if lo is not None else None,
            "n_above_upper": int((v > hi).sum()) if hi is not None else None,
        })
    return pd.DataFrame(rows)


def internal_consistency(clin: pd.DataFrame) -> dict:
    wbc, neut, lymph = (to_num(clin[CONT[k]]) for k in ("wbc", "neut", "lymph"))
    sbp, mapv = to_num(clin[CONT["sbp"]]), to_num(clin[CONT["map"]])
    a1c_bins = ["A1C.under6.5", "A1C.6.6to7.9", "A1C.8to9.9", "A1C.over10"]
    a1c_true = (clin[a1c_bins] == "TRUE").sum(axis=1)
    a1c_present = ~is_missing(clin["A1C.under6.5"])
    return {
        "neutrophils_gt_wbc": int((neut > wbc).sum()),
        "lymphocytes_gt_wbc": int((lymph > wbc).sum()),
        "neut_plus_lymph_gt_wbc": int((neut + lymph > wbc).sum()),
        "map_gt_sbp": int((mapv > sbp).sum()),
        "a1c_bins_true_count_distribution_when_present": a1c_true[a1c_present].value_counts().sort_index().to_dict(),
        "sodium_duplicate_columns_identical": bool((clin[CONT["na"]] == clin[CONT["na_dup"]]).all()),
    }


def missing_blocks(clin: pd.DataFrame, min_size: int = 2) -> list:
    """Groups of columns sharing an identical missingness mask (e.g. lab panels)."""
    masks = {}
    for c in clin.columns:
        m = is_missing(clin[c])
        if m.any():
            masks.setdefault(hashlib.md5(m.values.tobytes()).hexdigest(), []).append((c, int(m.sum())))
    return [{"n_missing": cols[0][1], "columns": [c for c, _ in cols]}
            for cols in masks.values() if len(cols) >= min_size]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", default="data/interim/clinical_audit")
    args = ap.parse_args()
    root = Path(args.root)
    out = root / args.out
    out.mkdir(parents=True, exist_ok=True)

    clin_path, meta_path = root / CLINICAL_FILE, root / METADATA_FILE
    clin = pd.read_csv(clin_path, dtype=str, keep_default_na=False)
    meta = pd.read_csv(meta_path, dtype=str, keep_default_na=False)
    clin_typed = pd.read_csv(clin_path, na_values=list(MISSING_TOKENS), keep_default_na=False)
    meta_typed = pd.read_csv(meta_path, na_values=list(MISSING_TOKENS), keep_default_na=False)

    summary = {"inputs": {}}
    for name, p, df in (("clinical", clin_path, clin), ("metadata", meta_path, meta)):
        summary["inputs"][name] = {"path": str(p.relative_to(root)), "sha256": sha256(p),
                                   "n_rows": len(df), "n_cols": df.shape[1]}

    column_profile(clin, clin_typed).to_csv(out / "clinical_column_profile.csv", index=False)
    column_profile(meta, meta_typed).to_csv(out / "metadata_column_profile.csv", index=False)

    # Identifiers and duplicates
    pid, sid = clin["to_patient_id"], meta["Subject ID"]
    summary["identifiers"] = {
        "clinical_to_patient_id_unique": bool(pid.is_unique),
        "clinical_n_patients": int(pid.nunique()),
        "clinical_id_pattern_A+6digits": bool(pid.str.fullmatch(r"A\d{6}").all()),
        "clinical_full_row_duplicates": int(clin.duplicated().sum()),
        "clinical_duplicate_rows_excluding_id": int(clin.drop(columns="to_patient_id").duplicated().sum()),
        "metadata_n_subjects": int(sid.nunique()),
        "metadata_full_row_duplicates": int(meta.duplicated().sum()),
        "metadata_duplicate_series_uid": int(meta["Series UID"].duplicated().sum()),
        "ids_in_clinical_not_metadata": int(len(set(pid) - set(sid))),
        "ids_in_metadata_not_clinical": int(len(set(sid) - set(pid))),
    }
    dup_series = meta[meta["Series UID"].duplicated(keep=False)]
    summary["identifiers"]["duplicate_series_rows_identical_except_download_timestamp"] = bool(
        dup_series.drop(columns="Download Timestamp").duplicated(keep=False).all()) if len(dup_series) else None

    # T0 column
    vsd = clin["visit_start_datetime"]
    parsed = pd.to_datetime(vsd.where(~is_missing(vsd)), format="%m/%d/%Y", errors="coerce")
    summary["visit_start_datetime"] = {
        "n_missing": int(is_missing(vsd).sum()),
        "n_unparseable_as_M/D/YYYY": int((parsed.isna() & ~is_missing(vsd)).sum()),
        "has_time_component": bool(vsd.str.contains(r"\d{1,2}:\d{2}").any()),
        "min": str(parsed.min().date()), "max": str(parsed.max().date()),
        "n_distinct": int(vsd.nunique()),
    }
    sd = pd.to_datetime(meta["Study Date"], format="%m-%d-%Y", errors="coerce")
    summary["metadata_study_date"] = {
        "n_unparseable_as_MM-DD-YYYY": int(sd.isna().sum()),
        "has_time_column_in_metadata": any(re.search("time", c, re.I) and "Download" not in c for c in meta.columns),
    }

    # Outcome-definition audit (outcome counts only; no predictor association)
    summary["outcome"] = {
        "last.status_counts": clin["last.status"].value_counts().to_dict(),
        "covid19_statuses_counts": clin["covid19_statuses"].value_counts().to_dict(),
    }
    pd.crosstab(clin["visit_concept_name"], clin["last.status"], margins=True).to_csv(
        out / "outcome_by_visit_type.csv")

    # Missingness structure by visit type (label-blind)
    miss_by_visit = clin.drop(columns=["to_patient_id"]).apply(is_missing).groupby(
        clin["visit_concept_name"]).mean().T.round(3)
    miss_by_visit.to_csv(out / "missing_rate_by_visit_type.csv")

    derived_consistency(clin).to_csv(out / "derived_binary_consistency.csv", index=False)
    numeric_ranges(clin).to_csv(out / "numeric_ranges_and_bounds.csv", index=False)
    summary["internal_consistency"] = internal_consistency(clin)
    summary["missing_blocks"] = missing_blocks(clin)

    with open(out / "audit_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
