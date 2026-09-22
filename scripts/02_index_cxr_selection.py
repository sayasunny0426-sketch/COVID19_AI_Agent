"""Index CXR selection (reconstruction). NOT a cohort fix, NOT a split.

Rules (D-020..D-023, D-025, D-026, D-028, D-038; researcher instruction 2026-09-20):
  1. Modality is CR or DX
  2. StudyDescription classified chest_candidate in the approved mapping (D-038)
  3. SeriesDescription is AP or PA (frontal)
  4. exclude_special_purpose studies are excluded (implied by rule 2)
  5. T0 = visit_start_datetime (date only; no admission time exists)
  6. CXR on a calendar date AFTER T0 is never used
  7. among eligible CXR on or before T0, take the date closest to T0
  8. on that date, take the series with the earliest AcquisitionTime
  9. if several series share that earliest AcquisitionTime, take the smallest SeriesNumber
 10. SeriesNumber is a reproducible tie-breaker only, never read as acquisition order
 11. the inclusion window (e.g. T0-2 days) is NOT applied here; the distance of the
     selected CXR from T0 is reported for every patient instead.

Inputs : data/interim/cxr_audit/cxr_file_level.csv (headers of all CR/DX files)
         docs/cxr_studydescription_mapping_final.csv (approved mapping)
         data/raw/clinical/患者データファイル.csv (visit_start_datetime)
Outputs: data/interim/cxr_audit/index_cxr_candidate_manifest.csv (one row per patient)
         data/interim/cxr_audit/index_cxr_selection_flow.json

Usage:
    python scripts/02_index_cxr_selection.py --project . --dicom-root "D:/manifest-1628608914773/COVID-19-NY-SBU"
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

FRONTAL = ["AP", "PA"]


def acq_seconds(t) -> float | None:
    if t is None or (isinstance(t, float) and pd.isna(t)):
        return None
    m = re.fullmatch(r"(\d{2})(\d{2})(\d{2}(?:\.\d+)?)", str(t).strip())
    return None if not m else int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=".")
    ap.add_argument("--dicom-root", default="")
    args = ap.parse_args()
    project = Path(args.project)
    out_dir = project / "data/interim/cxr_audit"

    files = pd.read_csv(out_dir / "cxr_file_level.csv", encoding="utf-8-sig", dtype=str)
    mapping = pd.read_csv(project / "docs/cxr_studydescription_mapping_final.csv", encoding="utf-8-sig")
    clin = pd.read_csv(project / "data/raw/clinical/患者データファイル.csv", dtype=str, keep_default_na=False)

    cls = dict(zip(mapping.StudyDescription, mapping.classification))
    t0 = clin[["to_patient_id", "visit_start_datetime"]].rename(
        columns={"to_patient_id": "PatientID", "visit_start_datetime": "T0_raw"})
    t0["T0"] = pd.to_datetime(t0.T0_raw, format="%m/%d/%Y", errors="coerce")
    assert t0.T0.notna().all() and t0.PatientID.is_unique

    df = files.copy()
    df["study_dt"] = pd.to_datetime(df.StudyDate, format="%Y%m%d", errors="coerce")
    df["classification"] = df.StudyDescription.map(cls)
    df["acq_sec"] = df.AcquisitionTime.map(acq_seconds)
    df["SeriesNumber_int"] = pd.to_numeric(df.SeriesNumber, errors="coerce")
    df = df.merge(t0[["PatientID", "T0"]], on="PatientID", how="left")
    df["delta_days"] = (df.study_dt - df.T0).dt.days

    flow = {"all_patients_clinical_file": int(len(t0)),
            "all_patients_dicom": int(files.PatientID.nunique())}
    step_cr_dx = df
    flow["with_CR_or_DX"] = int(step_cr_dx.PatientID.nunique())
    step_chest = step_cr_dx[step_cr_dx.classification == "chest_candidate"]
    flow["with_chest_candidate"] = int(step_chest.PatientID.nunique())
    step_frontal = step_chest[step_chest.SeriesDescription.isin(FRONTAL)]
    flow["with_frontal_AP_PA"] = int(step_frontal.PatientID.nunique())
    eligible = step_frontal[step_frontal.delta_days <= 0]
    flow["with_eligible_on_or_before_T0"] = int(eligible.PatientID.nunique())
    flow["excluded_only_after_T0"] = int(step_frontal.PatientID.nunique() - eligible.PatientID.nunique())
    flow["patients_without_any_CR_DX"] = int(len(t0) - files.PatientID.nunique())
    assert eligible.acq_sec.notna().all() and eligible.SeriesNumber_int.notna().all()

    # rule 7 -> 8 -> 9
    picked = (eligible
              .sort_values(["PatientID", "delta_days", "acq_sec", "SeriesNumber_int"],
                           ascending=[True, False, True, True])
              .groupby("PatientID", as_index=False)
              .first())

    # record how each tie was resolved (transparency for rule 9/10)
    chosen_day = eligible.merge(picked[["PatientID", "delta_days"]], on=["PatientID", "delta_days"])
    earliest = chosen_day.merge(
        chosen_day.groupby("PatientID").acq_sec.min().rename("min_sec").reset_index(), on="PatientID")
    tied = earliest[earliest.acq_sec == earliest.min_sec].groupby("PatientID").size()
    flow["selected_day_series_at_earliest_time"] = {int(k): int(v) for k, v in sorted(Counter(tied).items())}
    flow["patients_resolved_by_SeriesNumber_tiebreak"] = int((tied > 1).sum())

    root = args.dicom_root.rstrip("/\\")
    manifest = pd.DataFrame({
        "PatientID": picked.PatientID,
        "T0": picked.T0.dt.strftime("%Y-%m-%d"),
        "selected_StudyDate": picked.StudyDate,
        "selected_AcquisitionDate": picked.AcquisitionDate,
        "delta_days_from_T0": picked.delta_days.astype(int),
        "AcquisitionTime": picked.AcquisitionTime,
        "StudyDescription": picked.StudyDescription,
        "SeriesDescription": picked.SeriesDescription,
        "Modality": picked.Modality,
        "SeriesNumber": picked.SeriesNumber_int.astype(int),
        "StudyInstanceUID": picked.StudyInstanceUID,
        "SeriesInstanceUID": picked.SeriesInstanceUID,
        "SOPInstanceUID": picked.SOPInstanceUID,
        "source_filepath": (root + "\\" if root else "") + picked.file.astype(str),
    }).sort_values("PatientID")
    manifest.to_csv(out_dir / "index_cxr_candidate_manifest.csv", index=False, encoding="utf-8-sig")

    d = manifest.delta_days_from_T0
    flow["delta_days_distribution"] = {int(k): int(v) for k, v in sorted(Counter(d).items(), reverse=True)}
    flow["delta_days_summary"] = {
        "0": int((d == 0).sum()), "-1": int((d == -1).sum()), "-2": int((d == -2).sum()),
        "<=-3": int((d <= -3).sum()), "min": int(d.min()), "max": int(d.max()),
        "within_T0-2_to_T0": int((d >= -2).sum()), "outside_T0-2_window": int((d < -2).sum())}
    flow["selected_modality"] = {str(k): int(v) for k, v in Counter(manifest.Modality).items()}
    flow["selected_view"] = {str(k): int(v) for k, v in Counter(manifest.SeriesDescription).items()}
    flow["selected_study_description"] = {str(k): int(v) for k, v in Counter(manifest.StudyDescription).most_common()}
    flow["manifest_rows"] = int(len(manifest))
    flow["note"] = ("no inclusion window applied (rule 11); no cohort fixed; no split applied; "
                    "existing 1,277-patient manifest not consulted")
    with open(out_dir / "index_cxr_selection_flow.json", "w", encoding="utf-8") as fh:
        json.dump(flow, fh, ensure_ascii=False, indent=2)
    print(json.dumps(flow, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
