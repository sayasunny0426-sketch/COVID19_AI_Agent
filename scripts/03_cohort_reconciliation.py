"""Reconcile the reconstructed index-CXR manifest with the existing selected DICOMs.

Acceptance criteria (D-036 / D-037):
  1. the patient set matches the existing cohort exactly, and
  2. where the existing side carries index-CXR identifiers, the index CXR matches exactly
     (StudyInstanceUID / SeriesInstanceUID / SOPInstanceUID / study date, whichever exist).
A single mismatching patient stops the pipeline: no cohort fixing, no split, no correction
towards the existing side.

Read-only: the existing DICOMs are opened with stop_before_pixels=True; nothing is written,
moved or copied in that folder.

Inputs : --existing-dir  folder with the existing selected DICOM files
         data/interim/cxr_audit/index_cxr_manifest_window_T0m2_T0.csv
Outputs: data/interim/cxr_audit/existing_selected_dicom_headers.csv
         data/interim/cxr_audit/reconciliation_report.json
         data/interim/cxr_audit/reconciliation_differences.csv  (only if differences exist)

Usage:
    python scripts/03_cohort_reconciliation.py --project . --existing-dir "<path>"
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd
import pydicom

TAGS = ["PatientID", "StudyDate", "AcquisitionDate", "AcquisitionTime", "StudyDescription",
        "SeriesDescription", "Modality", "SeriesNumber", "StudyInstanceUID", "SeriesInstanceUID",
        "SOPInstanceUID"]
COMPARE = [("SeriesInstanceUID", "primary"), ("SOPInstanceUID", "primary"),
           ("StudyInstanceUID", "supporting"), ("StudyDate", "supporting"),
           ("AcquisitionTime", "supporting"), ("SeriesNumber", "supporting"),
           ("AcquisitionDate", "supporting"), ("StudyDescription", "supporting"),
           ("SeriesDescription", "supporting"), ("Modality", "supporting")]


def read_headers(folder: Path) -> tuple[pd.DataFrame, list]:
    rows, errors = [], []
    for f in sorted(folder.rglob("*")):
        if not f.is_file():
            continue
        try:
            ds = pydicom.dcmread(f, stop_before_pixels=True)
        except Exception as e:
            errors.append({"filepath": str(f), "error": f"{type(e).__name__}: {e}"})
            continue
        row = {"filepath": str(f)}
        for t in TAGS:
            v = getattr(ds, t, None)
            if v is None or (isinstance(v, str) and v.strip() == ""):
                row[t] = None
            elif t == "SeriesNumber":
                try:
                    row[t] = int(v)
                except (TypeError, ValueError):
                    row[t] = str(v)
            else:
                row[t] = str(v)
        rows.append(row)
    return pd.DataFrame(rows), errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=".")
    ap.add_argument("--existing-dir", required=True)
    args = ap.parse_args()
    project, existing_dir = Path(args.project), Path(args.existing_dir)
    d = project / "data/interim/cxr_audit"

    print(f"reading existing DICOM headers (read-only) from {existing_dir} ...", flush=True)
    ex, errors = read_headers(existing_dir)
    ex.to_csv(d / "existing_selected_dicom_headers.csv", index=False, encoding="utf-8-sig")
    # dtype=str throughout: AcquisitionTime like "083615.864" must not be parsed as a number
    # (that dropped leading/trailing zeros and produced 393 false mismatches; change_log CL-002)
    rec = pd.read_csv(d / "index_cxr_manifest_window_T0m2_T0.csv", encoding="utf-8-sig", dtype=str).rename(
        columns={"selected_StudyDate": "StudyDate", "selected_AcquisitionDate": "AcquisitionDate"})
    rec["SeriesNumber"] = pd.to_numeric(rec.SeriesNumber, errors="coerce")

    rep: dict = {"existing_dir": str(existing_dir), "unreadable_files": errors}
    rep["1_existing_file_count"] = {"files": int(len(ex)) + len(errors), "headers_read": int(len(ex)),
                                    "expected": 1277, "ok": int(len(ex)) == 1277}
    rep["2_existing_unique_patients"] = {"n": int(ex.PatientID.nunique()), "ok": ex.PatientID.nunique() == 1277,
                                         "duplicated_PatientID": int(ex.PatientID.duplicated().sum())}
    rep["3_reconstructed_unique_patients"] = {"n": int(rec.PatientID.nunique()),
                                              "ok": rec.PatientID.nunique() == 1277 == len(rec)}
    ex_ids, rec_ids = set(ex.PatientID), set(rec.PatientID)
    only_ex, only_rec = sorted(ex_ids - rec_ids), sorted(rec_ids - ex_ids)
    rep["4_patient_set"] = {"identical": not only_ex and not only_rec,
                            "only_existing_n": len(only_ex), "only_reconstructed_n": len(only_rec),
                            "only_existing": only_ex[:50], "only_reconstructed": only_rec[:50]}

    merged = ex.merge(rec, on="PatientID", suffixes=("_existing", "_reconstructed"), how="inner")
    diffs, per_patient_flags = [], {}
    for _, r in merged.iterrows():
        types = []
        for col, kind in COMPARE:
            a, b = r.get(f"{col}_existing"), r.get(f"{col}_reconstructed")
            if col == "SeriesNumber":
                a = None if pd.isna(a) else int(a)
                b = None if pd.isna(b) else int(b)
            else:
                a = None if (a is None or (isinstance(a, float) and pd.isna(a))) else str(a)
                b = None if (b is None or (isinstance(b, float) and pd.isna(b))) else str(b)
            if a != b:
                types.append(f"{col}({kind})")
        if types:
            per_patient_flags[r.PatientID] = types
            diffs.append({
                "PatientID": r.PatientID,
                "existing_SeriesInstanceUID": r.SeriesInstanceUID_existing,
                "reconstructed_SeriesInstanceUID": r.SeriesInstanceUID_reconstructed,
                "existing_StudyDate": r.StudyDate_existing,
                "reconstructed_StudyDate": r.StudyDate_reconstructed,
                "existing_AcquisitionTime": r.AcquisitionTime_existing,
                "reconstructed_AcquisitionTime": r.AcquisitionTime_reconstructed,
                "existing_SeriesNumber": r.SeriesNumber_existing,
                "reconstructed_SeriesNumber": r.SeriesNumber_reconstructed,
                "existing_SOPInstanceUID": r.SOPInstanceUID_existing,
                "reconstructed_SOPInstanceUID": r.SOPInstanceUID_reconstructed,
                "existing_StudyInstanceUID": r.StudyInstanceUID_existing,
                "reconstructed_StudyInstanceUID": r.StudyInstanceUID_reconstructed,
                "difference_type": ";".join(types),
            })
    rep["5_SeriesInstanceUID_mismatch_patients"] = int(sum(
        any(t.startswith("SeriesInstanceUID") for t in v) for v in per_patient_flags.values()))
    rep["6_SOPInstanceUID_mismatch_patients"] = int(sum(
        any(t.startswith("SOPInstanceUID") for t in v) for v in per_patient_flags.values()))
    rep["7_supporting_attribute_mismatches"] = dict(Counter(
        t for v in per_patient_flags.values() for t in v if t.endswith("(supporting)")))
    rep["patients_compared"] = int(len(merged))
    rep["fully_matching_patients"] = int(len(merged) - len(per_patient_flags))
    rep["patients_with_any_difference"] = int(len(per_patient_flags))
    rep["acceptance_D-037"] = {
        "patient_set_identical": rep["4_patient_set"]["identical"],
        "index_cxr_identical_for_all_patients": len(per_patient_flags) == 0,
        "passed": rep["4_patient_set"]["identical"] and len(per_patient_flags) == 0}

    if diffs:
        pd.DataFrame(diffs).to_csv(d / "reconciliation_differences.csv", index=False, encoding="utf-8-sig")
    with open(d / "reconciliation_report.json", "w", encoding="utf-8") as fh:
        json.dump(rep, fh, ensure_ascii=False, indent=2)
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    if not rep["acceptance_D-037"]["passed"]:
        print("\nSTOP: acceptance criteria not met (D-036/D-037). No cohort fixing, no split.")


if __name__ == "__main__":
    main()
