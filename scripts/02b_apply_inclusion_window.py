"""Apply the fixed CXR inclusion window (T0-2 days .. T0 calendar date) to the
reconstructed index-CXR candidates, and write the resulting manifest + exclusion log.

This is NOT a cohort fix: the patient set has not yet been compared with the existing
1,277-patient manifest (D-036 / D-037). No split is applied.

Inputs : data/interim/cxr_audit/index_cxr_candidate_manifest.csv (1,309 patients, no window)
         data/interim/cxr_audit/cxr_file_level.csv, docs/cxr_studydescription_mapping_final.csv,
         data/raw/clinical/患者データファイル.csv   (for the full exclusion flow)
Outputs: data/interim/cxr_audit/index_cxr_manifest_window_T0m2_T0.csv   (kept as the basis
         of the next step; the 1,309-row candidate manifest is left untouched)
         data/interim/cxr_audit/index_cxr_exclusion_log.csv
         data/interim/cxr_audit/index_cxr_manifest_window_T0m2_T0.sha256

Usage:
    python scripts/02b_apply_inclusion_window.py --project .
"""
import argparse
import hashlib
from pathlib import Path

import pandas as pd

WINDOW_MIN_DELTA = -2  # T0-2 days .. T0 (delta 0)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=".")
    project = Path(ap.parse_args().project)
    d = project / "data/interim/cxr_audit"

    # dtype=str: AcquisitionTime like "083615.864" must keep its leading/trailing zeros
    # (parsing it as a number corrupted the saved manifest; change_log CL-002)
    cand = pd.read_csv(d / "index_cxr_candidate_manifest.csv", encoding="utf-8-sig", dtype=str)
    cand["delta_days_from_T0"] = pd.to_numeric(cand.delta_days_from_T0, errors="raise")
    files = pd.read_csv(d / "cxr_file_level.csv", encoding="utf-8-sig", dtype=str)
    mapping = pd.read_csv(project / "docs/cxr_studydescription_mapping_final.csv", encoding="utf-8-sig")
    clin = pd.read_csv(project / "data/raw/clinical/患者データファイル.csv", dtype=str, keep_default_na=False)
    cls = dict(zip(mapping.StudyDescription, mapping.classification))

    inside = cand[cand.delta_days_from_T0 >= WINDOW_MIN_DELTA].sort_values("PatientID")
    outside = cand[cand.delta_days_from_T0 < WINDOW_MIN_DELTA].sort_values("delta_days_from_T0")
    out_path = d / "index_cxr_manifest_window_T0m2_T0.csv"
    inside.to_csv(out_path, index=False, encoding="utf-8-sig")
    digest = sha256(out_path)
    (d / "index_cxr_manifest_window_T0m2_T0.sha256").write_text(
        f"{digest}  index_cxr_manifest_window_T0m2_T0.csv\n", encoding="utf-8")

    # full exclusion flow, one row per excluded patient
    all_pat = set(clin.to_patient_id)
    with_cxr = set(files.PatientID)
    chest = set(files[files.StudyDescription.map(cls) == "chest_candidate"].PatientID)
    frontal = set(files[(files.StudyDescription.map(cls) == "chest_candidate")
                        & (files.SeriesDescription.isin(["AP", "PA"]))].PatientID)
    eligible = set(cand.PatientID)
    rows = []
    for p in sorted(all_pat - with_cxr):
        rows.append({"PatientID": p, "excluded_at_step": "1_no_CR_DX_series", "detail": "CR/DX の画像がない"})
    for p in sorted(with_cxr - chest):
        rows.append({"PatientID": p, "excluded_at_step": "2_no_chest_candidate_study",
                     "detail": "CR/DX はあるが chest_candidate の StudyDescription がない"})
    for p in sorted(chest - frontal):
        rows.append({"PatientID": p, "excluded_at_step": "3_no_frontal_AP_PA", "detail": "AP/PA の Series がない"})
    for p in sorted(frontal - eligible):
        rows.append({"PatientID": p, "excluded_at_step": "4_only_after_T0",
                     "detail": "frontal CXR が T0 より後の暦日にしかない"})
    for _, r in outside.iterrows():
        rows.append({"PatientID": r.PatientID, "excluded_at_step": "5_outside_window_T0m2_T0",
                     "detail": f"最も T0 に近い eligible CXR が delta={int(r.delta_days_from_T0)} 日"})
    log = pd.DataFrame(rows)
    log.to_csv(d / "index_cxr_exclusion_log.csv", index=False, encoding="utf-8-sig")

    print(f"window: delta in [{WINDOW_MIN_DELTA}, 0]")
    print(f"candidates (no window): {len(cand)}")
    print(f"inside window  -> manifest rows: {len(inside)}  patients: {inside.PatientID.nunique()}")
    print(f"outside window -> excluded     : {len(outside)}")
    print(f"exclusion log rows: {len(log)}  by step:")
    print(log.excluded_at_step.value_counts().sort_index().to_string())
    print(f"total excluded from {len(all_pat)} patients: {len(log)}  -> remaining {len(all_pat) - len(log)}")
    print(f"delta distribution in manifest: "
          f"{inside.delta_days_from_T0.value_counts().sort_index(ascending=False).to_dict()}")
    print(f"SHA256 {digest}")
    print(f"saved  {out_path}")
    print("NOT done: comparison with the existing 1,277-patient manifest, cohort fixing, split")


if __name__ == "__main__":
    main()
