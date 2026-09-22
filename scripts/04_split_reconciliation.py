"""Verify the existing fixed split manifest against the reconciled 1,277-patient cohort (D-036).

The split is NEVER regenerated: this script only checks the existing manifest and, if every
check passes, copies it into the project as the fixed split of record. No seed is chosen, no
patient is re-assigned. A single failed check stops the pipeline.

Checks
  0. SHA256 of the source manifest (recorded)
  1. Subject ID set identical to index_cxr_manifest_window_T0m2_T0.csv (1,277 patients)
  2. no patient appears in more than one split
  3. split sizes: train 1,021 / val 128 / test 128
  4. deaths per split: 135 / 17 / 17
  5. Subject ID == to_patient_id, no duplicates, last.status consistent with true_label
  6. labels agree with last.status in the clinical file

Usage:
    python scripts/04_split_reconciliation.py --project . --split-file "<path>"
"""
import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

import pandas as pd

EXPECTED_SIZES = {"train": 1021, "val": 128, "test": 128}
EXPECTED_DEATHS = {"train": 135, "val": 17, "test": 17}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=".")
    ap.add_argument("--split-file", required=True)
    ap.add_argument("--adopt", action="store_true", help="copy into data/splits/ when all checks pass")
    args = ap.parse_args()
    project, src = Path(args.project), Path(args.split_file)

    sp = pd.read_csv(src, dtype=str, keep_default_na=False)
    cohort = pd.read_csv(project / "data/interim/cxr_audit/index_cxr_manifest_window_T0m2_T0.csv",
                         encoding="utf-8-sig", dtype=str)
    clin = pd.read_csv(project / "data/raw/clinical/患者データファイル.csv", dtype=str, keep_default_na=False)

    rep: dict = {"source_file": str(src), "sha256": sha256(src), "rows": int(len(sp)),
                 "columns": list(sp.columns)}

    sp_ids, cohort_ids = set(sp["Subject ID"]), set(cohort.PatientID)
    only_split, only_cohort = sorted(sp_ids - cohort_ids), sorted(cohort_ids - sp_ids)
    rep["1_subject_id_set"] = {"identical": not only_split and not only_cohort,
                               "n_split": len(sp_ids), "n_cohort": len(cohort_ids),
                               "only_in_split": only_split[:20], "only_in_cohort": only_cohort[:20]}
    dup = sp["Subject ID"].duplicated().sum()
    per_patient_splits = sp.groupby("Subject ID")["split"].nunique()
    rep["2_no_patient_overlap"] = {"duplicated_subject_ids": int(dup),
                                   "patients_in_more_than_one_split": int((per_patient_splits > 1).sum()),
                                   "ok": dup == 0 and int((per_patient_splits > 1).sum()) == 0}
    sizes = {k: int(v) for k, v in Counter(sp.split).items()}
    rep["3_split_sizes"] = {"observed": sizes, "expected": EXPECTED_SIZES, "ok": sizes == EXPECTED_SIZES}
    deaths = {k: int(v) for k, v in sp[sp.true_label == "1"].split.value_counts().items()}
    rep["4_deaths_per_split"] = {"observed": deaths, "expected": EXPECTED_DEATHS, "ok": deaths == EXPECTED_DEATHS}
    label_map = {"deceased": "1", "discharged": "0"}
    bad_label = sp[sp.true_label != sp["last.status"].map(label_map)]
    rep["5_internal_consistency"] = {
        "subject_id_equals_to_patient_id": bool((sp["Subject ID"] == sp.to_patient_id).all()),
        "label_status_mismatches": int(len(bad_label)),
        "split_values": sorted(sp.split.unique()),
        "ok": bool((sp["Subject ID"] == sp.to_patient_id).all()) and len(bad_label) == 0}
    merged = sp.merge(clin[["to_patient_id", "last.status"]], on="to_patient_id",
                      suffixes=("_split", "_clinical"))
    disagree = merged[merged["last.status_split"] != merged["last.status_clinical"]]
    rep["6_labels_match_clinical_file"] = {"compared": int(len(merged)), "mismatches": int(len(disagree)),
                                           "examples": disagree.head(5).to_dict("records"),
                                           "ok": len(disagree) == 0 and len(merged) == len(sp)}
    rep["all_checks_passed"] = all(rep[k]["ok"] if "ok" in rep[k] else rep[k]["identical"]
                                   for k in ["1_subject_id_set", "2_no_patient_overlap", "3_split_sizes",
                                             "4_deaths_per_split", "5_internal_consistency",
                                             "6_labels_match_clinical_file"])

    if rep["all_checks_passed"] and args.adopt:
        dest_dir = project / "data/splits"
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / "COVID19_固定患者split_1277.csv"
        shutil.copy2(src, dest)
        (dest_dir / "COVID19_固定患者split_1277.sha256").write_text(
            f"{sha256(dest)}  COVID19_固定患者split_1277.csv\n", encoding="utf-8")
        rep["adopted_copy"] = {"path": str(dest), "sha256_after_copy": sha256(dest),
                               "identical_to_source": sha256(dest) == rep["sha256"]}

    out = project / "data/interim/cxr_audit/split_reconciliation_report.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(rep, fh, ensure_ascii=False, indent=2)
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    if not rep["all_checks_passed"]:
        print("\nSTOP: split checks failed. The split is NOT adopted and is NOT regenerated.")


if __name__ == "__main__":
    main()
