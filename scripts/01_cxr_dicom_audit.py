"""CXR DICOM header audit over all patients (implementation_plan stage 01).

Read-only: files on the SSD are opened with stop_before_pixels=True and are never
written, moved or copied. No index-CXR selection, no cohort work, no pixel hashing.
Purpose: understand the data structure and quantify missing / duplicate cases,
including the evidence needed for Q-B9(b) (missing AcquisitionTime) and
Q-B9(c) (series holding more than one image).

Scope decision (562,376 files in total, mostly CT slices):
  - every series: the FIRST file's header is read (series-level tags) and the number of
    files in the series folder is taken from the filesystem;
  - CR / DX series (the CXR scope): EVERY file's header is read (per-file tags).
  CT / MR / NM / PT slices are therefore counted but not read individually.

Outputs (local, git-ignored):
  data/interim/cxr_audit/series_level.csv           one row per series (all modalities)
  data/interim/cxr_audit/cxr_file_level.csv         one row per CR/DX file
  data/interim/cxr_audit/anomalies.csv              unreadable files / structure problems
  data/interim/cxr_audit/audit_summary.json         aggregates 1-14 + duplicate candidates

Usage:
    python scripts/01_cxr_dicom_audit.py --dicom-root "D:/manifest-1628608914773/COVID-19-NY-SBU"
"""
import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd
import pydicom

TAGS = ["PatientID", "StudyDate", "AcquisitionDate", "AcquisitionTime", "StudyDescription",
        "SeriesDescription", "Modality", "ViewPosition", "SeriesNumber", "InstanceNumber",
        "StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID"]
EXTRA = ["StudyTime", "SeriesTime", "ContentTime", "PhotometricInterpretation", "Manufacturer",
         "BodyPartExamined", "Rows", "Columns"]
CXR_MODALITIES = {"CR", "DX"}
DUP_WINDOW_SEC = 5.0


def read_header(path: Path) -> tuple[dict, str]:
    try:
        ds = pydicom.dcmread(path, stop_before_pixels=True)
    except Exception as e:
        return {}, f"{type(e).__name__}: {e}"
    row = {}
    for t in TAGS + EXTRA:
        v = getattr(ds, t, None)
        if v is None or (isinstance(v, str) and v.strip() == ""):
            row[t] = None
        elif t in ("SeriesNumber", "InstanceNumber", "Rows", "Columns"):
            try:
                row[t] = int(v)
            except (TypeError, ValueError):
                row[t] = str(v)
        else:
            row[t] = str(v)
    return row, ""


def acq_seconds(t) -> float | None:
    """DICOM TM 'HHMMSS.FFFFFF' -> seconds since midnight."""
    if t is None:
        return None
    m = re.fullmatch(r"(\d{2})(\d{2})(\d{2}(?:\.\d+)?)", str(t).strip())
    if not m:
        return None
    return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))


def scan(root: Path, out: Path, max_patients: int | None = None):
    series_rows, file_rows, anomalies = [], [], []
    patients = sorted(p for p in root.iterdir() if p.is_dir())
    if max_patients:
        patients = patients[:max_patients]
    t0 = time.time()
    for i, pat in enumerate(patients, 1):
        for study in sorted(d for d in pat.iterdir() if d.is_dir()):
            for series in sorted(d for d in study.iterdir() if d.is_dir()):
                files = sorted(f for f in series.iterdir() if f.is_file())
                stray = [f for f in series.iterdir() if not f.is_file() and not f.is_dir()]
                if any(d.is_dir() for d in series.iterdir()):
                    anomalies.append({"path": str(series), "issue": "series folder contains subfolders"})
                if not files:
                    anomalies.append({"path": str(series), "issue": "series folder has no files"})
                    continue
                head, err = read_header(files[0])
                if err:
                    anomalies.append({"path": str(files[0]), "issue": f"unreadable: {err}"})
                    continue
                modality = head.get("Modality")
                rec = {"patient_folder": pat.name, "study_folder": study.name, "series_folder": series.name,
                       "n_files_in_folder": len(files), "first_file": str(files[0].relative_to(root))}
                rec.update({k: head.get(k) for k in TAGS + EXTRA})
                series_rows.append(rec)
                if modality in CXR_MODALITIES:
                    for f in files:
                        row, err = read_header(f)
                        if err:
                            anomalies.append({"path": str(f), "issue": f"unreadable: {err}"})
                            continue
                        r = {"patient_folder": pat.name, "file": str(f.relative_to(root))}
                        r.update({k: row.get(k) for k in TAGS + EXTRA})
                        file_rows.append(r)
                    if head.get("PatientID") != pat.name:
                        anomalies.append({"path": str(series),
                                          "issue": f"PatientID {head.get('PatientID')} != folder {pat.name}"})
        if i % 100 == 0 or i == len(patients):
            print(f"  {i}/{len(patients)} patients, {len(series_rows)} series, "
                  f"{len(file_rows)} CXR files, {time.time() - t0:.0f}s", flush=True)
    return pd.DataFrame(series_rows), pd.DataFrame(file_rows), pd.DataFrame(anomalies)


def aggregates(series: pd.DataFrame, files: pd.DataFrame, root: Path, project: Path) -> dict:
    cxr_series = series[series["Modality"].isin(CXR_MODALITIES)]
    s: dict = {"scope": {"dicom_root": str(root), "note": "per-file headers read for CR/DX only"}}
    s["1_patients"] = {"folders": int(series["patient_folder"].nunique()),
                       "distinct_PatientID": int(series["PatientID"].nunique())}
    s["2_studies"] = {"all": int(series["StudyInstanceUID"].nunique()),
                      "cxr": int(cxr_series["StudyInstanceUID"].nunique())}
    s["3_series"] = {"all": int(len(series)), "distinct_SeriesInstanceUID": int(series["SeriesInstanceUID"].nunique()),
                     "cxr": int(len(cxr_series))}
    s["4_files"] = {"all_from_filesystem": int(series["n_files_in_folder"].sum()),
                    "cxr_headers_read": int(len(files))}

    def missing_counts(df, col, key="SeriesInstanceUID"):
        miss = df[df[col].isna()]
        return {"series": int(miss[key].nunique()), "patients": int(miss["PatientID"].nunique()),
                "files": int(len(miss)) if key == "file" else None}

    s["5_AcquisitionTime_missing_cxr"] = missing_counts(files, "AcquisitionTime", "file") | {
        "series": int(files[files["AcquisitionTime"].isna()]["SeriesInstanceUID"].nunique()),
        "patients": int(files[files["AcquisitionTime"].isna()]["PatientID"].nunique())}
    s["6_AcquisitionDate_missing_cxr"] = {
        "files": int(files["AcquisitionDate"].isna().sum()),
        "series": int(files[files["AcquisitionDate"].isna()]["SeriesInstanceUID"].nunique()),
        "patients": int(files[files["AcquisitionDate"].isna()]["PatientID"].nunique())}
    s["7_SeriesNumber_missing_cxr"] = {
        "files": int(files["SeriesNumber"].isna().sum()),
        "series": int(files[files["SeriesNumber"].isna()]["SeriesInstanceUID"].nunique()),
        "patients": int(files[files["SeriesNumber"].isna()]["PatientID"].nunique())}
    multi = series[series["n_files_in_folder"] > 1]
    multi_cxr = cxr_series[cxr_series["n_files_in_folder"] > 1]
    s["8_series_with_multiple_files"] = {
        "all_series": int(len(multi)), "all_patients": int(multi["PatientID"].nunique()),
        "cxr_series": int(len(multi_cxr)), "cxr_patients": int(multi_cxr["PatientID"].nunique()),
        "cxr_file_count_distribution": {int(k): int(v) for k, v in
                                        Counter(multi_cxr["n_files_in_folder"]).items()}}

    meta_path = project / "data/raw/clinical/metadata.csv"
    if meta_path.exists():
        meta = pd.read_csv(meta_path, dtype=str, keep_default_na=False)
        meta["n_images_meta"] = pd.to_numeric(meta["Number of Images"], errors="coerce")
        disk = series[["SeriesInstanceUID", "n_files_in_folder", "Modality"]].rename(
            columns={"Modality": "modality_disk"})
        m = meta.merge(disk, left_on="Series UID", right_on="SeriesInstanceUID", how="left")
        mism = m[m["n_files_in_folder"].notna() & (m["n_images_meta"] != m["n_files_in_folder"])]
        two = m[m["n_images_meta"] == 2]
        s["9_metadata_vs_filesystem"] = {
            "metadata_rows": int(len(meta)),
            "series_uid_not_found_on_disk": int(m["n_files_in_folder"].isna().sum()),
            "mismatched_counts": int(len(mism)),
            "rows_with_Number_of_Images_eq_2": int(len(two)),
            "of_those_file_count_is_2": int((two["n_files_in_folder"] == 2).sum()),
            "of_those_modality": {str(k): int(v) for k, v in Counter(two["modality_disk"].dropna()).items()},
            "example_mismatches": mism[["Series UID", "n_images_meta", "n_files_in_folder"]].head(10).to_dict("records"),
        }
    else:
        s["9_metadata_vs_filesystem"] = "metadata.csv not found"

    f = files.copy()
    f["acq_sec"] = f["AcquisitionTime"].map(acq_seconds)
    ser = f.drop_duplicates("SeriesInstanceUID")[
        ["PatientID", "StudyDate", "StudyInstanceUID", "SeriesInstanceUID", "SeriesNumber",
         "Modality", "SeriesDescription", "StudyDescription", "ViewPosition", "AcquisitionTime", "acq_sec"]]
    same_in_study = ser.groupby(["StudyInstanceUID", "AcquisitionTime"], dropna=True).size()
    s["10_same_acqtime_within_study"] = {
        "groups_with_2plus_series": int((same_in_study > 1).sum()),
        "series_involved": int(same_in_study[same_in_study > 1].sum()),
        "patients": int(ser.merge(same_in_study[same_in_study > 1].rename("n").reset_index(),
                                  on=["StudyInstanceUID", "AcquisitionTime"])["PatientID"].nunique()),
        "group_size_distribution": {int(k): int(v) for k, v in Counter(same_in_study[same_in_study > 1]).items()}}
    per_day = ser.groupby(["PatientID", "StudyDate"])["StudyInstanceUID"].nunique()
    s["11_multiple_studies_same_day"] = {
        "patient_days": int((per_day > 1).sum()),
        "patients": int(per_day[per_day > 1].reset_index()["PatientID"].nunique()),
        "studies_per_day_distribution": {int(k): int(v) for k, v in Counter(per_day).items()}}
    same_day_time = ser.groupby(["PatientID", "StudyDate", "AcquisitionTime"], dropna=True).size()
    s["12_same_day_same_acqtime_multiple_series"] = {
        "groups": int((same_day_time > 1).sum()),
        "series_involved": int(same_day_time[same_day_time > 1].sum()),
        "patients": int(same_day_time[same_day_time > 1].reset_index()["PatientID"].nunique())}
    s["13_modality_counts"] = {
        "series_all_modalities": {str(k): int(v) for k, v in Counter(series["Modality"].dropna()).items()},
        "cxr_files": {str(k): int(v) for k, v in Counter(files["Modality"].dropna()).items()},
        "cxr_patients_per_modality": {m_: int(cxr_series[cxr_series["Modality"] == m_]["PatientID"].nunique())
                                      for m_ in sorted(CXR_MODALITIES)}}
    s["14_descriptions"] = {
        "StudyDescription_cxr": {str(k): int(v) for k, v in
                                 Counter(cxr_series["StudyDescription"].fillna("<missing>")).most_common()},
        "SeriesDescription_cxr": {str(k): int(v) for k, v in
                                  Counter(cxr_series["SeriesDescription"].fillna("<missing>")).most_common()},
        "ViewPosition_cxr": {str(k): int(v) for k, v in
                             Counter(cxr_series["ViewPosition"].fillna("<missing>")).most_common()}}

    # duplicate-storage candidates: same patient + same StudyDate, acquisition times within DUP_WINDOW_SEC
    cand_groups, examples = [], []
    for (pid, date), g in ser.dropna(subset=["acq_sec"]).groupby(["PatientID", "StudyDate"]):
        g = g.sort_values("acq_sec")
        cluster = [g.iloc[0]]
        for _, row in g.iloc[1:].iterrows():
            if row["acq_sec"] - cluster[-1]["acq_sec"] <= DUP_WINDOW_SEC:
                cluster.append(row)
            else:
                if len(cluster) > 1:
                    cand_groups.append((pid, date, cluster))
                cluster = [row]
        if len(cluster) > 1:
            cand_groups.append((pid, date, cluster))
    def describe(cluster):
        mods = {r["Modality"] for r in cluster}
        return {"n_series": len(cluster), "same_study": len({r["StudyInstanceUID"] for r in cluster}) == 1,
                "modalities": sorted(mods), "cross_modality": len(mods) > 1,
                "same_series_desc": len({r["SeriesDescription"] for r in cluster}) == 1,
                "same_view": len({r["ViewPosition"] for r in cluster}) == 1,
                "identical_time": len({r["AcquisitionTime"] for r in cluster}) == 1,
                "max_gap_sec": round(max(r["acq_sec"] for r in cluster) - min(r["acq_sec"] for r in cluster), 3)}
    descs = [describe(c) for _, _, c in cand_groups]
    s["duplicate_candidates"] = {
        "window_seconds": DUP_WINDOW_SEC,
        "groups": len(cand_groups),
        "series_involved": int(sum(d["n_series"] for d in descs)),
        "patients": len({pid for pid, _, _ in cand_groups}),
        "identical_time": int(sum(d["identical_time"] for d in descs)),
        "within_same_study": int(sum(d["same_study"] for d in descs)),
        "cross_modality_CR_DX": int(sum(d["cross_modality"] for d in descs)),
        "same_series_description": int(sum(d["same_series_desc"] for d in descs)),
        "same_view_position": int(sum(d["same_view"] for d in descs)),
        "group_size_distribution": {int(k): int(v) for k, v in Counter(d["n_series"] for d in descs).items()},
    }
    for (pid, date, cluster), d in list(zip(cand_groups, descs))[:5]:
        examples.append({"PatientID": pid, "StudyDate": date, **d,
                         "series": [{"SeriesInstanceUID": r["SeriesInstanceUID"], "SeriesNumber": r["SeriesNumber"],
                                     "Modality": r["Modality"], "AcquisitionTime": r["AcquisitionTime"],
                                     "SeriesDescription": r["SeriesDescription"]} for r in cluster]})
    s["duplicate_candidates"]["examples"] = examples
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dicom-root", required=True)
    ap.add_argument("--project", default=".")
    ap.add_argument("--out", default="data/interim/cxr_audit")
    ap.add_argument("--max-patients", type=int, default=None, help="smoke test on the first N patients")
    args = ap.parse_args()
    root, project = Path(args.dicom_root), Path(args.project)
    out = project / args.out
    out.mkdir(parents=True, exist_ok=True)

    print(f"scanning {root} (read-only)...", flush=True)
    series, files, anomalies = scan(root, out, args.max_patients)
    for df, name in ((series, "series_level"), (files, "cxr_file_level"), (anomalies, "anomalies")):
        if not df.empty:
            df.to_csv(out / f"{name}.csv", index=False, encoding="utf-8-sig")
    summary = aggregates(series, files, root, project)
    summary["anomalies"] = {"rows": int(len(anomalies)),
                            "issues": {str(k): int(v) for k, v in Counter(anomalies["issue"]).items()} if len(anomalies) else {}}
    with open(out / "audit_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2, default=str)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    sys.exit(main())
