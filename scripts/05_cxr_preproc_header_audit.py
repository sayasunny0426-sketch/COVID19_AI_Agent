"""Audit the preprocessing-relevant DICOM headers of the 1,277 selected CXRs.

Read-only. No preprocessing decision is made here and no model is trained. Per-split
summaries are produced so that any decision that depends on pixel statistics can be based
on the Training set alone (Validation / Test values are reported for completeness only and
must not drive preprocessing choices).

Inputs : --dicom-dir  folder holding the selected DICOMs (one file per patient)
         data/splits/COVID19_固定患者split_1277.csv
Outputs: data/interim/cxr_preproc_audit/header_audit.csv
         data/interim/cxr_preproc_audit/header_audit_summary.json

Usage:
    python scripts/05_cxr_preproc_header_audit.py --project . --dicom-dir "<path>" [--decode]
"""
import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom

TAGS = ["PatientID", "Modality", "Manufacturer", "Rows", "Columns", "BitsAllocated", "BitsStored",
        "HighBit", "PixelRepresentation", "PhotometricInterpretation", "RescaleSlope",
        "RescaleIntercept", "RescaleType", "WindowCenter", "WindowWidth", "VOILUTFunction",
        "PresentationLUTShape", "PixelSpacing", "ImagerPixelSpacing", "PixelIntensityRelationship",
        "PixelIntensityRelationshipSign", "SamplesPerPixel", "SOPInstanceUID"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=".")
    ap.add_argument("--dicom-dir", required=True)
    ap.add_argument("--decode", action="store_true", help="also decode pixel_array (slow)")
    args = ap.parse_args()
    project, dicom_dir = Path(args.project), Path(args.dicom_dir)
    out = project / "data/interim/cxr_preproc_audit"
    out.mkdir(parents=True, exist_ok=True)

    split = pd.read_csv(project / "data/splits/COVID19_固定患者split_1277.csv", dtype=str)
    split_map = dict(zip(split["Subject ID"], split["split"]))

    files = sorted(f for f in dicom_dir.rglob("*") if f.is_file())
    rows, t0 = [], time.time()
    for i, f in enumerate(files, 1):
        rec = {"file": f.name, "decode_error": "", "read_error": ""}
        try:
            ds = pydicom.dcmread(f, stop_before_pixels=not args.decode)
        except Exception as e:
            rec["read_error"] = f"{type(e).__name__}: {e}"
            rows.append(rec)
            continue
        for t in TAGS:
            v = getattr(ds, t, None)
            if v is None:
                rec[t] = None
                rec[f"{t}_missing"] = True
                continue
            rec[f"{t}_missing"] = False
            if t in ("WindowCenter", "WindowWidth", "PixelSpacing", "ImagerPixelSpacing"):
                vals = list(v) if hasattr(v, "__iter__") and not isinstance(v, str) else [v]
                rec[t] = "|".join(str(x) for x in vals)
                rec[f"{t}_n_values"] = len(vals)
            else:
                rec[t] = str(v)
        rec["has_VOILUTSequence"] = "VOILUTSequence" in ds
        rec["has_ModalityLUTSequence"] = "ModalityLUTSequence" in ds
        rec["split"] = split_map.get(str(getattr(ds, "PatientID", "")), "<not_in_split>")
        if args.decode:
            try:
                arr = ds.pixel_array
                rec["pixel_dtype"] = str(arr.dtype)
                rec["pixel_shape"] = "x".join(str(x) for x in arr.shape)
                rec["pixel_min"] = int(np.min(arr))
                rec["pixel_max"] = int(np.max(arr))
                rec["pixel_mean"] = float(np.mean(arr))
            except Exception as e:
                rec["decode_error"] = f"{type(e).__name__}: {e}"
        rows.append(rec)
        if i % 100 == 0 or i == len(files):
            print(f"  {i}/{len(files)} files, {time.time() - t0:.0f}s", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out / "header_audit.csv", index=False, encoding="utf-8-sig")

    def counts(col):
        return {str(k): int(v) for k, v in Counter(df[col].fillna("<missing>")).most_common()}

    s: dict = {"n_files": int(len(df)), "dicom_dir": str(dicom_dir), "decoded": bool(args.decode)}
    s["read_errors"] = int((df.read_error != "").sum())
    s["split_counts"] = counts("split")
    for col in ["Modality", "Manufacturer", "PhotometricInterpretation", "BitsAllocated", "BitsStored",
                "HighBit", "PixelRepresentation", "SamplesPerPixel", "RescaleSlope", "RescaleIntercept",
                "RescaleType", "VOILUTFunction", "PresentationLUTShape", "PixelIntensityRelationship",
                "PixelIntensityRelationshipSign"]:
        s[col] = counts(col)
    s["missing_tags"] = {t: int(df.get(f"{t}_missing", pd.Series(dtype=bool)).sum()) for t in TAGS}
    s["has_VOILUTSequence"] = int(df.has_VOILUTSequence.sum())
    s["has_ModalityLUTSequence"] = int(df.has_ModalityLUTSequence.sum())
    for t in ["WindowCenter", "WindowWidth"]:
        n = df.get(f"{t}_n_values")
        s[f"{t}_value_count_distribution"] = ({int(k): int(v) for k, v in Counter(n.dropna()).items()}
                                              if n is not None else {})
        s[f"{t}_most_common_values"] = {str(k): int(v) for k, v in Counter(df[t].dropna()).most_common(8)}
    rc = (df.Rows.astype(str) + "x" + df.Columns.astype(str))
    s["Rows_x_Columns_top"] = {str(k): int(v) for k, v in Counter(rc).most_common(10)}
    s["distinct_Rows_x_Columns"] = int(rc.nunique())
    r = pd.to_numeric(df.Rows, errors="coerce")
    c = pd.to_numeric(df.Columns, errors="coerce")
    s["portrait_rows_gt_cols"] = int((r > c).sum())
    s["landscape_cols_gt_rows"] = int((c > r).sum())
    s["square"] = int((r == c).sum())
    ar = (r / c).dropna()
    s["aspect_ratio"] = {"min": round(float(ar.min()), 3), "max": round(float(ar.max()), 3),
                         "n_distinct": int(ar.round(3).nunique())}
    s["PixelSpacing_missing"] = int(df.get("PixelSpacing_missing", pd.Series(dtype=bool)).sum())
    if args.decode:
        s["decode_errors"] = int((df.decode_error != "").sum())
        s["pixel_dtype"] = counts("pixel_dtype")
        ok = df[df.decode_error == ""]
        s["pixel_value_range_all"] = {"min": int(ok.pixel_min.min()), "max": int(ok.pixel_max.max())}
        s["pixel_value_range_by_split"] = {
            sp: {"n": int(len(g)), "min": int(g.pixel_min.min()), "max": int(g.pixel_max.max()),
                 "mean_of_means": round(float(g.pixel_mean.mean()), 1)}
            for sp, g in ok.groupby("split")}
        s["note_on_pixel_stats"] = ("per-split values are descriptive; any preprocessing decision that "
                                    "depends on pixel statistics must use the Training set only")
        s["by_photometric_and_split"] = {
            f"{p}|{sp}": int(len(g)) for (p, sp), g in ok.groupby(["PhotometricInterpretation", "split"])}
    with open(out / "header_audit_summary.json", "w", encoding="utf-8") as fh:
        json.dump(s, fh, ensure_ascii=False, indent=2)
    print(json.dumps(s, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
