"""Task A preprocessing cache: DICOM -> deterministic pipeline -> 16-bit PNG (512x512) + round-trip QC.

Pipeline (docs/task_a_implementation_plan.md §2; image-local only, no dataset statistics,
so nothing is "fitted" and no information can cross the split boundary):
  1. read pixel_array (uint16)
  2. apply RescaleSlope / RescaleIntercept (identity on this data, applied for explicitness)
  3. invert if PhotometricInterpretation == MONOCHROME1 (does not occur here)
  4. per-image percentile clip (1-99) and min-max to [0,1]   <- DICOM WindowCenter/Width not used
  5. zero-pad to a square on the long side (aspect ratio preserved)
  6. resize to 512x512 (bilinear)
  7. quantise to uint16 and save as 16-bit PNG

Round-trip QC on every image: re-read the PNG and check dtype, shape, value range and that
the stored values equal the quantised pipeline output exactly. Any deviation stops the run.

Read-only with respect to the DICOM source; the existing processed_png_1277 is never touched
and (per the researcher's instruction of 2026-09-20) is not consulted at all.

Usage:
    python scripts/06_taskA_preprocess_cache.py --project . [--limit N] [--out <dir>]
"""
import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
from PIL import Image

TARGET = 512
PCT_LOW, PCT_HIGH = 1.0, 99.0
UINT16_MAX = 65535


def preprocess(ds) -> tuple[np.ndarray, dict]:
    """Return the float32 [0,1] image at TARGETxTARGET and a record of what was done."""
    arr = ds.pixel_array.astype(np.float32)
    rec = {"orig_rows": int(arr.shape[0]), "orig_cols": int(arr.shape[1]),
           "orig_min": float(arr.min()), "orig_max": float(arr.max())}
    slope = float(getattr(ds, "RescaleSlope", 1) or 1)
    intercept = float(getattr(ds, "RescaleIntercept", 0) or 0)
    rec["rescale_slope"], rec["rescale_intercept"] = slope, intercept
    arr = arr * slope + intercept
    photometric = str(getattr(ds, "PhotometricInterpretation", ""))
    rec["photometric"] = photometric
    rec["inverted"] = photometric == "MONOCHROME1"
    if rec["inverted"]:
        arr = arr.max() - arr
    lo, hi = np.percentile(arr, [PCT_LOW, PCT_HIGH])
    rec["p_low"], rec["p_high"] = float(lo), float(hi)
    if hi <= lo:  # degenerate image
        rec["degenerate"] = True
        arr = np.zeros_like(arr)
    else:
        rec["degenerate"] = False
        rec["frac_clipped_low"] = float((arr < lo).mean())
        rec["frac_clipped_high"] = float((arr > hi).mean())
        arr = np.clip(arr, lo, hi)
        arr = (arr - lo) / (hi - lo)
    h, w = arr.shape
    side = max(h, w)
    padded = np.zeros((side, side), dtype=np.float32)
    top, left = (side - h) // 2, (side - w) // 2
    padded[top:top + h, left:left + w] = arr
    rec["pad_top"], rec["pad_left"], rec["pad_side"] = int(top), int(left), int(side)
    rec["pad_fraction"] = float(1 - (h * w) / (side * side))
    resized = np.asarray(Image.fromarray(padded, mode="F").resize((TARGET, TARGET), Image.BILINEAR),
                         dtype=np.float32)
    rec["resized_min"], rec["resized_max"] = float(resized.min()), float(resized.max())
    return resized, rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=".")
    ap.add_argument("--out", default="data/processed/taskA_png512_16bit")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    project = Path(args.project)
    out_dir = project / args.out
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(project / "data/interim/cxr_audit/index_cxr_manifest_window_T0m2_T0.csv",
                           encoding="utf-8-sig", dtype=str)
    split = pd.read_csv(project / "data/splits/COVID19_固定患者split_1277.csv", dtype=str)
    df = manifest.merge(split[["Subject ID", "split", "true_label"]], left_on="PatientID",
                        right_on="Subject ID", how="inner")
    assert len(df) == len(manifest) == 1277, (len(df), len(manifest))
    assert df.PatientID.is_unique
    if args.limit:
        df = df.head(args.limit)

    rows, failures, t0 = [], [], time.time()
    for i, r in enumerate(df.itertuples(index=False), 1):
        rec = {"PatientID": r.PatientID, "split": r.split, "true_label": r.true_label,
               "source_filepath": r.source_filepath, "SOPInstanceUID": r.SOPInstanceUID}
        try:
            ds = pydicom.dcmread(r.source_filepath)
            if str(ds.SOPInstanceUID) != str(r.SOPInstanceUID):
                raise ValueError(f"SOPInstanceUID mismatch: {ds.SOPInstanceUID} != {r.SOPInstanceUID}")
            img, prec = preprocess(ds)
            rec.update(prec)
            quantised = np.rint(img * UINT16_MAX).astype(np.uint16)
            png_path = img_dir / f"{r.PatientID}.png"
            # no explicit mode: Pillow infers I;16 from uint16 (the mode= form is deprecated)
            Image.fromarray(quantised).save(png_path, optimize=True)

            # --- round-trip QC -------------------------------------------------
            reread = Image.open(png_path)
            back = np.array(reread)
            rec["png_mode"] = reread.mode
            rec["reread_dtype"] = str(back.dtype)
            rec["reread_shape"] = "x".join(str(x) for x in back.shape)
            back16 = back.astype(np.uint16) if back.dtype != np.uint16 else back
            rec["max_abs_diff"] = int(np.max(np.abs(back16.astype(np.int64) - quantised.astype(np.int64))))
            rec["exact_roundtrip"] = bool(rec["max_abs_diff"] == 0)
            rec["dtype_is_uint16"] = bool(back.dtype == np.uint16)
            rec["shape_ok"] = bool(back.shape == (TARGET, TARGET))
            rec["stored_min"], rec["stored_max"] = int(quantised.min()), int(quantised.max())
            rec["file_bytes"] = png_path.stat().st_size
            rec["sha256"] = hashlib.sha256(png_path.read_bytes()).hexdigest()
            ok = rec["exact_roundtrip"] and rec["shape_ok"] and not rec["degenerate"]
            rec["qc_pass"] = bool(ok)
            if not ok:
                failures.append(rec)
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {e}"
            rec["qc_pass"] = False
            failures.append(rec)
        rows.append(rec)
        if i % 100 == 0 or i == len(df):
            print(f"  {i}/{len(df)}  {time.time() - t0:.0f}s", flush=True)

    rep = pd.DataFrame(rows)
    rep.to_csv(out_dir / "preprocess_records.csv", index=False, encoding="utf-8-sig")

    ok = rep[rep.qc_pass]
    summary = {
        "images_written": int(len(ok)),
        "failures": int(len(failures)),
        "target_size": TARGET,
        "pipeline": {"percentile_clip": [PCT_LOW, PCT_HIGH], "pad": "zero, square on long side",
                     "resize": "bilinear", "dtype": "uint16 (16-bit PNG)",
                     "window_center_width_used": False, "dataset_statistics_used": False},
        "png_mode": {str(k): int(v) for k, v in Counter(rep.get("png_mode", pd.Series(dtype=str)).dropna()).items()},
        "reread_dtype": {str(k): int(v) for k, v in Counter(rep.get("reread_dtype", pd.Series(dtype=str)).dropna()).items()},
        "reread_shape": {str(k): int(v) for k, v in Counter(rep.get("reread_shape", pd.Series(dtype=str)).dropna()).items()},
        "exact_roundtrip_all": bool(rep.get("exact_roundtrip", pd.Series([False])).all()),
        "dtype_is_uint16_all": bool(rep.get("dtype_is_uint16", pd.Series([False])).all()),
        "max_abs_diff_max": int(rep.get("max_abs_diff", pd.Series([0])).max()),
        "photometric": {str(k): int(v) for k, v in Counter(rep.get("photometric", pd.Series(dtype=str)).dropna()).items()},
        "inverted_count": int(rep.get("inverted", pd.Series(dtype=bool)).sum()),
        "degenerate_count": int(rep.get("degenerate", pd.Series(dtype=bool)).sum()),
        "stored_min": {"min": int(ok.stored_min.min()), "max": int(ok.stored_min.max())} if len(ok) else {},
        "stored_max": {"min": int(ok.stored_max.min()), "max": int(ok.stored_max.max())} if len(ok) else {},
        "split_counts": {str(k): int(v) for k, v in Counter(ok.split).items()},
        "deaths_per_split": {str(k): int(v) for k, v in Counter(ok[ok.true_label == "1"].split).items()},
        "pad_fraction": {"min": round(float(ok.pad_fraction.min()), 4),
                         "median": round(float(ok.pad_fraction.median()), 4),
                         "max": round(float(ok.pad_fraction.max()), 4)} if len(ok) else {},
        "clipped_fraction_high": {"min": round(float(ok.frac_clipped_high.min()), 5),
                                  "max": round(float(ok.frac_clipped_high.max()), 5)} if len(ok) else {},
        "total_bytes": int(ok.file_bytes.sum()) if len(ok) else 0,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    with open(out_dir / "preprocess_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if failures:
        pd.DataFrame(failures).to_csv(out_dir / "preprocess_failures.csv", index=False, encoding="utf-8-sig")
        print(f"\nSTOP: {len(failures)} image(s) failed QC -> {out_dir / 'preprocess_failures.csv'}")


if __name__ == "__main__":
    main()
