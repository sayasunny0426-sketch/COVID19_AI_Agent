"""Task B: recompute ECE for both pipelines under one common definition.

Why this exists
---------------
The two pipelines measured calibration differently, so their reported ECE values are not on the
same scale:

  human-guided   expected_calibration_error(y, p, n_bins=10) in
                 Notebook/TaskB_03_ClinicalModeling.ipynb, cell 10:
                     bins = np.linspace(0.0, 1.0, n_bins + 1)
                     bin_ids = np.digitize(y_prob, bins[1:-1], right=True)
                 -> 10 EQUAL-WIDTH bins over [0, 1]

  AI-agent       calibration_table(y, p, n_bins=5) in scripts/27 and scripts/30:
                     edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
                 -> 5 EQUAL-COUNT (quantile) bins

Both then weight each bin by its share of patients and sum |mean predicted - observed rate|,
so the aggregation is the same; only the binning differs.

What this script does
---------------------
Recomputes ECE for all six models from the **saved patient-level Test predictions**, using one
function and one bin definition: **10 equal-width bins**, matching the current manuscript.

It is a post hoc descriptive comparison. It does not modify the official Test metrics of either
pipeline, and it is not used for model selection, retraining or any threshold change.

Usage:
    python scripts/34_taskB_harmonized_ece.py --project . \
        --drive-root "<DRIVE_ROOT>"
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from covid_mortality.evaluation.metrics import expected_calibration_error  # noqa: E402

H_DIR = "02_TaskB_臨床データ/05_Evaluation"
H_FILES = {"LR": "TaskB_LR_Test患者別予測.csv",
           "XGBoost": "TaskB_XGBoost_Test患者別予測.csv",
           "MLP": "TaskB_MLP_Test患者別予測.csv"}
A_COLS = {"LR": "prob_clinical_lr", "XGBoost": "prob_clinical_xgboost",
          "MLP": "prob_clinical_mlp"}
# each pipeline's own reported Test ECE, for the reproduction check
REPORTED = {("LR", "Human-guided"): 0.041302, ("XGBoost", "Human-guided"): 0.090264,
            ("MLP", "Human-guided"): 0.308812, ("LR", "AI-Agent"): 0.161595,
            ("XGBoost", "AI-Agent"): 0.022064, ("MLP", "AI-Agent"): 0.296720}
HARMONISED_BINS, HARMONISED_SCHEME = 10, "equal_width"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=".")
    ap.add_argument("--drive-root", required=True)
    args = ap.parse_args()
    project, drive = Path(args.project).resolve(), Path(args.drive_root)
    out = project / "results/taskB/comparison"
    out.mkdir(parents=True, exist_ok=True)

    agent = pd.read_csv(project / "results/taskB/evaluation/test_predictions_taskB.csv",
                        dtype={"subject_id": str}, encoding="utf-8-sig")
    preds: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    for m, f in H_FILES.items():
        h = pd.read_csv(drive / H_DIR / f, dtype={"Subject ID": str}, encoding="utf-8-sig")
        preds[(m, "Human-guided")] = (h.true_label.to_numpy(int), h["prob"].to_numpy(float))
        preds[(m, "AI-Agent")] = (agent.true_label.to_numpy(int),
                                  agent[A_COLS[m]].to_numpy(float))
        # the same patients must be behind both, or the comparison is meaningless
        if set(h["Subject ID"]) != set(agent.subject_id):
            print(f"STOP: {m} subject sets differ between pipelines")
            return 1
    print(f"loaded 6 model predictions over the same 128 Test patients "
          f"({int(agent.true_label.sum())} deaths)")

    rows = []
    for (model, pipeline), (y, p) in preds.items():
        own_bins, own_scheme = ((10, "equal_width") if pipeline == "Human-guided"
                                else (5, "equal_count"))
        own = expected_calibration_error(y, p, n_bins=own_bins, scheme=own_scheme)
        harm = expected_calibration_error(y, p, n_bins=HARMONISED_BINS,
                                          scheme=HARMONISED_SCHEME)
        reported = REPORTED[(model, pipeline)]
        rows.append({
            "model": model, "pipeline": pipeline,
            "original_ece_reported": reported,
            "original_ece_definition": f"{own_bins} {own_scheme.replace('_', '-')} bins",
            "original_ece_recomputed_here": round(float(own), 6),
            "reproduces_reported": bool(abs(own - reported) < 5e-4),
            "harmonized_ece": round(float(harm), 6),
            "harmonized_definition": f"{HARMONISED_BINS} equal-width bins over [0,1]",
            "difference_harmonized_minus_original": round(float(harm - reported), 6),
            "n": len(y), "events": int(y.sum()),
            "source_predictions": (f"{H_DIR}/{H_FILES[model]}" if pipeline == "Human-guided"
                                   else "results/taskB/evaluation/test_predictions_taskB.csv"),
        })
    res = pd.DataFrame(rows).sort_values(["model", "pipeline"])
    res.to_csv(out / "harmonized_ece.csv", index=False, encoding="utf-8-sig")
    print("\n=== ECE under each pipeline's own definition, and under the common one ===")
    print(res[["model", "pipeline", "original_ece_definition", "original_ece_reported",
               "original_ece_recomputed_here", "reproduces_reported",
               "harmonized_ece"]].to_string(index=False))

    # per-bin detail, so a reader can see where the calibration error sits
    detail = []
    for (model, pipeline), (y, p) in preds.items():
        edges = np.linspace(0.0, 1.0, HARMONISED_BINS + 1)
        ids = np.digitize(p, edges[1:-1], right=True)
        for b in range(HARMONISED_BINS):
            m = ids == b
            if not m.sum():
                continue
            detail.append({"model": model, "pipeline": pipeline, "bin": b,
                           "bin_low": round(float(edges[b]), 3),
                           "bin_high": round(float(edges[b + 1]), 3),
                           "n": int(m.sum()), "events": int(y[m].sum()),
                           "mean_predicted": round(float(p[m].mean()), 6),
                           "observed_rate": round(float(y[m].mean()), 6),
                           "abs_gap": round(float(abs(p[m].mean() - y[m].mean())), 6),
                           "weight": round(float(m.sum() / len(y)), 6)})
    pd.DataFrame(detail).to_csv(out / "harmonized_ece_bins.csv", index=False,
                                encoding="utf-8-sig")

    failed = res[~res.reproduces_reported]
    meta = {
        "generated": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "script": Path(__file__).name,
        "purpose": ("post hoc descriptive comparison of calibration under one common "
                    "definition; the official Test metrics of both pipelines are unchanged"),
        "definitions_found": {
            "human_guided": {
                "n_bins": 10, "scheme": "equal-width over [0,1]",
                "evidence": ("Notebook/TaskB_03_ClinicalModeling.ipynb cell 10: "
                             "def expected_calibration_error(y_true, y_prob, n_bins=10) with "
                             "bins = np.linspace(0.0, 1.0, n_bins + 1)")},
            "ai_agent": {
                "n_bins": 5, "scheme": "equal-count (quantile)",
                "evidence": ("scripts/27_taskB_select_and_freeze.py and "
                             "scripts/30_taskB_test_evaluation.py, calibration_table(): "
                             "edges = np.quantile(p, np.linspace(0, 1, n_bins + 1)) with "
                             "n_bins = 5")},
            "identical": False,
            "aggregation_identical": ("both weight each bin by its share of patients and sum "
                                      "|mean predicted - observed rate|"),
        },
        "harmonized_definition": {"n_bins": HARMONISED_BINS, "scheme": "equal-width over [0,1]",
                                  "rationale": "matches the current manuscript"},
        "reproduction_check": {"all_reproduced": bool(failed.empty),
                               "failures": failed.to_dict(orient="records")},
        "not_used_for": ["model selection", "retraining", "threshold change",
                         "any change to the official Test metrics"],
        "outputs": ["results/taskB/comparison/harmonized_ece.csv",
                    "results/taskB/comparison/harmonized_ece_bins.csv"],
    }
    (out / "harmonized_ece_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                                  encoding="utf-8")
    print(f"\nreproduction check: {'all 6 reproduce' if failed.empty else failed.to_string()}")
    print(f"wrote harmonized_ece.csv, harmonized_ece_bins.csv and the meta to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
