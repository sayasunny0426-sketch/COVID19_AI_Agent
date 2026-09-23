"""Task C step 4: quality control before the Test set is opened.

Reads no Test outcome, no Test prediction and no Test clinical value. The Test side is checked
from the split manifest alone -- ids and split labels -- plus the SHA256 of the Task A / Task B
Test prediction files, which proves they have not changed without reading what is in them.

Outputs:
    results/taskC/qc/pre_test_qc_report.csv / .json

Usage:
    python scripts/38_taskC_pre_test_qc.py --project . --taskA-runs <...>
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
from covid_mortality.fusion import taskC_fusion as tc  # noqa: E402

DEFERRED = [
    "Test transform / merge produces the frozen columns in the frozen order",
    "Test prediction count equals the Test patient count",
    "Test predictions aligned with subject_id and with the fixed labels",
    "no NaN / Inf in the Test fused predictions",
    "combined all-model file contains 128 patients with every probability column",
]


class QC:
    def __init__(self):
        self.rows = []

    def check(self, name, ok, detail=""):
        self.rows.append({"check": name, "result": "PASS" if ok else "FAIL", "detail": detail})
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))
        return ok

    @property
    def failures(self):
        return [r for r in self.rows if r["result"] == "FAIL"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=".")
    ap.add_argument("--taskA-runs", required=True)
    ap.add_argument("--taskA-eval", default="github_repo/results/taskA/evaluation")
    args = ap.parse_args()
    project, runs = Path(args.project).resolve(), Path(args.taskA_runs)
    a_eval = project / args.taskA_eval
    out = project / "results/taskC/qc"
    out.mkdir(parents=True, exist_ok=True)
    q = QC()

    frozen = json.loads((project / "results/taskC/modeling/final_selection_taskC.json")
                        .read_text(encoding="utf-8"))
    split = pd.read_csv(project / "data/splits/COVID19_固定患者split_1277.csv",
                        dtype=str, encoding="utf-8-sig")

    # ---- split manifest -----------------------------------------------------------------
    q.check("split manifest: subject id uniqueness",
            not split["Subject ID"].duplicated().any(),
            f"{split['Subject ID'].nunique()} unique of {len(split)}")
    sets = {s: set(split[split.split == s]["Subject ID"]) for s in ("train", "val", "test")}
    overlap = sum(len(a & b) for i, a in enumerate(sets.values())
                  for b in list(sets.values())[i + 1:])
    q.check("split manifest: no patient in two splits", overlap == 0, f"{overlap}")
    q.check("split manifest: cohort sizes unchanged",
            [len(sets[s]) for s in ("train", "val", "test")] == [1021, 128, 128])
    q.check("Test patients held out and untouched",
            len(sets["test"]) == 128 and not (sets["test"] & (sets["train"] | sets["val"])),
            "ids only; no Test outcome or prediction was read")

    # ---- Task A / Task B inputs unchanged since the freeze ------------------------------
    for role, rec in frozen["input_sha256"].items():
        p = project / rec["path"] if not Path(rec["path"]).is_absolute() else Path(rec["path"])
        if not p.exists():
            p = Path(rec["path"])
        ok = p.exists() and tc.sha256(p) == rec["sha256"]
        q.check(f"input unchanged since the freeze: {role}", ok, rec["sha256"][:16] + "…")

    q.check("Task A model was not retrained for Task C",
            frozen["cxr_component"]["retrained_for_taskC"] is False)
    q.check("Task B model was not retrained for Task C",
            frozen["clinical_component"]["retrained_for_taskC"] is False)
    taskB = json.loads((project / "results/taskB/modeling/final_selection_taskB.json")
                       .read_text(encoding="utf-8"))
    name = frozen["clinical_component"]["model"]
    art = project / "results/taskB/modeling" / taskB["models"][name]["artefact"]
    q.check("Task B clinical artefact hash matches its own frozen record",
            art.exists() and tc.sha256(art) == taskB["models"][name]["artefact_sha256"],
            taskB["models"][name]["artefact"])

    # ---- Validation inputs ---------------------------------------------------------------
    val = tc.load_split("val", runs / "lr3e-4_aug_b/seed42/val_predictions.csv",
                        project / "results/taskB/modeling/validation_predictions.csv",
                        split, tc.CLINICAL_COLUMNS)
    y = val.y
    probs = [c for c in val.frame.columns if c.startswith("prob_")]
    q.check("Validation: Task A and Task B ids aligned to the manifest",
            list(val.frame.subject_id) == val.order, f"{len(val.order)} patients")
    q.check("Validation: labels aligned between Task A and Task B", True,
            "load_split raises if they disagree")
    q.check("Validation: no duplicate ids", not val.frame.subject_id.duplicated().any())
    q.check("Validation: no missing predictions",
            int(val.frame[probs].isna().sum().sum()) == 0)
    q.check("Validation: no NaN / Inf",
            not np.isinf(val.frame[probs].to_numpy(float)).any())
    q.check("Validation: probabilities in [0,1]",
            bool(((val.frame[probs] >= 0) & (val.frame[probs] <= 1)).all().all()))
    q.check("Validation: prediction count correct", len(val.frame) == 128, f"{len(val.frame)}")
    q.check("Validation: death count correct", int(y.sum()) == 17, f"{int(y.sum())}")

    # ---- the fused Validation predictions reproduce from the frozen weight ---------------
    vp = pd.read_csv(project / "results/taskC/validation/validation_predictions_taskC.csv",
                     dtype={"subject_id": str}, encoding="utf-8-sig")
    w = frozen["fusion"]["w_clinical"]
    recomputed = tc.fuse(val.clinical(frozen["clinical_component"]["model"]), val.cxr(), w,
                         "probability")
    q.check("saved Validation fusion reproduces from the frozen weight",
            float(np.max(np.abs(recomputed - vp.prob_late_fusion.to_numpy(float)))) < 1e-12,
            f"max |difference| {np.max(np.abs(recomputed - vp.prob_late_fusion.to_numpy(float))):.2e}")
    q.check("Validation fusion row order matches the manifest",
            list(vp.subject_id) == val.order)

    # ---- the specification is complete and frozen ------------------------------------------
    for key, label in [("fusion", "fusion strategy and weights"),
                       ("threshold", "threshold"),
                       ("calibration_definition", "calibration definition"),
                       ("bootstrap", "bootstrap settings"),
                       ("prediction_schema", "output schema"),
                       ("modality_importance_plan", "modality importance method"),
                       ("delong_plan", "DeLong plan")]:
        q.check(f"frozen: {label}", bool(frozen.get(key)))
    q.check("frozen: ECE bin count and scheme are explicit",
            frozen["calibration_definition"]["ece_bins"] == 10
            and frozen["calibration_definition"]["ece_scheme"] == "equal_width",
            frozen["calibration_definition"]["statement"])
    q.check("frozen: threshold is not recomputed on Test",
            frozen["threshold"]["recomputed_on_test"] is False)
    q.check("frozen: weight was selected on Validation only",
            frozen["fusion"]["search"]["dataset_used"] == "Validation only"
            and frozen["fusion"]["search"]["test_used"] is False)
    q.check("frozen: the primary strategy was fixed before the search",
            frozen["fusion"]["strategy_fixed_before_search"] is True)
    q.check("frozen: records that Test predictions do not exist",
            frozen["test_predictions_generated"] is False)

    # ---- no Test output yet ------------------------------------------------------------------
    ev = project / "results/taskC/evaluation"
    q.check("no Task C Test prediction file exists",
            not (ev.exists() and list(ev.glob("test_*"))),
            "Test predictions have not been generated")
    q.check("no combined all-model prediction file exists",
            not (project / "results/comparison/test_predictions_all_models.csv").exists())
    q.check("no DeLong results exist",
            not list((project / "results/comparison").glob("delong_results.*")))

    res = pd.DataFrame(q.rows)
    res.to_csv(out / "pre_test_qc_report.csv", index=False, encoding="utf-8-sig")
    (out / "pre_test_qc_report.json").write_text(json.dumps({
        "generated": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "script": Path(__file__).name,
        "scope": ("Validation and the split manifest only. No Test outcome, prediction or "
                  "clinical value was read; the Task A / Task B Test files were hashed, not "
                  "opened."),
        "deferred_to_post_test_qc": DEFERRED,
        "frozen_sha256": tc.sha256(project / "results/taskC/modeling/final_selection_taskC.json"),
        "n_checks": len(q.rows), "n_failed": len(q.failures), "checks": q.rows},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{len(q.rows) - len(q.failures)}/{len(q.rows)} checks passed")
    print(f"deferred to post-Test QC: {len(DEFERRED)}")
    return 1 if q.failures else 0


if __name__ == "__main__":
    sys.exit(main())
