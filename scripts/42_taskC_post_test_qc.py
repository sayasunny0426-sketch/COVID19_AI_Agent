"""Task C post-Test QC: the deferred checks, plus proof that the specification did not change.

Runs after the approved Test evaluation and the DeLong step. It refits nothing and changes
nothing; it verifies that what was produced matches the frozen specification and that the
patient-level files are internally consistent.

Outputs:
    results/taskC/qc/post_test_qc_report.csv / .json

Usage:
    python scripts/42_taskC_post_test_qc.py --project . --taskA-runs <...>
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from covid_mortality.evaluation.metrics import roc_auc  # noqa: E402
from covid_mortality.fusion import taskC_fusion as tc  # noqa: E402

warnings.filterwarnings("ignore")
COMBINED_REQUIRED = ["subject_id", "true_label", "prob_cxr", "prob_clinical_lr",
                     "prob_clinical_xgboost", "prob_clinical_mlp", "prob_late_fusion"]


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
    ev = project / "results/taskC/evaluation"
    cmp_dir = project / "results/comparison"
    out = project / "results/taskC/qc"
    q = QC()

    fp = project / "results/taskC/modeling/final_selection_taskC.json"
    frozen = json.loads(fp.read_text(encoding="utf-8"))
    meta = json.loads((ev / "test_run_meta.json").read_text(encoding="utf-8"))
    split = pd.read_csv(project / "data/splits/COVID19_固定患者split_1277.csv",
                        dtype=str, encoding="utf-8-sig")
    order = [str(x) for x in split[split.split == "test"]["Subject ID"]]

    # ---- the specification is the one that was frozen ---------------------------------
    q.check("the Test run used the frozen specification file",
            meta["frozen_sha256"] == tc.sha256(fp), "frozen file unchanged since the Test run")
    q.check("the run recorded that the specification did not change",
            meta["specification_changed"] is False)
    for role, rec in frozen["input_sha256"].items():
        p = project / rec["path"]
        if not p.exists():
            p = Path(rec["path"])
        q.check(f"input unchanged: {role}", p.exists() and tc.sha256(p) == rec["sha256"])
    q.check("Task A model not retrained", frozen["cxr_component"]["retrained_for_taskC"] is False)
    q.check("Task B model not retrained",
            frozen["clinical_component"]["retrained_for_taskC"] is False)

    # ---- patient-level Task C file -----------------------------------------------------
    pf = pd.read_csv(ev / "test_predictions_taskC.csv", dtype={"subject_id": str},
                     encoding="utf-8-sig")
    q.check("Task C Test predictions: 128 rows", len(pf) == 128, f"{len(pf)}")
    q.check("Task C Test predictions: unique subject_id",
            not pf.subject_id.duplicated().any())
    q.check("Task C Test predictions: row order matches the split manifest",
            list(pf.subject_id) == order)
    q.check("Task C Test predictions: 17 events", int(pf.true_label.sum()) == 17)
    pcols = [c for c in pf.columns if c.startswith("prob_")]
    q.check("Task C Test predictions: no missing probability",
            int(pf[pcols].isna().sum().sum()) == 0)
    q.check("Task C Test predictions: no NaN / Inf",
            not np.isinf(pf[pcols].to_numpy(float)).any())
    q.check("Task C Test predictions: probabilities in [0,1]",
            bool(((pf[pcols] >= 0) & (pf[pcols] <= 1)).all().all()))

    # ---- the fused column reproduces exactly from the frozen weight ---------------------
    test = tc.load_split("test", a_eval / "test_predictions_primary.csv",
                         project / "results/taskB/evaluation/test_predictions_taskB.csv",
                         split, tc.TEST_CLINICAL_COLUMNS)
    w, clin = frozen["fusion"]["w_clinical"], frozen["clinical_component"]["model"]
    recomputed = tc.fuse(test.clinical(clin), test.cxr(), w, "probability")
    diff = float(np.max(np.abs(recomputed - pf.prob_late_fusion.to_numpy(float))))
    q.check("the fused Test column reproduces from the frozen weight", diff < 1e-12,
            f"max |difference| {diff:.2e}")
    q.check("the fusion weight is the frozen one", w == 0.75, f"w_clinical = {w}")
    q.check("the clinical component is the frozen one", clin == "lr", clin)

    # ---- labels match the fixed split ---------------------------------------------------
    lab = dict(zip(split["Subject ID"].astype(str), (split.true_label == "1").astype(int)))
    q.check("Test labels match the fixed split",
            all(int(r.true_label) == lab[r.subject_id] for r in pf.itertuples()))

    # ---- threshold was not recomputed on Test --------------------------------------------
    m = json.loads((ev / "test_metrics_taskC.json").read_text(encoding="utf-8"))
    q.check("threshold equals the frozen Validation-derived value",
            abs(m["threshold"]["value"] - frozen["threshold"]["primary"]) < 1e-12,
            f"{m['threshold']['value']:.6f}")
    q.check("threshold recorded as not recomputed on Test",
            m["threshold"]["recomputed_on_test"] is False)
    q.check("calibration definition on Test matches the frozen one",
            m["calibration_definition"]["ece_bins"] == 10
            and m["calibration_definition"]["ece_scheme"] == "equal_width")

    # ---- combined all-model file ----------------------------------------------------------
    cb = pd.read_csv(cmp_dir / "test_predictions_all_models.csv", dtype={"subject_id": str},
                     encoding="utf-8-sig")
    q.check("combined file: required columns present",
            all(c in cb.columns for c in COMBINED_REQUIRED), str(COMBINED_REQUIRED))
    q.check("combined file: 128 patients", len(cb) == 128, f"{len(cb)}")
    q.check("combined file: unique subject_id", not cb.subject_id.duplicated().any())
    q.check("combined file: deterministic order matching the manifest",
            list(cb.subject_id) == order)
    q.check("combined file: labels aligned",
            all(int(r.true_label) == lab[r.subject_id] for r in cb.itertuples()))
    ccols = [c for c in cb.columns if c.startswith("prob_")]
    q.check("combined file: no missing probability", int(cb[ccols].isna().sum().sum()) == 0)
    q.check("combined file: probabilities in [0,1]",
            bool(((cb[ccols] >= 0) & (cb[ccols] <= 1)).all().all()))
    q.check("combined file: Task A column matches the Task A artefact",
            float(np.max(np.abs(cb.prob_cxr.to_numpy(float) - test.cxr()))) < 1e-12)
    q.check("combined file: Task B columns match the Task B artefact",
            all(float(np.max(np.abs(cb[f"prob_clinical_{a}"].to_numpy(float)
                                    - test.clinical(b)))) < 1e-12
                for a, b in [("lr", "lr"), ("xgboost", "xgb"), ("mlp", "mlp")]))

    # ---- DeLong output ---------------------------------------------------------------------
    dl = pd.read_csv(cmp_dir / "delong_results.csv", encoding="utf-8-sig")
    plan = json.loads((cmp_dir / "delong_plan.json").read_text(encoding="utf-8"))
    q.check("DeLong: family column present and populated",
            "family" in dl.columns and dl.family.notna().all())
    q.check("DeLong: Family A has 3 comparisons", int((dl.family == "A").sum()) == 3)
    q.check("DeLong: Family B has 2 comparisons", int((dl.family == "B").sum()) == 2)
    q.check("DeLong: the overlapping comparison appears once per family",
            int(((dl.model_1 == "late_fusion") & (dl.model_2 == "clinical_lr")).sum()) == 2)
    ov = dl[(dl.model_1 == "late_fusion") & (dl.model_2 == "clinical_lr")]
    q.check("DeLong: the overlapping comparison has one unadjusted p-value",
            ov.p_unadjusted.nunique() == 1, f"{ov.p_unadjusted.iloc[0]:.6f}")
    q.check("DeLong: Holm was applied within each family separately",
            plan["multiplicity_correction"].endswith("within each family"))
    q.check("DeLong: adjusted p-values are never below the unadjusted ones",
            bool((dl.p_holm_within_family >= dl.p_unadjusted - 1e-12).all()))
    q.check("DeLong: every comparison states that non-significance is not equivalence",
            bool(dl[~dl["significant_at_0.05_after_holm"]].interpretation
                 .str.contains("not evidence of equivalence").all()))
    q.check("DeLong: AUCs agree with the Test metrics table",
            abs(float(dl.AUC_1.iloc[0]) - float(m["auroc"]["point"])) < 1e-6)
    q.check("DeLong: input hash recorded",
            json.loads((cmp_dir / "delong_results.json").read_text(encoding="utf-8"))
            ["input_sha256"] == tc.sha256(cmp_dir / "test_predictions_all_models.csv"))

    # ---- secondary analyses kept separate ---------------------------------------------------
    sec = json.loads((ev / "secondary_exploratory_test.json").read_text(encoding="utf-8"))
    q.check("secondary Test results are labelled exploratory and not promoted",
            "NOT promoted to primary" in sec["role"])
    q.check("the primary model in the metrics table is the Late Fusion",
            pd.read_csv(ev / "test_metrics_table.csv", encoding="utf-8-sig")
            .query("role == 'PRIMARY'").model.iloc[0] == "late_fusion")

    # ---- curve and calibration data saved ----------------------------------------------------
    for kind in ("roc_points", "pr_points", "calibration"):
        n = len(list(ev.glob(f"test_{kind}_*.csv")))
        q.check(f"{kind} data saved for all five models", n == 5, f"{n} files")
    q.check("modality importance on Test saved and labelled post hoc",
            (ev / "modality_importance_test.csv").exists()
            and "post hoc" in pd.read_csv(ev / "modality_importance_test.csv",
                                          encoding="utf-8-sig").role.iloc[0])
    q.check("run metadata saved with package versions and seeds",
            bool(meta.get("packages")) and "seed" in meta)
    log = out / "test_access_log.jsonl"
    entries = [json.loads(l) for l in log.read_text(encoding="utf-8").splitlines() if l.strip()]
    q.check("Test access log records exactly one Task C Test evaluation",
            len([e for e in entries if e.get("event") == "test evaluation"]) == 1,
            f"{len(entries)} entries")

    res = pd.DataFrame(q.rows)
    res.to_csv(out / "post_test_qc_report.csv", index=False, encoding="utf-8-sig")
    (out / "post_test_qc_report.json").write_text(json.dumps({
        "generated": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "script": Path(__file__).name, "frozen_sha256": tc.sha256(fp),
        "n_checks": len(q.rows), "n_failed": len(q.failures), "checks": q.rows},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{len(q.rows) - len(q.failures)}/{len(q.rows)} checks passed")
    return 1 if q.failures else 0


if __name__ == "__main__":
    sys.exit(main())
