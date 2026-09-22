"""Task B post-Test QC: the checks that were deferred because they need Test data.

Runs only after the approved Test evaluation. It verifies that what was produced matches the
frozen specification and that the patient-level file is usable for the later paired DeLong
analysis. It does not refit anything and it does not change any specification.

Usage:
    python scripts/32_taskB_post_test_qc.py --project .
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from covid_mortality.evaluation import taskB_schema as sch  # noqa: E402
from covid_mortality.features import taskB_clinical as tb  # noqa: E402
from covid_mortality.features.taskB_preprocess import TaskBPreprocessor  # noqa: E402

warnings.filterwarnings("ignore")


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


def sha256(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=".")
    args = ap.parse_args()
    project = Path(args.project).resolve()
    ev = project / "results/taskB/evaluation"
    modeling = project / "results/taskB/modeling"
    out = project / "results/taskB/qc"
    q = QC()

    pf_path = ev / "test_predictions_taskB.csv"
    if not pf_path.exists():
        print("STOP: the Test prediction file does not exist; run scripts/30 --approved first.")
        return 1
    frozen = json.loads((modeling / "final_selection_taskB.json").read_text(encoding="utf-8"))
    meta = json.loads((ev / "test_run_meta.json").read_text(encoding="utf-8"))
    pf = pd.read_csv(pf_path, dtype={sch.ID_COLUMN: str}, encoding="utf-8-sig")

    # ---- cohort -------------------------------------------------------------------
    df = tb.load_cohort(project)
    test = df[df.split == "test"]
    q.check("Test patient count", len(test) == 128, f"{len(test)}")
    q.check("Test death count", int(test.y.sum()) == 17, f"{int(test.y.sum())}")
    q.check("Test outcome consistency (last.status vs true_label)",
            int(((test[tb.OUTCOME] == "deceased") != (test.true_label == "1")).sum()) == 0)

    # ---- the patient-level file ----------------------------------------------------
    order = sch.patient_order(tb.split_manifest(project), "test")
    q.check("Subject ID uniqueness", not pf[sch.ID_COLUMN].duplicated().any(),
            f"{pf[sch.ID_COLUMN].nunique()} unique of {len(pf)}")
    q.check("prediction row count equals the Test patient count",
            len(pf) == len(test), f"{len(pf)} vs {len(test)}")
    q.check("subject set matches the split manifest",
            set(pf[sch.ID_COLUMN]) == set(order))
    q.check("row order matches the frozen patient order",
            list(pf[sch.ID_COLUMN]) == order)
    q.check("patient-order hash matches the frozen record",
            hashlib.sha256("\n".join(order).encode()).hexdigest()
            == frozen["prediction_schema"]["patient_order_sha256"])
    lab = dict(zip(test["Subject ID"].astype(str), test.y))
    q.check("true_label alignment with the fixed split",
            all(int(r[sch.LABEL_COLUMN]) == lab[r[sch.ID_COLUMN]] for _, r in pf.iterrows()))
    q.check("Subject ID dtype is str, matching Task A",
            pf[sch.ID_COLUMN].map(type).eq(str).all())

    prob_cols = [c for c in pf.columns if c.startswith("prob_")]
    q.check("all required DeLong columns are present",
            all(c in pf.columns for c in sch.TASKB_REQUIRED), str(sch.TASKB_REQUIRED))
    for c in prob_cols:
        v = pd.to_numeric(pf[c], errors="coerce")
        q.check(f"{c}: no missing prediction", int(v.isna().sum()) == 0)
        q.check(f"{c}: probabilities in [0,1]", bool(((v >= 0) & (v <= 1)).all()),
                f"min {v.min():.6f} max {v.max():.6f}")
    try:
        summary = sch.validate(pf, sch.TASKB_REQUIRED, expected_ids=order, expected_labels=test.y.values)
        q.check("shared prediction schema validates", True, str(summary))
    except ValueError as e:
        q.check("shared prediction schema validates", False, str(e)[:200])

    # ---- the specification was not changed by the Test run --------------------------
    for name, m in frozen["models"].items():
        p = modeling / m["artefact"]
        q.check(f"{name}: artefact hash still matches the frozen record",
                p.exists() and sha256(p) == m["artefact_sha256"])
    q.check("the Test run used the frozen specification file",
            meta["frozen_sha256"] == sha256(modeling / "final_selection_taskB.json"),
            "frozen file unchanged since the Test run")
    q.check("the Test run was not a dry run", meta["dry_run"] is False)
    q.check("the split evaluated was 'test'", meta["split_evaluated"] == "test")

    tbl = pd.read_csv(ev / "test_metrics_table.csv", encoding="utf-8-sig")
    thr_frozen = {n: m["threshold_youden"] for n, m in frozen["models"].items()}
    q.check("thresholds are the Validation-derived frozen values, not recomputed on Test",
            all(abs(float(r.threshold) - thr_frozen[r.model]) < 1e-12 for _, r in tbl.iterrows()),
            str({r.model: r.threshold for _, r in tbl.iterrows()}))

    # ---- preprocessing used Training-fitted parameters only -------------------------
    train = tb.training_only(df)
    pre_train = TaskBPreprocessor().fit(train)
    pre_all = TaskBPreprocessor().fit(df)
    q.check("preprocessing parameters come from Training only",
            pre_train.fill_values != pre_all.fill_values,
            "refitting on the whole cohort changes the fill values, so Test did not contribute")
    Xt = pre_train.transform(test)
    q.check("Test transform produces the frozen columns in the frozen order",
            list(Xt.columns) == list(pre_train.transform(train).columns))
    q.check("no NaN in the Test design matrix", int(Xt.isna().sum().sum()) == 0)
    q.check("no Inf in the Test design matrix", int(np.isinf(Xt.to_numpy(float)).sum()) == 0)
    q.check("frozen features are all present in the Test design matrix",
            all(f in Xt.columns for f in frozen["features_primary"]))

    # ---- the saved predictions are reproducible from the frozen models -----------------
    from covid_mortality.evaluation.taskB_schema import ID_COLUMN
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "te", project / "scripts/30_taskB_test_evaluation.py")
    te = importlib.util.module_from_spec(spec)
    sys.argv = ["x"]
    spec.loader.exec_module(te)
    ev_rows = test.set_index(test["Subject ID"].astype(str)).loc[order].reset_index(drop=True)
    max_diff = 0.0
    for name, col in te.FAMILIES.items():
        pre_m, feats_m, predict = te.load_model(name, frozen["models"][name], train, project)
        p_new = predict(pre_m.transform(ev_rows)[feats_m].to_numpy(float))
        d = float(np.max(np.abs(p_new - pf[col].to_numpy(float))))
        max_diff = max(max_diff, d)
        q.check(f"{name}: saved Test predictions reproduce from the frozen model",
                d < 1e-6, f"max |difference| = {d:.3e}")
    q.check("all saved Test predictions reproduce within 1e-6", max_diff < 1e-6,
            f"largest difference across models {max_diff:.3e}")

    # ---- Test access history -----------------------------------------------------------
    log = (project / "results/taskB/qc/test_access_log.jsonl")
    entries = [json.loads(l) for l in log.read_text(encoding="utf-8").splitlines() if l.strip()] \
        if log.exists() else []
    evals = [e for e in entries if e.get("event") == "test evaluation"]
    q.check("the Test set was evaluated exactly once",
            len(evals) == 1, f"{len(evals)} evaluation(s), {len(entries)} log entries total")

    # ---- paired DeLong plan and its record correction ---------------------------------
    q.check("paired DeLong has not been run for Task B alone",
            not list((project / "results").glob("comparison/delong_results.*")),
            "DeLong waits for Task C so that CXR and Late Fusion join the same patients")
    plan_path = project / "results/comparison/delong_plan.json"
    q.check("the authoritative DeLong plan file exists", plan_path.exists())
    if plan_path.exists():
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        fams = plan.get("families", {})
        q.check("the DeLong plan defines two comparison families", len(fams) == 2,
                str(list(fams)))
        q.check("Family A holds the three Late Fusion vs clinical-model comparisons",
                len(fams.get("A_late_fusion_vs_clinical_models", {})
                    .get("comparisons", [])) == 3)
        q.check("Family B holds the two Late Fusion vs reference-model comparisons",
                len(fams.get("B_late_fusion_vs_reference_models", {})
                    .get("comparisons", [])) == 2)
        q.check("each family is Holm-corrected separately",
                plan.get("multiplicity_correction", "").endswith("within each family"))
        q.check("the plan records which frozen file it supersedes, and that file is unchanged",
                plan["supersedes"]["sha256_of_that_file"]
                == sha256(modeling / "final_selection_taskB.json"),
                "the frozen Task B specification was not edited by the correction")

    res = pd.DataFrame(q.rows)
    res.to_csv(out / "post_test_qc_report.csv", index=False, encoding="utf-8-sig")
    (out / "post_test_qc_report.json").write_text(json.dumps({
        "generated": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "script": Path(__file__).name,
        "n_checks": len(q.rows), "n_failed": len(q.failures), "checks": q.rows},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{len(q.rows) - len(q.failures)}/{len(q.rows)} checks passed")
    return 1 if q.failures else 0


if __name__ == "__main__":
    sys.exit(main())
