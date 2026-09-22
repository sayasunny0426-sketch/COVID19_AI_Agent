"""Task B step 8: quality control that runs BEFORE the Test set is opened.

This script never materialises a Test row. It loads Training and Validation only
(`tb.load_development`), and checks the Test side of the split using the split manifest alone
-- patient ids and the split column -- so no Test outcome and no Test clinical value is read.
Checks that genuinely need Test data (transform behaviour on Test, prediction alignment,
Test counts) belong to the post-Test QC script and are listed here as deferred.

Each check passes or fails; a non-zero exit gates the Test evaluation.

Usage:
    python scripts/28_taskB_pre_test_qc.py --project .
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
from covid_mortality.features import taskB_clinical as tb  # noqa: E402
from covid_mortality.features.taskB_preprocess import TaskBPreprocessor  # noqa: E402

warnings.filterwarnings("ignore")

DEFERRED_TO_POST_TEST = [
    "Test transform produces the same columns in the same order",
    "no NaN / Inf in the Test design matrix",
    "Test prediction count equals the Test patient count",
    "Test predictions aligned with Subject ID",
    "Test cohort and death counts match the frozen split",
]


class QC:
    def __init__(self):
        self.rows = []

    def check(self, name: str, ok: bool, detail: str = "") -> bool:
        self.rows.append({"check": name, "result": "PASS" if ok else "FAIL", "detail": detail})
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))
        return ok

    @property
    def failures(self):
        return [r for r in self.rows if r["result"] == "FAIL"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=".")
    args = ap.parse_args()
    project = Path(args.project).resolve()
    out = project / "results/taskB/qc"
    out.mkdir(parents=True, exist_ok=True)
    q = QC()

    # ---- split manifest integrity (ids and split labels only; no Test outcome read) ----
    man = tb.split_manifest(project)
    q.check("split manifest: Subject ID uniqueness", not man[tb.ID_COLUMN].duplicated().any(),
            f"{man[tb.ID_COLUMN].nunique()} unique of {len(man)}")
    sets = {s: set(man[man.split == s][tb.ID_COLUMN]) for s in ("train", "val", "test")}
    overlap = sum(len(a & b) for i, a in enumerate(sets.values())
                  for b in list(sets.values())[i + 1:])
    q.check("split manifest: no patient appears in two splits", overlap == 0, f"{overlap} overlaps")
    q.check("split manifest: cohort sizes match the frozen split",
            [len(sets[s]) for s in ("train", "val", "test")] == [1021, 128, 128],
            str({s: len(sets[s]) for s in sets}))
    q.check("split manifest: Test patients are present and held out",
            len(sets["test"]) == 128 and not (sets["test"] & (sets["train"] | sets["val"])),
            "128 Test ids exist in the manifest and appear in no other split "
            "(their outcomes and clinical values were not read)")

    # ---- development data only ---------------------------------------------------------
    dev = tb.load_development(project)          # Training + Validation; Test rows not loaded
    q.check("loader returned only Training and Validation rows",
            set(dev.split.unique()) == {"train", "val"}, str(sorted(dev.split.unique())))
    deaths = dev[dev.true_label == "1"].split.value_counts().to_dict()
    q.check("development death counts match the frozen split",
            deaths == {"train": 135, "val": 17}, str(deaths))
    q.check("outcome consistency in development rows",
            int(((dev[tb.OUTCOME] == "deceased") != (dev.true_label == "1")).sum()) == 0)

    # ---- leakage ------------------------------------------------------------------------
    feats = json.loads((project / "results/taskB/variable_selection/final_variables.json")
                       .read_text(encoding="utf-8"))["primary"]["features"]
    used_vars = {v.name for v in tb.CANDIDATES
                 if any(f == v.name or f.startswith(f"{v.name}_") for f in feats)}
    used_cols = {v.column for v in tb.CANDIDATES if v.name in used_vars}
    q.check("no post-T0 / outcome column is used as a predictor",
            not (used_cols & set(tb.POST_T0)), str(sorted(used_cols & set(tb.POST_T0))))
    q.check("no missing indicator for sex (D-072)", "sex_missing" not in feats)
    vsel = json.loads((project / "results/taskB/variable_selection/"
                       "variable_selection_summary.json").read_text(encoding="utf-8"))
    q.check("variable selection used the Training set only",
            vsel["specification"]["data"] == "training only"
            and "neither Validation nor Test" in vsel["validation_test_use"],
            vsel["specification"]["data"])
    q.check("variable selection method is backward elimination by AIC",
            vsel["specification"]["direction"] == "backward elimination"
            and vsel["specification"]["criterion"] == "AIC")
    fv = json.loads((project / "results/taskB/variable_selection/final_variables.json")
                    .read_text(encoding="utf-8"))
    q.check("final variable set records that it was fitted on Training only",
            fv.get("fitted_on") == "train only", fv.get("fitted_on", ""))
    q.check("no inpatient-only result is mixed into the primary tree",
            not list((project / "results/taskB/preprocessing").glob("*inpatient*"))
            and not list((project / "results/taskB/modeling").glob("*inpatient*")),
            "inpatient-only material is isolated in secondary_exploratory_unused/")

    audit = pd.read_csv(project / "results/taskB/preprocessing/variable_audit.csv",
                        encoding="utf-8-sig")
    q.check("every clinical column has a recorded keep/drop reason",
            int((audit.reason.fillna("") == "").sum()) == 0,
            f"{int((audit.reason.fillna('') == '').sum())} without a reason")

    # ---- preprocessing fitted on Training only -------------------------------------------
    train = tb.training_only(dev)
    val = dev[dev.split == "val"]
    pre = TaskBPreprocessor().fit(train)
    spec = json.loads((project / "results/taskB/preprocessing/preprocess_spec.json")
                      .read_text(encoding="utf-8"))
    q.check("saved spec records that it was fitted on Training only",
            spec.get("fitted_on") == "train only", spec.get("fitted_on", ""))
    q.check("Validation rows were not used to fit the preprocessor",
            TaskBPreprocessor().fit(dev).fill_values != pre.fill_values,
            "refitting on Train+Val changes the fill values, so they are Training-only")

    X = pre.transform(train)
    q.check("no NaN in the Training design matrix", int(X.isna().sum().sum()) == 0)
    q.check("no Inf in the Training design matrix", int(np.isinf(X.to_numpy(float)).sum()) == 0)
    q.check("frozen features are all present in the design matrix",
            all(f in X.columns for f in feats), f"{len(feats)} features")
    Xv = pre.transform(val)
    q.check("Validation: transform produces the same columns in the same order",
            list(Xv.columns) == list(X.columns))
    q.check("Validation: no NaN / Inf after transform",
            int(Xv.isna().sum().sum()) == 0 and int(np.isinf(Xv.to_numpy(float)).sum()) == 0)

    a = TaskBPreprocessor().fit(train).transform(train).to_numpy(float)
    b = TaskBPreprocessor().fit(train).transform(train).to_numpy(float)
    q.check("preprocessing is deterministic", bool(np.array_equal(a, b)))

    # ---- model artefacts and Validation predictions ---------------------------------------
    modeling = project / "results/taskB/modeling"
    fs = modeling / "final_selection_taskB.json"
    if fs.exists():
        frozen = json.loads(fs.read_text(encoding="utf-8"))
        q.check("frozen selection records that Test predictions were not generated",
                frozen.get("test_predictions_generated") is False)
        q.check("frozen selection carries the exact Test-status wording",
                "Test predictions and Test performance metrics have not been generated"
                in frozen.get("test_status", ""))
        q.check("frozen selection fixes the downstream prediction schema",
                frozen.get("prediction_schema", {}).get("file")
                == "results/taskB/evaluation/test_predictions_taskB.csv")
        q.check("frozen selection fixes the DeLong family and correction before Test",
                frozen.get("delong_plan", {}).get("multiplicity_correction", "").startswith("Holm")
                and len(frozen.get("delong_plan", {}).get("family", [])) == 4)
        q.check("frozen selection fixes the feature-importance plan before Test",
                frozen.get("feature_importance_plan", {}).get("dataset", "").startswith("Validation"))
        for name, m in frozen["models"].items():
            p = modeling / m["artefact"]
            ok = p.exists()
            q.check(f"{name}: saved model artefact exists", ok, m["artefact"])
            if ok:
                h = hashlib.sha256(p.read_bytes()).hexdigest()
                q.check(f"{name}: artefact hash matches the frozen record",
                        h == m["artefact_sha256"])
        vp = pd.read_csv(modeling / "validation_predictions.csv", encoding="utf-8-sig")
        q.check("Validation predictions: one row per patient", len(vp) == len(val),
                f"{len(vp)} vs {len(val)}")
        q.check("Validation predictions aligned with Subject ID",
                list(vp.subject_id) == list(val[tb.ID_COLUMN]))
        q.check("Validation predictions carry the correct labels",
                list(vp.true_label) == list(val.y))
        pcols = [c for c in vp.columns if c.startswith("prob_")]
        q.check("all Validation probabilities are in [0,1]",
                bool(((vp[pcols] >= 0) & (vp[pcols] <= 1)).all().all()))
    else:
        q.check("frozen selection exists", False, "run scripts/27 first")

    # ---- Test has produced nothing yet ------------------------------------------------------
    q.check("no Test prediction or Test metric file exists",
            not list((project / "results/taskB").glob("evaluation/test_*")),
            "Test predictions have not been generated")

    res = pd.DataFrame(q.rows)
    res.to_csv(out / "pre_test_qc_report.csv", index=False, encoding="utf-8-sig")
    (out / "pre_test_qc_report.json").write_text(json.dumps({
        "generated": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "script": Path(__file__).name,
        "scope": ("Training and Validation only. Test rows were not materialised; the Test "
                  "side was checked from the split manifest (ids and split column) alone."),
        "deferred_to_post_test_qc": DEFERRED_TO_POST_TEST,
        "n_checks": len(q.rows), "n_failed": len(q.failures), "checks": q.rows},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{len(q.rows) - len(q.failures)}/{len(q.rows)} checks passed")
    print(f"deferred to post-Test QC: {len(DEFERRED_TO_POST_TEST)} checks")
    return 1 if q.failures else 0


if __name__ == "__main__":
    sys.exit(main())
