"""Task B step 6: search and fit the three model families (plus the secondary XGBoost).

Hyperparameters are scored by repeated stratified 5-fold x 3 CV inside the Training set;
the preprocessor is re-fitted on every fold. The winning condition per family is refit on the
whole Training set and scored once on Validation. Test is never read.

Search spaces are sized for n=1,021 with 135 events and are fixed here before any result is
seen; the reason for each range is in the comment next to it.

Usage:
    python scripts/26_taskB_train_models.py --project . --model all
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from covid_mortality.evaluation.metrics import average_precision, roc_auc  # noqa: E402
from covid_mortality.features import taskB_clinical as tb  # noqa: E402
from covid_mortality.features.taskB_preprocess import TaskBPreprocessor  # noqa: E402
from covid_mortality.training import taskB_models as tm  # noqa: E402

warnings.filterwarnings("ignore")
SEED = 42

# Logistic regression: L2 only. C spans four orders of magnitude around 1 so that both a
# nearly unpenalised and a strongly shrunk fit are covered; class_weight is included because
# faculty feedback §11 explicitly asks for the weighted model to be re-analysed.
LR_GRID = {"C": [0.01, 0.1, 1.0, 10.0], "class_weight": [None, "balanced"]}

# XGBoost and the MLP are searched in two pre-registered stages rather than as one full grid.
# A full grid would be 288 and 96 conditions; with 135 events the extra conditions buy noise,
# not information, and the staged form is the same one used for Task A.
#
# XGBoost: depth 2-3 only -- with 135 events, depth 4+ leaves too few events per leaf.
# scale_pos_weight 6.563 = 886/135 mirrors the LR "balanced" option.
XGB_STAGE1 = {"max_depth": [2, 3], "learning_rate": [0.03, 0.05, 0.1],
              "min_child_weight": [1, 5, 10]}                                  # 18
XGB_STAGE2 = {"subsample": [0.8, 1.0], "colsample_bytree": [0.8, 1.0],
              "scale_pos_weight": [1.0, 886 / 135], "reg_lambda": [1.0, 10.0]}  # 16
XGB_STAGE1_FIXED = {"subsample": 0.8, "colsample_bytree": 0.8,
                    "scale_pos_weight": 1.0, "reg_lambda": 1.0}
XGB_N_ESTIMATORS = 300

# MLP: deliberately smaller than the previous Task B (64->32). With 12 inputs and 135 events a
# wide network has no data to support it.
MLP_STAGE1 = {"hidden": [(16, 8), (32, 16)], "dropout": [0.2, 0.3],
              "lr": [1e-4, 3e-4, 1e-3]}                                        # 12
MLP_STAGE2 = {"weight_decay": [1e-4, 1e-3], "batch_size": [32, 64],
              "pos_weight": [1.0, 886 / 135]}                                  # 8
MLP_STAGE1_FIXED = {"weight_decay": 1e-4, "batch_size": 32, "pos_weight": 1.0}


def load(project: Path):
    df = tb.load_development(project)   # Training + Validation only
    train = tb.training_only(df)
    val = df[df.split == "val"].copy()
    sel = json.loads((project / "results/taskB/variable_selection/final_variables.json")
                     .read_text(encoding="utf-8"))
    return df, train, val, sel["primary"]["features"]


def val_metrics(y, p) -> dict:
    return {"val_auroc": round(float(roc_auc(y, p)), 6),
            "val_auprc": round(float(average_precision(y, p)), 6)}


def run_lr(train, val, feats, out):
    rows = []
    for C, cw in product(LR_GRID["C"], LR_GRID["class_weight"]):
        fp, make = tm.logistic_factory(C, cw)
        t0 = time.time()
        cv = tm.cv_score(train, feats, fp)
        pre = TaskBPreprocessor().fit(train)
        m = make().fit(pre.transform(train)[feats].to_numpy(float), train.y.values)
        pv = m.predict_proba(pre.transform(val)[feats].to_numpy(float))[:, 1]
        rows.append({"condition": f"C={C}_cw={cw}", "C": C, "class_weight": str(cw),
                     "cv_mean_auroc": round(cv.mean_auroc, 6), "cv_sd_auroc": round(cv.sd_auroc, 6),
                     "cv_mean_auprc": round(cv.mean_auprc, 6), **val_metrics(val.y.values, pv),
                     "seconds": round(time.time() - t0, 1)})
        print(f"  LR {rows[-1]['condition']:22s} CV {cv.mean_auroc:.4f}±{cv.sd_auroc:.4f} "
              f"val {rows[-1]['val_auroc']:.4f}")
    return pd.DataFrame(rows)


def _eval_xgb(params, train, val, feats, native_missing, stage):
    fp, make = tm.xgb_factory(params, XGB_N_ESTIMATORS)
    t0 = time.time()
    cv = tm.cv_score(train, feats, fp, native_missing=native_missing, scale=not native_missing)
    pre = TaskBPreprocessor(native_missing=native_missing, scale=not native_missing).fit(train)
    m = make().fit(pre.transform(train)[feats].to_numpy(float), train.y.values)
    pv = m.predict_proba(pre.transform(val)[feats].to_numpy(float))[:, 1]
    return {"stage": stage, "condition": "_".join(f"{k}={v}" for k, v in params.items()),
            **params, "cv_mean_auroc": round(cv.mean_auroc, 6),
            "cv_sd_auroc": round(cv.sd_auroc, 6), "cv_mean_auprc": round(cv.mean_auprc, 6),
            **val_metrics(val.y.values, pv), "seconds": round(time.time() - t0, 1)}


def run_xgb(train, val, feats, out, native_missing=False):
    if native_missing:
        # D-071: the secondary model differs from the primary ONLY in how missingness is
        # handled. It sees the same clinical variables, with missingness left as NaN for the
        # tree to split on, so the imputation-derived indicator column does not exist.
        feats = [f for f in feats if not f.endswith("_missing")]
        print(f"  native-missing feature set ({len(feats)}): {feats}")
    rows = []
    for combo in product(*XGB_STAGE1.values()):
        p = {**XGB_STAGE1_FIXED, **dict(zip(XGB_STAGE1, combo))}
        rows.append(_eval_xgb(p, train, val, feats, native_missing, 1))
        print(f"  [1] {rows[-1]['condition'][:52]:52s} CV {rows[-1]['cv_mean_auroc']:.4f}")
    best1 = max(rows, key=lambda r: r["cv_mean_auroc"])
    keep = {k: best1[k] for k in XGB_STAGE1}
    print(f"  stage 1 best: {keep} -> CV {best1['cv_mean_auroc']:.4f}")
    for combo in product(*XGB_STAGE2.values()):
        p = {**keep, **dict(zip(XGB_STAGE2, combo))}
        rows.append(_eval_xgb(p, train, val, feats, native_missing, 2))
        print(f"  [2] {rows[-1]['condition'][:52]:52s} CV {rows[-1]['cv_mean_auroc']:.4f}")
    return pd.DataFrame(rows)


def _eval_mlp(kw, train, val, feats, stage):
    fp, trainer = tm.mlp_factory(**kw)
    t0 = time.time()
    cv = tm.cv_score(train, feats, fp)
    pre = TaskBPreprocessor().fit(train)
    pv, _, best_epoch = trainer(pre.transform(train)[feats].to_numpy(float), train.y.values,
                                pre.transform(val)[feats].to_numpy(float))
    return {"stage": stage, "condition": "_".join(f"{k}={v}" for k, v in kw.items()),
            **{k: str(v) for k, v in kw.items()},
            "cv_mean_auroc": round(cv.mean_auroc, 6), "cv_sd_auroc": round(cv.sd_auroc, 6),
            "cv_mean_auprc": round(cv.mean_auprc, 6), **val_metrics(val.y.values, pv),
            "best_epoch": best_epoch, "seconds": round(time.time() - t0, 1)}


def run_mlp(train, val, feats, out):
    rows = []
    for combo in product(*MLP_STAGE1.values()):
        kw = {**MLP_STAGE1_FIXED, **dict(zip(MLP_STAGE1, combo))}
        rows.append(_eval_mlp(kw, train, val, feats, 1))
        print(f"  [1] {rows[-1]['condition'][:56]:56s} CV {rows[-1]['cv_mean_auroc']:.4f} "
              f"({rows[-1]['seconds']:.0f}s)")
    best1 = max(rows, key=lambda r: r["cv_mean_auroc"])
    keep = {"hidden": eval(best1["hidden"]), "dropout": float(best1["dropout"]),
            "lr": float(best1["lr"])}
    print(f"  stage 1 best: {keep} -> CV {best1['cv_mean_auroc']:.4f}")
    for combo in product(*MLP_STAGE2.values()):
        kw = {**keep, **dict(zip(MLP_STAGE2, combo))}
        rows.append(_eval_mlp(kw, train, val, feats, 2))
        print(f"  [2] {rows[-1]['condition'][:56]:56s} CV {rows[-1]['cv_mean_auroc']:.4f} "
              f"({rows[-1]['seconds']:.0f}s)")
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=".")
    ap.add_argument("--model", default="all", choices=["lr", "xgb", "xgb_native", "mlp", "all"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    project = Path(args.project).resolve()
    out = Path(args.out) if args.out else project / "results/taskB/modeling"
    out.mkdir(parents=True, exist_ok=True)

    df, train, val, feats = load(project)
    print(f"Training {len(train)} (deaths {int(train.y.sum())}), "
          f"Validation {len(val)} (deaths {int(val.y.sum())}). Test not read.")
    print(f"features ({len(feats)}): {feats}\n")

    jobs = {"lr": run_lr, "xgb": run_xgb, "mlp": run_mlp}
    todo = list(jobs) + ["xgb_native"] if args.model == "all" else [args.model]
    summary = {}
    for name in todo:
        print(f"=== {name} ===")
        t0 = time.time()
        if name == "xgb_native":
            res = run_xgb(train, val, feats, out, native_missing=True)
        else:
            res = jobs[name](train, val, feats, out)
        res = res.sort_values("cv_mean_auroc", ascending=False).reset_index(drop=True)
        res.to_csv(out / f"{name}_search.csv", index=False, encoding="utf-8-sig")
        best = res.iloc[0]
        summary[name] = {"n_conditions": len(res), "best_condition": best.condition,
                         "cv_mean_auroc": float(best.cv_mean_auroc),
                         "cv_sd_auroc": float(best.cv_sd_auroc),
                         "cv_mean_auprc": float(best.cv_mean_auprc),
                         "val_auroc": float(best.val_auroc), "val_auprc": float(best.val_auprc),
                         "minutes": round((time.time() - t0) / 60, 1)}
        print(f"  -> best {best.condition}: CV {best.cv_mean_auroc:.4f} "
              f"(sd {best.cv_sd_auroc:.4f}), Validation {best.val_auroc:.4f} "
              f"[{summary[name]['minutes']} min]\n")

    meta = {"generated": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
            "script": Path(__file__).name, "seed": SEED,
            "selection_metric": "mean ROC-AUC over repeated stratified 5-fold x 3 CV inside Training",
            "validation_role": "reported once per condition; family comparison and threshold only",
            "test_used": False, "features": feats, "n_features": len(feats),
            "grids": {"lr": {k: [str(x) for x in v] for k, v in LR_GRID.items()},
                      "xgb_stage1": {k: [str(x) for x in v] for k, v in XGB_STAGE1.items()},
                      "xgb_stage1_fixed": {k: str(v) for k, v in XGB_STAGE1_FIXED.items()},
                      "xgb_stage2": {k: [str(x) for x in v] for k, v in XGB_STAGE2.items()},
                      "xgb_n_estimators": XGB_N_ESTIMATORS,
                      "mlp_stage1": {k: [str(x) for x in v] for k, v in MLP_STAGE1.items()},
                      "mlp_stage1_fixed": {k: str(v) for k, v in MLP_STAGE1_FIXED.items()},
                      "mlp_stage2": {k: [str(x) for x in v] for k, v in MLP_STAGE2.items()}},
            "results": summary}
    (out / "search_summary.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                             encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
