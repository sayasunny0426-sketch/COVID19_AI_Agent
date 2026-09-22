"""Task B final Test evaluation -- IMPLEMENTED BUT NOT RUN.

This script opens the Test set. It refuses to do so unless two conditions hold:
  * the frozen specification exists and every artefact hash still matches it, and
  * `--approved` is passed, which stands for the researcher's explicit approval.

`--dry-run` exercises the identical code path against the Validation set, which is how the
script is verified before approval; it writes to a scratch directory and never reads Test.

Outputs (results/taskB/evaluation/), written only on an approved run:
    test_predictions_taskB.csv   the patient-level file the downstream paired DeLong needs
    test_metrics_taskB.json      per-model metrics with stratified bootstrap CIs
    test_metrics_table.csv       the same as one table
    test_roc_points_*.csv        ROC coordinates per model
    test_pr_points_*.csv         PR coordinates per model
    test_calibration_*.csv       calibration bins per model
    test_run_meta.json           environment, hashes, seeds, command
    test_access_log.jsonl        append-only record that the Test set was opened

Usage:
    python scripts/30_taskB_test_evaluation.py --project . --dry-run     # verification
    python scripts/30_taskB_test_evaluation.py --project . --approved    # after approval only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from covid_mortality.evaluation import taskB_schema as sch  # noqa: E402
from covid_mortality.evaluation.metrics import (average_precision, binary_rates,  # noqa: E402
                                                bootstrap_ci, brier_score, roc_auc, roc_curve)
from covid_mortality.features import taskB_clinical as tb  # noqa: E402
from covid_mortality.features.taskB_preprocess import TaskBPreprocessor  # noqa: E402
from covid_mortality.training import taskB_models as tm  # noqa: E402

warnings.filterwarnings("ignore")
SEED, N_BOOT, BOOT_SEED = 42, 2000, 12345
FAMILIES = {"lr": "prob_clinical_lr", "xgb": "prob_clinical_xgboost",
            "mlp": "prob_clinical_mlp", "xgb_native": "prob_clinical_xgboost_native"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def calibration_table(y, p, n_bins=5) -> pd.DataFrame:
    edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    idx = np.digitize(p, edges[1:-1], right=True)
    rows = []
    for b in range(n_bins):
        m = idx == b
        if m.sum():
            rows.append({"bin": b, "n": int(m.sum()), "mean_predicted": float(p[m].mean()),
                         "observed_rate": float(y[m].mean()), "events": int(y[m].sum())})
    return pd.DataFrame(rows)


def load_model(name: str, spec: dict, train: pd.DataFrame, project: Path):
    """Rebuild the frozen model. The preprocessor is refit on Training, never on Test."""
    native = name.endswith("native")
    pre = TaskBPreprocessor(native_missing=native, scale=not native).fit(train)
    feats = spec["features"]
    Xtr = pre.transform(train)[feats].to_numpy(float)
    modeling = project / "results/taskB/modeling"
    p = spec["params"]
    if name == "lr":
        import joblib
        model = joblib.load(modeling / spec["artefact"])
        return pre, feats, (lambda X: model.predict_proba(X)[:, 1])
    if name.startswith("xgb"):
        import xgboost as xgb
        model = xgb.XGBClassifier()
        model.load_model(str(modeling / spec["artefact"]))
        return pre, feats, (lambda X: model.predict_proba(X)[:, 1])
    import torch
    _, trainer = tm.mlp_factory(**{k: (tuple(v) if k == "hidden" else v)
                                   for k, v in p.items() if k != "best_epoch"})
    _, model, _ = trainer(Xtr, train.y.values, Xtr)
    state = torch.load(modeling / spec["artefact"], weights_only=True)
    model.load_state_dict(state)
    model.eval()

    def predict(X):
        with torch.no_grad():
            return torch.sigmoid(model(torch.tensor(X, dtype=torch.float32))).squeeze(1).numpy()

    return pre, feats, predict


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=".")
    ap.add_argument("--approved", action="store_true",
                    help="the researcher has explicitly approved opening the Test set")
    ap.add_argument("--dry-run", action="store_true",
                    help="run the identical path on Validation; Test is not read")
    ap.add_argument("--recompute-metrics", action="store_true",
                    help="rewrite the metric tables from the predictions already saved. No "
                         "model is refit and no prediction changes; used only to correct an "
                         "output-formatting defect in the saved tables.")
    args = ap.parse_args()
    project = Path(args.project).resolve()
    modeling = project / "results/taskB/modeling"
    frozen_path = modeling / "final_selection_taskB.json"

    if not frozen_path.exists():
        print("STOP: the frozen specification does not exist. Run scripts/27 first.")
        return 1
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))

    # every artefact must still be the one that was frozen
    for name, spec in frozen["models"].items():
        p = modeling / spec["artefact"]
        if not p.exists():
            print(f"STOP: {spec['artefact']} is missing")
            return 1
        if sha256(p) != spec["artefact_sha256"]:
            print(f"STOP: {spec['artefact']} has changed since it was frozen")
            return 1
    print(f"frozen specification verified ({len(frozen['models'])} models, hashes match)")

    if not args.approved and not args.dry_run:
        print("\nSTOP: the Test set is not opened without --approved.\n"
              "      Use --dry-run to verify this script against Validation.")
        return 1

    split = "val" if args.dry_run else "test"
    if args.dry_run:
        print("DRY RUN: evaluating on Validation. The Test set is not read.")
        df = tb.load_development(project)
        out = project / "results/taskB/evaluation/_dry_run"
    else:
        print("APPROVED RUN: opening the Test set.")
        df = tb.load_cohort(project)                       # all splits
        out = project / "results/taskB/evaluation"
        existing = list(out.glob("test_predictions_taskB.csv"))
        if existing and not args.recompute_metrics:
            print(f"STOP: {existing[0].name} already exists. The Test set is evaluated once.")
            return 1
        if args.recompute_metrics and not existing:
            print("STOP: --recompute-metrics needs an existing prediction file.")
            return 1
    out.mkdir(parents=True, exist_ok=True)

    train = tb.training_only(df)
    ev = df[df.split == split].copy()
    y = ev.y.values
    order = sch.patient_order(tb.split_manifest(project), split)
    ev = ev.set_index(ev["Subject ID"].astype(str)).loc[order].reset_index(drop=True)
    y = ev.y.values
    print(f"{split}: {len(ev)} patients, {int(y.sum())} deaths")

    preds = {sch.ID_COLUMN: ev["Subject ID"].astype(str).tolist(),
             sch.LABEL_COLUMN: y.astype(int)}
    saved = None
    if args.recompute_metrics:
        saved = pd.read_csv(out / "test_predictions_taskB.csv",
                            dtype={sch.ID_COLUMN: str}, encoding="utf-8-sig")
        if list(saved[sch.ID_COLUMN]) != order:
            print("STOP: the saved prediction file does not match the frozen patient order.")
            return 1
        print("recomputing metric tables from the saved predictions; no model is refit")

    rows, curves = [], {}
    for name, col in FAMILIES.items():
        if saved is not None:
            p = saved[col].to_numpy(float)
        else:
            pre, feats, predict = load_model(name, frozen["models"][name], train, project)
            p = predict(pre.transform(ev)[feats].to_numpy(float))
        preds[col] = p
        thr = frozen["models"][name]["threshold_youden"]
        preds[f"pred_{col.replace('prob_', '')}"] = (p >= thr).astype(int)
        auc_ci = bootstrap_ci(y, p, roc_auc, n_boot=N_BOOT, seed=BOOT_SEED, stratified=True)
        ap_ci = bootstrap_ci(y, p, average_precision, n_boot=N_BOOT, seed=BOOT_SEED,
                             stratified=True)
        auroc, lo, hi = auc_ci["point"], auc_ci["ci_low"], auc_ci["ci_high"]
        ap_, ap_lo, ap_hi = ap_ci["point"], ap_ci["ci_low"], ap_ci["ci_high"]
        cal = calibration_table(y, p)
        ece = float((cal.n / len(y) * (cal.mean_predicted - cal.observed_rate).abs()).sum())
        rates = binary_rates(y, p, thr)
        rows.append({"model": name, "role": frozen["models"][name].get("role", ""),
                     "n": len(y), "events": int(y.sum()),
                     "auroc": round(float(auroc), 6), "auroc_ci_low": round(float(lo), 6),
                     "auroc_ci_high": round(float(hi), 6),
                     "auprc": round(float(ap_), 6), "auprc_ci_low": round(float(ap_lo), 6),
                     "auprc_ci_high": round(float(ap_hi), 6),
                     "brier": round(float(brier_score(y, p)), 6), "ece_5bin": round(ece, 6),
                     # the rates dict also carries a rounded "threshold"; the frozen,
                     # full-precision value must win, so it is written last
                     **{k: round(v, 4) for k, v in rates.items()}, "threshold": thr})
        rc = roc_curve(y, p)
        curves[name] = rc
        cal.to_csv(out / f"{split}_calibration_{name}.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame({"fpr": rc["fpr"], "tpr": rc["tpr"],
                      "threshold": rc["thresholds"]}).to_csv(
            out / f"{split}_roc_points_{name}.csv", index=False, encoding="utf-8-sig")
        prec, rec = [], []
        for t in np.unique(p)[::-1]:
            pred = p >= t
            tp = float((pred & (y == 1)).sum())
            prec.append(tp / max(pred.sum(), 1))
            rec.append(tp / max((y == 1).sum(), 1))
        pd.DataFrame({"recall": rec, "precision": prec}).to_csv(
            out / f"{split}_pr_points_{name}.csv", index=False, encoding="utf-8-sig")
        print(f"  {name:11s} AUROC {auroc:.4f} ({lo:.4f}-{hi:.4f})  AUPRC {ap_:.4f}  "
              f"Brier {brier_score(y, p):.4f}  ECE {ece:.4f}")

    pf = pd.DataFrame(preds)
    summary = sch.validate(pf, sch.TASKB_REQUIRED, expected_ids=order, expected_labels=y)
    print(f"  prediction schema OK: {summary}")
    fname = "test_predictions_taskB.csv" if split == "test" else "val_predictions_dryrun.csv"
    if args.recompute_metrics:
        print(f"  {fname} left untouched (metrics-only recompute)")
    else:
        pf.to_csv(out / fname, index=False, encoding="utf-8-sig")
    pd.DataFrame(rows).to_csv(out / f"{split}_metrics_table.csv", index=False, encoding="utf-8-sig")

    meta = {"finished": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
            "script": Path(__file__).name, "split_evaluated": split, "dry_run": args.dry_run,
            "frozen_sha256": sha256(frozen_path), "seed": SEED,
            "n_bootstrap": N_BOOT, "bootstrap_seed": BOOT_SEED, "stratified_bootstrap": True,
            "python": platform.python_version(), "platform": platform.platform(),
            "packages": {m: __import__(m).__version__
                         for m in ("numpy", "pandas", "sklearn", "torch", "xgboost")},
            "prediction_file": fname, "patient_order_sha256":
                hashlib.sha256("\n".join(order).encode()).hexdigest(),
            "schema_check": summary}
    (out / f"{split}_run_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                                encoding="utf-8")
    if not args.dry_run and not args.recompute_metrics:
        with open(project / "results/taskB/qc/test_access_log.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"event": "test evaluation", **meta}, ensure_ascii=False) + "\n")
    print(f"\nwrote {len(list(out.glob('*')))} files to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
