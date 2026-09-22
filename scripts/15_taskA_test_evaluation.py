"""Task A final Test evaluation. Runs ONCE, only against a frozen final_selection.json.

Guards (all must pass, otherwise nothing is written):
  * final_selection.json exists and its recorded hashes still match (checkpoints, configs,
    Validation predictions, split manifest, index-CXR manifest);
  * the Test split has 128 patients / 17 deaths and does not overlap Train or Validation;
  * thresholds come from the frozen file and are never recomputed on Test;
  * existing Test predictions are not overwritten unless --allow-overwrite is given, and the
    attempt is logged either way.

--dry-run evaluates the VALIDATION split instead of Test with the identical code path, so the
whole pipeline can be verified before the single real Test run.

Outputs (docs/task_a_test_evaluation_plan.md §2): patient-level predictions (primary, per
seed, ensemble, combined), metrics json/csv, ROC/PR/calibration/bootstrap data, run metadata
and an append-only access log.

Usage:
    python scripts/15_taskA_test_evaluation.py --runs-root <...> --out-dir <...> --dry-run
    python scripts/15_taskA_test_evaluation.py --runs-root <...> --out-dir <...>
"""
import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from covid_mortality.data.taskA_dataset import TaskACXRDataset, build_loader, load_index  # noqa: E402
from covid_mortality.evaluation.metrics import (  # noqa: E402
    accuracy, average_precision, binary_rates, bootstrap_ci, brier_score, roc_auc, roc_curve)
from covid_mortality.models.taskA_resnet18 import ModelConfig, build_model  # noqa: E402

N_BOOT = 2000
CALIBRATION_BINS = 5


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_freeze(frozen: dict, project: Path) -> list[str]:
    """Re-check every hash recorded at freeze time."""
    problems = []
    for seed, info in frozen["runs"].items():
        for key, path_key in (("checkpoint_sha256", "checkpoint"),
                              ("config_sha256", None), ("val_predictions_sha256", None)):
            if path_key:
                p = Path(info[path_key])
            else:
                run_dir = Path(info["run_dir"])
                p = run_dir / ("config.json" if key == "config_sha256" else "val_predictions.csv")
            if not p.exists():
                problems.append(f"missing: {p}")
            elif sha256(p) != info[key]:
                problems.append(f"hash changed: {p}")
    for name, rel in (("split_manifest_sha256", "data/splits/COVID19_固定患者split_1277.csv"),
                      ("index_cxr_manifest_sha256",
                       "data/interim/cxr_audit/index_cxr_manifest_window_T0m2_T0.csv")):
        p = project / rel
        if not p.exists():
            problems.append(f"missing: {p}")
        elif sha256(p) != frozen["inputs"][name]:
            problems.append(f"hash changed: {p}")
    return problems


@torch.no_grad()
def predict(checkpoint: Path, dataset: TaskACXRDataset, device: torch.device,
            batch_size: int = 32) -> pd.DataFrame:
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model = build_model(ModelConfig(pretrained=False)).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    probs, labels, pids = [], [], []
    for batch in build_loader(dataset, batch_size=batch_size, seed=0):
        logits = model(batch["image"].to(device)).squeeze(1)
        probs.append(torch.sigmoid(logits).float().cpu().numpy())
        labels.append(batch["label"].numpy())
        pids.extend(batch["patient_id"])
    return pd.DataFrame({"subject_id": pids, "true_label": np.concatenate(labels).astype(int),
                         "prob": np.concatenate(probs)}).sort_values("subject_id").reset_index(drop=True)


def calibration_table(y: np.ndarray, p: np.ndarray, n_bins: int = CALIBRATION_BINS) -> pd.DataFrame:
    edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    idx = np.digitize(p, edges[1:-1], right=True)
    rows = []
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        rows.append({"bin": b, "n": int(m.sum()), "mean_predicted": float(p[m].mean()),
                     "observed_rate": float(y[m].mean()), "events": int(y[m].sum())})
    return pd.DataFrame(rows)


def expected_calibration_error(cal: pd.DataFrame, n: int) -> float:
    return float((cal.n / n * (cal.mean_predicted - cal.observed_rate).abs()).sum())


def evaluate(y: np.ndarray, p: np.ndarray, thresholds: dict, label: str) -> dict:
    out = {"model": label, "n": int(len(y)), "events": int(y.sum()),
           "prevalence": float(y.mean()),
           "auroc": bootstrap_ci(y, p, roc_auc, N_BOOT, stratified=True),
           "auprc": bootstrap_ci(y, p, average_precision, N_BOOT, stratified=True),
           "brier": brier_score(y, p), "accuracy_at_0.5": accuracy(y, p, 0.5)}
    for name, thr in thresholds.items():
        out[f"operating_point_{name}"] = (binary_rates(y, p, thr) if thr is not None
                                          and np.isfinite(thr) else None)
    cal = calibration_table(y, p)
    out["calibration_bins"] = cal.to_dict("records")
    out["ece"] = expected_calibration_error(cal, len(y))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=".")
    ap.add_argument("--runs-root", required=True)
    ap.add_argument("--frozen", default=None, help="path to final_selection.json")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--image-dir", default="data/processed/taskA_png512_16bit/images")
    ap.add_argument("--dry-run", action="store_true",
                    help="evaluate the Validation split instead of Test (same code path)")
    ap.add_argument("--allow-overwrite", action="store_true")
    args = ap.parse_args()

    project = Path(args.project)
    runs_root = Path(args.runs_root)
    frozen_path = Path(args.frozen) if args.frozen else runs_root / "final_selection.json"
    out_dir = Path(args.out_dir)
    split = "val" if args.dry_run else "test"
    started = time.time()

    if not frozen_path.exists():
        print(f"STOP: {frozen_path} not found. Freeze the selection first "
              f"(scripts/14_taskA_freeze_selection.py).")
        return 1
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    problems = verify_freeze(frozen, project)
    if problems:
        print("STOP: frozen state no longer matches:")
        for p in problems:
            print("  -", p)
        return 1

    index = load_index(project)          # asserts 1,277 / 1,021 / 128 / 128 and 135 / 17 / 17
    ds = TaskACXRDataset(index, project / args.image_dir, split)
    ids = {s: set(index[index.split == s].PatientID) for s in ("train", "val", "test")}
    assert not ids["test"] & ids["train"] and not ids["test"] & ids["val"], "split overlap"
    assert len(ds) == 128 and int(ds.labels().sum()) == 17, "unexpected evaluation split"

    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "test_access_log.jsonl"
    primary_file = out_dir / f"{split}_predictions_primary.csv"
    entry = {"timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "split_evaluated": split, "dry_run": bool(args.dry_run),
             "frozen_sha256": sha256(frozen_path), "script": Path(__file__).name,
             "script_sha256": sha256(Path(__file__)), "out_dir": str(out_dir),
             "existing_primary_predictions": primary_file.exists()}
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    if primary_file.exists() and not args.allow_overwrite:
        print(f"STOP: {primary_file} already exists. The Test set is evaluated once; "
              f"re-running needs --allow-overwrite and must be recorded in change_log.csv.")
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    thresholds = {
        "youden": float(frozen["primary_analysis"]["threshold"]["value"]),
        "sensitivity80": (float(frozen["primary_analysis"]["secondary_threshold"]["value"])
                          if frozen["primary_analysis"]["secondary_threshold"]["value"] is not None
                          else None)}
    print(f"evaluating split={split} on {device.type} with frozen thresholds {thresholds}")

    per_seed = {}
    for seed, info in frozen["runs"].items():
        per_seed[int(seed)] = predict(Path(info["checkpoint"]), ds, device)
        per_seed[int(seed)].to_csv(out_dir / f"{split}_predictions_seed{seed}.csv", index=False,
                                   encoding="utf-8-sig")
    primary_seed = int(frozen["primary_analysis"]["model"].split("seed")[-1])
    y = per_seed[primary_seed].true_label.to_numpy()
    p_primary = per_seed[primary_seed].prob.to_numpy()
    p_ens = np.mean([per_seed[s].prob.to_numpy() for s in sorted(per_seed)], axis=0)

    delta = index.set_index("PatientID").delta_days_from_T0.astype(int)
    primary = per_seed[primary_seed].copy()
    primary["model"] = frozen["primary_analysis"]["model"]
    primary["threshold_used"] = thresholds["youden"]
    primary["predicted_label"] = (primary.prob >= thresholds["youden"]).astype(int)
    primary["delta_days_from_T0"] = primary.subject_id.map(delta)
    primary.to_csv(primary_file, index=False, encoding="utf-8-sig")

    ens = per_seed[primary_seed][["subject_id", "true_label"]].copy()
    ens["prob"] = p_ens
    ens.to_csv(out_dir / f"{split}_predictions_ensemble.csv", index=False, encoding="utf-8-sig")
    combined = per_seed[primary_seed][["subject_id", "true_label"]].copy()
    for s in sorted(per_seed):
        combined[f"prob_seed{s}"] = per_seed[s].prob.to_numpy()
    combined["prob_ensemble"] = p_ens
    combined["prob_primary"] = p_primary
    combined.to_csv(out_dir / f"{split}_predictions_all_variants.csv", index=False,
                    encoding="utf-8-sig")

    metrics_primary = evaluate(y, p_primary, thresholds, frozen["primary_analysis"]["model"])
    secondary = {f"seed{s}": evaluate(y, per_seed[s].prob.to_numpy(), thresholds, f"seed{s}")
                 for s in sorted(per_seed)}
    secondary["three_seed_ensemble"] = evaluate(y, p_ens, thresholds, "ensemble(3 seeds)")
    # L-1 sensitivity analysis: exclude patients whose index CXR is on the T0 calendar date
    mask = primary.delta_days_from_T0.to_numpy() < 0
    if mask.sum() >= 10 and y[mask].sum() >= 2:
        secondary["L1_excluding_T0_day"] = evaluate(y[mask], p_primary[mask], thresholds,
                                                    "primary, index CXR before T0 day")
        secondary["L1_excluding_T0_day"]["n_excluded"] = int((~mask).sum())
    else:
        secondary["L1_excluding_T0_day"] = {"skipped": "too few patients or events after exclusion",
                                            "n_remaining": int(mask.sum()),
                                            "events_remaining": int(y[mask].sum())}

    (out_dir / f"{split}_metrics_primary.json").write_text(
        json.dumps(metrics_primary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / f"{split}_metrics_secondary.json").write_text(
        json.dumps(secondary, ensure_ascii=False, indent=2), encoding="utf-8")

    table = []
    for m in [metrics_primary] + [v for v in secondary.values() if "auroc" in v]:
        table.append({"model": m["model"], "n": m["n"], "events": m["events"],
                      "auroc": m["auroc"]["point"], "auroc_ci_low": m["auroc"]["ci_low"],
                      "auroc_ci_high": m["auroc"]["ci_high"], "auprc": m["auprc"]["point"],
                      "auprc_ci_low": m["auprc"]["ci_low"], "auprc_ci_high": m["auprc"]["ci_high"],
                      "brier": m["brier"], "ece": m["ece"],
                      **{f"youden_{k}": v for k, v in (m.get("operating_point_youden") or {}).items()
                         if k in ("sensitivity", "specificity", "ppv", "npv", "accuracy")}})
    pd.DataFrame(table).to_csv(out_dir / f"{split}_metrics_table.csv", index=False,
                               encoding="utf-8-sig")

    curves = []
    for name, probs in (("primary", p_primary), ("ensemble", p_ens)):
        c = roc_curve(y, probs)
        curves.append(pd.DataFrame({"model": name, "fpr": c["fpr"], "tpr": c["tpr"],
                                    "threshold": c["thresholds"]}))
    pd.concat(curves).to_csv(out_dir / f"{split}_roc_curve_points.csv", index=False,
                             encoding="utf-8-sig")
    order = np.argsort(-p_primary)
    tp = np.cumsum(y[order]); fp = np.cumsum(1 - y[order])
    pd.DataFrame({"threshold": p_primary[order], "precision": tp / np.maximum(tp + fp, 1e-12),
                  "recall": tp / max(y.sum(), 1)}).to_csv(
        out_dir / f"{split}_pr_curve_points.csv", index=False, encoding="utf-8-sig")
    calibration_table(y, p_primary).to_csv(out_dir / f"{split}_calibration_bins.csv", index=False,
                                           encoding="utf-8-sig")

    meta = {"finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "split_evaluated": split, "dry_run": bool(args.dry_run),
            "frozen_selection": str(frozen_path), "frozen_sha256": sha256(frozen_path),
            "thresholds": thresholds, "device": device.type,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "python": sys.version.split()[0], "torch": torch.__version__,
            "platform": platform.platform(), "seconds": round(time.time() - started, 1),
            "n_bootstrap": N_BOOT, "outputs": sorted(p.name for p in out_dir.glob(f"{split}_*"))}
    (out_dir / f"{split}_run_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                                    encoding="utf-8")
    print(json.dumps({"model": metrics_primary["model"], "split": split,
                      "auroc": metrics_primary["auroc"], "auprc": metrics_primary["auprc"],
                      "brier": metrics_primary["brier"], "ece": metrics_primary["ece"],
                      "youden_point": metrics_primary.get("operating_point_youden")},
                     ensure_ascii=False, indent=2, default=str))
    print(f"\noutputs -> {out_dir}")
    if args.dry_run:
        print("DRY RUN: Validation was evaluated; the Test split was never read.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
