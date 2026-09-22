"""Freeze the Task A model selection into final_selection.json (run BEFORE any Test evaluation).

Records, with hashes, exactly what will be evaluated on the Test set:
  primary   : lr3e-4_aug_b / seed 42          (pre-specified single seed, D-056)
  secondary : seed 43, seed 44                (reproducibility check)
  secondary : mean of the three seeds         (ensemble, reported but not the primary analysis)

The classification threshold is derived from the Validation predictions of the PRIMARY model
only (Youden index; a fixed-sensitivity threshold is also recorded as a secondary operating
point). Nothing here reads the Test split.

The Test evaluation script must refuse to run unless this file exists and every hash still
matches, so the frozen state cannot drift.

Usage (in Colab, with the run directory mounted):
    python scripts/14_taskA_freeze_selection.py \
        --runs-root "<.../AIagent_taskA_runs>" \
        --out "<.../AIagent_taskA_runs>/final_selection.json"
"""
import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from covid_mortality.evaluation.metrics import (  # noqa: E402
    accuracy, average_precision, binary_rates, brier_score, roc_auc,
    threshold_at_min_sensitivity, youden_threshold)

CONDITION = "lr3e-4_aug_b"
PRIMARY_SEED = 42
SECONDARY_SEEDS = [43, 44]
# Threshold rules fixed before the Test evaluation (D-057):
YOUDEN_TIE_BREAK = "lowest"   # among thresholds with maximal J, take the smallest -> keeps
                              # sensitivity highest, i.e. fewest missed deaths
TARGET_SENSITIVITY = 0.80     # exploratory secondary operating point only: no clinical
                              # justification has been set for this value yet


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_val(run_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(run_dir / "val_predictions.csv", encoding="utf-8-sig")
    if len(df) != 128:
        raise ValueError(f"{run_dir}: expected 128 Validation rows, got {len(df)}")
    return df.sort_values("subject_id").reset_index(drop=True)


def val_metrics(y: np.ndarray, p: np.ndarray, threshold: float | None = None) -> dict:
    out = {"auroc": roc_auc(y, p), "auprc": average_precision(y, p), "brier": brier_score(y, p),
           "accuracy_at_0.5": accuracy(y, p, 0.5), "n": int(len(y)), "events": int(y.sum())}
    if threshold is not None:
        out["at_threshold"] = binary_rates(y, p, threshold)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--project", default=".")
    args = ap.parse_args()
    runs_root, project = Path(args.runs_root), Path(args.project)
    out = Path(args.out) if args.out else runs_root / "final_selection.json"

    seeds = [PRIMARY_SEED] + SECONDARY_SEEDS
    runs, preds = {}, {}
    for seed in seeds:
        run_dir = runs_root / CONDITION / f"seed{seed}"
        ckpt = run_dir / "checkpoints/best.pt"
        for required in (ckpt, run_dir / "config.json", run_dir / "DONE.json",
                         run_dir / "val_predictions.csv", run_dir / "history.csv"):
            if not required.exists():
                raise FileNotFoundError(required)
        done = json.loads((run_dir / "DONE.json").read_text(encoding="utf-8"))
        cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        if done.get("smoke"):
            raise ValueError(f"{run_dir} is a smoke run")
        preds[seed] = load_val(run_dir)
        runs[seed] = {"run_dir": str(run_dir), "best_epoch": done["best"]["epoch"],
                      "val_auroc_at_best": done["best"]["val_auroc"],
                      "checkpoint": str(ckpt), "checkpoint_sha256": sha256(ckpt),
                      "config_sha256": sha256(run_dir / "config.json"),
                      "val_predictions_sha256": sha256(run_dir / "val_predictions.csv"),
                      "lr": cfg.get("lr"), "augment": cfg.get("augment"),
                      "unfreeze_schedule": cfg.get("unfreeze_schedule"),
                      "pos_weight": cfg.get("pos_weight"), "seed": seed}

    ids = [tuple(preds[s].subject_id) for s in seeds]
    if len(set(ids)) != 1:
        raise ValueError("Validation patient order differs between seeds")
    y = preds[PRIMARY_SEED].true_label.to_numpy()
    p_primary = preds[PRIMARY_SEED].prob.to_numpy()
    p_ens = np.mean([preds[s].prob.to_numpy() for s in seeds], axis=0)

    thr_youden = youden_threshold(y, p_primary, YOUDEN_TIE_BREAK)   # primary operating point
    thr_sens = threshold_at_min_sensitivity(y, p_primary, TARGET_SENSITIVITY)
    tied = [float(t) for t in np.unique(p_primary)
            if abs((binary_rates(y, p_primary, float(t))["sensitivity"]
                    + binary_rates(y, p_primary, float(t))["specificity"] - 1)
                   - (binary_rates(y, p_primary, thr_youden)["sensitivity"]
                      + binary_rates(y, p_primary, thr_youden)["specificity"] - 1)) < 1e-12]

    payload = {
        "frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "task": "A_CXR_ResNet18",
        "decision_log": ["D-044", "D-047", "D-052", "D-053", "D-054", "D-055", "D-056"],
        "condition": CONDITION,
        "primary_analysis": {
            "model": f"{CONDITION}/seed{PRIMARY_SEED}",
            "rationale": ("single pre-specified seed so that the comparison with the existing "
                          "Task A (seed 42 single model) isolates the effect of the training "
                          "conditions from any ensembling effect (D-056)"),
            "checkpoint": runs[PRIMARY_SEED]["checkpoint"],
            "checkpoint_sha256": runs[PRIMARY_SEED]["checkpoint_sha256"],
            "threshold": {"rule": "maximum Youden index J on Validation (primary model only)",
                          "tie_break": f"{YOUDEN_TIE_BREAK} threshold among ties "
                                       f"(keeps sensitivity highest at equal J)",
                          "n_tied_thresholds": len(tied), "tied_thresholds": tied[:10],
                          "value": thr_youden,
                          "validation_operating_point": binary_rates(y, p_primary, thr_youden),
                          "recomputed_on_test": False},
            "secondary_threshold": {
                "rule": f"highest threshold with Validation sensitivity >= {TARGET_SENSITIVITY} "
                        f"(keeps specificity as high as possible while holding that sensitivity)",
                "status": "exploratory secondary operating point; no clinical justification has "
                          "been set for the 0.80 sensitivity target",
                "value": thr_sens,
                "validation_operating_point": (binary_rates(y, p_primary, thr_sens)
                                               if np.isfinite(thr_sens) else None),
                "recomputed_on_test": False},
            "validation_metrics": val_metrics(y, p_primary, thr_youden)},
        "secondary_analyses": {
            "per_seed": {str(s): {"model": f"{CONDITION}/seed{s}",
                                  "checkpoint_sha256": runs[s]["checkpoint_sha256"],
                                  "validation_metrics": val_metrics(
                                      y, preds[s].prob.to_numpy())} for s in seeds},
            "three_seed_ensemble": {
                "members": [f"{CONDITION}/seed{s}" for s in seeds],
                "combination": "unweighted mean of predicted probabilities",
                "status": "secondary only; not the primary analysis (D-056)",
                "validation_metrics": val_metrics(y, p_ens)}},
        "runs": runs,
        "inputs": {
            "split_manifest_sha256": sha256(project / "data/splits/COVID19_固定患者split_1277.csv"),
            "index_cxr_manifest_sha256": sha256(
                project / "data/interim/cxr_audit/index_cxr_manifest_window_T0m2_T0.csv")},
        "preprocessing": {"cache": "AIagent_taskA_png512_16bit",
                          "pipeline": "percentile 1-99 clip -> [0,1] -> zero pad to square -> 512 -> 224 -> 3ch -> ImageNet norm",
                          "reference": "decision_log D-044"},
        "test_set": {"used_so_far": False,
                     "policy": "single evaluation after this file is frozen; every access logged"},
        "environment": {"python": sys.version.split()[0], "platform": platform.platform()},
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"frozen": str(out),
                      "primary": payload["primary_analysis"]["model"],
                      "threshold_youden": thr_youden,
                      "threshold_sensitivity80": thr_sens,
                      "val_auroc_primary": payload["primary_analysis"]["validation_metrics"]["auroc"],
                      "val_auroc_ensemble": payload["secondary_analyses"]["three_seed_ensemble"]["validation_metrics"]["auroc"]},
                     ensure_ascii=False, indent=2))
    print(f"\nSHA256 of final_selection.json: {sha256(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
