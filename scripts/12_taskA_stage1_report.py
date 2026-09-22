"""Report the state of Stage 1 runs (per-run table + artifact check). No model selection.

Model selection is only done after all 24 Stage-1 runs finish, on the mean Validation AUROC
over the 3 seeds (plan §8). This script therefore prints per-run numbers and the progress of
the grid, and explicitly refuses to rank conditions while runs are missing.

Test is never read.

Usage:
    python scripts/12_taskA_stage1_report.py --runs-root "<.../AIagent_taskA_runs>"
"""
import argparse
import json
from pathlib import Path

import pandas as pd

EXPECTED_RUNS = 24
ARTIFACTS = ["config.json", "history.csv", "val_predictions.csv",
             "checkpoints/best.pt", "checkpoints/last.pt", "DONE.json"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", required=True)
    ap.add_argument("--out", default=None, help="optional CSV path for the table")
    args = ap.parse_args()
    root = Path(args.runs_root)
    if not root.exists():
        print(f"runs root not found: {root}")
        return 1

    rows = []
    for done_path in sorted(root.glob("*/seed*/DONE.json")):
        run_dir = done_path.parent
        d = json.loads(done_path.read_text(encoding="utf-8"))
        best = d.get("best", {})
        hist = pd.read_csv(run_dir / "history.csv", encoding="utf-8-sig") if (run_dir / "history.csv").exists() else pd.DataFrame()
        missing = [a for a in ARTIFACTS if not (run_dir / a).exists()]
        preds = (pd.read_csv(run_dir / "val_predictions.csv", encoding="utf-8-sig")
                 if (run_dir / "val_predictions.csv").exists() else pd.DataFrame())
        rows.append({
            "condition": d.get("condition"), "seed": d.get("seed"),
            "epochs_run": d.get("epochs_run"), "best_epoch": best.get("epoch"),
            "best_val_auroc": best.get("val_auroc"), "best_val_auprc": best.get("val_auprc"),
            "best_val_loss": best.get("val_loss"), "best_val_accuracy": best.get("val_accuracy"),
            "early_stopped": d.get("early_stopped"),
            "mean_sec_per_epoch": d.get("mean_seconds_per_epoch"),
            "total_sec": d.get("seconds_total_all_epochs"),
            "peak_gpu_mem_mb": d.get("peak_gpu_mem_mb"), "device": d.get("device"),
            "gpu": d.get("gpu"), "amp": d.get("amp_enabled"), "smoke": d.get("smoke"),
            "val_pred_rows": len(preds), "history_rows": len(hist),
            "missing_artifacts": ";".join(missing) if missing else "",
        })

    if not rows:
        print(f"no completed runs under {root}")
        return 0
    df = pd.DataFrame(rows).sort_values(["condition", "seed"]).reset_index(drop=True)
    pd.set_option("display.width", 220)
    cols = ["condition", "seed", "epochs_run", "best_epoch", "best_val_auroc", "best_val_auprc",
            "best_val_loss", "early_stopped", "mean_sec_per_epoch", "total_sec", "peak_gpu_mem_mb",
            "val_pred_rows", "missing_artifacts"]
    print(df[cols].to_string(index=False))

    smoke_runs = df[df.smoke == True]  # noqa: E712
    if len(smoke_runs):
        print(f"\nWARNING: {len(smoke_runs)} run(s) are smoke runs and must not be used for selection")
    incomplete = df[(df.val_pred_rows != 128) | (df.missing_artifacts != "")]
    if len(incomplete):
        print("\nWARNING: runs with unexpected artifacts:")
        print(incomplete[["condition", "seed", "val_pred_rows", "missing_artifacts"]].to_string(index=False))

    print(f"\ncompleted runs: {len(df)} / {EXPECTED_RUNS}")
    print(f"total compute: {df.total_sec.sum() / 3600:.2f} h  "
          f"(mean {df.total_sec.mean() / 60:.1f} min/run, "
          f"{df.mean_sec_per_epoch.mean():.1f} s/epoch)")
    if len(df) < EXPECTED_RUNS:
        remaining = EXPECTED_RUNS - len(df)
        print(f"remaining runs: {remaining}  -> estimated {remaining * df.total_sec.mean() / 3600:.2f} h")
        print("\nSELECTION NOT PERFORMED: all 24 Stage-1 runs must finish first; conditions are then "
              "compared on the mean Validation AUROC over the 3 seeds (plan §8).")
    else:
        per_cond = df.groupby("condition").best_val_auroc.agg(["mean", "std", "count"])
        print("\nall Stage-1 runs complete. Mean Validation AUROC per condition "
              "(for the record; the selection step is a separate, logged decision):")
        print(per_cond.sort_values("mean", ascending=False).to_string())

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False, encoding="utf-8-sig")
        print(f"\ntable -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
