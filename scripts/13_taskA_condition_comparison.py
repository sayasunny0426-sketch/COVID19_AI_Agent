"""Regenerate the Stage 1 + Stage 2 condition comparison from the run artifacts, and apply
the pre-registered selection rule (D-054).

This is the reproducible path: given the 30 run directories it recomputes every number in
docs/task_a_condition_comparison.csv, so the recorded decision can be checked later without
trusting a pasted summary.

Selection rule (fixed before Stage 2 ran, D-054):
  improvement of mean Validation AUROC over the Stage 1 baseline < 0.005
      -> practical tolerance / parsimony margin: keep the simpler Stage 1 condition
  improvement >= 0.005
      -> take the condition with the highest mean Validation AUROC, and inspect seed SD,
         Val AUPRC, Val loss, learning curves and the best-epoch distribution for close calls

Test is never read.

Usage:
    python scripts/13_taskA_condition_comparison.py --runs-root "<.../AIagent_taskA_runs>" \
        --out docs/task_a_condition_comparison.csv
"""
import argparse
import json
from pathlib import Path

import pandas as pd

BASELINE = "lr3e-4_aug_b"
MARGIN = 0.005
STAGE2_CONDITIONS = ["lr3e-4_aug_b_warmup1", "lr3e-4_aug_b_staged"]


def collect(runs_root: Path) -> pd.DataFrame:
    rows = []
    for done in sorted(runs_root.glob("*/seed*/DONE.json")):
        d = json.loads(done.read_text(encoding="utf-8"))
        if d.get("smoke"):
            continue
        cfg_path = done.parent / "config.json"
        cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
        b = d.get("best", {})
        rows.append({"stage": cfg.get("stage", "stage2" if d["condition"] in STAGE2_CONDITIONS else "stage1"),
                     "condition": d["condition"], "seed": d["seed"],
                     "best_epoch": b.get("epoch"), "val_auroc": b.get("val_auroc"),
                     "val_auprc": b.get("val_auprc"), "val_loss": b.get("val_loss"),
                     "epochs_run": d.get("epochs_run"), "early_stopped": d.get("early_stopped"),
                     "unfreeze_schedule": json.dumps(cfg.get("unfreeze_schedule"), ensure_ascii=False),
                     "lr": cfg.get("lr"), "augment": cfg.get("augment")})
    return pd.DataFrame(rows)


def summarise(runs: pd.DataFrame) -> pd.DataFrame:
    g = runs.groupby(["stage", "condition"])
    out = g.agg(n_seeds=("seed", "count"),
                mean_val_auroc=("val_auroc", "mean"), sd_val_auroc=("val_auroc", "std"),
                mean_val_auprc=("val_auprc", "mean"), mean_val_loss=("val_loss", "mean"),
                best_epochs=("best_epoch", lambda s: ",".join(str(int(x)) for x in sorted(s)))
                ).reset_index()
    return out.sort_values("mean_val_auroc", ascending=False).reset_index(drop=True)


def apply_rule(summary: pd.DataFrame) -> dict:
    base = summary[summary.condition == BASELINE]
    if base.empty:
        return {"error": f"baseline {BASELINE} not found"}
    base_auroc = float(base.mean_val_auroc.iloc[0])
    cand = summary[summary.condition != BASELINE].copy()
    cand["improvement"] = cand.mean_val_auroc - base_auroc
    best = cand.sort_values("mean_val_auroc", ascending=False).head(1)
    improvement = float(best.improvement.iloc[0]) if len(best) else float("-inf")
    if improvement < MARGIN:
        decision = {"selected_condition": BASELINE, "reason": (
            f"best alternative improves mean Validation AUROC by {improvement:+.6f}, which is below "
            f"the pre-registered practical tolerance / parsimony margin of {MARGIN}; the simpler "
            f"Stage 1 condition is kept")}
    else:
        decision = {"selected_condition": str(best.condition.iloc[0]), "reason": (
            f"improves mean Validation AUROC by {improvement:+.6f} (>= {MARGIN}); inspect seed SD, "
            f"Val AUPRC, Val loss, learning curves and best-epoch distribution for close calls")}
    decision.update({"baseline": BASELINE, "baseline_mean_val_auroc": base_auroc,
                     "margin": MARGIN, "best_alternative": str(best.condition.iloc[0]) if len(best) else None,
                     "best_alternative_mean_val_auroc": float(best.mean_val_auroc.iloc[0]) if len(best) else None,
                     "improvement": improvement})
    return decision


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", required=True)
    ap.add_argument("--out", default="docs/task_a_condition_comparison.csv")
    ap.add_argument("--runs-out", default="docs/task_a_all_runs.csv")
    args = ap.parse_args()
    runs_root = Path(args.runs_root)
    runs = collect(runs_root)
    if runs.empty:
        print(f"no runs under {runs_root}")
        return 1
    summary = summarise(runs)
    decision = apply_rule(summary)
    pd.set_option("display.width", 200)
    print(summary.to_string(index=False))
    print("\nselection:", json.dumps(decision, ensure_ascii=False, indent=2))
    print(f"\nruns: {len(runs)} (stage1 {int((runs.stage == 'stage1').sum())}, "
          f"stage2 {int((runs.stage == 'stage2').sum())})")
    runs.sort_values(["stage", "condition", "seed"]).to_csv(args.runs_out, index=False, encoding="utf-8-sig")
    summary.to_csv(args.out, index=False, encoding="utf-8-sig")
    Path(args.out).with_suffix(".decision.json").write_text(
        json.dumps({"decision": decision, "n_runs": len(runs)}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"\nwrote {args.out}, {args.runs_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
