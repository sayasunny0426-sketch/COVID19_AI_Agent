"""Pre-flight check of the freeze -> Test-evaluation pipeline, without touching the Test split.

Builds a throw-away "runs" tree from untrained ResNet18 checkpoints (3 seeds) using the real
image cache and the real fixed split, freezes it, then runs the evaluation script in --dry-run
mode (Validation). This exercises every guard and every output path before the single real
Test run, and verifies that the guards actually stop the pipeline when they should.

Usage:
    python scripts/16_taskA_eval_pipeline_check.py --project .
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from covid_mortality.data.taskA_dataset import TaskACXRDataset, build_loader, load_index  # noqa: E402
from covid_mortality.models.taskA_resnet18 import ModelConfig, build_model  # noqa: E402

CONDITION = "lr3e-4_aug_b"
SEEDS = [42, 43, 44]
fails: list[str] = []


def check(name: str, ok: bool, detail="") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    if not ok:
        fails.append(name)


def make_fake_runs(project: Path, root: Path) -> None:
    """Untrained checkpoints + real Validation predictions, so the freeze step has real inputs."""
    index = load_index(project)
    val = TaskACXRDataset(index, project / "data/processed/taskA_png512_16bit/images", "val")
    device = torch.device("cpu")
    for seed in SEEDS:
        torch.manual_seed(seed)
        model = build_model(ModelConfig(pretrained=False)).to(device).eval()
        probs, labels, pids = [], [], []
        with torch.no_grad():
            for batch in build_loader(val, batch_size=32, seed=0):
                logits = model(batch["image"]).squeeze(1)
                probs.append(torch.sigmoid(logits).numpy())
                labels.append(batch["label"].numpy())
                pids.extend(batch["patient_id"])
        d = root / CONDITION / f"seed{seed}"
        (d / "checkpoints").mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "epoch": 1, "best": {}}, d / "checkpoints/best.pt")
        pd.DataFrame({"subject_id": pids, "true_label": np.concatenate(labels).astype(int),
                      "prob": np.concatenate(probs)}).sort_values("subject_id").to_csv(
            d / "val_predictions.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame([{"epoch": 1, "train_loss": 1.0, "val_loss": 1.0, "val_auroc": 0.5,
                       "val_auprc": 0.13, "val_accuracy": 0.87, "lr": 3e-4, "seconds": 1.0,
                       "backbone_trainable": True, "peak_gpu_mem_mb": None}]).to_csv(
            d / "history.csv", index=False, encoding="utf-8-sig")
        (d / "config.json").write_text(json.dumps(
            {"lr": 3e-4, "augment": "aug_b", "seed": seed, "unfreeze_schedule": None,
             "pos_weight": 6.563, "stage": "stage1"}, ensure_ascii=False), encoding="utf-8")
        (d / "DONE.json").write_text(json.dumps(
            {"status": "completed", "condition": CONDITION, "seed": seed, "smoke": False,
             "best": {"epoch": 1, "val_auroc": 0.5, "val_loss": 1.0, "val_auprc": 0.13}},
            ensure_ascii=False), encoding="utf-8")


def run(cmd: list[str], project: Path) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, *cmd], cwd=project, capture_output=True, text=True,
                          encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=".")
    args = ap.parse_args()
    project = Path(args.project).resolve()
    sandbox = project / "outputs/eval_pipeline_check"
    shutil.rmtree(sandbox, ignore_errors=True)
    runs_root = sandbox / "runs"
    out_dir = sandbox / "eval"
    runs_root.mkdir(parents=True, exist_ok=True)

    print("building throw-away runs from the real image cache (untrained weights)...")
    make_fake_runs(project, runs_root)

    # 1) evaluation must refuse to start without a frozen selection
    r = run(["scripts/15_taskA_test_evaluation.py", "--runs-root", str(runs_root),
             "--out-dir", str(out_dir), "--dry-run"], project)
    check("refuses_without_frozen_file", r.returncode == 1 and "Freeze the selection first" in r.stdout,
          r.stdout.strip().splitlines()[-1][:70] if r.stdout else r.stderr[:70])

    # 2) freeze
    r = run(["scripts/14_taskA_freeze_selection.py", "--project", ".", "--runs-root", str(runs_root)],
            project)
    frozen_path = runs_root / "final_selection.json"
    check("freeze_creates_file", r.returncode == 0 and frozen_path.exists(),
          r.stdout.strip().splitlines()[-1][:70] if r.stdout else r.stderr[-200:])
    frozen = json.loads(frozen_path.read_text(encoding="utf-8")) if frozen_path.exists() else {}
    pa = frozen.get("primary_analysis", {})
    check("primary_is_seed42", pa.get("model", "").endswith("seed42"), pa.get("model"))
    check("threshold_recorded", "value" in pa.get("threshold", {}),
          f"youden={pa.get('threshold', {}).get('value')}, tie_break="
          f"{pa.get('threshold', {}).get('tie_break', '')[:20]}")
    sec = pa.get("secondary_threshold", {})
    check("secondary_threshold_is_highest_rule", "highest threshold" in sec.get("rule", ""),
          f"value={sec.get('value')}")
    check("secondary_threshold_marked_exploratory", "exploratory" in sec.get("status", ""),
          sec.get("status", "")[:50])
    check("ensemble_marked_secondary",
          "secondary" in frozen.get("secondary_analyses", {})
          .get("three_seed_ensemble", {}).get("status", ""), "three_seed_ensemble")
    check("test_recorded_as_unused", frozen.get("test_set", {}).get("used_so_far") is False, "")

    # 3) dry run on Validation
    r = run(["scripts/15_taskA_test_evaluation.py", "--project", ".", "--runs-root", str(runs_root),
             "--out-dir", str(out_dir), "--dry-run"], project)
    check("dry_run_completes", r.returncode == 0, (r.stdout or r.stderr).strip().splitlines()[-1][:70])
    produced = sorted(p.name for p in out_dir.glob("*")) if out_dir.exists() else []
    expected = ["val_predictions_primary.csv", "val_predictions_ensemble.csv",
                "val_predictions_all_variants.csv", "val_metrics_primary.json",
                "val_metrics_secondary.json", "val_metrics_table.csv",
                "val_roc_curve_points.csv", "val_pr_curve_points.csv",
                "val_calibration_bins.csv", "val_run_meta.json", "test_access_log.jsonl"]
    check("all_outputs_written", all(e in produced for e in expected),
          f"{len([e for e in expected if e in produced])}/{len(expected)}")
    if (out_dir / "val_predictions_primary.csv").exists():
        pred = pd.read_csv(out_dir / "val_predictions_primary.csv", encoding="utf-8-sig")
        check("predictions_are_patient_level", len(pred) == 128 and pred.subject_id.is_unique,
              f"{len(pred)} rows")
        check("threshold_from_frozen_file",
              abs(pred.threshold_used.iloc[0] - pa["threshold"]["value"]) < 1e-12, "matches")
        check("delta_days_attached", pred.delta_days_from_T0.notna().all(),
              f"range {pred.delta_days_from_T0.min()}..{pred.delta_days_from_T0.max()}")
        m = json.loads((out_dir / "val_metrics_primary.json").read_text(encoding="utf-8"))
        check("bootstrap_ci_present",
              m["auroc"]["ci_low"] <= m["auroc"]["point"] <= m["auroc"]["ci_high"],
              f"AUROC {m['auroc']['point']:.3f} [{m['auroc']['ci_low']:.3f}, {m['auroc']['ci_high']:.3f}]")
        check("stratified_bootstrap_used", m["auroc"]["stratified"] is True, "")
        sec_m = json.loads((out_dir / "val_metrics_secondary.json").read_text(encoding="utf-8"))
        check("per_seed_and_ensemble_reported",
              all(k in sec_m for k in ("seed42", "seed43", "seed44", "three_seed_ensemble")),
              list(sec_m)[:5])
        check("L1_sensitivity_analysis_present", "L1_excluding_T0_day" in sec_m,
              str(sec_m["L1_excluding_T0_day"])[:60])

    # 4) re-running must be blocked, and the attempt logged
    n_before = len((out_dir / "test_access_log.jsonl").read_text(encoding="utf-8").splitlines())
    r = run(["scripts/15_taskA_test_evaluation.py", "--project", ".", "--runs-root", str(runs_root),
             "--out-dir", str(out_dir), "--dry-run"], project)
    n_after = len((out_dir / "test_access_log.jsonl").read_text(encoding="utf-8").splitlines())
    check("second_run_blocked", r.returncode == 1 and "evaluated once" in r.stdout,
          r.stdout.strip().splitlines()[-1][:60] if r.stdout else "")
    check("attempt_logged", n_after == n_before + 1, f"{n_before} -> {n_after} log lines")

    # 5) tampering with a checkpoint must stop the pipeline
    ckpt = Path(frozen["runs"]["43"]["checkpoint"])
    ckpt.write_bytes(ckpt.read_bytes() + b"0")
    r = run(["scripts/15_taskA_test_evaluation.py", "--project", ".", "--runs-root", str(runs_root),
             "--out-dir", str(sandbox / "eval2"), "--dry-run"], project)
    check("hash_mismatch_stops_run", r.returncode == 1 and "no longer matches" in r.stdout,
          r.stdout.strip().splitlines()[0][:60] if r.stdout else "")

    print(f"\n{len(fails)} failure(s). sandbox: {sandbox}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
