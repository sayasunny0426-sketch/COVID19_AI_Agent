"""Verify that a run survives a disconnect: interrupt after epoch 2, resume, finish at epoch 4.

Also checks that resuming with a changed configuration is refused, and that a completed run is
skipped. Uses a tiny debug configuration (smoke) on CPU/GPU alike; never touches Test.

Usage:
    python scripts/10_taskA_resume_test.py --project . --runs-root outputs/runs/taskA_resume_test
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from covid_mortality.training.taskA_trainer import TrainConfig, run_training  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=".")
    ap.add_argument("--runs-root", default="outputs/runs/taskA_resume_test")
    args = ap.parse_args()
    root = Path(args.runs_root)
    if root.exists():
        shutil.rmtree(root)
    base = dict(condition="resume_demo", seed=42, lr=1e-4, augment="aug_a", project=args.project,
                runs_root=str(root), max_epochs=4, batch_size=16, num_workers=0, smoke=True)
    failures = []

    def check(name, ok, detail=None):
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
        if not ok:
            failures.append(name)

    # 1) interrupted after epoch 2
    out1 = run_training(TrainConfig(**base, stop_after_epoch=2))
    run_dir = Path(out1["run_dir"])
    hist1 = pd.read_csv(run_dir / "history.csv", encoding="utf-8-sig")
    check("interrupted_without_DONE", out1["status"] == "interrupted"
          and not (run_dir / "DONE.json").exists(), out1["status"])
    check("history_flushed_per_epoch", list(hist1.epoch) == [1, 2], list(hist1.epoch))
    check("last_checkpoint_saved", (run_dir / "checkpoints/last.pt").exists(), "last.pt")

    # 2) resume with the same configuration
    out2 = run_training(TrainConfig(**base))
    hist2 = pd.read_csv(run_dir / "history.csv", encoding="utf-8-sig")
    check("resumed_and_completed", out2["status"] == "completed", out2["status"])
    check("history_continues", list(hist2.epoch) == [1, 2, 3, 4], list(hist2.epoch))
    check("no_epoch_recomputed", len(hist2) == 4 and hist2.epoch.is_unique, len(hist2))
    check("done_written", (run_dir / "DONE.json").exists(), "DONE.json")
    check("val_predictions_saved",
          len(pd.read_csv(run_dir / "val_predictions.csv", encoding="utf-8-sig")) > 0, "rows > 0")

    # 3) a completed run is skipped
    out3 = run_training(TrainConfig(**base))
    check("completed_run_skipped", out3["status"] == "skipped_completed", out3["status"])

    # 4) resuming with a changed configuration is refused
    (run_dir / "DONE.json").unlink()
    try:
        run_training(TrainConfig(**{**base, "lr": 3e-4}))
        check("changed_config_refused", False, "no exception raised")
    except RuntimeError as e:
        check("changed_config_refused", "configuration changed" in str(e), str(e)[:90])
    (run_dir / "DONE.json").write_text(json.dumps(out2, ensure_ascii=False), encoding="utf-8")

    print(f"\n{len(failures)} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
