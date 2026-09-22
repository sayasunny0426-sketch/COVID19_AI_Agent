"""Task A run manager: expands the pre-registered grid, skips finished runs, resumes the rest.

The grid is fixed in code (plan §5) so that the set of conditions cannot drift between
sessions: learning rate {1e-5, 3e-5, 1e-4, 3e-4} x augmentation {aug_a, aug_b} x seed
{42, 43, 44}. Stage 2 (head-only warm-up) is opt-in with --stage warmup.

Only Train and Validation are touched. Test is never loaded here.

Examples
    python scripts/09_taskA_run_training.py --list
    python scripts/09_taskA_run_training.py --smoke --runs-root outputs/runs/taskA_smoke
    python scripts/09_taskA_run_training.py --limit 1 \
        --runs-root "/content/drive/MyDrive/気合のCOVID19/01_TaskA_CXR/04_Training/AIagent_taskA_runs"

Peak GPU memory is recorded per run by the trainer (torch.cuda.max_memory_allocated).
"""
import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from covid_mortality.training.taskA_trainer import TrainConfig, run_training  # noqa: E402

LRS = [1e-5, 3e-5, 1e-4, 3e-4]
AUGS = ["aug_a", "aug_b"]
SEEDS = [42, 43, 44]

# Stage 2 (docs/task_a_stage2_design.md, approved 2026-09-21): the two conditions are fixed in
# code so they cannot drift. Both start from the Stage 1 best condition (lr 3e-4, aug_b) and
# change only the unfreezing schedule. Each group's LR is 0 while frozen, ramps over one epoch
# after it is unfrozen, then decays with cosine.
STAGE2 = [
    {"condition": "lr3e-4_aug_b_warmup1", "lr": 3e-4, "augment": "aug_b",
     "unfreeze_schedule": {"fc": 1, "layer4": 2, "rest": 2}},      # head 1 epoch -> full
    {"condition": "lr3e-4_aug_b_staged", "lr": 3e-4, "augment": "aug_b",
     "unfreeze_schedule": {"fc": 1, "layer4": 2, "rest": 4}},      # head -> +layer4 -> full
]


def lr_tag(lr: float) -> str:
    return f"lr{lr:.0e}".replace("e-0", "e-")


def build_grid(stage: str) -> list[dict]:
    grid = []
    if stage in ("main", "all"):
        for lr in LRS:
            for aug in AUGS:
                for seed in SEEDS:
                    grid.append({"condition": f"{lr_tag(lr)}_{aug}", "lr": lr, "augment": aug,
                                 "seed": seed, "freeze_backbone_epochs": 0,
                                 "unfreeze_schedule": None})
    if stage in ("stage2", "all"):
        for c in STAGE2:
            for seed in SEEDS:
                grid.append({**c, "seed": seed, "freeze_backbone_epochs": 0})
    return grid


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=".")
    ap.add_argument("--runs-root", default="outputs/runs/taskA")
    ap.add_argument("--image-dir", default="data/processed/taskA_png512_16bit/images")
    ap.add_argument("--stage", choices=["main", "stage2", "all"], default="main")
    ap.add_argument("--only-condition", default=None)
    ap.add_argument("--only-seed", type=int, default=None)
    ap.add_argument("--max-epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None, help="stop after N runs (for time-boxing)")
    ap.add_argument("--smoke", action="store_true", help="tiny debug run, never used for selection")
    ap.add_argument("--list", action="store_true", help="print the plan and exit")
    ap.add_argument("--force", action="store_true", help="re-run even if DONE.json exists")
    args = ap.parse_args()

    grid = build_grid(args.stage)
    if args.only_condition:
        grid = [g for g in grid if g["condition"] == args.only_condition]
    if args.only_seed is not None:
        grid = [g for g in grid if g["seed"] == args.only_seed]
    if args.smoke:
        grid = grid[:2]

    runs_root = Path(args.runs_root)
    status = []
    for g in grid:
        done = runs_root / g["condition"] / f"seed{g['seed']}" / "DONE.json"
        status.append({**g, "done": done.exists()})
    pending = [s for s in status if args.force or not s["done"]]
    print(f"runs planned: {len(status)}  already completed: {sum(s['done'] for s in status)}  "
          f"to run now: {len(pending)}")
    for s in status:
        print(f"  [{'DONE' if s['done'] else '    '}] {s['condition']}/seed{s['seed']}")
    if args.list:
        return 0

    results = []
    for i, g in enumerate(pending[: args.limit] if args.limit else pending, 1):
        cfg = TrainConfig(condition=g["condition"], seed=g["seed"], lr=g["lr"], augment=g["augment"],
                          freeze_backbone_epochs=g["freeze_backbone_epochs"],
                          unfreeze_schedule=g.get("unfreeze_schedule"), project=args.project,
                          image_dir=args.image_dir, runs_root=str(runs_root),
                          max_epochs=2 if args.smoke else args.max_epochs,
                          batch_size=args.batch_size, num_workers=args.num_workers, smoke=args.smoke)
        if args.force:
            (cfg.run_dir / "DONE.json").unlink(missing_ok=True)
        print(f"\n=== run {i}/{len(pending)}: {cfg.condition}/seed{cfg.seed} -> {cfg.run_dir}")
        out = run_training(cfg)
        print(json.dumps({k: v for k, v in out.items() if k != "config"}, ensure_ascii=False))
        results.append(out)

    summary_path = runs_root / "run_manager_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps({"planned": status, "results": results},
                                       ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\nsummary -> {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
