"""Stage 2 checks: per-group unfreezing, learning-rate ramp, and Stage 1 left unchanged.

The central requirement (approved design, 2026-09-21): when a backbone group is unfrozen it
must NOT receive the peak learning rate; it ramps from 0 over one epoch and then decays.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from covid_mortality.models.taskA_resnet18 import build_model, ModelConfig  # noqa: E402
from covid_mortality.training.taskA_trainer import (  # noqa: E402
    GROUP_ORDER, apply_unfreeze_schedule, group_lr_lambda, split_param_groups)

S2A = {"fc": 1, "layer4": 2, "rest": 2}     # head 1 epoch -> full fine-tuning
S2B = {"fc": 1, "layer4": 2, "rest": 4}     # head -> head+layer4 -> full
STEPS, EPOCHS = 32, 30


def test_param_groups_partition_all_parameters():
    model = build_model(ModelConfig(pretrained=False))
    groups = split_param_groups(model)
    n = sum(len(groups[g]) for g in GROUP_ORDER)
    assert n == len(list(model.named_parameters()))
    assert all(name.startswith("fc") for name, _ in groups["fc"])
    assert all(name.startswith("layer4") for name, _ in groups["layer4"])
    assert not any(name.startswith(("fc", "layer4")) for name, _ in groups["rest"])
    print(f"    fc={len(groups['fc'])} layer4={len(groups['layer4'])} rest={len(groups['rest'])}")


def test_lr_is_zero_while_frozen_and_ramps_after_unfreeze():
    for name, sched in (("S2-A", S2A), ("S2-B", S2B)):
        for g in GROUP_ORDER:
            fn = group_lr_lambda(sched[g], STEPS, EPOCHS)
            u = sched[g]
            # every step of every epoch before the unfreeze epoch is exactly 0
            for e in range(1, u):
                for s in range(STEPS):
                    assert fn((e - 1) * STEPS + s) == 0.0, f"{name}/{g}: lr != 0 while frozen"
            first = fn((u - 1) * STEPS)          # first step after unfreezing
            peak_epoch_end = fn((u - 1) * STEPS + STEPS - 1)
            assert first <= 1.0 / STEPS + 1e-9, f"{name}/{g}: lr jumps to {first} at unfreeze"
            assert abs(peak_epoch_end - 1.0) < 1e-9, f"{name}/{g}: ramp does not reach max"
            print(f"    {name}/{g:6s} unfreeze@{u}: first={first:.4f} end_of_ramp={peak_epoch_end:.3f}")


def test_lr_decays_to_zero_at_the_end():
    for g in GROUP_ORDER:
        fn = group_lr_lambda(S2B[g], STEPS, EPOCHS)
        assert fn(EPOCHS * STEPS - 1) < 0.01, "learning rate should decay to ~0"
    print("    final multiplier < 0.01 for all groups")


def test_backbone_never_sees_peak_at_unfreeze_moment():
    """The concrete failure mode of the Stage 1 schedule: OneCycle peaks at the end of epoch 1,
    which is exactly when a warm-up condition unfreezes the backbone."""
    rest = group_lr_lambda(S2A["rest"], STEPS, EPOCHS)
    at_unfreeze = rest((S2A["rest"] - 1) * STEPS)
    assert at_unfreeze < 0.05, at_unfreeze
    model = build_model(ModelConfig(pretrained=False))
    opt = torch.optim.AdamW([{"params": [p for _, p in split_param_groups(model)[g]], "lr": 3e-4}
                             for g in GROUP_ORDER], lr=3e-4)
    sch = torch.optim.lr_scheduler.LambdaLR(
        opt, [group_lr_lambda(S2A[g], STEPS, EPOCHS) for g in GROUP_ORDER])
    lrs = []
    for _ in range(2 * STEPS):      # two epochs
        lrs.append([pg["lr"] for pg in opt.param_groups])
        sch.step()
    assert lrs[0][2] == 0.0, "rest group must start at 0"
    assert max(lr[2] for lr in lrs[:STEPS]) == 0.0, "rest group must stay 0 during epoch 1"
    assert lrs[STEPS][2] <= 3e-4 / STEPS + 1e-12, "rest group must ramp, not jump"
    print(f"    epoch1 rest lr max = {max(lr[2] for lr in lrs[:STEPS]):.2e}; "
          f"first step of epoch2 = {lrs[STEPS][2]:.2e} (max 3.0e-04)")


def test_unfreeze_flags_per_epoch():
    model = build_model(ModelConfig(pretrained=False))
    expected = {1: (True, False, False), 2: (True, True, False), 4: (True, True, True)}
    for epoch, exp in expected.items():
        flags = apply_unfreeze_schedule(model, S2B, epoch)
        got = (flags["fc"], flags["layer4"], flags["rest"])
        assert got == exp, f"epoch {epoch}: {got} != {exp}"
        groups = split_param_groups(model)
        for g, want in zip(GROUP_ORDER, exp):
            assert all(p.requires_grad == want for _, p in groups[g])
    print("    S2-B flags: epoch1 fc only, epoch2-3 +layer4, epoch4+ all")


def test_stage1_path_unchanged():
    """With unfreeze_schedule=None the trainer must still build a single OneCycleLR."""
    import inspect

    from covid_mortality.training import taskA_trainer as t
    src = inspect.getsource(t.run_training)
    assert "if cfg.unfreeze_schedule:" in src and "OneCycleLR" in src
    assert 'schedule_name = "OneCycleLR(cos)"' in src
    cfg = t.TrainConfig(condition="stage1_like")
    assert cfg.unfreeze_schedule is None, "Stage 1 default must remain None"
    print("    default config -> unfreeze_schedule=None -> OneCycleLR branch")


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"[{name}]")
            try:
                fn()
                print("  PASS")
            except AssertionError as e:
                failures += 1
                print(f"  FAIL: {e}")
    print(f"\n{failures} failure(s)")
    sys.exit(1 if failures else 0)
