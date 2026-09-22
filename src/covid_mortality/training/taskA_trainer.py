"""Task A training engine (reusable module; no notebook logic inside).

One call to `run_training(cfg)` trains ONE run and writes everything it produces into
`cfg.run_dir`, flushing after every epoch so that a Colab disconnect loses at most the
epoch in progress:

    config.json          the exact configuration (+ code/data hashes, environment)
    history.csv          one row per epoch (losses, val AUROC/AUPRC/accuracy, lr, seconds)
    checkpoints/last.pt  model+optimizer+scheduler+RNG state -> resume
    checkpoints/best.pt  checkpoint selected by Val AUROC (plan §8)
    val_predictions.csv  patient-level predictions of the selected epoch
    DONE.json            written only on clean completion -> the runner skips the run

The engine is environment agnostic (CUDA if available, otherwise CPU) and never touches the
Test split: only "train" and "val" are ever loaded here.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from covid_mortality.data.taskA_dataset import (  # noqa: E402
    AUGMENTATIONS, TaskACXRDataset, build_loader, load_index)
from covid_mortality.evaluation.metrics import accuracy, average_precision, roc_auc  # noqa: E402
from covid_mortality.models.taskA_resnet18 import (  # noqa: E402
    ModelConfig, build_model, count_parameters, make_criterion, set_backbone_trainable)


@dataclass
class TrainConfig:
    # identity
    condition: str                      # e.g. "lr1e-4_augA"
    seed: int = 42
    # data
    project: str = "."
    image_dir: str = "data/processed/taskA_png512_16bit/images"
    runs_root: str = "outputs/runs/taskA"
    augment: str = "aug_a"              # none | aug_a | aug_b
    batch_size: int = 32
    num_workers: int = 2
    # optimisation
    lr: float = 1e-4
    weight_decay: float = 1e-4
    max_epochs: int = 30
    warmup_epochs: int = 1              # cosine warm-up (optimisation), see freeze_epochs below
    freeze_backbone_epochs: int = 0     # head-only epochs before unfreezing (Stage 1 path)
    # Stage 2 only: epoch (1-indexed) at which each parameter group starts training.
    # None -> Stage 1 behaviour (single OneCycleLR over all parameters), unchanged.
    # Each group's learning rate is 0 while frozen, ramps 0 -> lr over one epoch after it is
    # unfrozen, then decays with cosine; this prevents the peak learning rate from hitting a
    # backbone group at the moment it is unfrozen.
    unfreeze_schedule: dict | None = None   # e.g. {"fc": 1, "layer4": 2, "rest": 2}
    amp: bool = True
    # selection / stopping (Validation only)
    monitor: str = "val_auroc"
    early_stopping_patience: int = 8
    min_delta: float = 0.005
    # debugging only (never used for model selection)
    smoke: bool = False
    smoke_train_n: int = 48
    smoke_val_n: int = 32
    stop_after_epoch: int | None = None   # simulate a disconnect: stop without writing DONE.json
    extra: dict = field(default_factory=dict)

    @property
    def run_dir(self) -> Path:
        return Path(self.runs_root) / self.condition / f"seed{self.seed}"


def _git_commit(project: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=project,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "not_a_git_repo"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _rng_state() -> dict:
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def _restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


GROUP_ORDER = ("fc", "layer4", "rest")


def split_param_groups(model: nn.Module) -> dict:
    """Partition ResNet18 parameters into fc / layer4 / rest (Stage 2 only)."""
    groups: dict[str, list] = {g: [] for g in GROUP_ORDER}
    for name, p in model.named_parameters():
        if name.startswith("fc"):
            groups["fc"].append((name, p))
        elif name.startswith("layer4"):
            groups["layer4"].append((name, p))
        else:
            groups["rest"].append((name, p))
    return groups


def group_lr_lambda(unfreeze_epoch: int, steps_per_epoch: int, total_epochs: int):
    """0 while frozen -> linear ramp over one epoch after unfreezing -> cosine decay to 0."""
    start = (unfreeze_epoch - 1) * steps_per_epoch
    warm = steps_per_epoch
    total = total_epochs * steps_per_epoch

    def fn(step: int) -> float:
        if step < start:
            return 0.0
        if step < start + warm:
            return (step - start + 1) / warm
        denom = max(total - start - warm, 1)
        progress = min((step - start - warm) / denom, 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return fn


def apply_unfreeze_schedule(model: nn.Module, schedule: dict, epoch: int) -> dict:
    """Set requires_grad per group for this epoch; returns the trainable flags."""
    groups = split_param_groups(model)
    flags = {}
    for g in GROUP_ORDER:
        trainable = epoch >= int(schedule[g])
        flags[g] = trainable
        for _, p in groups[g]:
            p.requires_grad = trainable
    return flags


def environment_info(project: Path) -> dict:
    return {"python": sys.version.split()[0], "platform": platform.platform(),
            "torch": torch.__version__, "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "cuda_version": torch.version.cuda, "git_commit": _git_commit(project)}


@torch.no_grad()
def evaluate(model: nn.Module, loader, criterion: nn.Module, device: torch.device) -> dict:
    model.eval()
    probs, labels, pids, losses = [], [], [], []
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)
        logits = model(x).squeeze(1)
        losses.append(float(criterion(logits, y)) * len(y))
        probs.append(torch.sigmoid(logits).float().cpu().numpy())
        labels.append(y.float().cpu().numpy())
        pids.extend(batch["patient_id"])
    p = np.concatenate(probs); l = np.concatenate(labels)  # noqa: E741
    return {"loss": float(sum(losses) / len(l)), "auroc": roc_auc(l, p),
            "auprc": average_precision(l, p), "accuracy": accuracy(l, p, 0.5),
            "probs": p, "labels": l, "patient_ids": pids}


def run_training(cfg: TrainConfig) -> dict:
    project = Path(cfg.project).resolve()
    run_dir = cfg.run_dir
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    done_file = run_dir / "DONE.json"
    if done_file.exists():
        # spread first, then override the status: the payload also carries status="completed"
        return {**json.loads(done_file.read_text(encoding="utf-8")), "status": "skipped_completed"}

    set_all_seeds(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    index = load_index(project)
    image_dir = project / cfg.image_dir
    aug = AUGMENTATIONS[cfg.augment]
    train_ds = TaskACXRDataset(index, image_dir, "train", aug)
    val_ds = TaskACXRDataset(index, image_dir, "val")
    if cfg.smoke:  # debug only; recorded in the config so it can never be mistaken for a real run
        train_ds.df = train_ds.df.head(cfg.smoke_train_n).reset_index(drop=True)
        val_ds.df = val_ds.df.head(cfg.smoke_val_n).reset_index(drop=True)
    pos_weight = float((len(train_ds.df) - train_ds.df.label.sum()) / max(train_ds.df.label.sum(), 1))

    train_loader = build_loader(train_ds, cfg.batch_size, cfg.seed, cfg.num_workers)
    val_loader = build_loader(val_ds, cfg.batch_size, cfg.seed, cfg.num_workers)

    model = build_model(ModelConfig(pretrained=True)).to(device)
    criterion = make_criterion(pos_weight).to(device)
    steps = max(len(train_loader), 1)
    if cfg.unfreeze_schedule:  # Stage 2: one param group per unfreeze stage
        groups = split_param_groups(model)
        optimizer = torch.optim.AdamW(
            [{"params": [p for _, p in groups[g]], "lr": cfg.lr, "name": g} for g in GROUP_ORDER],
            lr=cfg.lr, weight_decay=cfg.weight_decay)
        sched = torch.optim.lr_scheduler.LambdaLR(
            optimizer, [group_lr_lambda(int(cfg.unfreeze_schedule[g]), steps, cfg.max_epochs)
                        for g in GROUP_ORDER])
        schedule_name = f"LambdaLR(per-group ramp+cos) {cfg.unfreeze_schedule}"
    else:  # Stage 1: unchanged
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=cfg.lr, total_steps=cfg.max_epochs * steps,
            pct_start=max(cfg.warmup_epochs / max(cfg.max_epochs, 1), 1e-3), anneal_strategy="cos",
            div_factor=10.0, final_div_factor=100.0)
        schedule_name = "OneCycleLR(cos)"
    use_amp = bool(cfg.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    config_payload = {**asdict(cfg), "device": device.type, "pos_weight": pos_weight,
                      "n_train": len(train_ds), "n_val": len(val_ds),
                      "deaths_train": int(train_ds.df.label.sum()),
                      "deaths_val": int(val_ds.df.label.sum()),
                      "model_parameters": count_parameters(model),
                      "environment": environment_info(project),
                      "inputs": {
                          "split_manifest": _sha256(project / "data/splits/COVID19_固定患者split_1277.csv"),
                          "index_manifest": _sha256(
                              project / "data/interim/cxr_audit/index_cxr_manifest_window_T0m2_T0.csv")},
                      "scheduler": schedule_name, "amp_enabled": use_amp,
                      "stage": "stage2" if cfg.unfreeze_schedule else "stage1"}
    (run_dir / "config.json").write_text(json.dumps(config_payload, ensure_ascii=False, indent=2,
                                                    default=str), encoding="utf-8")

    history_path = run_dir / "history.csv"
    start_epoch, best = 1, {"epoch": 0, "val_auroc": -np.inf, "val_loss": np.inf}
    last_ckpt = run_dir / "checkpoints/last.pt"
    if last_ckpt.exists():  # resume after a disconnect
        state = torch.load(last_ckpt, map_location=device, weights_only=False)
        # a changed configuration is a different run, not a resume: stop instead of silently
        # continuing with a schedule/optimiser that no longer matches the config
        fixed = ("condition", "seed", "lr", "weight_decay", "max_epochs", "batch_size", "augment",
                 "freeze_backbone_epochs", "warmup_epochs", "unfreeze_schedule", "smoke",
                 "smoke_train_n", "smoke_val_n")
        changed = {k: (state["config"].get(k), asdict(cfg)[k]) for k in fixed
                   if state["config"].get(k) != asdict(cfg)[k]}
        if changed:
            raise RuntimeError(
                f"cannot resume {run_dir}: configuration changed {changed}. "
                "Use a new condition name or --force to start over.")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        sched.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        _restore_rng(state["rng"])
        start_epoch = state["epoch"] + 1
        best = state["best"]
        print(f"resumed {run_dir} at epoch {start_epoch}", flush=True)

    epochs_without_improvement = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    run_started = time.time()
    for epoch in range(start_epoch, cfg.max_epochs + 1):
        t0 = time.time()
        if cfg.unfreeze_schedule:
            group_flags = apply_unfreeze_schedule(model, cfg.unfreeze_schedule, epoch)
        else:
            set_backbone_trainable(model, epoch > cfg.freeze_backbone_epochs)
            group_flags = {}
        model.train()
        running, seen = 0.0, 0
        for batch in train_loader:
            x = batch["image"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = criterion(model(x).squeeze(1), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            sched.step()
            running += float(loss.detach()) * len(y)
            seen += len(y)
        val = evaluate(model, val_loader, criterion, device)
        row = {"epoch": epoch, "train_loss": running / max(seen, 1), "val_loss": val["loss"],
               "val_auroc": val["auroc"], "val_auprc": val["auprc"], "val_accuracy": val["accuracy"],
               "lr": optimizer.param_groups[0]["lr"], "seconds": round(time.time() - t0, 2),
               "backbone_trainable": (group_flags.get("rest") if cfg.unfreeze_schedule
                                      else epoch > cfg.freeze_backbone_epochs),
               "peak_gpu_mem_mb": (round(torch.cuda.max_memory_allocated() / 2 ** 20, 1)
                                   if device.type == "cuda" else None)}
        if cfg.unfreeze_schedule:  # per-group learning rate and trainable flag (Stage 2)
            for g, pg in zip(GROUP_ORDER, optimizer.param_groups):
                row[f"lr_{g}"] = pg["lr"]
                row[f"trainable_{g}"] = group_flags[g]
        pd.DataFrame([row]).to_csv(history_path, mode="a", header=not history_path.exists(),
                                   index=False, encoding="utf-8-sig")

        improved = (val["auroc"] > best["val_auroc"] + cfg.min_delta) or (
            abs(val["auroc"] - best["val_auroc"]) <= cfg.min_delta and val["loss"] < best["val_loss"]
            and val["auroc"] >= best["val_auroc"])
        if improved:
            best = {"epoch": epoch, "val_auroc": float(val["auroc"]), "val_loss": float(val["loss"]),
                    "val_auprc": float(val["auprc"]), "val_accuracy": float(val["accuracy"])}
            torch.save({"model": model.state_dict(), "epoch": epoch, "best": best,
                        "config": asdict(cfg)}, run_dir / "checkpoints/best.pt")
            pd.DataFrame({"subject_id": val["patient_ids"], "true_label": val["labels"].astype(int),
                          "prob": val["probs"], "epoch": epoch, "condition": cfg.condition,
                          "seed": cfg.seed}).to_csv(run_dir / "val_predictions.csv", index=False,
                                                    encoding="utf-8-sig")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": sched.state_dict(), "scaler": scaler.state_dict(),
                    "epoch": epoch, "best": best, "rng": _rng_state(), "config": asdict(cfg)},
                   last_ckpt)
        print(f"[{cfg.condition}/seed{cfg.seed}] epoch {epoch}/{cfg.max_epochs} "
              f"train_loss={row['train_loss']:.4f} val_loss={row['val_loss']:.4f} "
              f"val_auroc={row['val_auroc']:.4f} ({row['seconds']}s)", flush=True)
        if cfg.stop_after_epoch is not None and epoch >= cfg.stop_after_epoch:
            print(f"stop_after_epoch={cfg.stop_after_epoch}: simulated interruption "
                  f"(no DONE.json written)", flush=True)
            return {"status": "interrupted", "condition": cfg.condition, "seed": cfg.seed,
                    "epochs_run": epoch, "best": best, "run_dir": str(run_dir)}
        if epochs_without_improvement >= cfg.early_stopping_patience:
            print(f"early stopping at epoch {epoch} (no improvement for "
                  f"{epochs_without_improvement} epochs)", flush=True)
            break

    full_history = pd.read_csv(history_path, encoding="utf-8-sig")
    payload = {"status": "completed", "condition": cfg.condition, "seed": cfg.seed,
               "best": best, "epochs_run": epoch, "run_dir": str(run_dir),
               "finished": time.strftime("%Y-%m-%dT%H:%M:%S"), "smoke": cfg.smoke,
               "early_stopped": bool(epochs_without_improvement >= cfg.early_stopping_patience),
               "device": device.type, "amp_enabled": use_amp,
               "gpu": environment_info(project)["gpu"],
               "seconds_this_session": round(time.time() - run_started, 1),
               "seconds_total_all_epochs": round(float(full_history.seconds.sum()), 1),
               "mean_seconds_per_epoch": round(float(full_history.seconds.mean()), 2),
               "peak_gpu_mem_mb": (float(np.nanmax(full_history.peak_gpu_mem_mb))
                                   if device.type == "cuda" else None)}
    done_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload
