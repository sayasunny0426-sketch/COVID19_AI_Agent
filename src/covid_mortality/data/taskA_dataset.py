"""Task A Dataset / DataLoader (index CXR -> in-hospital mortality).

Design (docs/task_a_implementation_plan.md §3):
  - one row per patient, one image per patient (1,277 patients);
  - the fixed split is read from file and never regenerated;
  - images come from the 16-bit PNG cache (512x512, per-image normalisation already applied),
    so nothing here is fitted on data and no statistic crosses a split boundary;
  - augmentation is available for Train only; Validation / Test are deterministic;
  - final tensor: float32, shape (3, 224, 224), ImageNet-normalised.

Pipeline inside __getitem__:
    16-bit PNG (uint16) -> float32 / 65535 -> [1,512,512] tensor
    -> (Train only) random affine + brightness/contrast jitter
    -> resize to 224 (bilinear, antialias)
    -> repeat to 3 channels -> ImageNet mean/std normalisation
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
UINT16_MAX = 65535.0
INPUT_SIZE = 224
CACHE_SIZE = 512
EXPECTED = {"train": (1021, 135), "val": (128, 17), "test": (128, 17)}


@dataclass(frozen=True)
class AugmentConfig:
    """Pre-registered augmentation conditions (plan §6). Train only."""
    name: str = "none"
    translate: float = 0.0        # fraction of image size
    scale: tuple[float, float] = (1.0, 1.0)
    degrees: float = 0.0
    brightness: float = 0.0       # +/- fraction
    contrast: float = 0.0         # +/- fraction

    @property
    def enabled(self) -> bool:
        return self.name != "none"


AUG_NONE = AugmentConfig("none")
AUG_A = AugmentConfig("aug_a", translate=0.03, scale=(0.97, 1.03), degrees=0.0,
                      brightness=0.05, contrast=0.05)
AUG_B = AugmentConfig("aug_b", translate=0.05, scale=(0.95, 1.05), degrees=7.0,
                      brightness=0.10, contrast=0.10)
AUGMENTATIONS = {c.name: c for c in (AUG_NONE, AUG_A, AUG_B)}


def load_index(project: Path) -> pd.DataFrame:
    """Join the index-CXR manifest with the fixed split; verify the fixed numbers."""
    manifest = pd.read_csv(project / "data/interim/cxr_audit/index_cxr_manifest_window_T0m2_T0.csv",
                           encoding="utf-8-sig", dtype=str)
    split = pd.read_csv(project / "data/splits/COVID19_固定患者split_1277.csv", dtype=str)
    df = manifest.merge(split[["Subject ID", "split", "true_label", "last.status"]],
                        left_on="PatientID", right_on="Subject ID", how="inner")
    if len(df) != 1277 or len(manifest) != 1277 or len(split) != 1277:
        raise ValueError(f"cohort size changed: manifest={len(manifest)} split={len(split)} joined={len(df)}")
    if not df.PatientID.is_unique:
        raise ValueError("PatientID is not unique")
    df["label"] = df.true_label.astype(int)
    for sp, (n, d) in EXPECTED.items():
        g = df[df.split == sp]
        if len(g) != n or int(g.label.sum()) != d:
            raise ValueError(f"split {sp}: n={len(g)} (expected {n}), deaths={int(g.label.sum())} (expected {d})")
    return df


class TaskACXRDataset(Dataset):
    def __init__(self, index: pd.DataFrame, image_dir: Path, split: str,
                 augment: AugmentConfig = AUG_NONE, input_size: int = INPUT_SIZE):
        if split not in EXPECTED:
            raise ValueError(f"unknown split: {split}")
        if augment.enabled and split != "train":
            raise ValueError(f"augmentation is Train-only, got split={split}, augment={augment.name}")
        self.split = split
        self.augment = augment
        self.input_size = input_size
        self.image_dir = Path(image_dir)
        self.df = index[index.split == split].sort_values("PatientID").reset_index(drop=True)
        missing = [p for p in self.df.PatientID if not (self.image_dir / f"{p}.png").exists()]
        if missing:
            raise FileNotFoundError(f"{len(missing)} cached image(s) missing, e.g. {missing[:3]}")

    def __len__(self) -> int:
        return len(self.df)

    def patient_ids(self) -> list[str]:
        return list(self.df.PatientID)

    def labels(self) -> np.ndarray:
        return self.df.label.to_numpy()

    def pos_weight(self) -> float:
        """(negatives / positives) for BCEWithLogitsLoss -- Train only by construction."""
        if self.split != "train":
            raise ValueError("pos_weight must be computed on the Training set only")
        pos = int(self.df.label.sum())
        return float((len(self.df) - pos) / pos)

    def read_raw(self, idx: int) -> np.ndarray:
        """16-bit PNG as uint16; used by the dataset and by the QC checks."""
        pid = self.df.PatientID.iloc[idx]
        arr = np.array(Image.open(self.image_dir / f"{pid}.png"))
        if arr.dtype != np.uint16:
            raise TypeError(f"{pid}: expected uint16 PNG, got {arr.dtype}")
        if arr.shape != (CACHE_SIZE, CACHE_SIZE):
            raise ValueError(f"{pid}: expected {CACHE_SIZE}x{CACHE_SIZE}, got {arr.shape}")
        return arr

    def _augment(self, img: torch.Tensor) -> torch.Tensor:
        from torchvision.transforms import v2
        from torchvision.transforms import functional as F
        a = self.augment
        img = v2.RandomAffine(degrees=a.degrees, translate=(a.translate, a.translate),
                              scale=a.scale, fill=0.0)(img)
        if a.brightness:
            factor = float(torch.empty(1).uniform_(1 - a.brightness, 1 + a.brightness))
            img = torch.clamp(img * factor, 0.0, 1.0)
        if a.contrast:
            factor = float(torch.empty(1).uniform_(1 - a.contrast, 1 + a.contrast))
            mean = img.mean()
            img = torch.clamp((img - mean) * factor + mean, 0.0, 1.0)
        return img

    def __getitem__(self, idx: int):
        from torchvision.transforms import functional as F
        arr = self.read_raw(idx)
        img = torch.from_numpy(arr.astype(np.float32) / UINT16_MAX).unsqueeze(0)  # [1,512,512] in [0,1]
        if self.augment.enabled:
            img = self._augment(img)
        img = F.resize(img, [self.input_size, self.input_size], antialias=True)
        img = img.repeat(3, 1, 1)
        img = F.normalize(img, mean=list(IMAGENET_MEAN), std=list(IMAGENET_STD))
        label = torch.tensor(float(self.df.label.iloc[idx]), dtype=torch.float32)
        return {"image": img, "label": label, "patient_id": self.df.PatientID.iloc[idx]}


def seed_worker(worker_id: int) -> None:
    """Deterministic per-worker seeding (torch generator seed is set by the DataLoader)."""
    import random
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_loader(dataset: TaskACXRDataset, batch_size: int = 32, seed: int = 42,
                 num_workers: int = 0) -> DataLoader:
    """Train loaders shuffle (seeded generator); Validation / Test are deterministic and ordered."""
    is_train = dataset.split == "train"
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=is_train, drop_last=False,
                      num_workers=num_workers, worker_init_fn=seed_worker if num_workers else None,
                      generator=generator if is_train else None, pin_memory=False)


def describe_config() -> dict:
    return {"input_size": INPUT_SIZE, "cache_size": CACHE_SIZE,
            "imagenet_mean": IMAGENET_MEAN, "imagenet_std": IMAGENET_STD,
            "augmentations": {k: asdict(v) for k, v in AUGMENTATIONS.items()},
            "expected_split_counts": EXPECTED}
