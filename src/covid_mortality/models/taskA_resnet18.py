"""Task A model: ImageNet-pretrained ResNet18 with a single-logit head.

Plan §4: torchvision ResNet18 (IMAGENET1K_V1), final layer replaced by Linear(512, 1) so that
BCEWithLogitsLoss(pos_weight=...) can be used for the imbalanced outcome. The Grad-CAM target
layer (plan §11) is the last convolutional block, exposed here so that training and
interpretation cannot drift apart.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


@dataclass(frozen=True)
class ModelConfig:
    pretrained: bool = True
    freeze_backbone_epochs: int = 0   # 0 = full fine-tuning from epoch 1
    dropout: float = 0.0


def build_model(cfg: ModelConfig = ModelConfig()) -> nn.Module:
    weights = ResNet18_Weights.IMAGENET1K_V1 if cfg.pretrained else None
    model = resnet18(weights=weights)
    in_features = model.fc.in_features  # 512
    model.fc = (nn.Sequential(nn.Dropout(cfg.dropout), nn.Linear(in_features, 1))
                if cfg.dropout > 0 else nn.Linear(in_features, 1))
    return model


def gradcam_target_layer(model: nn.Module) -> nn.Module:
    """Last conv block output (plan §11)."""
    return model.layer4[-1]


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    """Head-only warm-up support (plan §4, §5)."""
    for name, p in model.named_parameters():
        if not name.startswith("fc"):
            p.requires_grad = trainable


def count_parameters(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": int(total), "trainable": int(trainable)}


def make_criterion(pos_weight: float) -> nn.Module:
    """BCEWithLogitsLoss with the Train-derived pos_weight (plan §7)."""
    return nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32))


def extract_features(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """512-d penultimate features (kept for a possible Task C fusion; not used in Task A)."""
    modules = list(model.children())[:-1]
    backbone = nn.Sequential(*modules)
    return torch.flatten(backbone(x), 1)
