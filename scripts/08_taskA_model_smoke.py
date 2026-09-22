"""Smoke test of the Task A model wiring. No training, no evaluation on any split.

Verifies: pretrained weights load, head shape, forward pass on one real Validation batch,
loss computation with the Train-derived pos_weight, warm-up freezing, the Grad-CAM target
layer, and that gradients reach that layer (Grad-CAM prerequisite).

Usage:
    python scripts/08_taskA_model_smoke.py --project .
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from covid_mortality.data.taskA_dataset import TaskACXRDataset, build_loader, load_index  # noqa: E402
from covid_mortality.models.taskA_resnet18 import (  # noqa: E402
    ModelConfig, build_model, count_parameters, extract_features, gradcam_target_layer,
    make_criterion, set_backbone_trainable)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=".")
    args = ap.parse_args()
    project = Path(args.project)
    image_dir = project / "data/processed/taskA_png512_16bit/images"
    out: dict = {}
    failures = []

    def check(name, ok, detail=None):
        out[name] = {"pass": bool(ok), "detail": detail}
        if not ok:
            failures.append(name)
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")

    torch.manual_seed(42)
    index = load_index(project)
    val = TaskACXRDataset(index, image_dir, "val")
    train = TaskACXRDataset(index, image_dir, "train")
    batch = next(iter(build_loader(val, batch_size=4, seed=42)))

    model = build_model(ModelConfig(pretrained=True))
    params = count_parameters(model)
    check("pretrained_resnet18_built", params["total"] > 11_000_000, params)
    check("head_is_single_logit", model.fc.out_features == 1 and model.fc.in_features == 512,
          {"in": model.fc.in_features, "out": model.fc.out_features})

    model.eval()
    with torch.no_grad():
        logits = model(batch["image"])
    check("forward_shape", tuple(logits.shape) == (4, 1), list(logits.shape))
    check("forward_finite", bool(torch.isfinite(logits).all()),
          {"min": round(float(logits.min()), 4), "max": round(float(logits.max()), 4)})

    pw = train.pos_weight()
    criterion = make_criterion(pw)
    loss = criterion(logits.squeeze(1), batch["label"])
    check("loss_computes", bool(torch.isfinite(loss)), {"pos_weight": round(pw, 4),
                                                        "loss": round(float(loss), 4)})

    set_backbone_trainable(model, False)
    frozen = count_parameters(model)
    set_backbone_trainable(model, True)
    unfrozen = count_parameters(model)
    check("warmup_freeze_works", frozen["trainable"] == 513 and unfrozen["trainable"] == unfrozen["total"],
          {"head_only": frozen["trainable"], "all": unfrozen["trainable"]})

    target = gradcam_target_layer(model)
    acts, grads = {}, {}

    def save_act(module, inputs, output):  # hooks must return None, otherwise the value is used
        acts["a"] = output.detach()

    def save_grad(module, grad_in, grad_out):
        grads["g"] = grad_out[0].detach()

    h1 = target.register_forward_hook(save_act)
    h2 = target.register_full_backward_hook(save_grad)
    model.train()
    logit = model(batch["image"][:1])
    logit.sum().backward()
    h1.remove(); h2.remove()
    check("gradcam_layer_activations", tuple(acts["a"].shape) == (1, 512, 7, 7), list(acts["a"].shape))
    check("gradcam_layer_gradients", "g" in grads and bool(torch.isfinite(grads["g"]).all()),
          list(grads["g"].shape))

    with torch.no_grad():
        feats = extract_features(model.eval(), batch["image"][:2])
    check("feature_extractor_512d", tuple(feats.shape) == (2, 512), list(feats.shape))

    rep = project / "data/interim/taskA_model_smoke.json"
    rep.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{len(failures)} failure(s). report -> {rep}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
