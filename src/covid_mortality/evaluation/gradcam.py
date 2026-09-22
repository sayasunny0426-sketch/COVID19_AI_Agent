"""Grad-CAM for the Task A CXR model, plus objective checks on where the map puts its mass.

Grad-CAM (Selvaraju et al., ICCV 2017): weight each channel of the target layer's activation
by the spatially averaged gradient of the score with respect to that channel, sum, apply ReLU
and normalise. The target layer for this study is the last convolutional block of ResNet18
(`model.layer4[-1]`), fixed in models/taskA_resnet18.gradcam_target_layer.

The region metrics exist so that "does the model look at the lungs or at the burned-in text?"
can be answered with numbers as well as by eye: the cached images are zero-padded to a square,
so padding is exactly 0 and can be separated from the radiograph itself.

Rendering uses numpy + PIL only (no matplotlib), so the same code runs anywhere.
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import nn

# perceptually ordered blue -> cyan -> yellow -> red control points for the overlay
_CMAP = np.array([[0, 0, 128], [0, 0, 255], [0, 255, 255], [255, 255, 0], [255, 128, 0],
                  [255, 0, 0]], dtype=np.float32)


class GradCAM:
    """Grad-CAM for a single-logit model. Use as a context manager to guarantee hook removal."""

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None
        self._handles: list = []

    def __enter__(self) -> "GradCAM":
        def fwd(_m, _i, output):
            self._activations = output.detach()

        def bwd(_m, _gi, grad_output):
            self._gradients = grad_output[0].detach()

        self._handles = [self.target_layer.register_forward_hook(fwd),
                         self.target_layer.register_full_backward_hook(bwd)]
        return self

    def __exit__(self, *exc) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    def __call__(self, x: torch.Tensor) -> tuple[np.ndarray, float]:
        """x: (1, 3, H, W). Returns the CAM upsampled to (H, W) in [0, 1] and the probability."""
        if not self._handles:
            raise RuntimeError("use GradCAM as a context manager so the hooks are registered")
        self.model.zero_grad(set_to_none=True)
        was_training = self.model.training
        self.model.eval()
        logit = self.model(x).squeeze()
        logit.backward()
        if self._activations is None or self._gradients is None:
            raise RuntimeError("target layer produced no activations/gradients")
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)          # GAP over space
        cam = torch.relu((weights * self._activations).sum(dim=1, keepdim=True))
        cam = torch.nn.functional.interpolate(cam, size=x.shape[-2:], mode="bilinear",
                                              align_corners=False)[0, 0]
        cam = cam - cam.min()
        cam = cam / cam.max() if float(cam.max()) > 0 else cam
        if was_training:
            self.model.train()
        return cam.cpu().numpy(), float(torch.sigmoid(logit.detach()))


def colourise(cam: np.ndarray) -> np.ndarray:
    """CAM in [0,1] -> RGB uint8 using the fixed colour map above."""
    pos = np.clip(cam, 0, 1) * (len(_CMAP) - 1)
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, len(_CMAP) - 1)
    frac = (pos - lo)[..., None]
    return ((_CMAP[lo] * (1 - frac) + _CMAP[hi] * frac)).astype(np.uint8)


def overlay(gray01: np.ndarray, cam: np.ndarray, alpha: float = 0.40) -> Image.Image:
    """Blend the CAM over the grayscale image (both HxW, values in [0,1])."""
    base = np.repeat((np.clip(gray01, 0, 1) * 255).astype(np.uint8)[..., None], 3, axis=2)
    blended = (1 - alpha) * base.astype(np.float32) + alpha * colourise(cam).astype(np.float32)
    return Image.fromarray(blended.astype(np.uint8))


def region_metrics(cam: np.ndarray, gray01: np.ndarray, pad_eps: float = 1e-3) -> dict:
    """Where does the CAM put its mass? All fractions are of the total CAM mass.

    padding      : zero-padded area added to make the image square (never anatomy)
    border_band  : outer 10% ring of the radiograph itself (collimation edges, L/R markers,
                   'PORTABLE' text usually live here)
    central_50   : central 50% x 50% box (lungs/mediastinum for a frontal chest radiograph)
    """
    total = float(cam.sum())
    if total <= 0:
        return {"cam_mass": 0.0, "frac_in_padding": float("nan"),
                "frac_in_border_band": float("nan"), "frac_in_central_50": float("nan"),
                "centroid_x": float("nan"), "centroid_y": float("nan"),
                "peak_x": float("nan"), "peak_y": float("nan")}
    h, w = cam.shape
    content = gray01 > pad_eps
    frac_pad = float(cam[~content].sum() / total)

    ys, xs = np.nonzero(content)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    bh, bw = y1 - y0 + 1, x1 - x0 + 1
    inner = np.zeros_like(content)
    iy0, iy1 = y0 + int(0.10 * bh), y1 - int(0.10 * bh)
    ix0, ix1 = x0 + int(0.10 * bw), x1 - int(0.10 * bw)
    inner[iy0:iy1 + 1, ix0:ix1 + 1] = True
    border = content & ~inner
    frac_border = float(cam[border].sum() / total)

    central = np.zeros_like(content)
    cy0, cy1 = y0 + int(0.25 * bh), y0 + int(0.75 * bh)
    cx0, cx1 = x0 + int(0.25 * bw), x0 + int(0.75 * bw)
    central[cy0:cy1 + 1, cx0:cx1 + 1] = True
    frac_central = float(cam[central].sum() / total)

    yy, xx = np.mgrid[0:h, 0:w]
    cy = float((cam * yy).sum() / total) / h
    cx = float((cam * xx).sum() / total) / w
    py, px = np.unravel_index(int(np.argmax(cam)), cam.shape)
    return {"cam_mass": total, "frac_in_padding": frac_pad, "frac_in_border_band": frac_border,
            "frac_in_central_50": frac_central, "centroid_x": cx, "centroid_y": cy,
            "peak_x": px / w, "peak_y": py / h}


def annotate(img: Image.Image, lines: list[str], height: int = 46) -> Image.Image:
    """Add a caption strip under an image (PIL default font; no font files needed)."""
    out = Image.new("RGB", (img.width, img.height + height), "white")
    out.paste(img, (0, 0))
    draw = ImageDraw.Draw(out)
    for i, line in enumerate(lines[:3]):
        draw.text((4, img.height + 2 + i * 14), line, fill=(0, 0, 0))
    return out


def side_by_side(original: Image.Image, cam_overlay: Image.Image, gap: int = 6) -> Image.Image:
    out = Image.new("RGB", (original.width * 2 + gap, original.height), "white")
    out.paste(original, (0, 0))
    out.paste(cam_overlay, (original.width + gap, 0))
    return out


def grid(images: list[Image.Image], cols: int, gap: int = 8, bg="white",
         cells: int | None = None) -> Image.Image:
    """Lay images out in a grid. `cells` forces the number of cells, so a 4x4 panel can be
    produced with a blank cell when fewer than 16 cases exist (no case is substituted)."""
    n_cells = cells if cells is not None else len(images)
    if n_cells < len(images):
        raise ValueError("cells must be >= the number of images")
    rows = (n_cells + cols - 1) // cols
    w = max(i.width for i in images)
    h = max(i.height for i in images)
    canvas = Image.new("RGB", (cols * w + (cols + 1) * gap, rows * h + (rows + 1) * gap), bg)
    for idx, im in enumerate(images):
        r, c = divmod(idx, cols)
        canvas.paste(im, (gap + c * (w + gap), gap + r * (h + gap)))
    for idx in range(len(images), n_cells):      # mark the empty cells explicitly
        r, c = divmod(idx, cols)
        x, y = gap + c * (w + gap), gap + r * (h + gap)
        ImageDraw.Draw(canvas).rectangle([x, y, x + w - 1, y + h - 1], outline=(200, 200, 200))
        ImageDraw.Draw(canvas).text((x + 8, y + h // 2), "(no case)", fill=(150, 150, 150))
    return canvas
