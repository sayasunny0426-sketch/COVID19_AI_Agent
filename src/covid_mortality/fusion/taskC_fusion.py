"""Task C: multimodal fusion of the frozen Task A (CXR) and Task B (clinical) models.

Nothing here retrains or modifies Task A or Task B. Both components are used only through the
predictions they already produced, loaded read-only and identified by SHA256.

Fusion strategies (fixed before any Validation search, decision_log D-085):
    primary       probability-space weighted average
                      p_fused = w * p_clinical + (1 - w) * p_cxr
                  one free parameter, chosen on Validation.
    secondary     logit-space weighted average, logistic stacking, and the same primary form
                  with the XGBoost or MLP clinical component. These are computed and saved but
                  do NOT compete with the primary strategy for selection.

Why only one free parameter: neither Task A nor Task B saved Training-set predictions, and
neither can be re-run per fold without retraining a frozen model, so the only data on which a
combiner can be fitted is the Validation set -- 128 patients with 17 deaths. A single weight is
what that supports.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

CLINICAL_COLUMNS = {"lr": "prob_lr", "xgb": "prob_xgb", "mlp": "prob_mlp",
                    "xgb_native": "prob_xgb_native"}
TEST_CLINICAL_COLUMNS = {"lr": "prob_clinical_lr", "xgb": "prob_clinical_xgboost",
                         "mlp": "prob_clinical_mlp",
                         "xgb_native": "prob_clinical_xgboost_native"}
EPS = 1e-6          # keeps the logit transform finite at probabilities of 0 or 1


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def logit(p: np.ndarray, eps: float = EPS) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))


def fuse(p_clinical, p_cxr, w: float, space: str = "probability") -> np.ndarray:
    """Weighted average of two predicted probabilities. w is the clinical weight."""
    p_clinical = np.asarray(p_clinical, dtype=float)
    p_cxr = np.asarray(p_cxr, dtype=float)
    if space == "probability":
        return w * p_clinical + (1.0 - w) * p_cxr
    if space == "logit":
        return sigmoid(w * logit(p_clinical) + (1.0 - w) * logit(p_cxr))
    raise ValueError(f"unknown fusion space: {space!r}")


@dataclass
class Inputs:
    """Aligned Task A and Task B predictions for one split, in a fixed patient order."""
    split: str
    frame: pd.DataFrame            # subject_id, true_label, prob_cxr, prob_lr/xgb/mlp...
    order: list
    sources: dict                  # path -> sha256

    @property
    def y(self) -> np.ndarray:
        return self.frame.true_label.to_numpy(int)

    def cxr(self) -> np.ndarray:
        return self.frame.prob_cxr.to_numpy(float)

    def clinical(self, name: str) -> np.ndarray:
        return self.frame[f"prob_{name}"].to_numpy(float)


def load_split(split: str, cxr_path: Path, clinical_path: Path,
               split_manifest: pd.DataFrame, clinical_columns: dict) -> Inputs:
    """Merge the two frozen prediction files on subject_id, in split-manifest order.

    The two files do not share a row order, so the merge key is explicit and the resulting
    order is taken from the split manifest rather than from either input file.
    """
    cxr = pd.read_csv(cxr_path, dtype={"subject_id": str}, encoding="utf-8-sig")
    clin = pd.read_csv(clinical_path, dtype={"subject_id": str}, encoding="utf-8-sig")
    order = [str(x) for x in split_manifest[split_manifest.split == split]["Subject ID"]]

    for name, df in (("CXR", cxr), ("clinical", clin)):
        if df.subject_id.duplicated().any():
            raise ValueError(f"{name} predictions contain duplicate subject_id")
        if set(df.subject_id) != set(order):
            raise ValueError(f"{name} predictions do not cover exactly the {split} split")

    cols = {"subject_id": "subject_id", "true_label": "true_label"}
    a = cxr[["subject_id", "true_label", "prob"]].rename(columns={"prob": "prob_cxr"})
    keep = ["subject_id", "true_label"] + [c for c in clinical_columns.values()
                                           if c in clin.columns]
    b = clin[keep].rename(columns={v: f"prob_{k}" for k, v in clinical_columns.items()})
    m = a.merge(b, on="subject_id", suffixes=("_cxr", "_clin"))
    if len(m) != len(order):
        raise ValueError(f"merge produced {len(m)} rows for {len(order)} patients")
    lab_a, lab_b = m.true_label_cxr.to_numpy(int), m.true_label_clin.to_numpy(int)
    if not np.array_equal(lab_a, lab_b):
        raise ValueError("true_label disagrees between the CXR and clinical prediction files")
    m = m.drop(columns=["true_label_clin"]).rename(columns={"true_label_cxr": "true_label"})
    m = m.set_index("subject_id").loc[order].reset_index()

    probs = [c for c in m.columns if c.startswith("prob_")]
    if m[probs].isna().any().any():
        raise ValueError("missing predictions after the merge")
    if not ((m[probs] >= 0) & (m[probs] <= 1)).all().all():
        raise ValueError("predicted probabilities outside [0, 1]")
    return Inputs(split, m, order,
                  {str(cxr_path): sha256(cxr_path), str(clinical_path): sha256(clinical_path)})


def weight_grid(step: float = 0.05) -> np.ndarray:
    """w = 0 is CXR alone, w = 1 is the clinical model alone; both endpoints are informative."""
    n = int(round(1.0 / step))
    return np.round(np.linspace(0.0, 1.0, n + 1), 10)


def fit_logistic_stack(p_clinical, p_cxr, y, seed: int = 42):
    """Exploratory only: 3 parameters fitted on 17 Validation events."""
    from sklearn.linear_model import LogisticRegression
    X = np.column_stack([logit(p_clinical), logit(p_cxr)])
    # C=inf is the unpenalised fit; `penalty=None` is deprecated from scikit-learn 1.8
    m = LogisticRegression(C=np.inf, max_iter=5000, random_state=seed).fit(X, y)
    return m, (lambda pc, px: m.predict_proba(
        np.column_stack([logit(pc), logit(px)]))[:, 1])
