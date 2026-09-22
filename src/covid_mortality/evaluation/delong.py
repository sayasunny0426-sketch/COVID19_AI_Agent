"""Paired DeLong test for two correlated ROC curves (same patients, two models).

Algorithm
---------
DeLong, DeLong & Clarke-Pearson (1988), computed with the fast O(N log N) formulation of
Sun & Xu (2014), "Fast Implementation of DeLong's Algorithm for Comparing the Areas Under
Correlated Receiver Operating Characteristic Curves", IEEE Signal Processing Letters 21(11).
Midranks handle tied predicted probabilities, so the AUC computed here equals the
Mann-Whitney U estimate used elsewhere in this project (evaluation/metrics.roc_auc).

Implemented on numpy only (the normal CDF uses math.erf), so no scipy/sklearn version can
change the reported p-value.

The test is two-sided and unadjusted; any multiplicity correction is applied by the caller.
"""
from __future__ import annotations

import math

import numpy as np


def _midrank(x: np.ndarray) -> np.ndarray:
    """Midranks (ties share the average rank)."""
    order = np.argsort(x, kind="mergesort")
    sorted_x = x[order]
    n = len(x)
    ranks_sorted = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        ranks_sorted[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    ranks = np.empty(n, dtype=float)
    ranks[order] = ranks_sorted
    return ranks


def _fast_delong(preds_positives_first: np.ndarray, n_pos: int) -> tuple[np.ndarray, np.ndarray]:
    """Sun & Xu (2014). preds_positives_first: (n_models, n_samples) with positives first."""
    k, total = preds_positives_first.shape
    n_neg = total - n_pos
    pos = preds_positives_first[:, :n_pos]
    neg = preds_positives_first[:, n_pos:]
    tx = np.empty((k, n_pos))
    ty = np.empty((k, n_neg))
    tz = np.empty((k, total))
    for r in range(k):
        tx[r] = _midrank(pos[r])
        ty[r] = _midrank(neg[r])
        tz[r] = _midrank(preds_positives_first[r])
    aucs = tz[:, :n_pos].sum(axis=1) / n_pos / n_neg - (n_pos + 1.0) / (2.0 * n_neg)
    v01 = (tz[:, :n_pos] - tx) / n_neg            # structural components over positives
    v10 = 1.0 - (tz[:, n_pos:] - ty) / n_pos      # structural components over negatives
    sx = np.cov(v01, ddof=1)
    sy = np.cov(v10, ddof=1)
    cov = np.atleast_2d(sx) / n_pos + np.atleast_2d(sy) / n_neg
    return aucs, cov


def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def delong_roc_test(y_true, prob_a, prob_b, alpha: float = 0.05) -> dict:
    """Compare AUROC(prob_a) - AUROC(prob_b) on the same patients.

    Returns the two AUCs, their difference (a - b), the standard error, a two-sided
    unadjusted p-value and a Wald confidence interval for the difference.
    """
    y = np.asarray(y_true).astype(float).ravel()
    a = np.asarray(prob_a).astype(float).ravel()
    b = np.asarray(prob_b).astype(float).ravel()
    if not (len(y) == len(a) == len(b)):
        raise ValueError("inputs must have the same length")
    if not np.isin(y, (0.0, 1.0)).all():
        raise ValueError("y_true must be 0/1")
    n_pos = int(y.sum())
    if n_pos == 0 or n_pos == len(y):
        raise ValueError("both classes must be present")

    order = np.argsort(-y, kind="mergesort")       # positives first, order otherwise preserved
    preds = np.vstack((a, b))[:, order]
    aucs, cov = _fast_delong(preds, n_pos)
    var_diff = float(cov[0, 0] + cov[1, 1] - 2.0 * cov[0, 1])
    diff = float(aucs[0] - aucs[1])
    z_crit = 1.959963984540054 if abs(alpha - 0.05) < 1e-12 else _z_for(alpha)
    if var_diff <= 0:                               # identical (or perfectly concordant) models
        return {"auc_a": float(aucs[0]), "auc_b": float(aucs[1]), "difference": diff,
                "se": 0.0, "z": float("nan"), "p_value": 1.0 if diff == 0 else 0.0,
                "ci_low": diff, "ci_high": diff, "n": int(len(y)), "events": n_pos,
                "alpha": alpha, "note": "degenerate variance (models give identical rankings)"}
    se = math.sqrt(var_diff)
    z = diff / se
    return {"auc_a": float(aucs[0]), "auc_b": float(aucs[1]), "difference": diff, "se": se,
            "z": z, "p_value": 2.0 * (1.0 - _normal_cdf(abs(z))),
            "ci_low": diff - z_crit * se, "ci_high": diff + z_crit * se,
            "n": int(len(y)), "events": n_pos, "alpha": alpha,
            "method": "DeLong et al. 1988; fast implementation of Sun & Xu 2014 (midranks)"}


def _z_for(alpha: float) -> float:
    """Inverse normal CDF for 1 - alpha/2 (bisection; no scipy dependency)."""
    target = 1.0 - alpha / 2.0
    lo, hi = 0.0, 10.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if _normal_cdf(mid) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2
