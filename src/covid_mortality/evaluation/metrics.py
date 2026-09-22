"""Evaluation metrics implemented on numpy only.

No scikit-learn dependency: the same code then runs unchanged on Colab, on another CUDA
machine and locally, and the numbers cannot drift with a library version. Ties are handled
with mid-ranks, so AUROC equals the Mann-Whitney U statistic.
"""
from __future__ import annotations

import numpy as np


def _check(y_true: np.ndarray, y_score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true).astype(float).ravel()
    y_score = np.asarray(y_score).astype(float).ravel()
    if y_true.shape != y_score.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_score.shape}")
    if not np.isin(y_true, (0.0, 1.0)).all():
        raise ValueError("y_true must be 0/1")
    return y_true, y_score


def roc_auc(y_true, y_score) -> float:
    """Mann-Whitney U / rank based AUROC with mid-ranks for ties."""
    y_true, y_score = _check(y_true, y_score)
    n_pos, n_neg = int(y_true.sum()), int((1 - y_true).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(y_score, kind="mergesort")
    ranks = np.empty(len(y_score), dtype=float)
    sorted_scores = y_score[order]
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision(y_true, y_score) -> float:
    """Average precision: sum_k (R_k - R_{k-1}) * P_k over distinct thresholds.

    Step-wise estimator, identical to sklearn's average_precision_score (NOT the
    trapezoidal area under the PR curve). Samples sharing a score form one threshold
    group, otherwise tied scores would be credited an arbitrary within-group order.
    """
    y_true, y_score = _check(y_true, y_score)
    n_pos = y_true.sum()
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-y_score, kind="mergesort")
    y, s = y_true[order], y_score[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    last_of_group = np.append(np.diff(s) != 0, True)  # keep the last index of each tied group
    tp, fp = tp[last_of_group], fp[last_of_group]
    precision = tp / np.maximum(tp + fp, 1e-12)
    recall = tp / n_pos
    return float(np.sum(np.diff(np.concatenate([[0.0], recall])) * precision))


def roc_curve(y_true, y_score) -> dict:
    """FPR/TPR points (for plotting and for saving the curve data)."""
    y_true, y_score = _check(y_true, y_score)
    order = np.argsort(-y_score, kind="mergesort")
    y, s = y_true[order], y_score[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    keep = np.append(np.diff(s) != 0, True)
    tpr = np.concatenate([[0.0], tp[keep] / max(y_true.sum(), 1)])
    fpr = np.concatenate([[0.0], fp[keep] / max((1 - y_true).sum(), 1)])
    thr = np.concatenate([[np.inf], s[keep]])
    return {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "thresholds": thr.tolist()}


def brier_score(y_true, y_prob) -> float:
    y_true, y_prob = _check(y_true, y_prob)
    return float(np.mean((y_prob - y_true) ** 2))


def accuracy(y_true, y_prob, threshold: float = 0.5) -> float:
    y_true, y_prob = _check(y_true, y_prob)
    return float(np.mean((y_prob >= threshold).astype(float) == y_true))


def binary_rates(y_true, y_prob, threshold: float) -> dict:
    """Sensitivity / specificity / PPV / NPV at a fixed threshold."""
    y_true, y_prob = _check(y_true, y_prob)
    pred = (y_prob >= threshold).astype(float)
    tp = float(((pred == 1) & (y_true == 1)).sum())
    tn = float(((pred == 0) & (y_true == 0)).sum())
    fp = float(((pred == 1) & (y_true == 0)).sum())
    fn = float(((pred == 0) & (y_true == 1)).sum())
    div = lambda a, b: float(a / b) if b > 0 else float("nan")  # noqa: E731
    return {"threshold": float(threshold), "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "sensitivity": div(tp, tp + fn), "specificity": div(tn, tn + fp),
            "ppv": div(tp, tp + fp), "npv": div(tn, tn + fn),
            "accuracy": div(tp + tn, tp + tn + fp + fn)}


def youden_threshold(y_true, y_score, tie_break: str = "lowest") -> float:
    """Threshold maximising Youden's J = sensitivity + specificity - 1 (Validation only).

    tie_break fixes the choice when several thresholds share the maximal J. It must be set
    before the evaluation is run:
      "lowest"  -> the smallest such threshold: at equal J this keeps sensitivity highest,
                   i.e. fewest missed deaths (the default for this study)
      "highest" -> the largest such threshold: at equal J this keeps specificity highest
    """
    if tie_break not in ("lowest", "highest"):
        raise ValueError(f"unknown tie_break: {tie_break}")
    y_true, y_score = _check(y_true, y_score)
    candidates = np.unique(y_score)
    js = np.array([binary_rates(y_true, y_score, float(t))["sensitivity"]
                   + binary_rates(y_true, y_score, float(t))["specificity"] - 1 for t in candidates])
    best = candidates[np.isclose(js, js.max(), rtol=0, atol=1e-12)]
    return float(best.min() if tie_break == "lowest" else best.max())


def threshold_at_min_sensitivity(y_true, y_score, min_sensitivity: float = 0.80) -> float:
    """Highest threshold whose sensitivity is still >= min_sensitivity (Validation only).

    Sensitivity is non-increasing in the threshold, so taking the highest qualifying value
    keeps the required sensitivity while making specificity as high as possible.
    Returns NaN if no threshold reaches the required sensitivity.
    """
    y_true, y_score = _check(y_true, y_score)
    qualifying = [float(t) for t in np.unique(y_score)
                  if binary_rates(y_true, y_score, float(t))["sensitivity"] >= min_sensitivity]
    return max(qualifying) if qualifying else float("nan")


def bootstrap_ci(y_true, y_score, metric=roc_auc, n_boot: int = 2000, alpha: float = 0.05,
                 stratified: bool = True, seed: int = 12345) -> dict:
    """Patient-level bootstrap CI.

    Stratified resampling keeps the class composition of the set, which stabilises the
    interval when the number of events is small (Test has 17 deaths).
    """
    y_true, y_score = _check(y_true, y_score)
    rng = np.random.default_rng(seed)
    idx_pos = np.flatnonzero(y_true == 1)
    idx_neg = np.flatnonzero(y_true == 0)
    n = len(y_true)
    stats = []
    for _ in range(n_boot):
        if stratified:
            take = np.concatenate([rng.choice(idx_pos, len(idx_pos), replace=True),
                                   rng.choice(idx_neg, len(idx_neg), replace=True)])
        else:
            take = rng.integers(0, n, n)
        value = metric(y_true[take], y_score[take])
        if np.isfinite(value):
            stats.append(value)
    stats = np.array(stats)
    return {"point": float(metric(y_true, y_score)),
            "ci_low": float(np.percentile(stats, 100 * alpha / 2)),
            "ci_high": float(np.percentile(stats, 100 * (1 - alpha / 2))),
            "n_boot": int(len(stats)), "stratified": bool(stratified), "seed": int(seed)}


def calibration_bins(y_true, y_prob, n_bins: int = 10,
                     scheme: str = "equal_width") -> list[dict]:
    """Bin predicted probabilities and report the gap between prediction and outcome.

    `scheme` must be stated, because the two are not interchangeable:

      "equal_width"  edges at np.linspace(0, 1, n_bins + 1). Bins cover the probability
                     scale evenly, so a bin can hold very few patients -- or none, in which
                     case it contributes nothing.
      "equal_count"  edges at the quantiles of y_prob. Every bin holds roughly the same
                     number of patients, but the bins are narrow where predictions cluster.

    Reporting an ECE without its scheme and bin count makes it incomparable across studies.
    """
    y_true, y_prob = _check(y_true, y_prob)
    if scheme == "equal_width":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    elif scheme == "equal_count":
        edges = np.quantile(y_prob, np.linspace(0, 1, n_bins + 1))
        edges[0], edges[-1] = -np.inf, np.inf
    else:
        raise ValueError(f"unknown binning scheme: {scheme!r}")
    ids = np.digitize(y_prob, edges[1:-1], right=True)
    out = []
    for b in range(n_bins):
        m = ids == b
        if not m.sum():
            continue
        out.append({"bin": b, "n": int(m.sum()),
                    "mean_predicted": float(y_prob[m].mean()),
                    "observed_rate": float(y_true[m].mean()),
                    "events": int(y_true[m].sum())})
    return out


def expected_calibration_error(y_true, y_prob, n_bins: int = 10,
                               scheme: str = "equal_width") -> float:
    """Sum over bins of (bin share) x |mean predicted - observed rate|.

    The aggregation is the standard one; the result depends on `n_bins` and `scheme`, which
    is why both are explicit arguments with no silent default change.
    """
    y_true, y_prob = _check(y_true, y_prob)
    bins = calibration_bins(y_true, y_prob, n_bins=n_bins, scheme=scheme)
    n = len(y_true)
    return float(sum(b["n"] / n * abs(b["mean_predicted"] - b["observed_rate"]) for b in bins))
