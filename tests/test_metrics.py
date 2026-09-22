"""Unit tests for the numpy metrics (run: python -m pytest tests/ -q, or python tests/test_metrics.py)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from covid_mortality.evaluation.metrics import (  # noqa: E402
    accuracy, average_precision, binary_rates, bootstrap_ci, brier_score, roc_auc, roc_curve,
    youden_threshold)


def test_auc_perfect_and_reversed():
    y = np.array([0, 0, 1, 1])
    assert roc_auc(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert roc_auc(y, np.array([0.9, 0.8, 0.2, 0.1])) == 0.0


def test_auc_ties_use_midranks():
    # all scores equal -> AUROC 0.5
    assert roc_auc([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]) == 0.5
    # one tie between a positive and a negative
    assert abs(roc_auc([0, 1], [0.5, 0.5]) - 0.5) < 1e-12


def test_auc_known_value():
    y = [0, 0, 1, 1, 0, 1]
    s = [0.1, 0.4, 0.35, 0.8, 0.2, 0.6]
    # 3 positives x 3 negatives = 9 pairs; positives beat negatives in 8 of them
    assert abs(roc_auc(y, s) - 8 / 9) < 1e-12


def test_average_precision_perfect():
    assert abs(average_precision([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) - 1.0) < 1e-12


def test_average_precision_known():
    # ranked: 1,0,1,0 -> precision at recalls 0.5 and 1.0 = 1.0 and 2/3
    ap = average_precision([1, 0, 1, 0], [0.9, 0.8, 0.7, 0.6])
    assert abs(ap - (0.5 * 1.0 + 0.5 * (2 / 3))) < 1e-12


def test_roc_curve_monotone():
    c = roc_curve([0, 1, 0, 1], [0.2, 0.9, 0.4, 0.7])
    assert c["fpr"][0] == 0.0 and c["tpr"][0] == 0.0
    assert abs(c["fpr"][-1] - 1.0) < 1e-12 and abs(c["tpr"][-1] - 1.0) < 1e-12
    assert all(b >= a - 1e-12 for a, b in zip(c["fpr"], c["fpr"][1:]))


def test_rates_and_threshold():
    r = binary_rates([0, 0, 1, 1], [0.1, 0.6, 0.4, 0.9], 0.5)
    assert (r["tp"], r["fp"], r["tn"], r["fn"]) == (1.0, 1.0, 1.0, 1.0)
    assert abs(r["sensitivity"] - 0.5) < 1e-12 and abs(r["specificity"] - 0.5) < 1e-12
    assert youden_threshold([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 0.8


def test_brier_and_accuracy():
    assert abs(brier_score([1, 0], [1.0, 0.0])) < 1e-12
    assert accuracy([1, 0, 1, 0], [0.9, 0.1, 0.4, 0.2]) == 0.75


def test_bootstrap_ci_reproducible_and_bracketing():
    rng = np.random.default_rng(0)
    y = np.concatenate([np.ones(20), np.zeros(80)])
    s = np.concatenate([rng.normal(1.0, 1, 20), rng.normal(0.0, 1, 80)])
    a = bootstrap_ci(y, s, n_boot=300, seed=7)
    b = bootstrap_ci(y, s, n_boot=300, seed=7)
    assert a == b                      # same seed -> identical
    assert a["ci_low"] <= a["point"] <= a["ci_high"]
    assert a["stratified"] is True


def test_bootstrap_stratified_keeps_events():
    y = np.array([1] * 3 + [0] * 97)
    s = np.linspace(0, 1, 100)
    out = bootstrap_ci(y, s, n_boot=50, seed=1)
    assert out["n_boot"] == 50        # no resample degenerated to zero events


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"[PASS] {name}")
            except AssertionError as e:
                failures += 1
                print(f"[FAIL] {name}: {e}")
    print(f"\n{failures} failure(s)")
    sys.exit(1 if failures else 0)
