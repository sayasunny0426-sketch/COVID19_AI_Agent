"""Final reference check of the numpy metrics before training.

Two independent references:
  (1) hand-computed values on small datasets (no library involved);
  (2) scikit-learn, if it is installed (skipped otherwise, so the suite still runs on Colab
      without pinning a sklearn version).

DEFINITION FIXED HERE AND USED EVERYWHERE AFTERWARDS
  AUROC = Mann-Whitney U statistic with mid-ranks for ties (= sklearn roc_auc_score).
  AUPRC = **Average Precision**: sum over thresholds of (recall_k - recall_{k-1}) * precision_k
          (= sklearn average_precision_score). This is a step-wise estimator and is NOT the
          trapezoidal area under the PR curve (sklearn's auc(recall, precision)), which
          interpolates between points and is optimistically biased with few events.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from covid_mortality.evaluation.metrics import average_precision, roc_auc  # noqa: E402

try:
    from sklearn.metrics import (average_precision_score, precision_recall_curve, auc,
                                 roc_auc_score)
    HAVE_SKLEARN = True
except Exception:  # pragma: no cover
    HAVE_SKLEARN = False

CASES = {
    # name: (y_true, y_score, hand_auroc, hand_average_precision)
    "separable": ([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], 1.0, 1.0),
    "reversed": ([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1], 0.0, 0.5 * (1 / 3) + 0.5 * 0.5),
    "all_tied": ([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5], 0.5, 0.5),
    "8of9_pairs": ([0, 0, 1, 1, 0, 1], [0.1, 0.4, 0.35, 0.8, 0.2, 0.6], 8 / 9,
                   (1 / 3) * 1.0 + (1 / 3) * 1.0 + (1 / 3) * 0.75),
    # pairs: (0.9>0.8) (0.9>0.6) (0.7<0.8) (0.7>0.6) -> 3 of 4 = 0.75
    "alternating": ([1, 0, 1, 0], [0.9, 0.8, 0.7, 0.6], 0.75, 0.5 * 1.0 + 0.5 * (2 / 3)),
    "one_event": ([0, 0, 0, 1], [0.1, 0.2, 0.3, 0.25], 2 / 3, 0.5),
}


def test_hand_computed_values():
    for name, (y, s, auroc_ref, ap_ref) in CASES.items():
        got_auc, got_ap = roc_auc(y, s), average_precision(y, s)
        assert abs(got_auc - auroc_ref) < 1e-12, f"{name}: AUROC {got_auc} != {auroc_ref}"
        assert abs(got_ap - ap_ref) < 1e-12, f"{name}: AP {got_ap} != {ap_ref}"
        print(f"    {name:12s} AUROC={got_auc:.6f} (ref {auroc_ref:.6f})  "
              f"AP={got_ap:.6f} (ref {ap_ref:.6f})")


def test_matches_sklearn_on_random_data():
    if not HAVE_SKLEARN:
        print("    sklearn not installed -> skipped")
        return
    rng = np.random.default_rng(0)
    for i, (n, p_event) in enumerate([(50, 0.1), (128, 0.133), (1021, 0.132), (300, 0.5)]):
        y = (rng.random(n) < p_event).astype(int)
        if y.sum() in (0, n):
            continue
        s = np.clip(0.5 + 0.25 * y + rng.normal(0, 0.3, n), 0, 1)
        if i == 3:  # force ties, the case where estimators usually disagree
            s = np.round(s, 1)
        d_auc = abs(roc_auc(y, s) - roc_auc_score(y, s))
        d_ap = abs(average_precision(y, s) - average_precision_score(y, s))
        assert d_auc < 1e-10, f"n={n}: AUROC differs by {d_auc}"
        assert d_ap < 1e-10, f"n={n}: AP differs by {d_ap} (ties: {n - len(np.unique(s))})"
        print(f"    n={n:5d} events={int(y.sum()):4d}  AUROC diff="
              f"{abs(roc_auc(y, s) - roc_auc_score(y, s)):.2e}  AP diff="
              f"{abs(average_precision(y, s) - average_precision_score(y, s)):.2e}")


def test_average_precision_is_not_trapezoidal_pr_auc():
    """Document the choice: AP, not the trapezoidal area under the PR curve."""
    if not HAVE_SKLEARN:
        print("    sklearn not installed -> skipped")
        return
    y = np.array([1, 0, 1, 0, 0, 1, 0, 0, 0, 0])
    s = np.array([0.9, 0.85, 0.7, 0.6, 0.55, 0.5, 0.4, 0.3, 0.2, 0.1])
    ap = average_precision(y, s)
    prec, rec, _ = precision_recall_curve(y, s)
    trapezoid = auc(rec, prec)
    assert abs(ap - average_precision_score(y, s)) < 1e-12
    print(f"    average_precision={ap:.6f}  trapezoidal_pr_auc={trapezoid:.6f} "
          f"(difference {abs(ap - trapezoid):.6f}) -> AP is used")


def test_empty_class_returns_nan():
    assert np.isnan(roc_auc([0, 0, 0], [0.1, 0.2, 0.3]))
    assert np.isnan(average_precision([0, 0, 0], [0.1, 0.2, 0.3]))
    print("    degenerate cases return NaN rather than a misleading number")


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
    print(f"\nsklearn available: {HAVE_SKLEARN}\n{failures} failure(s)")
    sys.exit(1 if failures else 0)
