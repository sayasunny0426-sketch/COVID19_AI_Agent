"""Threshold rules fixed before the Test evaluation (D-057).

Primary        : Youden's J maximum on Validation; ties -> lowest threshold (keeps sensitivity).
Secondary      : highest threshold whose Validation sensitivity is still >= 0.80
                 (exploratory operating point; no clinical justification for 0.80 yet).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from covid_mortality.evaluation.metrics import (  # noqa: E402
    binary_rates, threshold_at_min_sensitivity, youden_threshold)


def test_youden_picks_the_separating_threshold():
    y = [0, 0, 1, 1]
    s = [0.1, 0.2, 0.8, 0.9]
    assert youden_threshold(y, s) == 0.8


def test_youden_tie_break_lowest_vs_highest():
    # scores 0.4 and 0.6 both give J = 0.5 here; 0.2 and 0.8 are worse
    y = np.array([0, 0, 1, 1, 0, 1])
    s = np.array([0.2, 0.4, 0.4, 0.6, 0.6, 0.8])
    js = {t: binary_rates(y, s, t)["sensitivity"] + binary_rates(y, s, t)["specificity"] - 1
          for t in np.unique(s)}
    tied = [t for t, j in js.items() if abs(j - max(js.values())) < 1e-12]
    low, high = youden_threshold(y, s, "lowest"), youden_threshold(y, s, "highest")
    assert low == min(tied) and high == max(tied), (tied, low, high)
    assert binary_rates(y, s, low)["sensitivity"] >= binary_rates(y, s, high)["sensitivity"]
    print(f"    tied thresholds {tied} -> lowest={low} highest={high}")


def test_youden_is_deterministic():
    rng = np.random.default_rng(3)
    y = (rng.random(128) < 0.133).astype(int)
    s = np.round(rng.random(128), 2)          # rounding creates ties on purpose
    assert youden_threshold(y, s) == youden_threshold(y, s)
    assert youden_threshold(y, s, "lowest") <= youden_threshold(y, s, "highest")


def test_secondary_threshold_is_the_highest_meeting_sensitivity():
    y = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    s = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.45, 0.4, 0.3, 0.2, 0.1])
    thr = threshold_at_min_sensitivity(y, s, 0.80)
    assert binary_rates(y, s, thr)["sensitivity"] >= 0.80
    higher = [t for t in np.unique(s) if t > thr]
    assert all(binary_rates(y, s, t)["sensitivity"] < 0.80 for t in higher), "not the highest"
    print(f"    threshold={thr} sens={binary_rates(y, s, thr)['sensitivity']:.2f} "
          f"spec={binary_rates(y, s, thr)['specificity']:.2f}")


def test_secondary_threshold_maximises_specificity_among_qualifying():
    rng = np.random.default_rng(11)
    y = (rng.random(128) < 0.133).astype(int)
    s = np.clip(0.2 + 0.5 * y + rng.normal(0, 0.15, 128), 0, 1)
    thr = threshold_at_min_sensitivity(y, s, 0.80)
    qualifying = [t for t in np.unique(s) if binary_rates(y, s, float(t))["sensitivity"] >= 0.80]
    specs = {t: binary_rates(y, s, float(t))["specificity"] for t in qualifying}
    assert abs(specs[thr] - max(specs.values())) < 1e-12
    print(f"    n_qualifying={len(qualifying)} chosen_spec={specs[thr]:.3f} "
          f"max_spec={max(specs.values()):.3f}")


def test_secondary_threshold_nan_when_unreachable():
    y = np.array([1, 0, 0, 0])
    s = np.array([0.1, 0.9, 0.8, 0.7])       # the only positive scores lowest
    thr = threshold_at_min_sensitivity(y, s, 0.80)
    assert not np.isnan(thr)                  # threshold 0.1 gives sensitivity 1.0
    assert np.isnan(threshold_at_min_sensitivity(np.array([0, 0, 0]), np.array([.1, .2, .3]), 0.8))


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
    print(f"\n{failures} failure(s)")
    sys.exit(1 if failures else 0)
