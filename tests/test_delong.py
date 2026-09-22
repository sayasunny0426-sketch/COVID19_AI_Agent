"""Unit tests for the paired DeLong implementation."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from covid_mortality.evaluation.delong import delong_roc_test  # noqa: E402
from covid_mortality.evaluation.metrics import bootstrap_ci, roc_auc  # noqa: E402


def _data(seed=0, n=128, n_events=17, sep_a=1.2, sep_b=0.8):
    rng = np.random.default_rng(seed)
    y = np.zeros(n, dtype=int)
    y[rng.choice(n, n_events, replace=False)] = 1
    a = 1 / (1 + np.exp(-(rng.normal(0, 1, n) + sep_a * y)))
    b = 1 / (1 + np.exp(-(rng.normal(0, 1, n) + sep_b * y)))
    return y, a, b


def test_aucs_match_project_metric():
    y, a, b = _data()
    r = delong_roc_test(y, a, b)
    assert abs(r["auc_a"] - roc_auc(y, a)) < 1e-12, (r["auc_a"], roc_auc(y, a))
    assert abs(r["auc_b"] - roc_auc(y, b)) < 1e-12
    print(f"    AUC a={r['auc_a']:.6f} b={r['auc_b']:.6f} (match metrics.roc_auc)")


def test_identical_models_give_zero_difference():
    y, a, _ = _data()
    r = delong_roc_test(y, a, a)
    assert abs(r["difference"]) < 1e-12 and r["p_value"] == 1.0
    print("    identical predictions -> difference 0, p = 1")


def test_sign_and_symmetry():
    y, a, b = _data()
    ab = delong_roc_test(y, a, b)
    ba = delong_roc_test(y, b, a)
    assert abs(ab["difference"] + ba["difference"]) < 1e-12
    assert abs(ab["p_value"] - ba["p_value"]) < 1e-12
    assert abs(ab["se"] - ba["se"]) < 1e-12
    print(f"    diff {ab['difference']:+.6f} vs {ba['difference']:+.6f}, p={ab['p_value']:.4f}")


def test_ci_brackets_difference_and_matches_p():
    y, a, b = _data(seed=3)
    r = delong_roc_test(y, a, b)
    assert r["ci_low"] <= r["difference"] <= r["ci_high"]
    crosses_zero = r["ci_low"] <= 0 <= r["ci_high"]
    assert crosses_zero == (r["p_value"] >= 0.05), (r["p_value"], r["ci_low"], r["ci_high"])
    print(f"    diff {r['difference']:+.4f} CI [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] "
          f"p={r['p_value']:.4f}")


def test_se_is_close_to_bootstrap_se():
    """Sanity check against an independent estimate of the same quantity."""
    y, a, b = _data(seed=5, n=400, n_events=80)
    r = delong_roc_test(y, a, b)
    rng = np.random.default_rng(1)
    idx_pos, idx_neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    diffs = []
    for _ in range(1500):
        take = np.concatenate([rng.choice(idx_pos, len(idx_pos), True),
                               rng.choice(idx_neg, len(idx_neg), True)])
        diffs.append(roc_auc(y[take], a[take]) - roc_auc(y[take], b[take]))
    boot_se = float(np.std(diffs, ddof=1))
    assert abs(r["se"] - boot_se) < 0.5 * max(r["se"], boot_se), (r["se"], boot_se)
    print(f"    DeLong SE {r['se']:.4f} vs bootstrap SE {boot_se:.4f}")


def test_ties_are_handled():
    y, a, b = _data(seed=9)
    a_t, b_t = np.round(a, 1), np.round(b, 1)          # many ties
    r = delong_roc_test(y, a_t, b_t)
    assert abs(r["auc_a"] - roc_auc(y, a_t)) < 1e-12
    assert abs(r["auc_b"] - roc_auc(y, b_t)) < 1e-12
    print(f"    with ties: AUC a={r['auc_a']:.4f} b={r['auc_b']:.4f} p={r['p_value']:.4f}")


def test_requires_both_classes():
    for y in (np.zeros(10, dtype=int), np.ones(10, dtype=int)):
        try:
            delong_roc_test(y, np.linspace(0, 1, 10), np.linspace(0, 1, 10))
            raise AssertionError("should have raised")
        except ValueError:
            pass
    print("    single-class input raises ValueError")


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
