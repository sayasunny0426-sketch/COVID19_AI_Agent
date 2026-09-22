"""Task B step 2: profile the candidate variables on Training and fix the preprocessing spec.

Training set only. The spec produced here (imputation value per variable, transformation,
scaling, missing indicators, implausible-value handling) is written to a JSON that later
scripts consume, so the fitted models cannot silently use a different specification than the
one documented -- the defect found in the previous Task B implementation.

Decision rule for imputation and transformation, pre-registered BEFORE looking at any
outcome association (faculty feedback §4: do not decide from a Shapiro-Wilk p-value alone):

    skew >= +1.0                 -> strongly RIGHT-skewed    -> log1p, then median imputation
    |skew| < 0.5                 -> roughly symmetric        -> mean imputation, no transform
    otherwise (incl. left-skew)  -> mildly skewed / left     -> median imputation, no transform

    log1p is applied only to right skew. A log transform does not correct left skew (it makes
    it worse: SpO2 goes from -2.10 to -2.46), and the reflection transform that would correct
    it flips the sign of the coefficient, which destroys the clinical reading of the odds
    ratio. Left-skewed variables therefore stay on their original scale.

    Shapiro-Wilk W and p are recorded as supporting evidence, never as the sole criterion.
    Histograms and Q-Q plots are written for every continuous variable so the shape can be
    inspected rather than inferred from a statistic.

Missing indicators are added for variables whose Training missing rate exceeds 10%, because
missingness here reflects test-ordering behaviour (ER 55-78% vs inpatient 0.4-14%) and is
itself associated with the outcome -- it is information, not noise.

Usage:
    python scripts/22_taskB_preprocess_spec.py --project .
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy import stats  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from covid_mortality.features import taskB_clinical as tb  # noqa: E402

MISSING_INDICATOR_THRESHOLD = 0.10
SKEW_SYMMETRIC, SKEW_STRONG = 0.5, 1.0


def decide(skew: float) -> tuple[str, str, str]:
    """(transform, imputation, rationale) from the pre-registered rule above."""
    if skew >= SKEW_STRONG:
        return ("log1p", "median",
                f"歪度 {skew:+.2f} で強い右裾を持つ。線形モデル（LR）と MLP のために log1p で"
                f"対数化し、補完は極端値に頑健な中央値を用いる")
    if abs(skew) < SKEW_SYMMETRIC:
        return ("none", "mean",
                f"歪度 {skew:+.2f} で左右対称に近く、外れ値の影響が小さい。"
                f"分布の中心を効率よく表す平均値を用いる")
    if skew <= -SKEW_STRONG:
        return ("none", "median",
                f"歪度 {skew:+.2f} で左裾が長い。log 変換は左歪みを悪化させ（適用すると "
                f"{skew:+.2f} → さらに負側）、補正のための反転変換は係数の符号が反転して"
                f"臨床的な解釈を失わせるため、変換しない。補完は中央値")
    return ("none", "median",
            f"歪度 {skew:+.2f} で軽度に歪む。極端値の影響を受けにくい中央値を用いる。"
            f"変換するほどの歪みではない")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=".")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    project = Path(args.project).resolve()
    out = Path(args.out) if args.out else project / "results/taskB/preprocessing"
    (out / "figures").mkdir(parents=True, exist_ok=True)

    df = tb.load_development(project)   # Training + Validation only
    train = tb.training_only(df)
    print(f"Training set only: {len(train)} patients, {int(train.y.sum())} deaths")

    # implausible values -> missing, before any statistic is computed
    fixed = 0
    for col, (lo, hi) in tb.IMPLAUSIBLE_BOUNDS.items():
        if col not in train.columns:
            continue
        v = pd.to_numeric(train[col], errors="coerce")
        bad = (v < lo) | (v > hi)
        if bad.any():
            train.loc[bad, col] = np.nan
            fixed += int(bad.sum())
    print(f"physiologically impossible values set to missing before profiling: {fixed}")

    continuous = [v for v in tb.CANDIDATES if v.kind == "continuous"]
    rows, spec_cont = [], {}
    for v in continuous:
        s = pd.to_numeric(train[v.column], errors="coerce")
        obs = s.dropna()
        skew = float(stats.skew(obs, bias=False))
        # Shapiro-Wilk is capped at 5000 observations; ours are far below that
        w, p = stats.shapiro(obs)
        transform, impute, rationale = decide(skew)
        tv = np.log1p(obs) if transform == "log1p" else obs
        skew_after = float(stats.skew(tv, bias=False))
        w_after, p_after = stats.shapiro(tv)
        fill = float(tv.median() if impute == "median" else tv.mean())
        indicator = float(s.isna().mean()) > MISSING_INDICATOR_THRESHOLD

        rows.append({
            "variable": v.name, "raw_column": v.column, "domain": v.domain,
            "n_observed": int(len(obs)), "missing_rate_train": round(float(s.isna().mean()), 4),
            "mean": round(float(obs.mean()), 4), "median": round(float(obs.median()), 4),
            "sd": round(float(obs.std()), 4),
            "iqr": round(float(obs.quantile(.75) - obs.quantile(.25)), 4),
            "min": round(float(obs.min()), 4), "p1": round(float(obs.quantile(.01)), 4),
            "p99": round(float(obs.quantile(.99)), 4), "max": round(float(obs.max()), 4),
            "skewness": round(skew, 4), "shapiro_W": round(float(w), 4),
            "shapiro_p": f"{p:.3e}",
            "transform": transform, "skewness_after": round(skew_after, 4),
            "shapiro_W_after": round(float(w_after), 4), "shapiro_p_after": f"{p_after:.3e}",
            "imputation": impute, "imputation_value_on_transformed_scale": round(fill, 6),
            "missing_indicator": indicator,
            "rationale": rationale,
            "indicator_rationale": (
                "Training 欠測率 >10%。欠測は検査オーダー行動（ER と入院で大きく異なる）を反映し、"
                "欠測の有無自体が予後情報を持ちうるため指示変数を追加する" if indicator else
                "Training 欠測率 <=10% のため指示変数は追加しない（パラメータ節約）"),
            "evidence_source": "data（Training の分布）＋ methodological consideration",
            "dataset_used_for_decision": "Training",
            "status": "fixed（Test 使用前は再検討可）",
        })
        spec_cont[v.name] = {"column": v.column, "transform": transform,
                             "imputation": impute, "fill_value": fill,
                             "missing_indicator": indicator}

        # figures: histogram (raw and transformed) + Q-Q plot, so the shape is inspectable
        fig, ax = plt.subplots(1, 3, figsize=(13, 3.4))
        ax[0].hist(obs, bins=40, color="#4878a8")
        ax[0].set_title(f"{v.name} raw (skew {skew:+.2f})")
        ax[1].hist(tv, bins=40, color="#7a9e5a")
        ax[1].set_title(f"{v.name} {transform} (skew {skew_after:+.2f})")
        stats.probplot(tv, dist="norm", plot=ax[2])
        ax[2].set_title(f"Q-Q after {transform}  (Shapiro W {w_after:.3f})")
        for a in ax:
            a.tick_params(labelsize=8)
        fig.suptitle(f"{v.name} — Training set only (n={len(obs)}, missing "
                     f"{s.isna().mean():.1%})", fontsize=10)
        fig.tight_layout()
        fig.savefig(out / "figures" / f"dist_{v.name}.png", dpi=110)
        plt.close(fig)

    cont_df = pd.DataFrame(rows)
    cont_df.to_csv(out / "continuous_distribution_summary.csv", index=False, encoding="utf-8-sig")
    print("\n=== continuous candidates (Training only) ===")
    print(cont_df[["variable", "n_observed", "missing_rate_train", "median", "skewness",
                   "shapiro_W", "transform", "skewness_after", "imputation",
                   "missing_indicator"]].to_string(index=False))

    # ---- categorical candidates ---------------------------------------------------
    crows, spec_cat = [], {}
    for v in tb.CANDIDATES:
        if v.kind == "continuous":
            continue
        if v.kind == "binary_threshold":
            floor = tb.ASSAY_FLOOR[v.column]
            raw = pd.to_numeric(train[v.column], errors="coerce")
            s = raw.where(raw.isna(), (raw > floor).map({True: "detectable", False: "undetectable"}))
        else:
            s = tb.comorbidity_count(train) if v.kind == "derived" else train[v.column]
        counts = s.value_counts(dropna=False).to_dict()
        miss = float(s.isna().mean())
        if v.kind == "binary_threshold":
            strategy, why, fill = "explicit_level", (
                f"Training の {int((pd.to_numeric(train[v.column], errors='coerce') <= tb.ASSAY_FLOOR[v.column]).sum())} / "
                f"{int(pd.to_numeric(train[v.column], errors='coerce').notna().sum())} "
                f"（{(pd.to_numeric(train[v.column], errors='coerce') <= tb.ASSAY_FLOOR[v.column]).sum() / max(pd.to_numeric(train[v.column], errors='coerce').notna().sum(), 1):.1%}）"
                f"が検出限界 {tb.ASSAY_FLOOR[v.column]} に集中し、連続値として扱うと測定系が持たない"
                f"精度を仮定することになる。検出可否の二値とし、欠測は Missing 水準として保持する。"
                f"この判断は値の分布と測定系のみに基づき、アウトカムとの関連は根拠にしていない"), None
        elif v.name == "comorbidity":
            strategy, why = "explicit_level", (
                "カルテ抽出ブロックが全欠測の患者には NotAbstracted 水準を与える。"
                "抽出の有無自体が死亡率と関連する（9.6% 対 14.0%）ため、"
                "最頻値で埋めると『併存症が少ない』という誤った情報を与える")
            fill = None
        elif miss == 0:
            strategy, why, fill = "none", "Training に欠測なし", None
        elif miss <= 0.05:
            fill = str(s.mode(dropna=True).iloc[0])
            strategy, why = "mode", (
                f"欠測率 {miss:.1%} と低く、匿名化に伴う欠測で系統性が想定しにくいため"
                f"Training の最頻値（{fill}）で補完する")
        else:
            strategy, why, fill = "explicit_level", (
                f"欠測率 {miss:.1%} と高く、欠測に情報がある可能性があるため Missing 水準を設ける")
        crows.append({"variable": v.name, "raw_column": v.column, "kind": v.kind,
                      "missing_rate_train": round(miss, 4),
                      "levels_train": json.dumps({str(k): int(n) for k, n in counts.items()},
                                                 ensure_ascii=False),
                      "imputation": strategy, "fill_value": fill, "rationale": why,
                      "encoding": "one-hot（drop_first=True、参照水準は下表）",
                      "evidence_source": "data（Training の欠測構造）＋ clinical knowledge",
                      "dataset_used_for_decision": "Training", "status": "fixed"})
        spec_cat[v.name] = {"column": v.column, "imputation": strategy, "fill_value": fill}
    cat_df = pd.DataFrame(crows)
    cat_df.to_csv(out / "categorical_summary.csv", index=False, encoding="utf-8-sig")
    print("\n=== categorical / derived candidates (Training only) ===")
    print(cat_df[["variable", "missing_rate_train", "imputation", "fill_value"]].to_string(index=False))
    print("\nlevels:")
    for _, r in cat_df.iterrows():
        print(f"  {r.variable:12s} {r.levels_train}")

    # ---- reference levels for one-hot ---------------------------------------------
    # The reference is the clinically lowest-risk / most common level, so odds ratios read
    # as "relative to the reference", which is what a clinical reader expects.
    reference = {"age": "[18,59]", "sex": "FEMALE", "comorbidity": "0"}
    spec = {
        "generated": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "script": Path(__file__).name,
        "fitted_on": "train only",
        "n_train": len(train), "deaths_train": int(train.y.sum()),
        "implausible_bounds": {k: list(v) for k, v in tb.IMPLAUSIBLE_BOUNDS.items()},
        "implausible_set_to_missing_train": fixed,
        "continuous": spec_cont,
        "categorical": spec_cat,
        "onehot_reference_levels": reference,
        "onehot_drop_first": True,
        "onehot_drop_first_rationale": (
            "LR では全水準を残すと切片と完全共線になり係数が一意に定まらない。"
            "旧 Task B は drop='first' を指定せず全水準を保持していたが、"
            "本実装では参照水準を臨床的に低リスク側に固定して解釈可能な OR を得る"),
        "scaling": {"applied_to": ["logistic_regression", "mlp"],
                    "method": "StandardScaler（Training で fit）",
                    "not_applied_to": ["xgboost"],
                    "rationale": "決定木は単調変換に不変で scaling の恩恵がなく、"
                                 "生値のままの方が分割点が解釈しやすい"},
        "missing_indicator_threshold": MISSING_INDICATOR_THRESHOLD,
        "skew_thresholds": {"symmetric": SKEW_SYMMETRIC, "strong": SKEW_STRONG},
        "leakage_guard": "すべての統計量は Training のみで算出。Validation / Test は参照していない",
    }
    (out / "preprocess_spec.json").write_text(json.dumps(spec, ensure_ascii=False, indent=2),
                                              encoding="utf-8")

    n_ind = int(cont_df.missing_indicator.sum())
    n_params = sum(v.params for v in tb.CANDIDATES) + n_ind
    print(f"\nmissing indicators: {n_ind}")
    print(f"estimated parameters incl. indicators: {n_params} "
          f"(budget {int(train.y.sum()) // 10})")
    print(f"wrote preprocess_spec.json and {len(list((out / 'figures').glob('*.png')))} figures")
    return 0


if __name__ == "__main__":
    sys.exit(main())
