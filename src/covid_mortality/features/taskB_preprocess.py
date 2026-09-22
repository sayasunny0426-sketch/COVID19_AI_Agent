"""Task B preprocessing: one fitted object that every model shares.

Logistic regression, the gradient-boosted trees and the MLP are given the *same* feature
matrix, so a performance difference between them is a difference between model families and
not between preprocessing pipelines. The parameters are estimated on the Training set only
and then applied unchanged to Validation and Test.

Order of operations (fixed in decision_log D-070 / D-073):
    1. physiologically impossible values -> missing (the patient is never dropped)
    2. troponin -> "detectable" / "undetectable" (78.2% of Training values sit at the assay floor)
    3. comorbidity count -> 0 / 1 / 2+ / NotAbstracted
    4. log1p on the right-skewed continuous variables
    5. missing indicator for the variables that passed the D-070 criteria (lymph only)
    6. imputation with the Training median or mean, on the transformed scale
    7. one-hot with fixed categories and a fixed reference level
    8. standardisation of the continuous columns (linear models only; trees see raw values)

`native_missing=True` skips steps 5-6 and 8 and leaves NaN in place, for the pre-specified
secondary XGBoost model (D-071).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import taskB_clinical as tb

# Fixed in D-073. The reference level is the clinically lowest-risk / most common one, so
# odds ratios read as "relative to the reference".
CATEGORY_LEVELS = {
    "age": ["[18,59]", "(59,74]", "(74,90]"],
    "sex": ["FEMALE", "MALE"],
    "troponin": ["undetectable", "detectable"],
    "comorbidity": ["0", "1", "2+", "NotAbstracted"],
}
REFERENCE = {"age": "[18,59]", "sex": "FEMALE", "troponin": "undetectable", "comorbidity": "0"}
LOG1P_MIN_SKEW = 1.0


@dataclass
class TaskBPreprocessor:
    """Fit on Training, transform anything. Nothing is estimated outside Training."""

    native_missing: bool = False
    scale: bool = True
    missing_indicators: tuple[str, ...] = ("lymph",)
    transforms: dict = field(default_factory=dict)      # name -> "log1p" | "none"
    fill_values: dict = field(default_factory=dict)     # name -> value on the transformed scale
    fill_strategy: dict = field(default_factory=dict)   # name -> "median" | "mean"
    category_fill: dict = field(default_factory=dict)   # name -> level used for missing
    scaler: dict = field(default_factory=dict)          # name -> (mean, sd)
    feature_names_: list = field(default_factory=list)
    fitted_: bool = False

    # ---- raw column extraction ---------------------------------------------------
    def _raw(self, df: pd.DataFrame) -> pd.DataFrame:
        """One tidy column per candidate variable, before any imputation."""
        out = pd.DataFrame(index=df.index)
        for v in tb.CANDIDATES:
            if v.kind == "derived":
                out[v.name] = tb.comorbidity_count(df)
            elif v.kind == "binary_threshold":
                floor = tb.ASSAY_FLOOR[v.column]
                raw = pd.to_numeric(df[v.column], errors="coerce")
                out[v.name] = np.where(raw.isna(), None,
                                       np.where(raw > floor, "detectable", "undetectable"))
                out[v.name] = out[v.name].replace({None: np.nan})
            elif v.kind == "continuous":
                s = pd.to_numeric(df[v.column], errors="coerce")
                lo, hi = tb.IMPLAUSIBLE_BOUNDS.get(v.column, (-np.inf, np.inf))
                out[v.name] = s.where((s >= lo) & (s <= hi))     # step 1
            else:
                out[v.name] = df[v.column]
        return out

    # ---- fit ---------------------------------------------------------------------
    def fit(self, train: pd.DataFrame) -> "TaskBPreprocessor":
        from scipy import stats

        raw = self._raw(train)
        for v in tb.CANDIDATES:
            if v.kind != "continuous":
                if v.kind in ("categorical", "binary"):
                    s = raw[v.name]
                    if s.isna().any() and v.name not in CATEGORY_LEVELS.get(v.name, []):
                        # low-missingness categorical -> Training mode (D-073)
                        self.category_fill[v.name] = str(s.mode(dropna=True).iloc[0])
                continue
            s = raw[v.name].dropna()
            skew = float(stats.skew(s, bias=False))
            tf = "log1p" if skew >= LOG1P_MIN_SKEW else "none"
            self.transforms[v.name] = tf
            t = np.log1p(s) if tf == "log1p" else s
            strategy = "mean" if abs(skew) < 0.5 and tf == "none" else "median"
            self.fill_strategy[v.name] = strategy
            self.fill_values[v.name] = float(t.mean() if strategy == "mean" else t.median())

        X = self._build(raw)
        if self.scale and not self.native_missing:
            for c in self._continuous_columns():
                mu, sd = float(X[c].mean()), float(X[c].std(ddof=0))
                self.scaler[c] = (mu, sd if sd > 0 else 1.0)
        self.feature_names_ = list(X.columns)
        self.fitted_ = True
        return self

    def _continuous_columns(self) -> list[str]:
        return [v.name for v in tb.CANDIDATES if v.kind == "continuous"]

    # ---- transform ---------------------------------------------------------------
    def _build(self, raw: pd.DataFrame) -> pd.DataFrame:
        cols = {}
        for v in tb.CANDIDATES:
            s = raw[v.name]
            if v.kind == "continuous":
                t = np.log1p(s) if self.transforms.get(v.name) == "log1p" else s
                if v.name in self.missing_indicators and not self.native_missing:
                    cols[f"{v.name}_missing"] = s.isna().astype(float)
                cols[v.name] = t if self.native_missing else t.fillna(self.fill_values[v.name])
            else:
                levels = CATEGORY_LEVELS[v.name]
                filled = s
                if v.name in self.category_fill:
                    filled = s.fillna(self.category_fill[v.name])
                if self.native_missing and filled.isna().any():
                    # trees can split on NaN; encode as all-zero dummies
                    pass
                ref = REFERENCE[v.name]
                for lv in levels:
                    if lv == ref:
                        continue
                    cols[f"{v.name}_{lv}"] = (filled == lv).astype(float)
        X = pd.DataFrame(cols, index=raw.index)
        # stable, readable column order: continuous, indicators, then dummies
        cont = [c for c in self._continuous_columns() if c in X.columns]
        ind = [c for c in X.columns if c.endswith("_missing")]
        dummies = [c for c in X.columns if c not in cont + ind]
        return X[cont + ind + dummies]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted_:
            raise RuntimeError("TaskBPreprocessor.transform called before fit")
        X = self._build(self._raw(df))
        if self.scale and not self.native_missing:
            for c, (mu, sd) in self.scaler.items():
                X[c] = (X[c] - mu) / sd
        X = X[self.feature_names_]
        if not self.native_missing:
            bad = int(X.isna().sum().sum()) + int(np.isinf(X.to_numpy(dtype=float)).sum())
            if bad:
                raise ValueError(f"{bad} NaN/Inf remain after preprocessing")
        return X

    def fit_transform(self, train: pd.DataFrame) -> pd.DataFrame:
        return self.fit(train).transform(train)

    # ---- variable <-> feature mapping, for group-wise variable selection ----------
    def groups(self) -> dict[str, list[str]]:
        """Original clinical variable -> the feature columns it expands into."""
        g: dict[str, list[str]] = {}
        for v in tb.CANDIDATES:
            members = [c for c in self.feature_names_
                       if c == v.name or c.startswith(f"{v.name}_")]
            if members:
                g[v.name] = members
        return g

    def to_dict(self) -> dict:
        return {"native_missing": self.native_missing, "scale": self.scale,
                "missing_indicators": list(self.missing_indicators),
                "transforms": self.transforms, "fill_strategy": self.fill_strategy,
                "fill_values": self.fill_values, "category_fill": self.category_fill,
                "category_levels": CATEGORY_LEVELS, "reference_levels": REFERENCE,
                "scaler": {k: list(v) for k, v in self.scaler.items()},
                "feature_names": self.feature_names_,
                "fitted_on": "train only"}

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
                              encoding="utf-8")
