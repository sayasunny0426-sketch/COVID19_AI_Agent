"""Task B: the three model families and the search harness they share.

Split discipline (decision_log D-071):
  * hyperparameters are scored by repeated stratified cross-validation **inside the Training
    set**. The Validation set has 17 deaths; using it to pick among dozens of conditions, to
    early-stop, and to set the threshold -- as the previous Task B implementation did -- makes
    its own estimate optimistic. Training-internal CV has 135 events and is re-usable.
  * the preprocessor is re-fitted on each CV training fold, so no imputation value, scaling
    constant or category level crosses a fold boundary.
  * Validation is used afterwards, once per family, to compare families and fix the threshold.
  * Test is never seen here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator

import numpy as np
import pandas as pd

from ..evaluation.metrics import average_precision, roc_auc
from ..features.taskB_preprocess import TaskBPreprocessor

SEED = 42


def repeated_stratified_folds(y: np.ndarray, n_splits: int = 5, n_repeats: int = 3,
                              seed: int = SEED) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Deterministic repeated stratified K-fold, implemented directly to avoid version drift."""
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    for r in range(n_repeats):
        p, n = rng.permutation(pos), rng.permutation(neg)
        p_folds = np.array_split(p, n_splits)
        n_folds = np.array_split(n, n_splits)
        for k in range(n_splits):
            va = np.concatenate([p_folds[k], n_folds[k]])
            tr = np.setdiff1d(np.arange(len(y)), va, assume_unique=False)
            yield np.sort(tr), np.sort(va)


@dataclass
class CVResult:
    mean_auroc: float
    sd_auroc: float
    mean_auprc: float
    fold_auroc: list


def cv_score(train_df: pd.DataFrame, features: list[str], fit_predict: Callable,
             native_missing: bool = False, scale: bool = True,
             n_splits: int = 5, n_repeats: int = 3, seed: int = SEED) -> CVResult:
    """Mean CV ROC-AUC of one hyperparameter condition, preprocessor re-fit per fold."""
    y = train_df.y.values
    aurocs, auprcs = [], []
    for tr_idx, va_idx in repeated_stratified_folds(y, n_splits, n_repeats, seed):
        tr, va = train_df.iloc[tr_idx], train_df.iloc[va_idx]
        pre = TaskBPreprocessor(native_missing=native_missing, scale=scale).fit(tr)
        Xtr, Xva = pre.transform(tr)[features], pre.transform(va)[features]
        p = fit_predict(Xtr.to_numpy(float), y[tr_idx], Xva.to_numpy(float))
        aurocs.append(roc_auc(y[va_idx], p))
        auprcs.append(average_precision(y[va_idx], p))
    return CVResult(float(np.mean(aurocs)), float(np.std(aurocs, ddof=1)),
                    float(np.mean(auprcs)), [round(float(a), 6) for a in aurocs])


# ---------------------------------------------------------------------------------
# Model families. Each returns a (fit_predict, refit) pair so the search and the final
# refit on the whole Training set use exactly the same code path.
# ---------------------------------------------------------------------------------
def logistic_factory(C: float, class_weight, max_iter: int = 5000, seed: int = SEED):
    from sklearn.linear_model import LogisticRegression

    def make():
        return LogisticRegression(penalty="l2", C=C, solver="lbfgs",
                                  class_weight=class_weight, max_iter=max_iter,
                                  random_state=seed)

    def fit_predict(Xtr, ytr, Xva):
        m = make().fit(Xtr, ytr)
        return m.predict_proba(Xva)[:, 1]

    return fit_predict, make


def xgb_factory(params: dict, n_estimators: int, seed: int = SEED):
    import xgboost as xgb

    def make():
        return xgb.XGBClassifier(
            objective="binary:logistic", eval_metric="auc", tree_method="hist",
            n_estimators=n_estimators, random_state=seed, n_jobs=2, **params)

    def fit_predict(Xtr, ytr, Xva):
        m = make().fit(Xtr, ytr, verbose=False)
        return m.predict_proba(Xva)[:, 1]

    return fit_predict, make


def mlp_factory(hidden: tuple[int, int], dropout: float, lr: float, weight_decay: float,
                batch_size: int, pos_weight: float, max_epochs: int = 200,
                patience: int = 20, seed: int = SEED):
    """Small network: 12 inputs and 135 events do not support a wide one."""
    import torch
    import torch.nn as nn

    def build(input_dim: int):
        torch.manual_seed(seed)
        return nn.Sequential(
            nn.Linear(input_dim, hidden[0]), nn.BatchNorm1d(hidden[0]), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden[0], hidden[1]), nn.BatchNorm1d(hidden[1]), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden[1], 1))

    def _train(Xtr, ytr, Xin, yin=None):
        """Train with an inner split for early stopping; returns model and best epoch."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        dev = torch.device("cpu")
        model = build(Xtr.shape[1]).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32))
        xt = torch.tensor(Xtr, dtype=torch.float32)
        yt = torch.tensor(ytr, dtype=torch.float32).unsqueeze(1)
        # inner early-stopping split, stratified, from the fold-training data only
        rng = np.random.default_rng(seed)
        pos, neg = np.flatnonzero(ytr == 1), np.flatnonzero(ytr == 0)
        n_p, n_n = max(1, int(0.2 * len(pos))), max(1, int(0.2 * len(neg)))
        es = np.concatenate([rng.permutation(pos)[:n_p], rng.permutation(neg)[:n_n]])
        keep = np.setdiff1d(np.arange(len(ytr)), es)
        xk, yk = xt[keep], yt[keep]
        xe, ye = xt[es], ytr[es]
        best, best_state, best_epoch, bad = -np.inf, None, 0, 0
        for epoch in range(1, max_epochs + 1):
            model.train()
            perm = torch.randperm(len(xk))
            for i in range(0, len(xk), batch_size):
                b = perm[i:i + batch_size]
                if len(b) < 2:
                    continue
                opt.zero_grad()
                loss = crit(model(xk[b]), yk[b])
                loss.backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                pe = torch.sigmoid(model(xe)).squeeze(1).numpy()
            a = roc_auc(ye, pe)
            if a > best + 1e-4:
                best, best_epoch, bad = a, epoch, 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            p = torch.sigmoid(model(torch.tensor(Xin, dtype=torch.float32))).squeeze(1).numpy()
        return p, model, best_epoch

    def fit_predict(Xtr, ytr, Xva):
        return _train(Xtr, ytr, Xva)[0]

    return fit_predict, _train
