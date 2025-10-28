from __future__ import annotations

import numpy as np
import torch


# --------------------------
# 0) Utilities
# --------------------------
def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)


def kfold_train_test_indices(n: int, k: int = 5, seed: int = 42):
    """Yields (train_idx, test_idx) for outer K-fold CV (test = held-out fold)."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    folds = np.array_split(perm, k)
    for i in range(k):
        test_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        yield train_idx, test_idx


# --------------------------
# 1) Metrics
# --------------------------
def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((y_true == y_pred).mean())


def auc_roc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    try:
        return float(roc_auc_score(y_true, y_prob))
    except ValueError:
        return np.nan
