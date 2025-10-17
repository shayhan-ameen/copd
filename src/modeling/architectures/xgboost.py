from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

# XGBoost
from xgboost import XGBRegressor

# Your dataset builder
from src.modeling.ragged_timeseries import TimeSeriesDataset


# --------------------------
# 0) Utilities
# --------------------------
def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)


def kfold_train_test_indices(n: int, k: int = 5, seed: int = 42):
    """
    Yields (train_idx, test_idx) for outer K-fold CV (test = held-out fold).
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    folds = np.array_split(perm, k)
    for i in range(k):
        test_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        yield train_idx, test_idx


# --------------------------
# 1) Sequence → Tabular features
# --------------------------
def last_observed_per_feature(X: np.ndarray, M: np.ndarray) -> np.ndarray:
    """
    For each feature d, return the last observed value over time (NaN if never observed).
    X, M: (T, D)
    """
    _, D = X.shape
    out = np.full(D, np.nan, dtype=float)
    for d in range(D):
        obs_idx = np.where(M[:, d] == 1)[0]
        if obs_idx.size:
            t_last = obs_idx[-1]
            out[d] = float(X[t_last, d])
    return out


def mean_observed_per_feature(X: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Mean over observed entries per feature (NaN if never observed)."""
    num = (X * M).sum(axis=0)
    den = M.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mu = num / den
    mu[den == 0] = np.nan
    return mu


def count_observed_per_feature(M: np.ndarray) -> np.ndarray:
    """Counts of observed timesteps per feature."""
    return M.sum(axis=0).astype(float)


def build_tabular_from_dataset(
    ds: TimeSeriesDataset,
    *,
    include_last: bool = True,
    include_mean: bool = True,
    include_count: bool = True,
    add_length: bool = True,
    add_span_days: bool = True,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """
    One row per patient:
      Features: last_[f], mean_[f], count_[f], seq_length, span_days
      Target: y
    """
    rows, ys, pids = [], [], []
    D = ds[0]["X"].shape[1]

    feat_names = []
    if include_last:
        feat_names += [f"last_f{d}" for d in range(D)]
    if include_mean:
        feat_names += [f"mean_f{d}" for d in range(D)]
    if include_count:
        feat_names += [f"count_f{d}" for d in range(D)]
    if add_length:
        feat_names.append("seq_length")
    if add_span_days:
        feat_names.append("span_days")

    for i in range(len(ds)):
        item = ds[i]
        pid = item.get("pid") if isinstance(item, dict) else None
        X = item["X"].numpy()  # (T, D)
        M = item["M"].numpy()  # (T, D)
        DT = item["DT"].numpy()  # (T,)
        T = int(item["length"].item())
        y = float(item["y"].item())

        feats = []
        if include_last:
            feats.extend(last_observed_per_feature(X, M))
        if include_mean:
            feats.extend(mean_observed_per_feature(X, M))
        if include_count:
            feats.extend(count_observed_per_feature(M))
        if add_length:
            feats.append(float(T))
        if add_span_days:
            feats.append(float(DT.sum()))

        rows.append(feats)
        ys.append(y)
        pids.append(pid)

    X_df = pd.DataFrame(rows, columns=feat_names, dtype=float)
    y_s = pd.Series(ys, name="y", dtype=float)
    return X_df, y_s, feat_names, pids


# --------------------------
# 2) Metrics
# --------------------------
def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    err = y_true - y_pred
    return float(np.mean(err * err))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mse(y_true, y_pred)))


# --------------------------
# 3) Train (no validation, no early stopping)
# --------------------------
def train_xgb(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    *,
    params: dict,
    seed: int = 42,
) -> XGBRegressor:
    """
    Plain XGBRegressor.fit on the training set.
    """
    model = XGBRegressor(
        n_estimators=params.get("n_estimators", 1000),
        learning_rate=params.get("learning_rate", 0.05),
        max_depth=params.get("max_depth", 6),
        min_child_weight=params.get("min_child_weight", 1.0),
        subsample=params.get("subsample", 0.9),
        colsample_bytree=params.get("colsample_bytree", 0.9),
        reg_alpha=params.get("reg_alpha", 0.0),
        reg_lambda=params.get("reg_lambda", 1.0),
        objective="reg:squarederror",
        tree_method=params.get("tree_method", "hist"),  # "gpu_hist" if GPU build installed
        n_jobs=params.get("n_jobs", 0),
        random_state=seed,
        # eval_metric can be set but won’t be used without eval_set
        eval_metric=params.get("eval_metric", "rmse"),
    )
    model.fit(X_tr, y_tr)
    return model


# --------------------------
# 4) K-fold CV driver (train/test only)
# --------------------------
def run_xgb_cv(
    pkl_path: str = "data/processed/COPD_PATIENTS_DATA.pkl",
    out_dir: str = "models/xgb_cv_no_es",
    *,
    k_folds: int = 5,
    seed: int = 42,
):
    """
    Outer K-fold CV (train/test only, no validation).
    Saves model + feature importance + per-fold metrics + summary.
    """
    set_seed(seed)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Build dataset (include age/gender/dt if desired in aggregates)
    # ds = TimeSeriesDataset(
    #     pkl_path,
    #     include_age=True,
    #     include_gender=True,
    #     include_dt_feature=True,
    # )
    ds = TimeSeriesDataset(compute=False)

    N = len(ds)
    if N < k_folds:
        raise ValueError(f"Dataset too small for {k_folds}-fold CV: N={N}")

    # Build tabular features once
    X_df, y_s, feature_names, pids = build_tabular_from_dataset(ds)
    pids_all = np.array(pids)
    X_all = X_df.values
    y_all = y_s.values

    print(f"Tabular shape: X={X_all.shape}, y={y_all.shape}, features={len(feature_names)}")

    # Reasonable defaults; tune as needed
    xgb_params = dict(
        n_estimators=1200,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=1.0,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.0,
        reg_lambda=1.0,
        tree_method="hist",  # switch to "gpu_hist" if you have GPU build
        n_jobs=0,
    )

    fold_metrics = []
    all_fold_preds = []

    for fold, (train_idx, test_idx) in enumerate(
        kfold_train_test_indices(N, k=k_folds, seed=seed), start=1
    ):
        fold_dir = out_path / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        X_tr, y_tr = X_all[train_idx], y_all[train_idx]
        X_te, y_te = X_all[test_idx], y_all[test_idx]

        # Train (no val)
        model = train_xgb(X_tr, y_tr, params=xgb_params, seed=seed + fold)

        # Save model
        model.save_model(str(fold_dir / "model.json"))

        # Predict
        y_tr_pred = model.predict(X_tr)
        y_te_pred = model.predict(X_te)

        # Metrics
        tr_mse, tr_rmse = mse(y_tr, y_tr_pred), rmse(y_tr, y_tr_pred)
        te_mse, te_rmse = mse(y_te, y_te_pred), rmse(y_te, y_te_pred)

        # Save per-fold metrics
        with open(fold_dir / "metrics.txt", "w") as f:
            f.write(f"train_mse={tr_mse:.6f}, train_rmse={tr_rmse:.6f}\n")
            f.write(f"test_mse={te_mse:.6f}, test_rmse={te_rmse:.6f}\n")

        # Feature importance (gain)
        booster = model.get_booster()
        score = booster.get_score(importance_type="gain")
        imp_rows = [(feature_names[i], score.get(f"f{i}", 0.0)) for i in range(len(feature_names))]
        imp_df = pd.DataFrame(imp_rows, columns=["feature", "gain"]).sort_values(
            "gain", ascending=False
        )
        imp_df.to_csv(fold_dir / "feature_importance_gain.csv", index=False)

        # # (Optional) Save test predictions
        # pd.DataFrame({"idx": test_idx, "y_true": y_te, "y_pred": y_te_pred}).to_csv(
        #     fold_dir / "test_predictions.csv", index=False
        # )

        fold_metrics.append((tr_mse, te_mse, tr_rmse, te_rmse))
        print(f"[Fold {fold}] Test RMSE={te_rmse:.4f} (MSE={te_mse:.4f})")

        # NEW: build per-fold predictions DataFrame, save it, and keep a copy for the aggregate file
        df_pred = pd.DataFrame({"pid": pids_all[test_idx], "y_true": y_te, "y_pred": y_te_pred})
        df_pred.to_csv(
            fold_dir / "test_predictions.csv", index=False
        )  # keeps existing per-fold file
        df_pred["fold"] = fold  # tag fold (only needed for the combined CSV)
        all_fold_preds.append(df_pred)

    # Summary
    # NEW: concatenate all folds, add absolute error, sort descending, and save once at out_path
    all_df = pd.concat(all_fold_preds, ignore_index=True)
    all_df["error"] = (all_df["y_true"] - all_df["y_pred"]).abs()
    all_df_sorted = all_df.sort_values("error", ascending=False)
    all_df_sorted.to_csv(out_path / "all_folds_test_predictions_sorted_by_error.csv", index=False)

    arr = np.array(fold_metrics)  # cols: tr_mse, te_mse, tr_rmse, te_rmse
    tr_mse_avg, te_mse_avg = arr[:, 0].mean(), arr[:, 1].mean()
    tr_rmse_avg, te_rmse_avg = arr[:, 2].mean(), arr[:, 3].mean()

    with open(out_path / "cv_summary.txt", "w") as f:
        for i, (tr_m, te_m, tr_r, te_r) in enumerate(fold_metrics, start=1):
            f.write(
                f"fold_{i}: train_mse={tr_m:.6f}, test_mse={te_m:.6f}, "
                f"train_rmse={tr_r:.6f}, test_rmse={te_r:.6f}\n"
            )
        f.write("\nAverages:\n")
        f.write(f"train_mse={tr_mse_avg:.6f}, test_mse={te_mse_avg:.6f}\n")
        f.write(f"train_rmse={tr_rmse_avg:.6f}, test_rmse={te_rmse_avg:.6f}\n")

    print("\n=== XGBoost 5-fold CV (no early stopping) Summary ===")
    print(f"Avg Test RMSE: {te_rmse_avg:.6f} | Avg Test MSE: {te_mse_avg:.6f}")


# --------------------------
# 5) CLI
# --------------------------
if __name__ == "__main__":
    run_xgb_cv(
        pkl_path="data/processed/COPD_PATIENTS_DATA.pkl",
        out_dir="models/xgb_cv_no_es",
        k_folds=5,
        seed=42,
    )
