# src/classification/models/rf_classifier.py
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.classification.build_tabular_dataset import build_tabular_data_from_pkl
from src.classification.utils import accuracy, auc_roc, kfold_train_test_indices, set_seed
from src.config import PROCESSED_DATA_DIR


# --- tiny helper: replace columns that are entirely NaN with zeros (before median impute)
class AllNaNToZero(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self._all_nan_cols_ = np.isnan(X).all(axis=0)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        if getattr(self, "_all_nan_cols_", None) is None:
            raise RuntimeError("Transformer not fitted")
        X[:, self._all_nan_cols_] = 0.0
        return X


# ------------------------------------------------------------
# 1) Train a single Random Forest classifier (no validation)
# ------------------------------------------------------------
def train_rf_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    params: dict,
    seed: int = 42,
) -> Pipeline:
    """Train a RandomForest binary classifier inside an imputation pipeline."""
    rf = RandomForestClassifier(
        n_estimators=params.get("n_estimators", 500),
        max_depth=params.get("max_depth", None),
        min_samples_split=params.get("min_samples_split", 2),
        min_samples_leaf=params.get("min_samples_leaf", 1),
        max_features=params.get("max_features", "sqrt"),
        bootstrap=params.get("bootstrap", True),
        oob_score=params.get("oob_score", False),
        class_weight=params.get("class_weight", None),  # e.g., "balanced"
        n_jobs=params.get("n_jobs", -1),
        random_state=seed,
    )

    pipe = Pipeline(
        steps=[
            ("allnan_to_zero", AllNaNToZero()),
            ("imputer", SimpleImputer(strategy="median")),
            ("rf", rf),
        ]
    )
    pipe.fit(X_train, y_train)
    return pipe


# ------------------------------------------------------------
# 2) Run K-Fold Cross-Validation
# ------------------------------------------------------------
def run_rf_classifier_cv(
    pkl_path: Path | str = PROCESSED_DATA_DIR / "NEW_COPD_PATIENTS_DATA.pkl",
    out_dir: str = "models/RandomForest_classifier",
    *,
    k_folds: int = 5,
    seed: int = 42,
):
    set_seed(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X_df, _, y_cls, feat_names, pids = build_tabular_data_from_pkl(pkl_path)
    X, y = X_df.values, y_cls.values
    pids = np.array(pids)
    N = len(X)

    if N < k_folds:
        raise ValueError(f"Dataset too small for {k_folds}-fold CV: N={N}")

    print(f"\nDataset loaded: X={X.shape}, y={y.shape}, features={len(feat_names)}")

    params = dict(
        n_estimators=500,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        bootstrap=True,
        oob_score=False,
        class_weight=None,  # set "balanced" if needed
        n_jobs=-1,
    )

    fold_metrics, all_preds = [], []

    for fold, (train_idx, test_idx) in enumerate(
        kfold_train_test_indices(N, k=k_folds, seed=seed), start=1
    ):
        print(f"\n--- Fold {fold}/{k_folds} ---")
        fold_dir = out_dir / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        # Train
        model = train_rf_classifier(X_train, y_train, params=params, seed=seed + fold)
        joblib.dump(model, fold_dir / "model.joblib")

        # Predict (pipeline handles imputation on X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        # Metrics
        acc, auc = accuracy(y_test, y_pred), auc_roc(y_test, y_prob)
        fold_metrics.append((acc, auc))
        print(f"Accuracy={acc:.4f}, AUC={auc:.4f}")

        with open(fold_dir / "metrics.txt", "w") as f:
            f.write(f"accuracy={acc:.6f}, auc={auc:.6f}\n")

        # Feature importance (from RF step)
        rf_step = model.named_steps["rf"]
        imp = getattr(rf_step, "feature_importances_", None)
        if imp is not None:
            imp_df = pd.DataFrame({"feature": feat_names, "gini_importance": imp}).sort_values(
                "gini_importance", ascending=False
            )
            imp_df.to_csv(fold_dir / "feature_importance_gini.csv", index=False)

        # Save predictions
        df_pred = pd.DataFrame(
            {
                "pid": pids[test_idx],
                "y_true": y_test,
                "y_pred": y_pred,
                "y_prob": y_prob,
                "fold": fold,
            }
        )
        df_pred.to_csv(fold_dir / "test_predictions.csv", index=False)
        all_preds.append(df_pred)

    # Aggregate
    all_df = pd.concat(all_preds, ignore_index=True)
    all_df.to_csv(out_dir / "all_folds_predictions.csv", index=False)

    arr = np.array(fold_metrics)
    mean_acc, mean_auc = arr[:, 0].mean(), arr[:, 1].mean()

    with open(out_dir / "cv_summary.txt", "w") as f:
        for i, (acc, auc) in enumerate(fold_metrics, start=1):
            f.write(f"fold_{i}: accuracy={acc:.6f}, auc={auc:.6f}\n")
        f.write(f"\nAverage Accuracy={mean_acc:.6f}, Average AUC={mean_auc:.6f}\n")

    print("\n=== RandomForest 5-Fold CV Summary ===")
    print(f"Avg Accuracy: {mean_acc:.4f} | Avg AUC: {mean_auc:.4f}")


if __name__ == "__main__":
    run_rf_classifier_cv(
        pkl_path=PROCESSED_DATA_DIR / "NEW_COPD_PATIENTS_DATA.pkl",
        out_dir="models/RandomForest_classifier",
        k_folds=5,
        seed=42,
    )
