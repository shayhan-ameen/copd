# src/classification/xgb_classifier.py
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from src.classification.build_tabular_dataset import build_tabular_data_from_pkl
from src.classification.utils import accuracy, auc_roc, kfold_train_test_indices, set_seed
from src.config import PROCESSED_DATA_DIR


# ------------------------------------------------------------
# 1) Train a single XGBoost classifier (no validation)
# ------------------------------------------------------------
def train_xgb_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    params: dict,
    seed: int = 42,
) -> XGBClassifier:
    """Train an XGBoost binary classifier."""
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric=params.get("eval_metric", "logloss"),
        tree_method=params.get("tree_method", "hist"),
        random_state=seed,
        n_jobs=params.get("n_jobs", 0),
        # Core hyperparameters
        n_estimators=params.get("n_estimators", 1000),
        learning_rate=params.get("learning_rate", 0.05),
        max_depth=params.get("max_depth", 6),
        min_child_weight=params.get("min_child_weight", 1.0),
        subsample=params.get("subsample", 0.9),
        colsample_bytree=params.get("colsample_bytree", 0.9),
        reg_alpha=params.get("reg_alpha", 0.0),
        reg_lambda=params.get("reg_lambda", 1.0),
        gamma=params.get("gamma", 0.0),  # delete
    )
    model.fit(X_train, y_train)
    return model


# ------------------------------------------------------------
# 2) Run K-Fold Cross-Validation
# ------------------------------------------------------------
def run_xgb_classifier_cv(
    pkl_path: Path | str = PROCESSED_DATA_DIR / "NEW_COPD_PATIENTS_DATA.pkl",
    out_dir: str = "models/XGBoost_classifier",
    *,
    k_folds: int = 5,
    seed: int = 42,
):
    """
    Run K-fold cross-validation for binary COPD classification.
    Saves models, metrics, feature importance, and predictions.
    """
    # Initialize
    set_seed(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------
    X_df, _, y_cls, feat_names, pids = build_tabular_data_from_pkl(pkl_path)
    X, y = X_df.values, y_cls.values
    pids = np.array(pids)
    N = len(X)

    if N < k_folds:
        raise ValueError(f"Dataset too small for {k_folds}-fold CV: N={N}")

    print(f"\nDataset loaded: X={X.shape}, y={y.shape}, features={len(feat_names)}")

    # --------------------------------------------------------
    # Default XGBoost parameters
    # --------------------------------------------------------
    params = dict(
        learning_rate=0.005,
        max_depth=5,
        min_child_weight=1.0,
        subsample=0.65,
        colsample_bytree=0.75,
        n_estimators=250,
        gamma=0.4,
        reg_alpha=0.0,
        reg_lambda=1.0,
        tree_method="hist",
        n_jobs=0,
    )

    fold_metrics, all_preds = [], []

    # --------------------------------------------------------
    # K-Fold Loop
    # --------------------------------------------------------
    for fold, (train_idx, test_idx) in enumerate(
        kfold_train_test_indices(N, k=k_folds, seed=seed), start=1
    ):
        print(f"\n--- Fold {fold}/{k_folds} ---")

        fold_dir = out_dir / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        # Train model
        model = train_xgb_classifier(X_train, y_train, params=params, seed=seed + fold)
        model.save_model(fold_dir / "model.json")

        # Predict on test set
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        # Compute metrics
        acc, auc = accuracy(y_test, y_pred), auc_roc(y_test, y_prob)
        fold_metrics.append((acc, auc))
        print(f"Accuracy={acc:.4f}, AUC={auc:.4f}")

        # Save per-fold metrics
        with open(fold_dir / "metrics.txt", "w") as f:
            f.write(f"accuracy={acc:.6f}, auc={auc:.6f}\n")

        # Feature importance (gain)
        booster = model.get_booster()
        score = booster.get_score(importance_type="gain")
        imp_df = pd.DataFrame(
            [(feat_names[i], score.get(f"f{i}", 0.0)) for i in range(len(feat_names))],
            columns=["feature", "gain"],
        ).sort_values("gain", ascending=False)
        imp_df.to_csv(fold_dir / "feature_importance_gain.csv", index=False)

        # Save test predictions
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

    # --------------------------------------------------------
    # 3) Aggregate Results
    # --------------------------------------------------------
    all_df = pd.concat(all_preds, ignore_index=True)
    all_df.to_csv(out_dir / "all_folds_predictions.csv", index=False)

    arr = np.array(fold_metrics)
    mean_acc, mean_auc = arr[:, 0].mean(), arr[:, 1].mean()

    with open(out_dir / "cv_summary.txt", "w") as f:
        for i, (acc, auc) in enumerate(fold_metrics, start=1):
            f.write(f"fold_{i}: accuracy={acc:.6f}, auc={auc:.6f}\n")
        f.write(f"\nAverage Accuracy={mean_acc:.6f}, Average AUC={mean_auc:.6f}\n")

    print("\n=== XGBoost 5-Fold CV Summary ===")
    print(f"Avg Accuracy: {mean_acc:.4f} | Avg AUC: {mean_auc:.4f}")


# ------------------------------------------------------------
# 4) Command-line Entry
# ------------------------------------------------------------
if __name__ == "__main__":
    run_xgb_classifier_cv(
        pkl_path=PROCESSED_DATA_DIR / "NEW_COPD_PATIENTS_DATA.pkl",
        out_dir="models/XGBoost_classifier",
        k_folds=5,
        seed=42,
    )


# OLD CODE BELOW - DO NOT USE
# xgb_classifier.py
# from __future__ import annotations

# from pathlib import Path

# import numpy as np
# import pandas as pd

# # XGBoost
# from xgboost import XGBClassifier

# from src.classification.build_tabular_dataset import build_tabular_data_from_pkl
# from src.classification.utils import accuracy, auc_roc, kfold_train_test_indices, set_seed
# from src.config import PROCESSED_DATA_DIR


# # --------------------------
# # Train (no validation)
# # --------------------------
# def train_xgb_classifier(
#     X_tr: np.ndarray,
#     y_tr: np.ndarray,
#     *,
#     params: dict,
#     seed: int = 42,
# ) -> XGBClassifier:
#     """Plain XGBClassifier.fit on the training set."""
#     model = XGBClassifier(
#         n_estimators=params.get("n_estimators", 1000),
#         learning_rate=params.get("learning_rate", 0.05),
#         max_depth=params.get("max_depth", 6),
#         min_child_weight=params.get("min_child_weight", 1.0),
#         subsample=params.get("subsample", 0.9),
#         colsample_bytree=params.get("colsample_bytree", 0.9),
#         reg_alpha=params.get("reg_alpha", 0.0),
#         reg_lambda=params.get("reg_lambda", 1.0),
#         objective="binary:logistic",
#         tree_method=params.get("tree_method", "hist"),  # or "gpu_hist"
#         n_jobs=params.get("n_jobs", 0),
#         random_state=seed,
#         eval_metric=params.get("eval_metric", "logloss"),
#     )
#     model.fit(X_tr, y_tr)
#     return model


# # --------------------------
# # K-fold CV driver
# # --------------------------
# def run_xgb_classifier_cv(
#     pkl_path: Path | str = PROCESSED_DATA_DIR / "NEW_COPD_PATIENTS_DATA.pkl",
#     out_dir: str = "models/xgb_classifier_cv",
#     *,
#     k_folds: int = 5,
#     seed: int = 42,
# ):
#     """
#     Outer K-fold CV (train/test only, no early stopping).
#     Saves model, feature importance, metrics, and predictions.
#     """
#     set_seed(seed)
#     out_path = Path(out_dir)
#     out_path.mkdir(parents=True, exist_ok=True)

#     X_df, y_reg, y_cls, feat_names, pids = build_tabular_data_from_pkl(pkl_path)
#     pids_all = np.array(pids)
#     X_all = X_df.values
#     y_all = y_cls.values
#     N = len(X_all)

#     if N < k_folds:
#         raise ValueError(f"Dataset too small for {k_folds}-fold CV: N={N}")

#     print(f"Tabular shape: X={X_all.shape}, y={y_all.shape}, features={len(feat_names)}")

#     xgb_params = dict(
#         n_estimators=1000,
#         learning_rate=0.05,
#         max_depth=6,
#         min_child_weight=1.0,
#         subsample=0.9,
#         colsample_bytree=0.9,
#         reg_alpha=0.0,
#         reg_lambda=1.0,
#         tree_method="hist",
#         n_jobs=0,
#     )

#     fold_metrics = []
#     all_fold_preds = []

#     for fold, (train_idx, test_idx) in enumerate(
#         kfold_train_test_indices(N, k=k_folds, seed=seed), start=1
#     ):
#         fold_dir = out_path / f"fold_{fold}"
#         fold_dir.mkdir(parents=True, exist_ok=True)

#         X_tr, y_tr = X_all[train_idx], y_all[train_idx]
#         X_te, y_te = X_all[test_idx], y_all[test_idx]

#         model = train_xgb_classifier(X_tr, y_tr, params=xgb_params, seed=seed + fold)

#         model.save_model(str(fold_dir / "model.json"))

#         # Predict
#         y_prob = model.predict_proba(X_te)[:, 1]
#         y_pred = (y_prob >= 0.5).astype(int)

#         # Metrics
#         acc = accuracy(y_te, y_pred)
#         auc = auc_roc(y_te, y_prob)

#         fold_metrics.append((acc, auc))
#         print(f"[Fold {fold}] Accuracy={acc:.4f}, AUC={auc:.4f}")

#         # Save metrics
#         with open(fold_dir / "metrics.txt", "w") as f:
#             f.write(f"test_accuracy={acc:.6f}, test_auc={auc:.6f}\n")

#         # Feature importance
#         booster = model.get_booster()
#         score = booster.get_score(importance_type="gain")
#         imp_rows = [(feat_names[i], score.get(f"f{i}", 0.0)) for i in range(len(feat_names))]
#         imp_df = pd.DataFrame(imp_rows, columns=["feature", "gain"]).sort_values(
#             "gain", ascending=False
#         )
#         imp_df.to_csv(fold_dir / "feature_importance_gain.csv", index=False)

#         # Save test predictions
#         df_pred = pd.DataFrame(
#             {"pid": pids_all[test_idx], "y_true": y_te, "y_pred": y_pred, "y_prob": y_prob}
#         )
#         df_pred["fold"] = fold
#         df_pred.to_csv(fold_dir / "test_predictions.csv", index=False)
#         all_fold_preds.append(df_pred)

#     # Summary
#     all_df = pd.concat(all_fold_preds, ignore_index=True)
#     all_df.to_csv(out_path / "all_folds_predictions.csv", index=False)

#     arr = np.array(fold_metrics)  # acc, auc
#     acc_avg, auc_avg = arr[:, 0].mean(), arr[:, 1].mean()

#     with open(out_path / "cv_summary.txt", "w") as f:
#         for i, (acc, auc) in enumerate(fold_metrics, start=1):
#             f.write(f"fold_{i}: accuracy={acc:.6f}, auc={auc:.6f}\n")
#         f.write("\nAverages:\n")
#         f.write(f"accuracy={acc_avg:.6f}, auc={auc_avg:.6f}\n")

#     print("\n=== XGBoost Classifier 5-fold CV Summary ===")
#     print(f"Avg Accuracy: {acc_avg:.4f} | Avg AUC: {auc_avg:.4f}")


# # --------------------------
# # CLI
# # --------------------------
# if __name__ == "__main__":
#     run_xgb_classifier_cv(
#         pkl_path=PROCESSED_DATA_DIR / "NEW_COPD_PATIENTS_DATA.pkl",
#         out_dir="models/XGBoost_classifier",
#         k_folds=5,
#         seed=42,
#     )
