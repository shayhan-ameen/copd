# src/classification/mlp_classifier.py
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import KFold, train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.classification.build_tabular_dataset import build_tabular_data_from_pkl
from src.classification.utils import set_seed
from src.config import PROCESSED_DATA_DIR


# ----------------------------
# Model: 1 hidden (32, ReLU) + logits
# ----------------------------
class MLPBinaryClassifier(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.hidden = nn.Linear(input_dim, 32)
        self.relu = nn.ReLU()
        self.output = nn.Linear(32, 1)  # logits

    def forward(self, x):
        x = self.relu(self.hidden(x))
        return self.output(x)  # logits


# ----------------------------
# Train / Eval
# ----------------------------
def train_epoch(model, loader, criterion, optimizer, device, grad_clip: float | None = 1.0):
    model.train()
    total_loss = 0.0
    for Xb, yb in loader:
        Xb, yb = Xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(Xb)
        loss = criterion(logits, yb)
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item() * len(Xb)
    return total_loss / len(loader.dataset)


def eval_loss(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            logits = model(Xb)
            loss = criterion(logits, yb)
            total_loss += loss.item() * len(Xb)
    return total_loss / len(loader.dataset)


def evaluate(model, loader, device):
    model.eval()
    probs_all, ys_all = [], []
    with torch.no_grad():
        for Xb, yb in loader:
            Xb = Xb.to(device)
            logits = model(Xb)
            probs = torch.sigmoid(logits)  # convert logits → prob
            probs_all.append(probs.cpu().numpy())
            ys_all.append(yb.numpy())
    y_prob = np.concatenate(probs_all).ravel()
    y_true = np.concatenate(ys_all).ravel()
    y_pred = (y_prob >= 0.5).astype(int)
    acc = accuracy_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = np.nan
    return acc, auc, y_true, y_pred, y_prob


# ----------------------------
# Utilities: per-fold impute + scale
# ----------------------------
def fit_imputer_and_scaler(X_train: np.ndarray):
    """
    Fit per-fold imputer+scaler without np.nanmean warnings.
    - Columns that are all-NaN get mean=0.0
    - Then compute mu/sigma on the imputed train
    """
    valid_counts = np.sum(~np.isnan(X_train), axis=0)
    sums = np.nansum(X_train, axis=0)  # no warning on all-NaN
    col_mean = sums / np.maximum(valid_counts, 1)
    col_mean[valid_counts == 0] = 0.0  # explicit default for all-NaN cols

    X_imp = np.where(np.isnan(X_train), col_mean, X_train)

    mu = X_imp.mean(axis=0)
    sigma = X_imp.std(axis=0, ddof=0)
    sigma = np.where(sigma < 1e-8, 1.0, sigma)  # avoid div-by-zero

    n_all_nan = int((valid_counts == 0).sum())
    if n_all_nan:
        print(f"  [Info] {n_all_nan} columns were all-NaN in this train fold (imputed with 0).")

    return {
        "col_mean": col_mean.astype(np.float32),
        "mu": mu.astype(np.float32),
        "sigma": sigma.astype(np.float32),
    }


def apply_imputer_and_scaler(X: np.ndarray, stats: dict):
    X = np.where(np.isnan(X), stats["col_mean"], X)
    X = (X - stats["mu"]) / stats["sigma"]
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)
    return X


# ----------------------------
# CV Driver (with early stopping)
# ----------------------------
def run_mlp_classifier_cv(
    pkl_path: Path | str = PROCESSED_DATA_DIR / "NEW_COPD_PATIENTS_DATA.pkl",
    out_dir: str = "models/MLP_classifier",
    *,
    k_folds: int = 5,
    seed: int = 42,
    batch_size: int = 32,
    lr: float = 1e-3,
    epochs: int = 500,  # ← total epochs
    weight_decay: float = 1e-4,
    patience: int = 30,  # ← early stopping patience
    val_size: float = 0.1,  # 10% of the training fold for validation
):
    """
    MLP (1 hidden layer, 32 ReLU units) for binary classification with K-Fold CV.
    Includes per-fold NaN imputation + standardization and early stopping on val loss.
    """
    set_seed(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X_df, _, y_cls, feat_names, pids = build_tabular_data_from_pkl(pkl_path)
    X = X_df.values.astype(np.float32)
    y = y_cls.values.astype(np.float32).reshape(-1, 1)
    pids = np.array(pids)
    N, D = X.shape
    print(f"\nDataset loaded: X={X.shape}, y={y.shape}, features={len(feat_names)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
    fold_metrics, all_preds = [], []

    for fold, (tr_idx, te_idx) in enumerate(kf.split(X), start=1):
        print(f"\n--- Fold {fold}/{k_folds} ---")

        # Split train fold into train/val (stratified)
        X_tr_raw, y_tr_raw = X[tr_idx], y[tr_idx].ravel()
        X_te_raw, y_te = X[te_idx], y[te_idx]

        X_tr_raw, X_val_raw, y_tr, y_val = train_test_split(
            X_tr_raw, y_tr_raw, test_size=val_size, random_state=seed + fold, stratify=y_tr_raw
        )
        y_tr = y_tr.reshape(-1, 1).astype(np.float32)
        y_val = y_val.reshape(-1, 1).astype(np.float32)

        # ------- Impute + Standardize (fit on train, apply to val/test) -------
        stats = fit_imputer_and_scaler(X_tr_raw)
        X_tr = apply_imputer_and_scaler(X_tr_raw, stats)
        X_val = apply_imputer_and_scaler(X_val_raw, stats)
        X_te = apply_imputer_and_scaler(X_te_raw, stats)

        # Safety checks
        assert np.isfinite(X_tr).all() and np.isfinite(X_val).all() and np.isfinite(X_te).all()
        assert set(np.unique(y_tr)).issubset({0.0, 1.0}) and set(np.unique(y_val)).issubset(
            {0.0, 1.0}
        )

        # DataLoaders
        train_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
            batch_size=batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)),
            batch_size=batch_size,
            shuffle=False,
        )
        test_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_te), torch.from_numpy(y_te)),
            batch_size=batch_size,
            shuffle=False,
        )

        # Model, loss, optimizer
        model = MLPBinaryClassifier(input_dim=D).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        # -------- Early stopping on validation loss --------
        best_val = np.inf
        best_state = None
        bad_epochs = 0

        for epoch in range(1, epochs + 1):
            train_loss = train_epoch(
                model, train_loader, criterion, optimizer, device, grad_clip=1.0
            )
            val_loss = eval_loss(model, val_loader, criterion, device)

            # if epoch % 10 == 0 or epoch == 1:
            #     print(f"Epoch {epoch}/{epochs} - train: {train_loss:.6f} | val: {val_loss:.6f}")

            if val_loss < best_val - 1e-6:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    print(f"  Early stopping at epoch {epoch} (best val loss={best_val:.6f})")
                    break

        # Restore best weights before test
        if best_state is not None:
            model.load_state_dict(best_state)

        # Evaluate on test set
        acc, auc, y_true, y_pred, y_prob = evaluate(model, test_loader, device)
        print(f"Fold {fold}: Accuracy={acc:.4f}, AUC={auc:.4f}")
        fold_metrics.append((acc, auc))

        # Save predictions
        df_pred = pd.DataFrame(
            {
                "pid": pids[te_idx],
                "y_true": y_true,
                "y_pred": y_pred,
                "y_prob": y_prob,
                "fold": fold,
            }
        )
        (Path(out_dir) / f"fold_{fold}_predictions.csv").write_text(df_pred.to_csv(index=False))

        # Save model weights
        torch.save(model.state_dict(), Path(out_dir) / f"fold_{fold}_model.pt")

    # Aggregate
    all_df = pd.concat(
        all_preds := [
            pd.read_csv(Path(out_dir) / f"fold_{i}_predictions.csv") for i in range(1, k_folds + 1)
        ],
        ignore_index=True,
    )
    all_df.to_csv(Path(out_dir) / "all_folds_predictions.csv", index=False)

    arr = np.array(fold_metrics)
    mean_acc, mean_auc = arr[:, 0].mean(), arr[:, 1].mean()
    with open(Path(out_dir) / "cv_summary.txt", "w") as f:
        for i, (acc, auc) in enumerate(fold_metrics, start=1):
            f.write(f"fold_{i}: accuracy={acc:.6f}, auc={auc:.6f}\n")
        f.write(f"\nAverage Accuracy={mean_acc:.6f}, Average AUC={mean_auc:.6f}\n")

    print("\n=== MLP 5-Fold CV Summary (Early Stopping) ===")
    print(f"Avg Accuracy: {mean_acc:.4f} | Avg AUC: {mean_auc:.4f}")


if __name__ == "__main__":
    run_mlp_classifier_cv(
        pkl_path=PROCESSED_DATA_DIR / "NEW_COPD_PATIENTS_DATA.pkl",
        out_dir="models/MLP_classifier",
        k_folds=5,
        seed=42,
        batch_size=32,
        lr=1e-3,
        epochs=500,  # ← total training epochs
        weight_decay=1e-4,
        patience=20,  # ← early stopping patience
        val_size=0.10,  # 10% of training fold for validation
    )
