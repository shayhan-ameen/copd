from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from torch import nn
from torch.utils.data import DataLoader, Subset

# If these live elsewhere, adjust imports accordingly
from src.modeling.ragged_timeseries import TimeSeriesDataset, collate_grud


# ----------------------------
# 1) Simple GRU Regressor
# ----------------------------
class SimpleGRURegressor(nn.Module):
    """
    Minimal GRU regressor:
      - Input: (B, T, Din)
      - Uses pack_padded_sequence with 'lengths'
      - Output: scalar yhat per sequence
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = False,
        fc_hidden: int | None = None,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        out_h = hidden_size * (2 if bidirectional else 1)

        if fc_hidden is None:
            self.head = nn.Linear(out_h, 1)
        else:
            self.head = nn.Sequential(
                nn.Linear(out_h, fc_hidden),
                nn.ReLU(),
                nn.Linear(fc_hidden, 1),
            )

    def forward(self, X: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        X: (B, T, Din)
        lengths: (B,) int64 lengths (descending order not required; enforce_sorted=False)
        """
        # Sanity check (remove after debugging for speed)
        # assert lengths.device.type == "cpu", f"lengths on {lengths.device}, expected cpu"
        packed = nn.utils.rnn.pack_padded_sequence(
            X,
            lengths,
            batch_first=True,
            enforce_sorted=False,
            # enforce_sorted=True,
        )
        _, h_n = self.gru(packed)  # h_n: (num_layers * num_directions, B, H)
        last = h_n[-1]  # final layer, last direction → (B, Hout)
        yhat = self.head(last).squeeze(-1)  # (B,)
        return yhat


# ----------------------------
# 2) Helpers
# ----------------------------
def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)


def batch_to_device(
    batch: dict[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    # return {k: v.to(device) for k, v in batch.items()}
    out = {}
    for k, v in batch.items():
        if k == "lengths":
            out[k] = v  # keep on CPU for pack_padded_sequence
        elif torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def make_loader(ds, idxs: Iterable[int], batch_size: int, shuffle: bool) -> DataLoader:
    subset = Subset(ds, list(idxs))
    nw = 2  # min(max((os.cpu_count() or 8) - 2, 4), 16)
    pf = 8  # prefetch 6 batches/worker
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        collate_fn=collate_grud,
        # num_workers=nw,
        # persistent_workers=(nw > 0),  # NEW: keep workers alive across epochs
        # prefetch_factor=pf if nw > 0 else None,  # NEW: more batches ready ahead of time
        # prefetch_factor=pf,  # NEW: more batches ready ahead of time
        # drop_last=shuffle, # OPTIONAL: True for train to avoid tiny last batch
    )


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


def split_train_val(idxs: np.ndarray, val_frac: float = 0.1, seed: int = 42):
    """
    Randomly split given indices into (train_inner, val_inner).
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(idxs)
    n = len(perm)
    n_val = max(1, int(round(n * val_frac)))
    val_inner = perm[:n_val]
    train_inner = perm[n_val:]
    return train_inner, val_inner


# ----------------------------
# 3) Train / Eval (with optional X||M)
# ----------------------------
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optim: torch.optim.Optimizer,
    device: torch.device,
    *,
    concat_XM: bool = True,
):
    model.train()
    total = 0.0
    n = 0
    for batch in loader:
        batch = batch_to_device(batch, device)
        X, M, lengths, y = batch["X"], batch["M"], batch["lengths"], batch["y"]

        X_in = torch.cat([X, M], dim=-1) if concat_XM else X

        optim.zero_grad()
        yhat = model(X_in, lengths)
        loss = nn.functional.mse_loss(yhat, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()

        bs = X.size(0)
        total += loss.item() * bs
        n += bs
    return total / max(n, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    concat_XM: bool = True,
):
    model.eval()
    total = 0.0
    n = 0
    for batch in loader:
        batch = batch_to_device(batch, device)
        X, M, lengths, y = batch["X"], batch["M"], batch["lengths"], batch["y"]
        X_in = torch.cat([X, M], dim=-1) if concat_XM else X
        yhat = model(X_in, lengths)
        loss = nn.functional.mse_loss(yhat, y)
        bs = X.size(0)
        total += loss.item() * bs
        n += bs
    return total / max(n, 1)


# ----------------------------
# 4) K-Fold CV (outer train/test) + inner val + early stopping
# ----------------------------
def run_k_fold_cv_earlystop(
    pkl_path: str = "data/processed/COPD_PATIENTS_DATA.pkl",
    out_dir: str = "models/exp_simple_gru_cv_es",
    *,
    batch_size: int = 64,
    hidden_size: int = 64,
    num_layers: int = 1,
    dropout: float = 0.0,
    bidirectional: bool = False,
    fc_hidden: int | None = 64,
    epochs: int = 100,
    lr: float = 3e-4,
    seed: int = 42,
    concat_XM: bool = True,  # True → feed [X||M]; False → only X
    val_frac: float = 0.1,  # inner validation fraction from outer-train
    patience: int = 5,  # early stopping patience on val MSE
    cv: int = 5,  # number of outer folds
):
    """
    Outer K-Fold CV:
      - Split dataset into (train, test).
      - From 'train', create (train_inner, val_inner) by val_frac.
      - Train with early stopping on validation MSE (patience).
      - Save best-by-val checkpoint, then evaluate on test.
      - Record train/val/test MSE per fold + overall summary.
    """
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ds = TimeSeriesDataset(compute=False)
    N = len(ds)
    print(f"Dataset size: N={N} samples")
    if N < cv:
        raise ValueError(f"Dataset too small for {cv}-fold CV: N={N}")

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Infer base D (to set GRU input_size) from a single sample
    tmp_loader = DataLoader(Subset(ds, [0]), batch_size=1, collate_fn=collate_grud)
    tmp_batch = next(iter(tmp_loader))
    D = tmp_batch["X"].shape[-1]
    Din = D * 2 if concat_XM else D
    T = int(tmp_batch["lengths"][0].item())
    print(f"Base feature dim D={D} → GRU input_size={Din} (concat_XM={concat_XM})")

    fold_train_mse, fold_val_mse, fold_test_mse = [], [], []

    for fold, (train_idx, test_idx) in enumerate(
        kfold_train_test_indices(N, k=cv, seed=seed), start=1
    ):
        print(f"\n=== Fold {fold} ===")
        fold_dir = out_path / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        # Inner split: train → (train_inner, val_inner)
        train_inner, val_inner = split_train_val(
            np.array(train_idx), val_frac=val_frac, seed=seed + fold
        )

        # DataLoaders
        train_loader = make_loader(ds, train_inner, batch_size=batch_size, shuffle=True)
        val_loader = make_loader(ds, val_inner, batch_size=batch_size, shuffle=False)
        test_loader = make_loader(ds, test_idx, batch_size=batch_size, shuffle=False)

        # Model / Optim
        model = SimpleGRURegressor(
            input_size=Din,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            fc_hidden=fc_hidden,
        ).to(device)
        optim = torch.optim.Adam(model.parameters(), lr=lr)

        # Early stopping bookkeeping
        best_val = float("inf")
        best_epoch = -1
        no_improve = 0

        # For reporting: track the *best* train/val (at the same epoch as best val)
        best_train_at_val = None

        for epoch in range(1, epochs + 1):
            tr = train_one_epoch(model, train_loader, optim, device, concat_XM=concat_XM)
            va = evaluate(model, val_loader, device, concat_XM=concat_XM)
            # print(f"[Fold {fold} | Epoch {epoch:03d}] train MSE={tr:.6f} | val MSE={va:.6f}")

            if va < best_val - 1e-8:  # tiny tolerance
                best_val = va
                best_epoch = epoch
                best_train_at_val = tr
                no_improve = 0

                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "input_size": Din,
                        "hidden_size": hidden_size,
                        "num_layers": num_layers,
                        "dropout": dropout,
                        "bidirectional": bidirectional,
                        "fc_hidden": fc_hidden,
                        "concat_XM": concat_XM,
                    },
                    fold_dir / "best.pt",
                )
                with open(fold_dir / "metrics_val.txt", "w") as f:
                    f.write(f"best_val_mse={best_val:.6f}\n")
                    f.write(f"best_epoch={best_epoch}\n")
                    f.write(f"train_mse_at_best_val={best_train_at_val:.6f}\n")
            else:
                no_improve += 1
                if no_improve >= patience:
                    # print(
                    #     f"Early stopping (patience={patience}) at epoch {epoch}. Best epoch={best_epoch}."
                    # )
                    break

        # Load best and evaluate on TEST set
        ckpt = torch.load(fold_dir / "best.pt", map_location=device)
        model.load_state_dict(ckpt["state_dict"])

        train_mse_final = (
            best_train_at_val
            if best_train_at_val is not None
            else evaluate(model, train_loader, device, concat_XM=concat_XM)
        )
        val_mse_final = (
            best_val
            if best_val < float("inf")
            else evaluate(model, val_loader, device, concat_XM=concat_XM)
        )
        test_mse = evaluate(model, test_loader, device, concat_XM=concat_XM)

        with open(fold_dir / "metrics_final.txt", "w") as f:
            f.write(f"train_mse_at_best={train_mse_final:.6f}\n")
            f.write(f"val_mse_best={val_mse_final:.6f}\n")
            f.write(f"test_mse={test_mse:.6f}\n")

        print(
            f"Fold {fold} → best_epoch={best_epoch} | train@best={train_mse_final:.6f} | val_best={val_mse_final:.6f} | test={test_mse:.6f}"
        )

        fold_train_mse.append(float(train_mse_final))
        fold_val_mse.append(float(val_mse_final))
        fold_test_mse.append(float(test_mse))

    # Summary across folds
    def stats(x: list[float]) -> tuple[float, float]:
        return float(np.mean(x)), float(np.std(x, ddof=0))

    train_avg, train_std = stats(fold_train_mse)
    val_avg, val_std = stats(fold_val_mse)
    test_avg, test_std = stats(fold_test_mse)

    # --- NEW: per-fold RMSE + summary ---
    fold_train_rmse = [float(m) ** 0.5 for m in fold_train_mse]
    fold_val_rmse = [float(m) ** 0.5 for m in fold_val_mse]
    fold_test_rmse = [float(m) ** 0.5 for m in fold_test_mse]

    train_rmse_avg, train_rmse_std = stats(fold_train_rmse)
    val_rmse_avg, val_rmse_std = stats(fold_val_rmse)
    test_rmse_avg, test_rmse_std = stats(fold_test_rmse)

    with open(out_path / "cv_summary.txt", "w") as f:
        for i, (tr, va, te, trr, var, ter) in enumerate(
            zip(
                fold_train_mse,
                fold_val_mse,
                fold_test_mse,
                fold_train_rmse,
                fold_val_rmse,
                fold_test_rmse,
                strict=False,
            ),
            start=1,
        ):
            f.write(f"fold_{i}_train_mse={tr:.6f} | val_mse={va:.6f} | test_mse={te:.6f}\n")
            f.write(f"fold_{i}_train_rmse={trr:.6f} | val_rmse={var:.6f} | test_rmse={ter:.6f}\n")
        f.write(f"avg_train_mse={train_avg:.6f} ± {train_std:.6f}\n")
        f.write(f"avg_val_mse={val_avg:.6f} ± {val_std:.6f}\n")
        f.write(f"avg_test_mse={test_avg:.6f} ± {test_std:.6f}\n")
        f.write(f"avg_train_rmse={train_rmse_avg:.6f} ± {train_rmse_std:.6f}\n")
        f.write(f"avg_val_rmse={val_rmse_avg:.6f} ± {val_rmse_std:.6f}\n")
        f.write(f"avg_test_rmse={test_rmse_avg:.6f} ± {test_rmse_std:.6f}\n")

    # print(f"\n=== {cv}-fold CV (Early Stopping) Summary ===")
    # print(
    #     f"Train RMSE: {train_rmse_avg:.2f}±{train_rmse_std:.2f} (MSE: {train_avg:.2f} ± {train_std:.2f})"
    # )
    # print(f"Val RMSE: {val_rmse_avg:.2f}±{val_rmse_std:.2f} (MSE: {val_avg:.2f} ± {val_std:.2f})")
    # print(
    #     f"Test RMSE: {test_rmse_avg:.2f}±{test_rmse_std:.2f} (MSE: {test_avg:.2f} ± {test_std:.2f})"
    # )

    logger.success("Model:\n{}", model)
    logger.success(
        f"Test RMSE: {test_rmse_avg:.2f}±{test_rmse_std:.2f} (MSE: {test_avg:.2f} ± {test_std:.2f})"
    )

    # from torchinfo import summary

    # model is already .to(device)
    # X_dummy = torch.randn(1, T, Din, device=device)  # CUDA
    # lengths_dummy = torch.tensor([T], dtype=torch.long)  # CPU

    # summary(
    #     model,
    #     input_data=(X_dummy, lengths_dummy),
    #     device=None,  # <- IMPORTANT: don't let torchinfo move inputs
    # )

    # from loguru import logger

    # logger.warning


# ----------------------------
# 5) CLI entry
# ----------------------------
if __name__ == "__main__":
    run_k_fold_cv_earlystop(
        pkl_path="data/processed/COPD_PATIENTS_DATA.pkl",
        out_dir="models/exp_simple_gru_cv_es",
        batch_size=128,  # 64,
        hidden_size=64,
        num_layers=2,  #! try 2 or 3 layers too
        dropout=0.4,
        bidirectional=False,
        fc_hidden=64,  # set None to use a single Linear
        epochs=500,  # upper bound; early stopping will usually stop sooner
        lr=3e-4,
        seed=42,
        concat_XM=True,  # feed [X||M]; recommended when zeros denote missing
        val_frac=0.1,  # 10% of outer-train becomes inner-val
        patience=30,  # stop if no val improvement for 5 epochs
        cv=5,  # 5-fold cross-validation
    )
