# retain.py
from __future__ import annotations

import math
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from torch import Tensor, nn
from torch.utils.data import DataLoader, Subset

from src.modeling.ragged_timeseries import TimeSeriesDataset, collate_grud

# If these live elsewhere, adjust imports accordingly


# Your project loaders/collate (same as in grud.py)


# ----------------------------
# Utilities
# ----------------------------
def lengths_to_mask(lengths: Tensor, T: int) -> Tensor:
    """(B,) -> (B,T) bool where True = valid timestep."""
    device = lengths.device
    rng = torch.arange(T, device=device).unsqueeze(0)  # (1,T)
    return rng < lengths.unsqueeze(1)  # (B,T)


def masked_mean(x: Tensor, mask: Tensor, dim: int) -> Tensor:
    """
    x:    (B,T,D)
    mask: (B,T) bool (True for valid)
    returns: (B,D) mean over valid positions; 0 if none valid.
    """
    m = mask.unsqueeze(-1).type_as(x)  # (B,T,1)
    num = (x * m).sum(dim=dim)  # (B,D)
    den = m.sum(dim=dim).clamp_min(1e-8)  # (B,1)->(B,D) via broadcast
    return num / den


def last_valid(x: Tensor, lengths: Tensor) -> Tensor:
    """
    x: (B,T,D), lengths: (B,)
    returns (B,D) picking x[b, lengths[b]-1]
    """
    B, T, D = x.shape
    idx = (lengths - 1).clamp_min(0).view(B, 1, 1).expand(B, 1, D)  # (B,1,D)
    return x.gather(1, idx).squeeze(1)  # (B,D)


# ----------------------------
# Positional / time encodings
# ----------------------------


class SinusoidalPositionalEncoding(nn.Module):
    """
    Standard transformer positional encoding (index-based).
    Produces (B,T,D) given T and d_model, then adds to token embeddings.
    """

    def __init__(self, d_model: int, max_len: int = 4096, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)  # (T,D)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)  # (max_len, D)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B,T,D)
        T = x.size(1)
        x = x + self.pe[:T].unsqueeze(0).to(x.dtype)  # (1,T,D)
        return self.dropout(x)


class ContinuousTimeEncoding(nn.Module):
    """
    Continuous-time sinusoidal encoding using cumulative time stamps (sum of DT).
    - dt: (B,T)  time gaps; first step can be 0.
    Enc(t) = [sin(w_k * tau_t), cos(w_k * tau_t)]_k with tau_t = cumsum(dt).
    """

    def __init__(self, d_model: int, dropout: float = 0.0):
        super().__init__()
        # We create D frequencies; if D is odd, last cos slice will be shorter (fine).
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        # frequency basis like transformer positional but applied to tau
        self.register_buffer(
            "freq",
            torch.exp(
                torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
            ),
        )  # (⌈D/2⌉,)

    def forward(self, x: Tensor, dt: Tensor) -> Tensor:
        """
        x:  (B,T,D)
        dt: (B,T)
        returns x + f(tau) with tau = cumsum(dt) along time.
        """
        tau = torch.cumsum(dt, dim=1)  # (B,T)
        # build sinusoidal embedding at runtime to match batch T
        B, T, D = x.shape
        freq = self.freq.to(x.device)  # (F,)
        # (B,T,1)*(1,1,F) -> (B,T,F)
        arg = tau.unsqueeze(-1) * freq.view(1, 1, -1)
        pe = torch.zeros(B, T, D, device=x.device, dtype=x.dtype)
        pe[:, :, 0::2] = torch.sin(arg)
        pe[:, :, 1::2] = torch.cos(arg)
        return self.dropout(x + pe)


# ----------------------------
# Time-Series Transformer
# ----------------------------


class TimeSeriesTransformerRegressor(nn.Module):
    """
    A clean Transformer encoder for ragged multivariate time series.

    Forward:
        y_hat = model(X, M, DT, lengths)

    Options:
      - concat_XM:   feed [X||M] (explicit missingness)
      - use_dt_feat: append DT as a channel to inputs
      - pos_encoding: 'sinusoidal' (index-based) or 'continuous' (uses DT cumsum)
      - pooling: 'mean' | 'last' | 'cls'
    """

    def __init__(
        self,
        input_size: int,  # D (or D*2 if concat_XM handled externally)
        d_model: int = 128,  # must be divisible by nhead
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        *,
        concat_XM: bool = False,
        use_dt_feat: bool = False,  # append DT as 1 more channel
        pos_encoding: Literal[sinusoidal, continuous] = "sinusoidal",
        pooling: Literal[mean, last, cls] = "mean",
        max_len: int = 4096,
        head_hidden: int = 64,
    ):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.concat_XM = concat_XM
        self.use_dt_feat = use_dt_feat
        self.pooling = pooling
        self.pos_encoding_type = pos_encoding

        in_dim = input_size + (1 if use_dt_feat else 0)

        # Optional CLS token
        self.use_cls = pooling == "cls"
        if self.use_cls:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))  # (1,1,D)

        # Project inputs to model dimension
        self.in_proj = nn.Linear(in_dim, d_model)
        self.in_drop = nn.Dropout(dropout)

        # Positional encodings
        if pos_encoding == "sinusoidal":
            self.posenc = SinusoidalPositionalEncoding(d_model, max_len=max_len, dropout=dropout)
            self.ctenc = None
        elif pos_encoding == "continuous":
            self.posenc = None
            self.ctenc = ContinuousTimeEncoding(d_model, dropout=dropout)
        else:
            raise ValueError("pos_encoding must be 'sinusoidal' or 'continuous'")

        # Transformer encoder (batch_first=True)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # Regression head
        self.head = nn.Sequential(
            nn.Linear(d_model, head_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),
        )

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in)
                    nn.init.uniform_(m.bias, -bound, bound)
        if self.use_cls:
            nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

    def _build_inputs(self, X: Tensor, M: Tensor, DT: Tensor) -> Tensor:
        """
        Optionally concat mask and/or DT, then project to d_model.
        """
        feats = [X]
        if self.concat_XM:
            feats.append(M)
        if self.use_dt_feat:
            feats.append(DT.unsqueeze(-1))  # (B,T,1)
        Xin = torch.cat(feats, dim=-1)  # (B,T,in_dim)
        return self.in_drop(self.in_proj(Xin))  # (B,T,D)

    def forward(self, X: Tensor, M: Tensor, DT: Tensor, lengths: Tensor) -> Tensor:
        """
        X: (B,T,D), M: (B,T,D) 0/1, DT: (B,T), lengths: (B,)
        returns: (B,)
        """
        B, T, _ = X.shape
        key_padding_mask = ~lengths_to_mask(lengths, T)  # (B,T) True for PAD

        h = self._build_inputs(X, M, DT)  # (B,T,D_model)

        # Positional / time encoding
        if self.pos_encoding_type == "sinusoidal":
            h = self.posenc(h)  # adds index-based PE
        else:
            h = self.ctenc(h, DT)  # adds continuous-time PE

        # Optional CLS token
        if self.use_cls:
            cls = self.cls_token.expand(B, -1, -1)  # (B,1,D)
            h = torch.cat([cls, h], dim=1)  # (B,1+T,D)
            # pad mask grows with a valid CLS at position 0
            pad = torch.zeros(B, 1, dtype=torch.bool, device=h.device)
            key_padding_mask = torch.cat([pad, key_padding_mask], dim=1)  # (B,1+T)

        # Encode
        h = self.encoder(h, src_key_padding_mask=key_padding_mask)  # (B,1+T,D) or (B,T,D)

        # Pool to a single vector
        if self.pooling == "cls":
            pooled = h[:, 0, :]  # (B,D)
        elif self.pooling == "last":
            # if CLS not used, sequences start at idx 0; otherwise last valid index shifts by +1
            if self.use_cls:
                pooled = last_valid(h[:, 1:, :], lengths)  # ignore CLS for last-valid
            else:
                pooled = last_valid(h, lengths)
        else:  # 'mean'
            if self.use_cls:
                pooled = masked_mean(h[:, 1:, :], ~key_padding_mask[:, 1:], dim=1)
            else:
                pooled = masked_mean(h, ~key_padding_mask, dim=1)

        y_hat = self.head(pooled).squeeze(-1)  # (B,)
        return y_hat


# ----------------------------
# 2) Helpers
# ----------------------------


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)


def batch_to_device(
    batch: dict[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}
    # out = {}
    # for k, v in batch.items():
    #     if k == "lengths":
    #         out[k] = v  # keep on CPU for pack_padded_sequence
    #     elif torch.is_tensor(v):
    #         out[k] = v.to(device, non_blocking=True)
    #     else:
    #         out[k] = v
    # return out


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
# 3) Train / Eval
# ----------------------------
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optim: torch.optim.Optimizer,
    device: torch.device,
    *,
    concat_XM: bool = False,
):
    model.train()
    total = 0.0
    n = 0
    for batch in loader:
        batch = batch_to_device(batch, device)
        X, M, DT, lengths, y = batch["X"], batch["M"], batch["DT"], batch["lengths"], batch["y"]

        # X_in = torch.cat([X, M], dim=-1) if concat_XM else X

        optim.zero_grad()
        yhat = model(X, M, DT, lengths)
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
    concat_XM: bool = False,
):
    model.eval()
    total = 0.0
    n = 0
    for batch in loader:
        batch = batch_to_device(batch, device)
        X, M, DT, lengths, y = batch["X"], batch["M"], batch["DT"], batch["lengths"], batch["y"]
        yhat = model(X, M, DT, lengths)
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
    out_dir: str = "models/exp_TS_Transformer_cv_es",
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
    concat_XM: bool = False,  # True → feed [X||M]; False → only X
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
        # model = TimeSeriesTransformerRegressor(
        #     input_size=Din,  # Din = D or D*2 if you set concat_XM=True in your loop
        #     d_model=128,  # must be divisible by nhead
        #     nhead=8,
        #     num_layers=4,
        #     dim_feedforward=256,
        #     dropout=0.1,
        #     concat_XM=concat_XM,  # keep in sync with how you compute Din
        #     use_dt_feat=False,  # set True to append DT as a channel
        #     pos_encoding="continuous",  # "sinusoidal" or "continuous" (uses DT cumsum)
        #     pooling="mean",  # "mean" | "last" | "cls"
        #     head_hidden=64,
        # ).to(device)

        model = TimeSeriesTransformerRegressor(
            input_size=Din,  # Din = D or D*2 if you set concat_XM=True in your loop
            d_model=128,  # must be divisible by nhead
            nhead=8,
            num_layers=num_layers,
            dim_feedforward=256,
            dropout=dropout,
            concat_XM=concat_XM,  # keep in sync with how you compute Din
            use_dt_feat=False,  # set True to append DT as a channel
            pos_encoding="continuous",  # "sinusoidal" or "continuous" (uses DT cumsum)
            pooling="mean",  # "mean" | "last" | "cls"
            head_hidden=64,
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
                    print(
                        f"Early stopping (patience={patience}) at epoch {epoch}. Best epoch={best_epoch}."
                    )
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

        # print(
        #     f"Fold {fold} → best_epoch={best_epoch} | train@best={train_mse_final:.6f} | val_best={val_mse_final:.6f} | test={test_mse:.6f}"
        # )

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
    # install()
    # import pretty_errors

    # # # `configure` can be omitted if you're satisfied with default settings
    # # pretty_errors.configure()
    # pretty_errors.configure(
    #     filename_display=pretty_errors.FILENAME_EXTENDED,
    #     line_number_first=True,
    #     display_link=True,
    #     line_color=pretty_errors.RED + "> " + pretty_errors.default_config.line_color,
    #     code_color="  " + pretty_errors.default_config.line_color,
    #     truncate_code=True,
    #     display_locals=True,
    # )

    # import better_exceptions

    # better_exceptions.MAX_LENGTH = None
    # # Check if you TERM variable is set to `xterm`, if not set below variable - https://github.com/Qix-/better-exceptions/issues/8
    # better_exceptions.SUPPORTS_COLOR = True
    # better_exceptions.hook()

    # run_k_fold_cv_earlystop(
    #     pkl_path="data/processed/COPD_PATIENTS_DATA.pkl",
    #     out_dir="models/exp_tstransformer_cv_es",
    #     batch_size=128,  # 64,
    #     hidden_size=64,
    #     num_layers=2,  #! try 2 or 3 layers too
    #     dropout=0.1,
    #     bidirectional=False,
    #     fc_hidden=64,  # set None to use a single Linear
    #     epochs=500,  # upper bound; early stopping will usually stop sooner
    #     lr=3e-4,
    #     seed=42,
    #     concat_XM=False,  # feed [X||M]; recommended when zeros denote missing
    #     val_frac=0.1,  # 10% of outer-train becomes inner-val
    #     patience=30,  # stop if no val improvement for 5 epochs
    #     cv=5,  # 5-fold cross-validation
    # ) Test RMSE: 0.31±0.01 (MSE: 0.09 ± 0.01)

    run_k_fold_cv_earlystop(
        pkl_path="data/processed/COPD_PATIENTS_DATA.pkl",
        out_dir="models/exp_tstransformer_cv_es",
        batch_size=128,  # 64,
        hidden_size=64,
        num_layers=2,  #! try 2 or 3 layers too
        dropout=0.1,
        bidirectional=False,
        fc_hidden=64,  # set None to use a single Linear
        epochs=5000,  # upper bound; early stopping will usually stop sooner
        lr=3e-4,
        seed=42,
        concat_XM=False,  # feed [X||M]; recommended when zeros denote missing
        val_frac=0.1,  # 10% of outer-train becomes inner-val
        patience=20,  # stop if no val improvement for 5 epochs
        cv=5,  # 5-fold cross-validation
    )
