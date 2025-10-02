# retain.py
from __future__ import annotations

import math
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Subset

from src.modeling.ragged_timeseries import TimeSeriesDataset, collate_grud

# If these live elsewhere, adjust imports accordingly


# Your project loaders/collate (same as in grud.py)

# ----------------------------
# Utilities
# ----------------------------


def lengths_to_mask(lengths: Tensor, T: int) -> Tensor:
    """
    lengths: (B,)  int
    returns: (B, T) bool mask, True for valid positions [0..len-1]
    """
    device = lengths.device
    rng = torch.arange(T, device=device).unsqueeze(0)  # (1,T)
    return rng < lengths.unsqueeze(1)  # (B,T)


def reverse_padded(x: Tensor, lengths: Tensor) -> Tensor:
    """
    Reverse each sequence along time for the first 'length' steps; keep the padded
    region at the end. Works for any trailing shape.
      x:       (B, T, ...)
      lengths: (B,)
    returns:   (B, T, ...)
    """
    B, T = x.size(0), x.size(1)
    idx_range = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)  # (B,T)
    # For valid timesteps t < len: map to (len-1 - t); else keep t
    rev_idx = (lengths.unsqueeze(1) - 1 - idx_range).clamp_min(0)
    rev_idx = torch.where(idx_range < lengths.unsqueeze(1), rev_idx, idx_range)  # (B,T)
    # Expand rev_idx to match x's trailing dims for gather
    gather_idx = rev_idx.view(B, T, *([1] * (x.dim() - 2))).expand_as(x)
    return x.gather(dim=1, index=gather_idx)


def masked_softmax(logits: Tensor, mask: Tensor, dim: int = -1, eps: float = 1e-9) -> Tensor:
    """
    logits: (B, T)
    mask:   (B, T) boolean; False positions are ignored
    returns normalized probabilities over valid positions (sum to 1 per row with any True).
    If a row has no True, returns all zeros for that row.
    """
    # Put very negative where mask is False
    neg_inf = torch.finfo(logits.dtype).min
    masked = torch.where(
        mask, logits, torch.tensor(neg_inf, device=logits.device, dtype=logits.dtype)
    )
    # For rows with all False, softmax would be NaN → guard by replacing with zeros after
    alphas = torch.softmax(masked, dim=dim)
    alphas = torch.where(mask, alphas, torch.zeros_like(alphas))
    # Re-normalize each row to sum 1 over valid entries (if any valid)
    denom = alphas.sum(dim=dim, keepdim=True).clamp_min(eps)
    return alphas / denom


# ----------------------------
# RETAIN core
# ----------------------------


class RETAINRegressor(nn.Module):
    """
    RETAIN (Choi et al. 2016) adapted for continuous multivariate time series.

    Forward contract (drop-in for your loops):
        y_hat = model(X, M, DT, lengths)
      where:
        X:       (B, T, D)  padded inputs (use zeros for missing)
        M:       (B, T, D)  binary mask 1=observed, 0=missing (optional; see concat_XM)
        DT:      (B, T)     unused here (kept for signature parity)
        lengths: (B,)       actual sequence lengths

    Key steps:
      1) Reverse-time processing.
      2) Build per-step embeddings e_t = tanh(W_e x_t).
      3) Two GRUs over reversed e: one for α-weights (attention over time),
         one for β-vectors (feature-wise importance).
      4) α = softmax(w_α^T h_α); β = tanh(W_β h_β).
      5) Context c = Σ_t α_t * (β_t ⊙ e_t).
      6) Regress y from c via an MLP head.
    """

    def __init__(
        self,
        input_size: int,  # D (or D*2 if you plan to pass [X||M])
        emb_size: int = 128,  # size of per-step embedding e_t
        attn_hidden: int = 64,  # hidden for α-GRU and β-GRU
        head_hidden: int = 64,  # MLP hidden for the regression head
        dropout: float = 0.1,
        concat_XM: bool = False,  # if True, we'll internally concat [X||M]
    ):
        super().__init__()
        self.concat_XM = concat_XM
        self.input_size = input_size
        self.emb_size = emb_size
        self.attn_hidden = attn_hidden

        # Step embedding
        self.emb = nn.Linear(input_size, emb_size)
        self.emb_act = nn.Tanh()
        self.emb_dropout = nn.Dropout(dropout)

        # Two GRUs run on reversed sequence (packed)
        self.alpha_rnn = nn.GRU(emb_size, attn_hidden, batch_first=True)
        self.beta_rnn = nn.GRU(emb_size, attn_hidden, batch_first=True)

        # α and β projections
        self.alpha_fc = nn.Linear(attn_hidden, 1)  # → scalar per step
        self.beta_fc = nn.Linear(attn_hidden, emb_size)  # → vector per step
        self.beta_act = nn.Tanh()

        # Regression head from context vector
        self.head = nn.Sequential(
            nn.Linear(emb_size, head_hidden),
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

    def _pack_sort(self, x: Tensor, lengths: Tensor):
        lengths_sorted, sort_idx = lengths.sort(descending=True)
        x_sorted = x.index_select(0, sort_idx)
        packed = pack_padded_sequence(
            x_sorted, lengths_sorted.cpu(), batch_first=True, enforce_sorted=True
        )
        inv_idx = torch.empty_like(sort_idx)
        inv_idx[sort_idx] = torch.arange(sort_idx.size(0), device=sort_idx.device)
        return packed, sort_idx, inv_idx, lengths_sorted

    def forward(self, X: Tensor, M: Tensor, DT: Tensor, lengths: Tensor) -> Tensor:
        # Optionally concatenate mask as features
        if self.concat_XM:
            Xin = torch.cat([X, M], dim=-1)  # (B,T,D*2)
        else:
            Xin = X  # (B,T,D)

        B, T, _ = Xin.shape
        device = Xin.device

        # Per-step embedding
        e = self.emb_act(self.emb(Xin))  # (B,T,E)
        e = self.emb_dropout(e)

        # Reverse time; build valid-time masks for softmax later
        time_mask = lengths_to_mask(lengths, T)  # (B,T) bool
        e_rev = reverse_padded(e, lengths)  # (B,T,E)
        mask_rev = reverse_padded(time_mask.float(), lengths).bool()  # (B,T)

        # Pack (need to sort by length)
        packed_e, sort_idx, unsort_idx, lengths_sorted = self._pack_sort(e_rev, lengths)

        # RNNs (alpha and beta)
        alpha_out_packed, _ = self.alpha_rnn(packed_e)  # packed
        beta_out_packed, _ = self.beta_rnn(packed_e)

        # Unpack back to padded sequences (B_sorted, Tmax, H)
        alpha_out, _ = pad_packed_sequence(alpha_out_packed, batch_first=True, total_length=T)
        beta_out, _ = pad_packed_sequence(beta_out_packed, batch_first=True, total_length=T)

        # Restore original batch order
        alpha_out = alpha_out.index_select(0, unsort_idx)  # (B,T,H_a)
        beta_out = beta_out.index_select(0, unsort_idx)  # (B,T,H_b)

        # α logits per step (on reversed time)
        alpha_logits = self.alpha_fc(alpha_out).squeeze(-1)  # (B,T)
        # Masked softmax over time (reversed axis)
        alpha = masked_softmax(alpha_logits, mask_rev, dim=1)  # (B,T)

        # β vectors per step
        beta = self.beta_act(self.beta_fc(beta_out))  # (B,T,E)

        # Context: c = Σ_t α_t * (β_t ⊙ e_t)  (all in reversed-time alignment)
        attn_term = beta * e_rev  # (B,T,E)
        c = torch.sum(alpha.unsqueeze(-1) * attn_term, dim=1)  # (B,E)

        # Regression
        y_hat = self.head(c).squeeze(-1)  # (B,)
        return y_hat

    @torch.no_grad()
    def forward_with_attention(
        self, X: Tensor, M: Tensor, DT: Tensor, lengths: Tensor
    ) -> tuple[Tensor, Tensor]:
        """
        Returns (y_hat, alpha) where alpha is the reverse-time attention over steps (B,T).
        Useful for inspecting which timesteps mattered.
        """
        if self.concat_XM:
            Xin = torch.cat([X, M], dim=-1)
        else:
            Xin = X

        B, T, _ = Xin.shape
        e = self.emb_act(self.emb(Xin))
        time_mask = lengths_to_mask(lengths, T)
        e_rev = reverse_padded(e, lengths)
        mask_rev = reverse_padded(time_mask.float(), lengths).bool()

        packed_e, sort_idx, unsort_idx, _ = self._pack_sort(e_rev, lengths)
        alpha_out_packed, _ = self.alpha_rnn(packed_e)
        beta_out_packed, _ = self.beta_rnn(packed_e)

        alpha_out, _ = pad_packed_sequence(alpha_out_packed, batch_first=True, total_length=T)
        beta_out, _ = pad_packed_sequence(beta_out_packed, batch_first=True, total_length=T)

        alpha_out = alpha_out.index_select(0, unsort_idx)
        beta_out = beta_out.index_select(0, unsort_idx)

        alpha_logits = self.alpha_fc(alpha_out).squeeze(-1)
        alpha = masked_softmax(alpha_logits, mask_rev, dim=1)

        beta = self.beta_act(self.beta_fc(beta_out))
        c = torch.sum(alpha.unsqueeze(-1) * (beta * e_rev), dim=1)
        y_hat = self.head(c).squeeze(-1)
        return y_hat, alpha


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
    out_dir: str = "models/exp_GRU_D_cv_es",
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
        model = RETAINRegressor(
            input_size=Din,  # Din = D or D*2 as you already compute
            emb_size=128,
            attn_hidden=64,
            head_hidden=64,
            dropout=0.1,
            concat_XM=concat_XM,  # if you want to feed [X||M]
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

    run_k_fold_cv_earlystop(
        pkl_path="data/processed/COPD_PATIENTS_DATA.pkl",
        out_dir="models/exp_retain_cv_es",
        batch_size=128,  # 64,
        hidden_size=64,
        num_layers=2,  #! try 2 or 3 layers too
        dropout=0.1,
        bidirectional=False,
        fc_hidden=64,  # set None to use a single Linear
        epochs=500,  # upper bound; early stopping will usually stop sooner
        lr=3e-4,
        seed=42,
        concat_XM=False,  # feed [X||M]; recommended when zeros denote missing
        val_frac=0.1,  # 10% of outer-train becomes inner-val
        patience=30,  # stop if no val improvement for 5 epochs
        cv=5,  # 5-fold cross-validation
    )
