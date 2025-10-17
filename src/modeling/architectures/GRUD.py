# grud.py
from __future__ import annotations

import math
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch
from loguru import logger

# from rich.traceback import install, import pretty_errors, import better_exceptions
from torch import Tensor, nn
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Subset

# If these live elsewhere, adjust imports accordingly
from src.modeling.ragged_timeseries import TimeSeriesDataset, collate_grud

# ----------------------------
# 1) GRU D Regressor
# ----------------------------
#
# ---------- Utilities ----------


def compute_feature_deltas(mask: Tensor, dt: Tensor) -> Tensor:
    """
    Compute feature-wise time since last observation Δ_x for each step.
    Args:
        mask: (B, T, D) with 1 if observed at step t for feature d, else 0
        dt:   (B, T) time gap between step t-1 and t (first step can be 0). Same units as you want decay in (e.g., days).

    Returns:
        deltas_x: (B, T, D) cumulative time since last obs per feature.
    """
    B, T, D = mask.shape
    device = mask.device
    deltas = torch.zeros(B, T, D, device=device)
    # running Δ since last observation (per feature)
    running = torch.zeros(B, D, device=device)
    for t in range(T):
        # add gap
        running = running + dt[:, t].unsqueeze(-1)
        # reset where observed now
        running = running * (1.0 - mask[:, t])  # if observed (mask=1), Δ becomes 0
        deltas[:, t] = running
    return deltas


def lengths_to_packed(x: Tensor, lengths: Tensor) -> PackedSequence:
    """
    Pack variable-length batch for RNNs.
    Args:
        x: (B, T, *) padded batch
        lengths: (B,) actual lengths (int), descending or not (we sort here)

    Returns:
        packed sequence + sort info (returned via attributes)
    """
    # sort by length desc for packing
    lengths_sorted, sort_idx = lengths.sort(descending=True)
    x_sorted = x.index_select(0, sort_idx)
    packed = pack_padded_sequence(
        x_sorted, lengths_sorted.cpu(), batch_first=True, enforce_sorted=True
    )
    packed.sort_idx = sort_idx
    # provide inverse index to restore order later
    inv_idx = torch.empty_like(sort_idx)
    inv_idx[sort_idx] = torch.arange(sort_idx.size(0), device=sort_idx.device)
    packed.unsort_idx = inv_idx
    packed.lengths_sorted = lengths_sorted
    return packed


# ---------- GRU-D core ----------


class GRUDCell(nn.Module):
    """
    GRU-D (Che et al., 2018) cell.
    Handles:
      - Input decay toward feature means with feature-wise Δ_x
      - Hidden state decay with step-wise Δ_h
      - Concatenates imputed x_hat, mask m_t, and Δ_x (or Δ_h) to the gate inputs

    Shapes:
      x_t:     (B, D)  raw inputs (NaN where missing or any placeholder)
      m_t:     (B, D)  1 if observed at t, else 0
      delta_x: (B, D)  time since last obs per feature
      delta_h: (B,)    time gap for hidden decay (e.g., same as dt[:, t])
      h_{t-1}: (B, H)

    Learnables:
      - gamma_x = exp(-relu(W_x * delta_x + b_x))  -> (B, D)
      - gamma_h = exp(-relu(W_h * delta_h + b_h))  -> (B, H)
    """

    def __init__(self, input_size: int, hidden_size: int, use_feature_delta_for_input: bool = True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_feature_delta_for_input = use_feature_delta_for_input

        # mean of each feature for input decay target (register as buffer; set later)
        self.register_buffer("x_mean", torch.zeros(1, input_size))

        # decay nets: linear -> ReLU -> exp(-.)
        self.Wx = nn.Linear(input_size, input_size)  # feature-wise decay
        self.Wh = nn.Linear(1, hidden_size)  # hidden decay from scalar Δ_h

        # GRU-like gates will read concatenated vector:
        #   concat = [x_hat (D), m_t (D), delta_in (D if feature-wise else 1)]
        delta_in_dim = input_size if use_feature_delta_for_input else 1
        gate_in = input_size + input_size + delta_in_dim

        self.Wz = nn.Linear(gate_in, hidden_size)
        self.Uz = nn.Linear(hidden_size, hidden_size, bias=False)

        self.Wr = nn.Linear(gate_in, hidden_size)
        self.Ur = nn.Linear(hidden_size, hidden_size, bias=False)

        self.Wn = nn.Linear(gate_in, hidden_size)
        self.Un = nn.Linear(hidden_size, hidden_size, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in)
                    nn.init.uniform_(m.bias, -bound, bound)

    @torch.jit.export
    def set_input_means(self, x_mean: Tensor):
        """
        Set per-feature means used as decay target when features are missing.
        Args:
            x_mean: (D,) or (1, D)
        """
        x_mean = x_mean.view(1, -1)
        self.x_mean = x_mean.to(self.x_mean.device)

    def forward(
        self,
        x_t: Tensor,  # (B, D) raw (fill NaN with 0 here; we will use mask)
        m_t: Tensor,  # (B, D) 0/1 observed mask
        delta_x: Tensor,  # (B, D) feature-wise Δ
        delta_h: Tensor,  # (B,)   scalar step Δ for hidden decay
        h_prev: Tensor,  # (B, H)
    ) -> Tensor:
        B, D = x_t.shape

        # --- Input decay ---
        # gamma_x in [0, 1], larger Δ -> smaller gamma (more decay toward mean)
        gx = torch.exp(-torch.relu(self.Wx(delta_x)))
        # impute: if observed -> keep x_t ; if missing -> decay prev x toward mean
        # we don't track prev x explicitly; common implementation: decay towards mean directly
        x_hat = m_t * x_t + (1.0 - m_t) * (gx * x_t + (1.0 - gx) * self.x_mean.expand(B, -1))
        # Note: using x_t in both terms is okay if x_t already carries last valid value;
        # if not, feed pre-imputed "last observation carried forward" as x_t for better fidelity.

        # --- Hidden decay ---
        gh = torch.exp(-torch.relu(self.Wh(delta_h.view(B, 1))))  # (B, H)
        h_tilde = gh * h_prev

        # Build gate input
        delta_in = delta_x if self.use_feature_delta_for_input else delta_h.view(B, 1)
        gate_in = torch.cat([x_hat, m_t, delta_in], dim=-1)

        z = torch.sigmoid(self.Wz(gate_in) + self.Uz(h_tilde))
        r = torch.sigmoid(self.Wr(gate_in) + self.Ur(h_tilde))
        n = torch.tanh(self.Wn(gate_in) + self.Un(r * h_tilde))
        h = (1.0 - z) * h_tilde + z * n
        return h


class GRUD(nn.Module):
    """
    Multi-layer GRU-D with packed-sequence support.
    Expects padded inputs and a lengths vector, will internally pack/unpack.

    Inputs (padded):
      X:      (B, T, D)    raw inputs (fill NaNs with 0 before calling)
      M:      (B, T, D)    mask 1=observed else 0
      DT:     (B, T)       gap between t-1 and t (first step can be 0)
      lengths:(B,)         actual sequence lengths (<= T)

    Forward returns:
      outputs: (B, T, H) padded hidden states (last layer)
      h_last:  (B, H)    last valid hidden per sequence (gathered via lengths)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bidirectional: bool = False,
        use_feature_delta_for_input: bool = True,
    ):
        super().__init__()
        assert not bidirectional, (
            "GRU-D is typically defined unidirectional for forecasting; extend if needed."
        )
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.layers = nn.ModuleList(
            [
                GRUDCell(
                    input_size if l == 0 else hidden_size, hidden_size, use_feature_delta_for_input
                )
                for l in range(num_layers)
            ]
        )

    @torch.no_grad()
    def set_input_means(self, means: Tensor):
        """
        Set per-feature means for all layers' input decay targets.
        means: (D,)
        """
        self.layers[0].set_input_means(means)

    def forward(self, X: Tensor, M: Tensor, DT: Tensor, lengths: Tensor) -> tuple[Tensor, Tensor]:
        B, T, D = X.shape
        device = X.device

        # Δ_x feature-wise
        Delta_x = compute_feature_deltas(M, DT)  # (B, T, D)
        # For hidden decay, we use scalar Δ_h per step (DT)
        Delta_h = DT  # (B, T)

        # Pack everything using the same sort
        packed_X = lengths_to_packed(X, lengths)  # stores sort/unsort idx
        sort_idx, unsort_idx = packed_X.sort_idx, packed_X.unsort_idx
        lengths_sorted = packed_X.lengths_sorted
        # apply same sorting to others
        M_sorted = M.index_select(0, sort_idx)
        Dx_sorted = Delta_x.index_select(0, sort_idx)
        Dh_sorted = Delta_h.index_select(0, sort_idx)

        # now iterate over timesteps using pad_packed_sequence to get slices
        # we could also unbind packed data; simpler: work un-packed per step respecting lengths
        # Convert to lists by time
        Xp, _ = pad_packed_sequence(packed_X, batch_first=True)  # (Bsorted, Tmax, D)

        h = [
            torch.zeros(M_sorted.size(0), self.hidden_size, device=device)
            for _ in range(self.num_layers)
        ]
        outputs = []

        for t in range(Xp.size(1)):
            x_t = Xp[:, t, :]
            m_t = M_sorted[:, t, :]
            dx_t = Dx_sorted[:, t, :]
            dh_t = Dh_sorted[:, t]

            # For sequences shorter than t, we should not update (mask out using lengths)
            valid = (t < lengths_sorted).float().unsqueeze(-1)  # (Bsorted, 1)

            in_t = x_t
            for l, cell in enumerate(self.layers):
                # h[l] = cell(
                #     in_t,
                #     m_t if l == 0 else torch.ones_like(m_t),
                #     dx_t if l == 0 else torch.zeros_like(dx_t),
                #     dh_t,
                #     h[l],
                # )

                # For layer 0, use real mask/deltas over features (D).
                # For higher layers, use ones/zeros matching the current layer input (H).
                m_l = m_t if l == 0 else torch.ones_like(in_t)  # (B, D) or (B, H)
                dx_l = dx_t if l == 0 else torch.zeros_like(in_t)
                h[l] = cell(in_t, m_l, dx_l, dh_t, h[l])

                # keep hidden for valid sequences only
                h[l] = valid * h[l] + (1 - valid) * h[l].detach()  # freeze past end
                in_t = h[l]  # next layer input

            outputs.append(h[-1].unsqueeze(1))

        H_all = torch.cat(outputs, dim=1)  # (Bsorted, Tmax, H)
        # Unsort back to original batch order
        H_all = H_all.index_select(0, unsort_idx)

        # Gather last hidden per sequence (last valid index = lengths-1)
        last_idx = (
            (lengths - 1).clamp(min=0).view(B, 1, 1).expand(B, 1, self.hidden_size)
        )  # (B,1,H)
        h_last = H_all.gather(1, last_idx).squeeze(1)  # (B,H)
        return H_all, h_last


# ---------- Simple regression head (e.g., y at target time) ----------


class GRUDRegressor(nn.Module):
    """
    Encoder-only GRU-D + MLP head for regression on the last hidden state.
    """

    def __init__(
        self, input_size: int, hidden_size: int, num_layers: int = 1, head_hidden: int = 64
    ):
        super().__init__()
        self.encoder = GRUD(input_size, hidden_size, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, 1),
        )

    @torch.no_grad()
    def set_input_means(self, means: Tensor):
        self.encoder.set_input_means(means)

    def forward(self, X: Tensor, M: Tensor, DT: Tensor, lengths: Tensor) -> Tensor:
        _, h_last = self.encoder(X, M, DT, lengths)
        return self.head(h_last).squeeze(-1)


# ----------------------------
# 2) Helpers
# ----------------------------


def compute_feature_means(loader: DataLoader) -> torch.Tensor:
    """
    Average of observed values per feature across the dataset.
    Uses mask M to ignore missing values. Returns (D,) tensor.
    """
    num = None
    den = None
    for batch in loader:
        X, M = batch["X"], batch["M"]  # (B,T,D)
        obs_sum = (X * M).sum(dim=(0, 1))  # (D,)
        obs_cnt = (M > 0).sum(dim=(0, 1)).clamp(min=1)  # avoid zero-division
        if num is None:
            num, den = obs_sum, obs_cnt
        else:
            num += obs_sum
            den += obs_cnt
    means = (num / den).float()
    means[torch.isnan(means)] = 0.0
    return means


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
        # model = SimpleGRURegressor(
        #     input_size=Din,
        #     hidden_size=hidden_size,
        #     num_layers=num_layers,
        #     dropout=dropout,
        #     bidirectional=bidirectional,
        #     fc_hidden=fc_hidden,
        # ).to(device)
        model = GRUDRegressor(input_size=Din, hidden_size=hidden_size, num_layers=num_layers).to(
            device
        )

        # set input means for GRU-D decay
        means = compute_feature_means(train_loader).to(device)
        model.set_input_means(means)

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
    # install()
    import pretty_errors

    # # `configure` can be omitted if you're satisfied with default settings
    # pretty_errors.configure()
    pretty_errors.configure(
        filename_display=pretty_errors.FILENAME_EXTENDED,
        line_number_first=True,
        display_link=True,
        line_color=pretty_errors.RED + "> " + pretty_errors.default_config.line_color,
        code_color="  " + pretty_errors.default_config.line_color,
        truncate_code=True,
        display_locals=True,
    )

    # import better_exceptions

    # better_exceptions.MAX_LENGTH = None
    # # Check if you TERM variable is set to `xterm`, if not set below variable - https://github.com/Qix-/better-exceptions/issues/8
    # better_exceptions.SUPPORTS_COLOR = True
    # better_exceptions.hook()

    run_k_fold_cv_earlystop(
        pkl_path="data/processed/COPD_PATIENTS_DATA.pkl",
        out_dir="models/exp_GRU_D_cv_es",
        batch_size=128,  # 64,
        hidden_size=64,
        num_layers=3,  #! try 2 or 3 layers too
        dropout=0.4,
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
