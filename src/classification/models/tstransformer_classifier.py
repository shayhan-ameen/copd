# src/classification/models/tstransformer_classifier.py
from __future__ import annotations

import math
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from loguru import logger
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from torch import Tensor, nn
from torch.utils.data import DataLoader, Subset

# Adjust these imports if they live elsewhere
from src.modeling.ragged_timeseries import TimeSeriesDataset, collate_grud


# ----------------------------
# Utilities
# ----------------------------
def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)


def lengths_to_mask(lengths: Tensor, T: int) -> Tensor:
    """(B,) -> (B,T) bool where True = valid timestep."""
    rng = torch.arange(T, device=lengths.device).unsqueeze(0)  # (1,T)
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


def batch_to_device(batch: dict[str, object], device: torch.device) -> dict[str, object]:
    """Move only tensors; keep Python lists (like pid) as-is."""
    out: dict[str, object] = {}
    for k, v in batch.items():
        out[k] = v.to(device) if torch.is_tensor(v) else v
    return out


def make_loader(ds, idxs: Iterable[int], batch_size: int, shuffle: bool) -> DataLoader:
    subset = Subset(ds, list(idxs))
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        collate_fn=collate_grud,
    )


def get_or_create_folds(y_bin: np.ndarray, k: int, seed: int, path: Path | str | None):
    """
    If 'path' exists, load (train_idx, test_idx) list for reproducible splits.
    Else, create StratifiedKFold splits, save, and return.
    """
    if path is not None:
        path = Path(path)
        if path.exists():
            splits = np.load(path, allow_pickle=True).tolist()
            return [(np.array(tr), np.array(te)) for tr, te in splits]

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    splits = [(tr, te) for tr, te in skf.split(np.arange(len(y_bin)), y_bin)]

    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, np.array(splits, dtype=object))
    return splits


def stratified_train_val_split(
    train_idx: np.ndarray, y_bin_all: np.ndarray, val_frac: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Stratified inner split to avoid single-class validation (which makes AUC NaN)."""
    y = y_bin_all[train_idx]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
    tr_inner_idx, val_inner_idx = next(sss.split(train_idx, y))
    return train_idx[tr_inner_idx], train_idx[val_inner_idx]


# ----------------------------
# Positional / time encodings
# ----------------------------
class SinusoidalPositionalEncoding(nn.Module):
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
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "freq",
            torch.exp(
                torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
            ),
        )  # (⌈D/2⌉,)

    def forward(self, x: Tensor, dt: Tensor) -> Tensor:
        tau = torch.cumsum(dt, dim=1)  # (B,T)
        B, T, D = x.shape
        freq = self.freq.to(x.device)  # (F,)
        arg = tau.unsqueeze(-1) * freq.view(1, 1, -1)  # (B,T,F)
        pe = torch.zeros(B, T, D, device=x.device, dtype=x.dtype)
        pe[:, :, 0::2] = torch.sin(arg)
        pe[:, :, 1::2] = torch.cos(arg)
        return self.dropout(x + pe)


# ----------------------------
# Time-Series Transformer (Classifier)
# ----------------------------
class TimeSeriesTransformerClassifier(nn.Module):
    """
    Transformer encoder for ragged multivariate time series (binary classification).

    Forward:
        logits = model(X, M, DT, lengths)   # logits for BCEWithLogitsLoss

    Options:
      - concat_XM:   feed [X||M] (explicit missingness)
      - use_dt_feat: append DT as a channel to inputs
      - pos_encoding: 'sinusoidal' (index-based) or 'continuous' (uses DT cumsum)
      - pooling: 'mean' | 'last' | 'cls'
    """

    def __init__(
        self,
        input_size: int,  # D (base X features; M is handled internally if concat_XM=True)
        d_model: int = 128,  # must be divisible by nhead
        nhead: int = 8,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        *,
        concat_XM: bool = True,
        use_dt_feat: bool = True,
        pos_encoding: Literal["sinusoidal", "continuous"] = "continuous",
        pooling: Literal["mean", "last", "cls"] = "cls",
        max_len: int = 4096,
        head_hidden: int | None = 64,
    ):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.concat_XM = concat_XM
        self.use_dt_feat = use_dt_feat
        self.pooling = pooling
        self.pos_encoding_type = pos_encoding

        # Build input projection
        in_dim = input_size + (input_size if concat_XM else 0) + (1 if use_dt_feat else 0)
        self.in_proj = nn.Linear(in_dim, d_model)
        self.in_drop = nn.Dropout(dropout)

        # CLS token (optional)
        self.use_cls = pooling == "cls"
        if self.use_cls:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))  # (1,1,D)

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

        # Classification head → logits
        if head_hidden is None:
            self.head = nn.Linear(d_model, 1)
        else:
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
        returns: logits (B,)
        """
        B, T, _ = X.shape
        key_padding_mask = ~lengths_to_mask(lengths, T)  # (B,T) True for PAD

        h = self._build_inputs(X, M, DT)  # (B,T,D_model)

        # Positional / time encoding
        if self.pos_encoding_type == "sinusoidal":
            h = self.posenc(h)
        else:
            h = self.ctenc(h, DT)

        # Optional CLS token
        if self.use_cls:
            cls = self.cls_token.expand(B, 1, -1)  # (B,1,D)
            h = torch.cat([cls, h], dim=1)  # (B,1+T,D)
            pad = torch.zeros(B, 1, dtype=torch.bool, device=h.device)
            key_padding_mask = torch.cat([pad, key_padding_mask], dim=1)  # (B,1+T)

        # Encode
        h = self.encoder(h, src_key_padding_mask=key_padding_mask)  # (B,1+T,D) or (B,T,D)

        # Pool to a single vector
        if self.pooling == "cls":
            pooled = h[:, 0, :]  # (B,D)
        elif self.pooling == "last":
            pooled = last_valid(h[:, 1:, :], lengths) if self.use_cls else last_valid(h, lengths)
        else:  # 'mean'
            pooled = (
                masked_mean(h[:, 1:, :], ~key_padding_mask[:, 1:], dim=1)
                if self.use_cls
                else masked_mean(h, ~key_padding_mask, dim=1)
            )

        logits = self.head(pooled).squeeze(-1)  # (B,)
        return logits


# ----------------------------
# Train / Eval / Predict (+ robust label binarization)
# ----------------------------
def _binarize(y: Tensor, threshold: float) -> Tensor:
    """
    Convert targets to {0,1} with 1 if y < threshold, else 0.
    Works whether y is already 0/1 or continuous in [0,1].
    """
    # If already {0,1}, leave as-is:
    if torch.all((y == 0) | (y == 1)):
        return y.float()
    return (y < threshold).float()


def compute_pos_weight(loader: DataLoader, threshold: float) -> torch.Tensor:
    """Compute pos_weight = (#neg / #pos) for BCEWithLogitsLoss from (possibly continuous) y."""
    pos = 0
    total = 0
    for batch in loader:
        y = batch["y"]
        yb = _binarize(y, threshold)
        pos += int(yb.sum().item())
        total += int(yb.numel())
    neg = total - pos
    w = (neg / max(1, pos)) if pos > 0 else 1.0
    return torch.tensor([w], dtype=torch.float32)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optim: torch.optim.Optimizer,
    device: torch.device,
    *,
    y_threshold: float,
    pos_weight: torch.Tensor | None = None,
):
    model.train()
    total = 0.0
    n = 0
    for batch in loader:
        b = batch_to_device(batch, device)
        X, M, DT, lengths, y = b["X"], b["M"], b["DT"], b["lengths"], b["y"]
        yb = _binarize(y, y_threshold)

        optim.zero_grad()
        logits = model(X, M, DT, lengths)  # (B,)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, yb, pos_weight=pos_weight)
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
    y_threshold: float,
):
    model.eval()
    total_loss = 0.0
    n = 0
    all_prob = []
    all_true = []
    correct = 0

    for batch in loader:
        b = batch_to_device(batch, device)
        X, M, DT, lengths, y = b["X"], b["M"], b["DT"], b["lengths"], b["y"]
        yb = _binarize(y, y_threshold)  # (B,)

        logits = model(X, M, DT, lengths)  # (B,)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, yb)

        prob = torch.sigmoid(logits)  # (B,)
        pred = (prob >= 0.5).long()
        correct += (pred.cpu() == yb.long().cpu()).sum().item()

        all_prob.append(prob.cpu())
        all_true.append(yb.cpu())

        bs = X.size(0)
        total_loss += loss.item() * bs
        n += bs

    avg_loss = total_loss / max(n, 1)
    y_true = torch.cat(all_true).numpy()
    y_prob = torch.cat(all_prob).numpy()

    try:
        auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")
    except Exception:
        auc = float("nan")

    acc = correct / max(n, 1)
    return avg_loss, acc, auc


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device, *, y_threshold: float):
    model.eval()
    ys_cont, ys_bin, yps, yls, pids = [], [], [], [], []
    for batch in loader:
        b = batch_to_device(batch, device)
        X, M, DT, lengths, y = b["X"], b["M"], b["DT"], b["lengths"], b["y"]
        yb = _binarize(y, y_threshold)
        logits = model(X, M, DT, lengths)
        prob = torch.sigmoid(logits)

        ys_cont.append(y.detach().cpu().numpy())
        ys_bin.append(yb.detach().cpu().numpy())
        yps.append(prob.detach().cpu().numpy())
        yls.append(logits.detach().cpu().numpy())
        pids.extend(b.get("pid", [None] * X.size(0)))

    return (
        np.concatenate(ys_bin, axis=0),
        np.concatenate(ys_cont, axis=0),
        np.concatenate(yps, axis=0),
        np.concatenate(yls, axis=0),
        np.array(pids, dtype=object),
    )


# ----------------------------
# Cross-Validation driver (early stop on AUC, fallback to BCE)
# ----------------------------
def run_k_fold_cv_earlystop(
    pkl_path: str = "data/processed/COPD_PATIENTS_DATA.pkl",
    out_dir: str = "models/exp_ts_transformer_cls",
    *,
    folds_file: str | Path | None = "models/shared_folds.npy",
    batch_size: int = 128,
    d_model: int = 128,
    nhead: int = 8,
    num_layers: int = 2,
    dim_feedforward: int = 256,
    dropout: float = 0.1,
    epochs: int = 1000,
    lr: float = 3e-4,
    seed: int = 42,
    concat_XM: bool = True,
    use_dt_feat: bool = True,
    pos_encoding: str = "continuous",
    pooling: str = "cls",
    val_frac: float = 0.1,
    patience: int = 30,
    cv: int = 5,
    use_pos_weight: bool = False,
    y_threshold: float = 0.7,  # <---- consistent binarization everywhere
):
    """
    Stratified K-Fold CV with early stopping on validation AUC (fallback to BCE if AUC is NaN).
    Saves per-fold predictions and an aggregate all-folds file.
    """
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ds = TimeSeriesDataset(compute=True)  # builds ragged cache from pkl
    N = len(ds)
    print(f"Dataset size: N={N} samples")
    assert N > cv, f"Dataset too small for {cv}-fold CV: N={N}"

    # For reproducible stratified folds: 1 if y < threshold else 0
    y_cont_all = np.array([float(s["y"]) for s in ds._cache], dtype=float)
    y_bin_all = (y_cont_all < y_threshold).astype(int)
    splits = get_or_create_folds(y_bin_all, cv, seed, folds_file)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Infer base D from a single sample (X only; M/DT added internally as configured)
    tmp = DataLoader(Subset(ds, [0]), batch_size=1, collate_fn=collate_grud)
    tb = next(iter(tmp))
    D = tb["X"].shape[-1]
    print(f"Base feature dim D={D} (concat_XM={concat_XM}, use_dt_feat={use_dt_feat})")

    fold_val_bce, fold_test_bce = [], []
    fold_val_accs, fold_val_aucs = [], []
    fold_test_accs, fold_test_aucs = [], []
    all_fold_preds = []

    for fold, (train_idx, test_idx) in enumerate(splits, start=1):
        print(f"\n=== Fold {fold} ===")
        fold_dir = out_path / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        # ----- Stratified inner split: train → (train_inner, val_inner) -----
        train_inner, val_inner = stratified_train_val_split(
            np.array(train_idx), y_bin_all, val_frac=val_frac, seed=seed + fold
        )
        print(
            f"  inner train class ratio: {y_bin_all[train_inner].mean():.3f}, "
            f"val class ratio: {y_bin_all[val_inner].mean():.3f}"
        )

        # DataLoaders
        train_loader = make_loader(ds, train_inner, batch_size=batch_size, shuffle=True)
        val_loader = make_loader(ds, val_inner, batch_size=batch_size, shuffle=False)
        test_loader = make_loader(ds, test_idx, batch_size=batch_size, shuffle=False)

        # Model / Optim
        model = TimeSeriesTransformerClassifier(
            input_size=D,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            concat_XM=concat_XM,
            use_dt_feat=use_dt_feat,
            pos_encoding=pos_encoding,  # "sinusoidal" or "continuous"
            pooling=pooling,  # "mean" | "last" | "cls"
            head_hidden=64,
        ).to(device)
        optim = torch.optim.Adam(model.parameters(), lr=lr)

        pos_weight = (
            compute_pos_weight(train_loader, y_threshold).to(device) if use_pos_weight else None
        )

        # Early stopping on validation AUC (fallback to BCE if AUC is NaN)
        best_auc = -np.inf
        best_val_bce = np.inf
        best_epoch = -1
        no_improve = 0
        saved_once = False
        eps = 1e-6

        for epoch in range(1, epochs + 1):
            _ = train_one_epoch(
                model, train_loader, optim, device, y_threshold=y_threshold, pos_weight=pos_weight
            )
            va_bce, va_acc, va_auc = evaluate(model, val_loader, device, y_threshold=y_threshold)

            auc_is_finite = np.isfinite(va_auc)
            improved_auc = auc_is_finite and (va_auc > best_auc + eps)
            improved_bce = (not auc_is_finite) and (va_bce < best_val_bce - eps)

            if improved_auc or improved_bce or (epoch == 1 and not saved_once):
                if improved_auc:
                    best_auc = va_auc
                if improved_bce or (not auc_is_finite):
                    best_val_bce = min(best_val_bce, va_bce)
                best_epoch = epoch
                no_improve = 0
                saved_once = True

                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "D": D,
                        "d_model": d_model,
                        "nhead": nhead,
                        "num_layers": num_layers,
                        "dim_feedforward": dim_feedforward,
                        "dropout": dropout,
                        "concat_XM": concat_XM,
                        "use_dt_feat": use_dt_feat,
                        "pos_encoding": pos_encoding,
                        "pooling": pooling,
                        "y_threshold": y_threshold,
                    },
                    fold_dir / "best.pt",
                )
                with open(fold_dir / "metrics_val.txt", "w") as f:
                    f.write(f"best_val_bce={va_bce:.6f}\n")
                    f.write(f"best_val_auc={va_auc}\n")  # may be NaN
                    f.write(f"best_epoch={best_epoch}\n")
                    f.write(f"val_acc_at_best={va_acc:.6f}\n")
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"Early stopping at epoch {epoch}. Best epoch={best_epoch}.")
                    break

        # Ensure we have a checkpoint even if nothing improved
        best_path = fold_dir / "best.pt"
        if not best_path.exists():
            torch.save({"state_dict": model.state_dict(), "y_threshold": y_threshold}, best_path)

        # Load best and evaluate on VAL/TEST
        ckpt = torch.load(best_path, map_location=device)
        if "state_dict" in ckpt:
            model.load_state_dict(ckpt["state_dict"])
        # trust runtime y_threshold param (or read from ckpt if you prefer)
        val_bce_final, val_acc_final, val_auc_final = evaluate(
            model, val_loader, device, y_threshold=y_threshold
        )
        test_bce, test_acc, test_auc = evaluate(model, test_loader, device, y_threshold=y_threshold)

        # Save test predictions
        y_true_bin, y_true_cont, y_prob, y_logit, pids = predict(
            model, test_loader, device, y_threshold=y_threshold
        )
        y_pred = (y_prob >= 0.5).astype(int)
        df_pred = pd.DataFrame(
            {
                "pid": pids,
                "y_true": y_true_bin,
                "y_true_cont": y_true_cont,  # for auditing the thresholding
                "y_pred": y_pred,
                "y_prob": y_prob,
                "y_logit": y_logit,
                "fold": fold,
            }
        )
        df_pred.to_csv(fold_dir / "test_predictions.csv", index=False)
        all_fold_preds.append(df_pred)

        with open(fold_dir / "metrics_final.txt", "w") as f:
            f.write(f"val_bce_best={val_bce_final:.6f}\n")
            f.write(f"val_acc_best={val_acc_final:.6f}\n")
            f.write(f"val_auc_best={val_auc_final}\n")
            f.write(f"test_bce={test_bce:.6f}\n")
            f.write(f"test_acc={test_acc:.6f}\n")
            f.write(f"test_auc={test_auc:.6f}\n")

        print(
            f"Fold {fold} → best_epoch={best_epoch} | "
            f"val: bce={val_bce_final:.6f}, acc={val_acc_final:.4f}, auc={val_auc_final} | "
            f"test: bce={test_bce:.6f}, acc={test_acc:.4f}, auc={test_auc:.4f}"
        )

        # collect stats
        fold_val_bce.append(float(val_bce_final))
        fold_test_bce.append(float(test_bce))
        fold_val_accs.append(float(val_acc_final))
        fold_val_aucs.append(float(val_auc_final) if np.isfinite(val_auc_final) else float("nan"))
        fold_test_accs.append(float(test_acc))  # <-- was missing before
        fold_test_aucs.append(float(test_auc) if np.isfinite(test_auc) else float("nan"))

    # Summary across folds
    def stats(x: list[float]) -> tuple[float, float]:
        if not x:
            return float("nan"), float("nan")
        arr = np.array(x, dtype=float)
        return float(np.nanmean(arr)), float(np.nanstd(arr, ddof=0))

    val_bce_avg, val_bce_std = stats(fold_val_bce)
    test_bce_avg, test_bce_std = stats(fold_test_bce)
    val_acc_avg, val_acc_std = stats(fold_val_accs)
    val_auc_avg, val_auc_std = stats(fold_val_aucs)
    test_acc_avg, test_acc_std = stats(fold_test_accs)
    test_auc_avg, test_auc_std = stats(fold_test_aucs)

    with open(out_path / "cv_summary.txt", "w") as f:
        for i, (vB, tB, vA, vU, tA, tU) in enumerate(
            zip(
                fold_val_bce,
                fold_test_bce,
                fold_val_accs,
                fold_val_aucs,
                fold_test_accs,
                fold_test_aucs,
                strict=False,
            ),
            start=1,
        ):
            f.write(
                f"fold_{i}: val_bce={vB:.6f} | test_bce={tB:.6f} | "
                f"val_acc={vA:.4f} | val_auc={vU} | "
                f"test_acc={tA:.4f} | test_auc={tU}\n"
            )
        f.write(
            f"\nAverages (± SD):\n"
            f"val_bce={val_bce_avg:.6f} ± {val_bce_std:.6f}\n"
            f"test_bce={test_bce_avg:.6f} ± {test_bce_std:.6f}\n"
            f"val_acc={val_acc_avg:.4f} ± {val_acc_std:.4f}\n"
            f"val_auc={val_auc_avg} ± {val_auc_std}\n"
            f"test_acc={test_acc_avg:.4f} ± {test_acc_std:.4f}\n"
            f"test_auc={test_auc_avg} ± {test_auc_std}\n"
        )

    # Combine fold predictions
    all_df = pd.concat(all_fold_preds, ignore_index=True)
    all_df.to_csv(out_path / "all_folds_predictions.csv", index=False)

    logger.success(
        "Test Summary → BCE={:.6f}±{:.6f} | ACC={:.4f}±{:.4f} | AUC={:.4f}±{:.4f}",
        test_bce_avg,
        test_bce_std,
        test_acc_avg,
        test_acc_std,
        test_auc_avg,
        test_auc_std,
    )


# ----------------------------
# CLI
# ----------------------------
if __name__ == "__main__":
    run_k_fold_cv_earlystop(
        pkl_path="data/processed/COPD_PATIENTS_DATA.pkl",
        out_dir="models/TS_Transformer_cls",
        folds_file="models/shared_folds.npy",
        batch_size=128,
        d_model=128,
        nhead=8,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.1,
        epochs=1000,
        lr=3e-4,
        seed=42,
        concat_XM=True,  # include mask channel
        use_dt_feat=True,  # include DT channel
        pos_encoding="continuous",
        pooling="cls",  # try "mean" as an alternative; often strong on irregular data
        val_frac=0.1,
        patience=30,
        cv=5,
        use_pos_weight=False,  # set True if classes are imbalanced
        y_threshold=0.7,  # <— matches your fold stratification
    )
