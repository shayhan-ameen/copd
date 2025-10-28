# src/modeling/exp_simple_grua_classifier.py
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.utils.data import DataLoader, Subset

from src.classification.build_timeseries import TimeSeriesDataset, collate_grud


# ----------------------------
# 0) Repro & small helpers
# ----------------------------
def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)


def batch_to_device(
    batch: dict[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for k, v in batch.items():
        if k == "lengths":
            out[k] = v  # keep on CPU (safer for pack_padded_sequence)
        elif torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
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


# ----------------------------
# 1) Temporal Additive Attention
# ----------------------------
class TemporalAttention(nn.Module):
    """
    Additive (Bahdanau) attention over time with padding mask.
    H: (B, T, H), mask: (B, T) boolean (True=valid timestep)
    Returns: context (B, H), attn weights (B, T)
    """

    def __init__(self, hidden_dim: int, attn_dim: int = 64, dropout: float = 0.0):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, attn_dim, bias=True)
        self.v = nn.Linear(attn_dim, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, H: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # score e_t = v^T tanh(W h_t)
        Z = self.proj(self.dropout(H))  # (B,T,attn_dim)
        e = self.v(torch.tanh(Z)).squeeze(-1)  # (B,T)
        e = e.masked_fill(~mask, float("-inf"))  # mask paddings
        alpha = torch.softmax(e, dim=1)  # (B,T)
        ctx = torch.sum(alpha.unsqueeze(-1) * H, dim=1)  # (B,H)
        return ctx, alpha


# ----------------------------
# 2) GRU Classifier (masked-mean OR attention) + aux features
# ----------------------------
class SimpleGRUClassifier(nn.Module):
    """
    GRU + temporal pooling:
      - Choose between masked mean pooling and additive attention
      - Optional aux features (e.g., [length, span]) concatenated to pooled h
      - Output: scalar logit per sequence
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = False,
        fc_hidden: int | None = 64,
        aux_dim: int = 0,  # e.g., 2 for [length, span]
        use_attention: bool = True,  # toggle attention pooling
        attn_dim: int = 64,
        attn_dropout: float = 0.1,
    ):
        super().__init__()
        self.aux_dim = aux_dim
        self.use_attention = use_attention

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        out_h = hidden_size * (2 if bidirectional else 1)

        if self.use_attention:
            self.attn = TemporalAttention(out_h, attn_dim=attn_dim, dropout=attn_dropout)

        in_head = out_h + aux_dim
        if fc_hidden is None:
            self.head = nn.Linear(in_head, 1)
        else:
            self.head = nn.Sequential(
                nn.Linear(in_head, fc_hidden),
                nn.ReLU(),
                nn.Linear(fc_hidden, 1),
            )

    def forward(
        self,
        X: torch.Tensor,  # (B, T, Din) on device
        lengths: torch.Tensor,  # (B,) on CPU (for packing)
        aux: torch.Tensor | None,  # (B, aux_dim) on device or None
    ) -> torch.Tensor:
        # pack requires CPU lengths on some PyTorch builds; keep it CPU-safe:
        packed = nn.utils.rnn.pack_padded_sequence(
            X, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        out, _ = self.gru(packed)  # GRU on device
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)  # (B,T,H) on device

        B, T, H = out.shape
        # make a device-aligned copy of lengths for masking/denominator
        lengths_dev = lengths.to(out.device)
        mask = torch.arange(T, device=out.device).unsqueeze(0) < lengths_dev.unsqueeze(1)  # (B,T)

        if self.use_attention:
            h, _ = self.attn(out, mask)  # (B,H)
        else:
            h = (out * mask.unsqueeze(-1)).sum(1) / lengths_dev.clamp(min=1).unsqueeze(-1)  # (B,H)

        if self.aux_dim and aux is not None:
            h = torch.cat([h, aux], dim=-1)  # (B, H+aux_dim)

        logits = self.head(h).squeeze(-1)  # (B,)
        return logits


# ----------------------------
# 3) Loss & metrics helpers
# ----------------------------
def _bce_loss(logits: torch.Tensor, y: torch.Tensor, pos_weight: torch.Tensor | None = None):
    return nn.functional.binary_cross_entropy_with_logits(logits, y.float(), pos_weight=pos_weight)


def compute_pos_weight(loader: DataLoader) -> torch.Tensor:
    pos = 0
    total = 0
    for batch in loader:
        y = batch["y"]
        if y.ndim > 1:
            y = y.view(-1)
        pos += int(y.sum().item())
        total += int(y.numel())
    neg = total - pos
    w = (neg / max(1, pos)) if pos > 0 else 1.0
    return torch.tensor([w], dtype=torch.float32)


# ----------------------------
# 4) Mask-aware standardization for sequences
# ----------------------------
@torch.no_grad()
def fit_ts_scaler(loader: DataLoader) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute per-feature mean/std over observed entries only (where M==1).
    Returns (mu, std) in shape (D,).
    """
    sum_ = None
    sumsq_ = None
    cnt_ = None

    for b in loader:
        X, M = b["X"], b["M"]  # (B,T,D)
        XM = X * M
        s = XM.sum(dim=(0, 1))  # (D,)
        ss = (XM * X).sum(dim=(0, 1))  # (D,)
        c = M.sum(dim=(0, 1))  # (D,)

        if sum_ is None:
            sum_ = s
            sumsq_ = ss
            cnt_ = c
        else:
            sum_ += s
            sumsq_ += ss
            cnt_ += c

    mu = torch.where(cnt_ > 0, sum_ / torch.clamp(cnt_, min=1), torch.zeros_like(sum_))
    var = torch.where(
        cnt_ > 0,
        torch.clamp(sumsq_ / torch.clamp(cnt_, min=1) - mu * mu, min=1e-8),
        torch.ones_like(mu),
    )
    std = torch.sqrt(var)
    return mu.float(), std.float()


def apply_ts_scaler(batch: dict[str, torch.Tensor], mu: torch.Tensor, std: torch.Tensor):
    X, M = batch["X"], batch["M"]
    X = (X - mu) / std  # broadcast (D,) → (B,T,D)
    X = torch.where(M.bool(), X, torch.zeros_like(X))  # keep zeros where missing
    batch["X"] = X
    return batch


# ----------------------------
# 5) Train / Eval (concat X||DT[||M]) + aux (length, span)
# ----------------------------
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optim: torch.optim.Optimizer,
    device: torch.device,
    *,
    concat_XM: bool,
    mu: torch.Tensor,
    std: torch.Tensor,
    pos_weight: torch.Tensor | None = None,
):
    model.train()
    total = 0.0
    n = 0
    for batch in loader:
        batch = batch_to_device(batch, device)
        batch = apply_ts_scaler(batch, mu.to(device), std.to(device))

        X, M, DT, lengths, y = batch["X"], batch["M"], batch["DT"], batch["lengths"], batch["y"]
        if y.ndim > 1:
            y = y.view(-1)

        # Append DT as a channel
        X_aug = torch.cat([X, DT.unsqueeze(-1)], dim=-1)  # (B,T,D+1)
        X_in = torch.cat([X_aug, M], dim=-1) if concat_XM else X_aug

        # Aux features: [length, span]
        aux = torch.stack([lengths.float().to(device), batch["span"].to(device)], dim=1)  # (B,2)

        optim.zero_grad()
        logits = model(X_in, lengths, aux)  # (B,)
        loss = _bce_loss(logits, y, pos_weight=pos_weight)
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
    concat_XM: bool,
    mu: torch.Tensor,
    std: torch.Tensor,
):
    model.eval()
    total_loss = 0.0
    n = 0
    all_prob = []
    all_true = []
    correct = 0

    for batch in loader:
        batch = batch_to_device(batch, device)
        batch = apply_ts_scaler(batch, mu.to(device), std.to(device))

        X, M, DT, lengths, y = batch["X"], batch["M"], batch["DT"], batch["lengths"], batch["y"]
        if y.ndim > 1:
            y = y.view(-1)

        X_aug = torch.cat([X, DT.unsqueeze(-1)], dim=-1)
        X_in = torch.cat([X_aug, M], dim=-1) if concat_XM else X_aug
        aux = torch.stack([lengths.float().to(device), batch["span"].to(device)], dim=1)

        logits = model(X_in, lengths, aux)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, y.float())

        prob = torch.sigmoid(logits)
        pred = (prob >= 0.5).long()
        correct += (pred.cpu() == y.long().cpu()).sum().item()

        all_prob.append(prob.cpu())
        all_true.append(y.cpu())

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


# ----------------------------
# 6) K-Fold CV (outer stratified) + inner val + early stop on AUC
# ----------------------------
def run_k_fold_cv_earlystop(
    pkl_path: str = "data/processed/COPD_PATIENTS_DATA.pkl",  # kept for interface parity
    out_dir: str = "models/GRU_classifier",
    *,
    folds_file: str | Path | None = "models/shared_folds.npy",  # shared with MLP
    batch_size: int = 128,
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.4,
    bidirectional: bool = False,
    fc_hidden: int | None = 64,
    epochs: int = 500,
    lr: float = 3e-4,
    seed: int = 42,
    concat_XM: bool = True,  # feed [X||DT||M] if True; else [X||DT]
    val_frac: float = 0.1,
    patience: int = 30,  # early stop if no val AUC improvement
    cv: int = 5,
    use_pos_weight: bool = False,
    use_attention: bool = True,  # <— enable attention pooling
    attn_dim: int = 64,
    attn_dropout: float = 0.1,
):
    """
    Outer Stratified K-Fold CV with early stopping on validation AUC.
    Uses mask-aware scaling, DT channel, (attention or masked-mean) pooling, and aux features.
    """
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ds = TimeSeriesDataset(compute=True)
    N = len(ds)
    print(f"Dataset size: N={N} samples")
    assert N > cv, f"Dataset too small for {cv}-fold CV: N={N}"

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Determine base D and resulting GRU input size Din
    tmp_loader = DataLoader(Subset(ds, [0]), batch_size=1, collate_fn=collate_grud)
    tmp_batch = next(iter(tmp_loader))
    D = tmp_batch["X"].shape[-1]  # base feature dim
    Din = (D + 1) + (D if concat_XM else 0)  # +1 for DT channel, +D for masks if concat_XM
    print(f"Base feature dim D={D} → GRU input_size={Din} (concat_XM={concat_XM}; +DT)")

    # Binary labels for stratification, same threshold as collate_grud
    y_bin = np.array([1 if float(s["y"]) < 0.7 else 0 for s in ds._cache], dtype=int)
    splits = get_or_create_folds(y_bin, cv, seed, folds_file)

    # per-fold collectors
    fold_train_bce, fold_val_bce, fold_test_bce = [], [], []
    fold_val_accs, fold_val_aucs = [], []
    fold_test_accs, fold_test_aucs = [], []

    for fold, (train_idx, test_idx) in enumerate(splits, start=1):
        print(f"\n=== Fold {fold} ===")
        fold_dir = out_path / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        # Inner split train → (train_inner, val_inner)
        rng = np.random.default_rng(seed + fold)
        perm = rng.permutation(train_idx)
        n_val = max(1, int(round(len(perm) * val_frac)))
        val_inner = perm[:n_val]
        train_inner = perm[n_val:]

        # DataLoaders
        train_loader = make_loader(ds, train_inner, batch_size=batch_size, shuffle=True)
        val_loader = make_loader(ds, val_inner, batch_size=batch_size, shuffle=False)
        test_loader = make_loader(ds, test_idx, batch_size=batch_size, shuffle=False)

        # Fit scaler on train_inner only
        mu, std = fit_ts_scaler(train_loader)

        # Model / Optim (aux_dim=2: [length, span])
        model = SimpleGRUClassifier(
            input_size=Din,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            fc_hidden=fc_hidden,
            aux_dim=2,
            use_attention=use_attention,
            attn_dim=attn_dim,
            attn_dropout=attn_dropout,
        ).to(device)
        optim = torch.optim.Adam(model.parameters(), lr=lr)

        pos_weight = compute_pos_weight(train_loader).to(device) if use_pos_weight else None

        # Early stopping on validation AUC
        best_auc = -1.0
        best_epoch = -1
        no_improve = 0
        best_train_bce_at_best = None

        for epoch in range(1, epochs + 1):
            tr_bce = train_one_epoch(
                model,
                train_loader,
                optim,
                device,
                concat_XM=concat_XM,
                mu=mu,
                std=std,
                pos_weight=pos_weight,
            )
            va_bce, va_acc, va_auc = evaluate(
                model, val_loader, device, concat_XM=concat_XM, mu=mu, std=std
            )

            improved = va_auc > best_auc + 1e-6  # maximize AUC
            if improved:
                best_auc = va_auc
                best_epoch = epoch
                best_train_bce_at_best = tr_bce
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
                        "aux_dim": 2,
                        "use_attention": use_attention,
                        "attn_dim": attn_dim,
                        "attn_dropout": attn_dropout,
                        "mu": mu,  # save scaler for reproducibility
                        "std": std,
                    },
                    fold_dir / "best.pt",
                )
                with open(fold_dir / "metrics_val.txt", "w") as f:
                    f.write(f"best_val_bce={va_bce:.6f}\n")
                    f.write(f"best_val_auc={va_auc:.6f}\n")
                    f.write(f"best_epoch={best_epoch}\n")
                    f.write(f"train_bce_at_best_val={best_train_bce_at_best:.6f}\n")
                    f.write(f"val_acc_at_best={va_acc:.6f}\n")
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

        # Load best and evaluate on TRAIN/VAL/TEST
        ckpt = torch.load(fold_dir / "best.pt", map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        mu_best, std_best = ckpt["mu"], ckpt["std"]

        train_bce_final = (
            best_train_bce_at_best
            if best_train_bce_at_best is not None
            else evaluate(
                model, train_loader, device, concat_XM=concat_XM, mu=mu_best, std=std_best
            )[0]
        )
        val_bce_final, val_acc_final, val_auc_final = evaluate(
            model, val_loader, device, concat_XM=concat_XM, mu=mu_best, std=std_best
        )
        test_bce, test_acc, test_auc = evaluate(
            model, test_loader, device, concat_XM=concat_XM, mu=mu_best, std=std_best
        )

        with open(fold_dir / "metrics_final.txt", "w") as f:
            f.write(f"train_bce_at_best={train_bce_final:.6f}\n")
            f.write(f"val_bce_best={val_bce_final:.6f}\n")
            f.write(f"val_acc_best={val_acc_final:.6f}\n")
            f.write(f"val_auc_best={val_auc_final:.6f}\n")
            f.write(f"test_bce={test_bce:.6f}\n")
            f.write(f"test_acc={test_acc:.6f}\n")
            f.write(f"test_auc={test_auc:.6f}\n")

        print(
            f"Fold {fold} → best_epoch={best_epoch} | "
            f"train_bce@best={train_bce_final:.6f} | "
            f"val_bce_best={val_bce_final:.6f} (acc={val_acc_final:.4f}, auc={val_auc_final:.4f}) | "
            f"test: bce={test_bce:.6f}, acc={test_acc:.4f}, auc={test_auc:.4f}"
        )

        # collect per-fold metrics
        fold_train_bce.append(float(train_bce_final))
        fold_val_bce.append(float(val_bce_final))
        fold_test_bce.append(float(test_bce))
        fold_val_accs.append(float(val_acc_final))
        fold_val_aucs.append(float(val_auc_final))
        fold_test_accs.append(float(test_acc))
        fold_test_aucs.append(float(test_auc))

    # Summary across folds
    def stats(x: list[float]) -> tuple[float, float]:
        return float(np.mean(x)), float(np.std(x, ddof=0))

    train_bce_avg, train_bce_std = stats(fold_train_bce)
    val_bce_avg, val_bce_std = stats(fold_val_bce)
    test_bce_avg, test_bce_std = stats(fold_test_bce)

    val_acc_avg, val_acc_std = stats(fold_val_accs)
    val_auc_avg, val_auc_std = stats(fold_val_aucs)
    test_acc_avg, test_acc_std = stats(fold_test_accs)
    test_auc_avg, test_auc_std = stats(fold_test_aucs)

    with open(Path(out_dir) / "cv_summary.txt", "w") as f:
        for i, (trb, vab, teb, vA, vU, tA, tU) in enumerate(
            zip(
                fold_train_bce,
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
                f"fold_{i}: "
                f"train_bce={trb:.6f} | val_bce={vab:.6f} | test_bce={teb:.6f} | "
                f"val_acc={vA:.4f} | val_auc={vU:.4f} | "
                f"test_acc={tA:.4f} | test_auc={tU:.4f}\n"
            )

        f.write(
            f"\nAverages (± SD):\n"
            f"train_bce={train_bce_avg:.6f} ± {train_bce_std:.6f}\n"
            f"val_bce={val_bce_avg:.6f} ± {val_bce_std:.6f}\n"
            f"test_bce={test_bce_avg:.6f} ± {test_bce_std:.6f}\n"
            f"val_acc={val_acc_avg:.4f} ± {val_acc_std:.4f}\n"
            f"val_auc={val_auc_avg:.4f} ± {val_auc_std:.4f}\n"
            f"test_acc={test_acc_avg:.4f} ± {test_acc_std:.4f}\n"
            f"test_auc={test_auc_avg:.4f} ± {test_auc_std:.4f}\n"
        )

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
# 7) CLI entry
# ----------------------------
if __name__ == "__main__":
    run_k_fold_cv_earlystop(
        pkl_path="data/processed/COPD_PATIENTS_DATA.pkl",
        out_dir="models/GRUA_classifier",
        folds_file="models/shared_folds.npy",  # <— shared with MLP for fair comparison
        batch_size=128,
        hidden_size=64,
        num_layers=2,
        dropout=0.4,
        bidirectional=False,
        fc_hidden=64,
        epochs=500,  # upper bound; early stopping will stop sooner
        lr=3e-4,
        seed=42,
        concat_XM=True,  # feed [X||DT||M]; recommended
        val_frac=0.1,  # 10% of outer-train becomes inner-val
        patience=30,  # early stop if no val AUC improvement for 30 epochs
        cv=5,  # 5-fold cross-validation
        use_pos_weight=False,
        use_attention=True,  # <— attention pooling enabled
        attn_dim=64,
        attn_dropout=0.1,
    )
