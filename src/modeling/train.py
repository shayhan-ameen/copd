# from pathlib import Path

# import typer
# from loguru import logger
# from tqdm import tqdm

# from src.config import MODELS_DIR, PROCESSED_DATA_DIR

# app = typer.Typer()


# @app.command()
# def main(
#     # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
#     features_path: Path = PROCESSED_DATA_DIR / "features.csv",
#     labels_path: Path = PROCESSED_DATA_DIR / "labels.csv",
#     model_path: Path = MODELS_DIR / "model.pkl",
#     # -----------------------------------------
# ):
#     # ---- REPLACE THIS WITH YOUR OWN CODE ----
#     logger.info("Training some model...")
#     for i in tqdm(range(10), total=10):
#         if i == 5:
#             logger.info("Something happened for iteration 5.")
#     logger.success("Modeling training complete.")
#     # -----------------------------------------


# if __name__ == "__main__":
#     app()


# Note: You can run this script from the command line as follows:
#       python src/modeling/train.py

from __future__ import annotations

import os
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

from src.modeling.architectures.GRUD import GRUDRegressor
from src.modeling.ragged_timeseries import COPDGRUDDataset, collate_grud


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


def train_one_epoch(model, loader, optim, device):
    model.train()
    total = 0.0
    for batch in loader:
        X = batch["X"].to(device)
        M = batch["M"].to(device)
        DT = batch["DT"].to(device)
        L = batch["lengths"].to(device)
        y = batch["y"].to(device)

        optim.zero_grad()
        yhat = model(X, M, DT, L)
        loss = nn.functional.mse_loss(yhat, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        total += loss.item() * X.size(0)
    return total / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total = 0.0
    for batch in loader:
        X = batch["X"].to(device)
        M = batch["M"].to(device)
        DT = batch["DT"].to(device)
        L = batch["lengths"].to(device)
        y = batch["y"].to(device)
        yhat = model(X, M, DT, L)
        loss = nn.functional.mse_loss(yhat, y)
        total += loss.item() * X.size(0)
    return total / len(loader.dataset)


def train_main(
    pkl_path: str = "data/processed/COPD_PATIENTS_DATA.pkl",
    out_dir: str = "models/exp_grud",
    batch_size: int = 64,
    hidden_size: int = 64,
    num_layers: int = 1,
    epochs: int = 20,
    lr: float = 3e-4,
    val_frac: float = 0.1,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    ds = COPDGRUDDataset(pkl_path, include_age=True, include_gender=False, include_dt_feature=False)

    # with open("data/processed/COPDGRUDDataset.pkl", "wb") as f:
    #     pickle.dump(ds, f, protocol=pickle.HIGHEST_PROTOCOL)

    # with open("data/processed/COPDGRUDDataset.pkl", "rb") as f:
    #     ds = pickle.load(f)

    # split
    n_total = len(ds)
    n_val = max(1, int(n_total * val_frac))
    n_train = n_total - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val])

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_grud,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_grud,
    )

    # model
    # infer D from one batch
    batch0 = next(iter(train_loader))
    D = batch0["X"].shape[-1]
    model = GRUDRegressor(input_size=D, hidden_size=hidden_size, num_layers=num_layers).to(device)

    # set input means for GRU-D decay
    means = compute_feature_means(train_loader).to(device)
    model.set_input_means(means)

    optim = torch.optim.Adam(model.parameters(), lr=lr)

    best_val = float("inf")
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        tr = train_one_epoch(model, train_loader, optim, device)
        va = evaluate(model, val_loader, device)
        print(f"[Epoch {epoch:03d}] train MSE={tr:.4f} | val MSE={va:.4f}")

        # save best
        if va < best_val:
            best_val = va
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "feature_means": means.cpu(),
                    "D": D,
                    "hidden_size": hidden_size,
                    "num_layers": num_layers,
                },
                os.path.join(out_dir, "best.pt"),
            )
            with open(os.path.join(out_dir, "metrics.txt"), "w") as f:
                f.write(f"val_mse={va:.6f}\n")

    print(f"Done. Best val MSE={best_val:.4f}  →  {out_dir}/best.pt")


if __name__ == "__main__":
    train_main()
