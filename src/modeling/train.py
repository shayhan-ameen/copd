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

import torch

from src.modeling.architectures.GRUD import GRUDRegressor

# Suppose you have padded history batches:
# X:  (B, T, D)      values (NaN→0)
# M:  (B, T, D)      mask (1 if observed else 0)
# DT: (B, T)         time gap between steps (e.g., in days)
# L:  (B,)           true lengths per sequence

B, T, D = 8, 20, 128
X = torch.randn(B, T, D)
M = (torch.rand(B, T, D) > 0.2).float()
DT = torch.zeros(B, T)
DT[:, 1:] = torch.randint(1, 5, (B, T - 1)).float()  # example gaps
L = torch.randint(low=T // 2, high=T, size=(B,))

model = GRUDRegressor(input_size=D, hidden_size=64, num_layers=1)
# set per-feature means (e.g., dataset means)
model.set_input_means(X[M.bool()].reshape(-1, D).nanmean(dim=0).fill_(0.0))  # demo

yhat = model(X, M, DT, L)  # (B,)
loss = ((yhat - torch.randn(B)) ** 2).mean()
loss.backward()
