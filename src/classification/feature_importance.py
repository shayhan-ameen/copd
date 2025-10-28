from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# ----------------------------------------------------------
# 1. Load and merge per-fold feature importance CSVs
# ----------------------------------------------------------
models = ["XGBoost_classifier"]

for model in models:
    print(f"Evaluating model: {model}")

    base_dir = Path(f"models/{model}")
    fold_dirs = sorted(base_dir.glob("fold_*"))
    imp_all = []

    for fd in fold_dirs:
        f = fd / "feature_importance_gain.csv"
        if f.exists():
            imp_df = pd.read_csv(f)
            imp_df["fold"] = fd.name
            imp_all.append(imp_df)

    if not imp_all:
        raise FileNotFoundError("No feature_importance_gain.csv files found in fold directories.")

    imp_all_df = pd.concat(imp_all, ignore_index=True)

    # ----------------------------------------------------------
    # 2. Aggregate mean gain across folds
    # ----------------------------------------------------------
    agg = (
        imp_all_df.groupby("feature", as_index=False)["gain"]
        .mean()
        .sort_values("gain", ascending=False)
    )
    agg["norm_gain"] = agg["gain"] / agg["gain"].max()

    # ----------------------------------------------------------
    # 3. Plot top-N features
    # ----------------------------------------------------------
    TOP_N = 20
    top_df = agg.head(TOP_N)

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(8, 6))

    # plot bars on ax
    bars = ax.barh(
        y=top_df["feature"][::-1],
        width=top_df["gain"][::-1],
        color=plt.cm.viridis(top_df["norm_gain"][::-1]),
    )

    ax.set_xlabel("Feature Importance", fontsize=12)
    ax.set_ylabel("")
    ax.set_title("Feature Importance of XGBoost Model", fontsize=13)

    # colorbar attached to this Axes
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=0, vmax=1))
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Normalized Importance", fontsize=11)

    plt.tight_layout()
    plt.savefig(base_dir / "feature_importance.png", dpi=300)
    plt.show()
