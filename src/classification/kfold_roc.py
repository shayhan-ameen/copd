from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve

models = ["XGBoost_classifier"]

for model in models:
    print(f"Evaluating model: {model}")
    # -----------------------------
    # Load predictions
    # -----------------------------
    csv_path = Path(f"models/{model}/all_folds_predictions.csv")
    df = pd.read_csv(csv_path)

    # -----------------------------
    # Initialize
    # -----------------------------
    folds = sorted(df["fold"].unique())
    tprs, aucs = [], []
    mean_fpr = np.linspace(0, 1, 100)

    plt.figure(figsize=(7, 6))

    # -----------------------------
    # Compute ROC for each fold
    # -----------------------------
    for fold in folds:
        d = df[df["fold"] == fold]
        y_true = d["y_true"].values
        y_prob = d["y_prob"].values

        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)

        # interpolate for mean curve
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)

        plt.plot(fpr, tpr, lw=1, alpha=0.4, label=f"Fold {fold} (AUC = {roc_auc:.2f})")

    # -----------------------------
    # Compute mean + std
    # -----------------------------
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    plt.plot(
        mean_fpr,
        mean_tpr,
        color="navy",
        label=f"Mean ROC (AUC = {mean_auc:.2f} ± {std_auc:.2f})",
        lw=2,
        alpha=0.9,
    )

    # ±1 std shading
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color="grey",
        alpha=0.2,
        label="± 1 std. dev.",
    )

    # -----------------------------
    # Plot details
    # -----------------------------
    plt.plot([0, 1], [0, 1], linestyle="--", color="r", lw=2, label="Chance")
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("K-Fold Cross Validation ROC (XGBoost)")
    plt.legend(loc="lower right", fontsize=9)
    plt.tight_layout()

    # Save figure
    plt.savefig(f"models/{model}/kfold_mean_roc.png", dpi=300)
    plt.show()
