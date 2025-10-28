from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# -----------------------------
# Load predictions
# -----------------------------

# models = ["XGBoost_classifier", "MLP_classifier", RandomForest_classifier, GRU_classifier, TS_Transformer_cls]
models = ["XGBoost_classifier"]


for model in models:
    print(f"Evaluating model: {model}")

    csv_path = Path(f"models/{model}/all_folds_predictions.csv")
    df = pd.read_csv(csv_path)

    y_true = df["y_true"].values
    y_pred = df["y_pred"].values
    y_prob = df["y_prob"].values

    # -----------------------------
    # Check label distribution
    # -----------------------------
    total_samples = len(y_true)
    count_1 = np.sum(y_true == 1)
    count_0 = np.sum(y_true == 0)

    print(f"\nLabel distribution for {model}:")
    print(f"  Total samples: {total_samples}")
    print(f"  Class 1 (Positive): {count_1} ({count_1 / total_samples:.2%})")
    print(f"  Class 0 (Negative): {count_0} ({count_0 / total_samples:.2%})\n")

    # -----------------------------
    # Compute main metrics
    # -----------------------------
    auc = roc_auc_score(y_true, y_prob)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # -----------------------------
    # Bootstrap 95% Confidence Intervals
    # -----------------------------
    def bootstrap_ci(metric_fn, y_true, y_pred_or_prob, n_boot=2000, alpha=0.95, is_prob=False):
        rng = np.random.default_rng(42)
        stats = []
        n = len(y_true)
        for _ in range(n_boot):
            idx = rng.integers(0, n, n)
            y_t = y_true[idx]
            y_p = y_pred_or_prob[idx]
            try:
                if is_prob:
                    val = metric_fn(y_t, y_p)
                else:
                    val = metric_fn(y_t, y_p)
                stats.append(val)
            except Exception:
                continue
        lower = np.percentile(stats, (1 - alpha) / 2 * 100)
        upper = np.percentile(stats, (1 + alpha) / 2 * 100)
        return (lower, upper)

    auc_ci = bootstrap_ci(roc_auc_score, y_true, y_prob, is_prob=True)
    acc_ci = bootstrap_ci(accuracy_score, y_true, y_pred)
    prec_ci = bootstrap_ci(precision_score, y_true, y_pred)
    rec_ci = bootstrap_ci(recall_score, y_true, y_pred)
    f1_ci = bootstrap_ci(f1_score, y_true, y_pred)

    # -----------------------------
    # Summary table
    # -----------------------------
    # summary = pd.DataFrame(
    #     {
    #         "Metric": ["AUC", "Accuracy", "Precision", "Recall", "F1-Score"],
    #         "Value": [auc, acc, prec, rec, f1],
    #         "95% CI (lower)": [auc_ci[0], acc_ci[0], prec_ci[0], rec_ci[0], f1_ci[0]],
    #         "95% CI (upper)": [auc_ci[1], acc_ci[1], prec_ci[1], rec_ci[1], f1_ci[1]],
    #     }
    # )

    # # Print and save
    # print(summary.to_string(index=False, float_format="%.4f"))
    # summary.to_csv(f"models/{model}/metrics_with_CI.csv", index=False)

    summary = pd.DataFrame(
        {
            "AUC (95% CI)": [f"{auc:.3f}", f"({auc_ci[0]:.3f}–{auc_ci[1]:.3f})"],
            "F1-Score": [f"{f1:.3f}", f"({f1_ci[0]:.3f}–{f1_ci[1]:.3f})"],
            "Accuracy": [f"{acc:.3f}", f"({acc_ci[0]:.3f}–{acc_ci[1]:.3f})"],
            "Precision": [f"{prec:.3f}", f"({prec_ci[0]:.3f}–{prec_ci[1]:.3f})"],
            "Recall": [f"{rec:.3f}", f"({rec_ci[0]:.3f}–{rec_ci[1]:.3f})"],
        }
    )

    # # Print as 2-line style table
    print(summary.to_string(index=False, float_format="%.4f"))
    summary.to_csv(f"models/{model}/metrics_summary_95CI.csv", index=False)
