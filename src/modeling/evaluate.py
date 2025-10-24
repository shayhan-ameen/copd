from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score

from src.config import FIGURES_DIR

# Example: Replace these with your actual data
df = pd.read_csv("models/xgb_cv_no_es/all_folds_test_predictions_sorted_by_error.csv")
# df = pd.read_csv("models/exp_tstransformer_cv_es/all_folds_test_predictions_sorted_by_error.csv")
print("XGBoost")

"exp_tstransformer_cv_es"
"all_folds_test_predictions_sorted_by_error.csv"

# y_true = np.array([...])  # true values
# y_pred = np.array([...])  # model predictions

y_true = np.array(df["y_true"])  # true values
y_pred = np.array(df["y_pred"])  # model predictions

# Compute MSE and RMSE
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)

# Compute variance, mean, and range of true target
var_y = np.var(y_true)
mean_y = np.mean(y_true)
range_y = np.max(y_true) - np.min(y_true)

# 1. Normalized MSE (NMSE)
nmse = mse / var_y

# 2. Relative RMSE
rel_rmse_range = rmse / range_y
rel_rmse_mean = rmse / mean_y

# 3. R² Score
r2 = r2_score(y_true, y_pred)

# Print results
print(f"MSE: {mse:.6f}")
print(f"RMSE: {rmse:.6f}")
print(f"NMSE: {nmse:.6f}")
print(f"Relative RMSE (range): {rel_rmse_range * 100:.2f}%")
print(f"Relative RMSE (mean): {rel_rmse_mean * 100:.2f}%")
print(f"R² Score: {r2:.4f}")


# Plot histogram
# Compute relative errors (as percentage)
# Compute relative errors
df["relative_error"] = (df["y_pred"] - df["y_true"]) / df["y_true"] * 100

# Plot
plt.figure(figsize=(10, 6))
sns.histplot(df["relative_error"], bins=40, kde=True, color="steelblue")
plt.title("Histogram of Relative Errors (%) - XGBoost")
plt.xlabel("Relative Error (%)")
plt.ylabel("Count")
plt.axvline(0, color="red", linestyle="--", linewidth=1.5, label="Zero Error Line")

# Add formula text
plt.text(
    0.05,
    0.95,
    r"$\mathrm{Relative\ Error} = \frac{\hat{y} - y}{y} \times 100$",
    transform=plt.gca().transAxes,
    fontsize=12,
    verticalalignment="top",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.7),
)

plt.legend()
plt.tight_layout()

# Save
output_dir = Path(FIGURES_DIR / "other/relative_error.png")
plt.savefig(output_dir, dpi=300)


# import matplotlib.pyplot as plt
# import seaborn as sns

# # Set the style and figure size
# # plt.style.use('seaborn')
# plt.figure(figsize=(10, 6))

# # Create histogram with KDE
# sns.histplot(data=target, bins=20, kde=True, color="skyblue")

# # Customize the plot
# plt.xlabel("Target Value (FEV1/FVC)", fontsize=12)
# plt.ylabel("Frequency", fontsize=12)
# plt.title("Distribution of Target Variable (FEV1/FVC)", fontsize=14, pad=15)

# # Add grid and customize appearance
# plt.grid(True, alpha=0.3)
# sns.despine(left=False, bottom=False)

# # Adjust layout and display
# plt.tight_layout()
# plt.show()
