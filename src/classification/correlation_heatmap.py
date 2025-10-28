import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ----------------------------------------------------------
# 1. Example: load your feature dataset
# ----------------------------------------------------------
# Replace this with your actual processed dataset
df = pd.read_csv("models/xgb_classifier_cv/all_folds_predictions.csv")
# Or directly use your clinical dataset, e.g.:
# df = pd.read_csv("data/processed/copd_features.csv")

# ----------------------------------------------------------
# 2. Select COPD-related numeric features
# ----------------------------------------------------------
# Example subset (replace with actual COPD feature columns from your dataset)
features = [
    "height",
    "weight",
    "Age",
    "Sex",
    "Pre-FVC (L)",
    "Post-FVC (L)",
    "Pre-FEV1 (L)",
    "Post-FEV1 (L)",
    "Pre-FEV1/FVC",
    "Post-FEV1/FVC",
    "DLCO_Pred",
    "FEF25_75_Meas",
    "FEF25_75_perc_Pred",
    "Smoker",
    "Cough",
    "Dyspnea",
    "COPD",
]

# Keep only columns that exist in df
cols = [c for c in features if c in df.columns]
df_sub = df[cols].copy()

# Convert categorical variables if needed
if "Sex" in df_sub.columns:
    df_sub["Sex"] = df_sub["Sex"].map({"M": 1, "F": 0})

# ----------------------------------------------------------
# 3. Compute correlation matrix
# ----------------------------------------------------------
corr = df_sub.corr(method="pearson").round(2)

# ----------------------------------------------------------
# 4. Plot upper-triangle correlation heatmap
# ----------------------------------------------------------
mask = np.triu(np.ones_like(corr, dtype=bool))

plt.figure(figsize=(12, 8))
sns.set(style="white")

# Custom diverging colormap (blue → white → red)
cmap = sns.diverging_palette(240, 10, as_cmap=True)

ax = sns.heatmap(
    corr,
    mask=mask,
    cmap=cmap,
    annot=True,
    fmt=".2f",
    center=0,
    vmin=-1,
    vmax=1,
    square=True,
    linewidths=0.5,
    cbar_kws={"shrink": 0.8, "label": "Correlation Coefficient"},
    annot_kws={"size": 8, "color": "black"},
)

plt.xticks(rotation=45, ha="right", fontsize=9)
plt.yticks(rotation=0, fontsize=9)
plt.title("Correlation Coefficients among COPD Features", fontsize=13, pad=15)
plt.tight_layout()

# ----------------------------------------------------------
# 5. Save and show
# ----------------------------------------------------------
plt.savefig("figures/copd_feature_correlation_heatmap.png", dpi=300)
plt.show()

print("✅ Saved figure: figures/copd_feature_correlation_heatmap.png")
