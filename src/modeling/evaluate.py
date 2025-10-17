import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

# Example: Replace these with your actual data
y_true = np.array([...])  # true values
y_pred = np.array([...])  # model predictions

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
