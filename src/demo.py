from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict
import pandas as pd
 

# FIGURES_DIR = Path("reports/figures")

def detect_anomalies(dfx: pd.DataFrame,
                     date_col: str = "Prescription Date",
                     value_col: str = "Result Numerical Value",
                     measure_col: str = "Measurement",
                     variable_col: str = "Variable",
                     z_thresh: float = 3.5,
                     jump_thresh_per_month: Dict[str, float] | None = None
                    ) -> Tuple[pd.DataFrame, str]:
    """
    Returns:
      flagged_df (same rows as dfx, with anomaly columns)
      title_tag  (e.g., 'Missing value+Out of valid range+MAD+Jump' for this patient)

    Flags:
      - Anomaly_Missing: value == 0 or NaN
      - Anomaly_Range: outside clinical ranges
          FEV1 (Meas/Post_Meas): 0.2–10.0
          FVC  (Meas/Post_Meas): 0.3–10.0
          DLCO (Meas/Post_Meas): 0.3–50
          FEV1/FVC ratio (Meas/Post_Meas): 0.2–1.2
          %Pred ( %Pred/Post_%Pred ): 0–200
      - Outlier_MAD: |robust_z| > z_thresh (per-measurement series)
      - Outlier_Jump: month-normalized step exceeds measurement threshold
    """
    jump_thresh_per_month = jump_thresh_per_month or {
        "FEV1": 0.30, "FVC": 0.40, "DLCO": 3.0, "FEV1/FVC": 0.08, "DLCO/VA": 0.60,
    }
    abs_vars   = {"Meas", "Post_Meas"}
    perc_vars  = {"%Pred", "Post_%Pred"}

    df2 = dfx.copy()

    # Ensure numeric + datetime for calculations
    v = pd.to_numeric(df2[value_col], errors="coerce")
    dt = pd.to_datetime(df2[date_col], errors="coerce")
    df2["_val_"] = v
    df2["_date_"] = dt

    # 1) Missing (as requested: treat 0 as missing; also NaN is missing)
    df2["Anomaly_Missing"] = v.isna() | (v == 0)

    # 2) Valid ranges
    rng_flag = pd.Series(False, index=df2.index)

    def _violate(series_mask, low, high):
        if not series_mask.any(): 
            return pd.Series(False, index=df2.index)
        s = v.where(series_mask)
        return (s < low) | (s > high)

    # Absolute measurements (Meas/Post_Meas)
    m = df2[measure_col].astype(str)
    var = df2[variable_col].astype(str)

    mask_abs = var.isin(abs_vars)
    rng_flag |= _violate(mask_abs & (m == "FEV1"), 0.2, 10.0)
    rng_flag |= _violate(mask_abs & (m == "FVC"),  0.3, 10.0)
    rng_flag |= _violate(mask_abs & (m == "DLCO"), 0.3, 50.0)

    # Ratio FEV1/FVC (absolute)
    rng_flag |= _violate(mask_abs & (m == "FEV1/FVC"), 0.2, 1.2)

    # Percent predicted
    mask_perc = var.isin(perc_vars)
    rng_flag |= _violate(mask_perc, 0.0, 200.0)

    df2["Anomaly_Range"] = rng_flag.fillna(False)

    # 3) Outliers (MAD + Jump) per measurement series
    out_mad  = pd.Series(False, index=df2.index)
    out_jump = pd.Series(False, index=df2.index)

    for mi, g in df2.groupby(measure_col, dropna=False):
        g = g.sort_values("_date_")
        vv = g["_val_"]
        dd = g["_date_"]

        # MAD
        med = vv.median()
        mad = float(np.median(np.abs(vv - med))) if len(vv) else 0.0
        mad = mad if mad > 0 else 1e-9
        robust_z = 0.6745 * (vv - med) / mad
        out_mad.loc[g.index] = robust_z.abs() > z_thresh

        # Jump per month
        dv = vv.diff().abs()
        dt_days = dd.diff().dt.days
        dt_days = dt_days.where(dt_days > 0, 1)  # avoid 0/NaN/<=0
        months = dt_days / 30.0
        thr = jump_thresh_per_month.get(str(mi), np.inf)
        out_jump.loc[g.index] = (dv / months) > thr

    df2["Outlier_MAD"]  = out_mad.fillna(False)
    df2["Outlier_Jump"] = out_jump.fillna(False)
    df2["Outlier"]      = df2["Outlier_MAD"] | df2["Outlier_Jump"]

    # Compose row-wise tags
    def _row_tags(row):
        tags = []
        if row["Anomaly_Missing"]: tags.append("Missing value")
        if row["Anomaly_Range"]:   tags.append("Out of valid range")
        if row["Outlier_MAD"]:     tags.append("MAD")
        if row["Outlier_Jump"]:    tags.append("Jump")
        return " | ".join(tags)

    df2["Anomaly_Tags"] = df2.apply(_row_tags, axis=1)
    # Short tag for tight annotations
    df2["Anomaly_Tags_Short"] = (df2["Anomaly_Tags"]
                                 .str.replace("Missing value", "Miss", regex=False)
                                 .str.replace("Out of valid range", "Range", regex=False))

    df2["Anomaly_Any"] = df2[["Anomaly_Missing","Anomaly_Range","Outlier_MAD","Outlier_Jump"]].any(axis=1)

    # Build patient-level title tag
    present = []
    anomaly_flag = df2["Anomaly_Any"].any()
    if df2["Anomaly_Missing"].any(): present.append("Missing value")
    if df2["Anomaly_Range"].any():   present.append("Out of valid range")
    if df2["Outlier_MAD"].any():     present.append("MAD")
    if df2["Outlier_Jump"].any():    present.append("Jump")
    title_tag = "+".join(present)

    # Clean temp cols for plotting (keep dates numeric as original)
    df2 = df2.drop(columns=["_val_", "_date_"])

    return df2, title_tag, anomaly_flag

df = pd.read_csv(Path(r"D:\Research\Project_COPD\COPD\data\interim\ALL_PRESCRIPTION_DATA_FILTERED.csv"))
df["Prescription Date"] = pd.to_datetime(df["Prescription Date"], format="%Y%m%d", errors="coerce")
df["Result Numerical Value"] = pd.to_numeric(df["Result Numerical Value"], errors="coerce")

pid = 29903

dfx = df[df["Patient Number"] == pid]
dfx_flagged, title_tag, anomaly_flag = detect_anomalies(dfx)