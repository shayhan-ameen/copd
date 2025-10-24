# src/features.py (wide-format version)
from __future__ import annotations

import os
import pickle
from collections import OrderedDict
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from src.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR

# -------------------------------------------------------------------
# Expect PATIENT_MAP and IGNORE_PATIENT_MAP to be imported/defined
# exactly as you posted in your message.
# -------------------------------------------------------------------
# from src.patient_map import PATIENT_MAP, IGNORE_PATIENT_MAP
# (Or just keep them in this file above.)


#  ----------------------------
# 0) Clinical feature template
# ----------------------------
# Use OrderedDict to lock a deterministic traversal order (important for models).

PATIENT_MAP: dict[str, list[str]] = OrderedDict(
    {
        "basic_info": ["H_no", "number_of_visit"],
        "test_result": [
            # "dt_gap
            # "Sex"
            "ht",
            "bw",
            "Age",
            "DL_Adj_Meas",
            "DL_Adj_Pred",
            "DL_Adj_perc_Pred",
            "DLCO_VA_Meas",
            "DLCO_VA_Pred",
            "DLCO_VA_perc_Pred",
            "DLCO_Meas",
            "DLCO_Pred",
            "DLCO_perc_Pred",
            "ERV_Meas",
            "ERV_Pred",
            "ERV_perc_Pred",
            "FEF25_75_Meas",
            "FEF25_75_Pred",
            "FEF25_75_perc_Pred",
            "FEF25_75_perc_Meas",
            "FEF25_75_perc_perc_Pred",
            "FEF25_perc_Meas",
            "FEF25_perc_Pred",
            "FEF25_perc_perc_Pred",
            "FEF50_perc_Meas",
            "FEF50_perc_Pred",
            "FEF50_perc_perc_Pred",
            "FEF75_perc_Meas",
            "FEF75_perc_Pred",
            "FEF75_perc_perc_Pred",
            "FEF_FIF50_Meas",
            "FEF_FIF50_Pred",
            "FEF_FIF50_perc_Pred",
            "FET100_Meas",
            "FET100_perc_Meas",
            "FET100_perc_Pred",
            "FET100_perc_perc_Pred",
            "FEV1_FVC_Meas",
            "FEV1_FVC_perc_Pred",
            "FEV1_FVC_Pred",
            "FEV1_Meas",
            "FEV1_Pred",
            "FEV1_perc_Pred",
            "FIV1_Meas",
            "FIV1_Pred",
            "FIV1_perc_Pred",
            "FIVC_Meas",
            "FIVC_Pred",
            "FIVC_perc_Pred",
            "FRC_Meas",
            "FRC_Pred",
            "FRC_perc_Pred",
            "FVC_Meas",
            "FVC_Pred",
            "FVC_perc_Pred",
            "IC_Meas",
            "IC_Pred",
            "IC_perc_Pred",
            "IVC_Meas",
            "IVC_Pred",
            "IVC_perc_Pred",
            "PEF_Meas",
            "PEF_Pred",
            "PEF_perc_Pred",
            "PIF_Meas",
            "PIF_Pred",
            "PIF_perc_Pred",
            "RV_Meas",
            "RV_Pred",
            "RV_TLC_Meas",
            "RV_TLC_Pred",
            "RV_TLC_perc_Pred",
            "RV_perc_Pred",
            "TLC_Meas",
            "TLC_Pred",
            "TLC_perc_Pred",
            "VA_Meas",
            "VA_Pred",
            "VA_perc_Pred",
            "VC_Meas",
            "VC_Pred",
            "VC_perc_Pred",
            "postFEF25_75Meas",
            "postFEF25_75_percChg",
            "postFEF25_75_percPred",
            "postFEF25_75_perc_Meas",
            "postFEF25_75_perc_perc_Chg.",
            "postFEF25_75_perc_perc_Pred",
            "postFEF25_perc_Meas",
            "postFEF25_perc_perc_Chg.",
            "postFEF25_perc_perc_Pred",
            "postFEF50_perc_Meas",
            "postFEF50_perc_perc_Chg.",
            "postFEF50_perc_perc_Pred",
            "postFEF75_perc_Meas",
            "postFEF75_perc_perc_Chg.",
            "postFEF75_perc_perc_Pred",
            "postFEF_FIF50_Meas",
            "postFEF_FIF50_perc_Chg.",
            "postFEF_FIF50_perc_Pred",
            "postFET100Meas",
            "postFET100_percChg",
            "postFET100_perc_Meas",
            "postFET100_perc_perc_Chg.",
            "postFET100_perc_perc_Pred",
            "postFEV1_FVC_Meas",
            "postFEV1FVC_percChg",
            "postFEV1_FVC_perc_Pred",
            "postFEV1_Meas",
            "postFEV1_FVC_perc_Chg.",
            "postFEV1_percChg",
            "postFEV1_perc_Pred",
            "postFEV1_perc_Chg.",
            "postFIV1_Meas",
            "postFIV1_perc_Chg.",
            "postFIV1_perc_Pred",
            "postFIVC_Meas",
            "postFIVC_percChg",
            "postFIVC_perc_Pred",
            "postFIVC_perc_Chg.",
            "postFVC_Meas",
            "postFVC_percChg",
            "postFVC_perc_Pred",
            "postFVC_perc_Chg.",
            "postPEF_Meas",
            "postPEF_percChg",
            "postPEF_perc_Pred",
            "postPEF_perc_Chg.",
            "postPIF_perc_Chg.",
            "postPIF_perc_Pred",
            "postPPIF_Meas",
        ],
        "treatment_info": [
            "Inhaler_Name",
            "Drug_Class",
            "Device_Type",
            "Inhaler_Start_Date",
            "Inhaler_End_Date",
            "PFT_Timepoint",
            "Exacerbation",
            "Exacerbation_Type",
            "Exacerbation_Date",
            "Exacerbation_Treatment",
            "Exacerbation_2",
            "Exacerbation_Type_2",
            "Exacerbation_Date_2",
            "Exacerbation_Treatment_2",
        ],
        "target_info": [
            "y",
            "y_index",
            "y_source",
            "y_date",
        ],
    }
)

PATIENT_MAP_CONTROLLER = {
    "basic_info": "basic_info",
    "test_result": "test_result",
    "treatment_info": "treatment_info",
    "target_info": "target_info",
    "ignore": {
        "basic_info": ["H_no", "Sex", "dob", "number_of_visit"],
        "test_result": [
            "PFT_date",
            "DL_Adj_Pred",
            "DLCO_VA_Pred",
            "DLCO_Pred",
            "ERV_Pred",
            "FEF25_75_Pred",
            "FEF_FIF50_Pred",
            "FEV1_FVC_Pred",
            "FEV1_Pred",
            "FIV1_Pred",
            "FIVC_Pred",
            "FRC_Pred",
            "FVC_Pred",
            "IC_Pred",
            "IVC_Pred",
            "PEF_Pred",
            "PIF_Pred",
            "RV_Pred",
            "RV_TLC_Pred",
            "TLC_Pred",
            "VA_Pred",
            "VC_Pred",
        ],
        "treatment_info": [
            "Exacerbation_2",
            "Exacerbation_Type_2",
            "Exacerbation_Date_2",
            "Exacerbation_Treatment_2",
        ],
    },
}


# ----------------------------
# 0) Helpers
# ----------------------------
def _norm(s: str) -> str:
    return (s or "").strip().lower()


def _age_years_from(dob: pd.Timestamp | None, on_date: pd.Timestamp | None) -> int | None:
    if pd.isna(dob) or pd.isna(on_date):
        return None
    return int((on_date.date() - dob.date()).days // 365)


def _parse_date_any(x) -> pd.Timestamp | None:
    if pd.isna(x):
        return pd.NaT
    ts = pd.to_datetime(x, errors="coerce")
    if pd.isna(ts):
        # Try YYYYMMDD first (your classic format), then fallback
        ts = pd.to_datetime(x, format="%Y%m%d", errors="coerce")

    return ts


# def _maybe_scale_ratio(value: float | None, colname: str) -> float | None:
#     """
#     Convert obvious percent ratios into [0,1] if they look like 70.0 for FEV1/FVC etc.
#     Trigger only for FEV1/FVC (pre/post) measured columns.
#     """
#     if value is None or pd.isna(value):
#         return None
#     name = _norm(colname)
#     is_fev1fvc_meas = name in {"fev1_fvc_meas", "postfev1_fvc_meas"}
#     if is_fev1fvc_meas and value > 1.5:  # likely in [0,100]
#         return float(value) / 100.0
#     return float(value)


def _coerce_numeric_inplace(df: pd.DataFrame, cols: Iterable[str]) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")


def _coerce_dates_inplace(df: pd.DataFrame, cols: Iterable[str]) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = df[c].apply(_parse_date_any)


# ----------------------------
# 2) Per-visit feature extraction (wide)
# ----------------------------
def _make_visit_feature_record(
    visit_df,
    *gender_val: Any,
    dt_gap: int,
    test_cols: list[str],
    tol: float = 1e-6,
) -> dict[str, Any]:
    """
    Builds a per-visit dictionary:
      - 'Age' (use row['Age'] if present else compute from dob)
      - 'Gender'
      - 'days_to_target'
      - 'test_result': subset of PATIENT_MAP['test_result']
      - 'treatment_info': subset of PATIENT_MAP['treatment_info']
    """
    # Age
    # visit_date = row.get(date_col, pd.NaT)
    # age_from_dob = _age_years_from(dob_val, visit_date)
    # age_val = row.get("Age", None)
    # if pd.isna(age_val) or age_val is None:
    #     age_val = age_from_dob

    # Copy test_result columns
    out: dict[str, Any] = {}

    out[dt_gap] = (int(dt_gap),)

    out["Gender"] = gender_val

    for col in test_cols:
        vals = visit_df[col].copy().dropna()
        vals = [v for v in vals if v - 0.0 >= tol]
        out[cpl] = None if pd.isna(val) else val
        if vals.empty:
            out[col] = None
        else:
            out[col] = float(vals.iloc[-1])

    # -------------------
    # Inhaler Section
    # -------------------
    df_inh = visit_df.copy().dropna(subset=["Inhaler_Name"])

    if df_inh.empty:
        out["Inhaler_Name"] = None
        out["Drug_Class"] = None
        out["Device_Type"] = None
        out["Inhaler_Duration"] = None
    else:
        # Ensure datetime conversion
        df_inh["Inhaler_Start_Date"] = pd.to_datetime(df_inh["Inhaler_Start_Date"], errors="coerce")
        df_inh["Inhaler_End_Date"] = pd.to_datetime(df_inh["Inhaler_End_Date"], errors="coerce")

        # Sort by end date to ensure latest inhaler is chosen
        df_inh = df_inh.sort_values("Inhaler_End_Date", ascending=True)

        # Select the last (most recent) record
        row = df_inh.iloc[-1]

        # Compute duration in days (handle missing safely)
        if pd.notna(row["Inhaler_End_Date"]) and pd.notna(row["Inhaler_Start_Date"]):
            duration_days = (row["Inhaler_End_Date"] - row["Inhaler_Start_Date"]).days
        else:
            duration_days = None

        # Assign to dict
        out["Inhaler_Name"] = row["Inhaler_Name"]
        out["Drug_Class"] = row["Drug_Class"]
        out["Device_Type"] = row["Device_Type"]
        out["Inhaler_Duration"] = duration_days

    # -------------------
    # Exacerbation Section
    # -------------------
    df_ex = visit_df.copy().dropna(subset=["Exacerbation"])

    if df_ex.empty:
        out["Exacerbation"] = None
        out["Exacerbation_Type"] = None
        out["Exacerbation_Treatment"] = None
    else:
        # Sort by date if there's an Exacerbation_Date column
        if "Exacerbation_Date" in df_ex.columns:
            df_ex = df_ex.sort_values("Exacerbation_Date", ascending=True)

        # Select last (most recent) record
        row = df_ex.iloc[-1]

        # Assign to dict
        out["Exacerbation"] = row["Exacerbation"]
        out["Exacerbation_Type"] = row["Exacerbation_Type"]
        out["Exacerbation_Treatment"] = row["Exacerbation_Treatment"]

    return out


# ----------------------------
# 3) Find y (target) by walking backward
# ----------------------------


def _find_y_backward(
    visit_groups: list[pd.DataFrame],
    date_order: list[pd.Timestamp],
    target_col: str = "FEV1_FVC_Meas",
) -> tuple[int | None, float | None, str | None, pd.Timestamp | None]:
    """
    Walk backward over visits. Return (y_index, y_value, y_source_test) where y is FEV1 Meas,
    preferring Post_BD then Pre_PFT. If not found or y_index==0, return (None, None, None).
    """
    for idx in range(len(visit_groups) - 1, -1, -1):
        visit_df = visit_groups[idx]
        vals = visit_df[target_col].dropna()
        if vals.empty:
            # return (None, None, None, None)
            continue
        y_val = float(vals.iloc[-1])  #! Pick last or mean?
        if target_col.lower() == "fev1_fvc_meas":
            y_val = float(y_val) / 100.0
        return (idx, y_val, target_col, date_order[idx])
    return (None, None, None, None)


# ----------------------------
# 4) Build single patient record (wide)
# ----------------------------
def _build_single_patient_record_wide(
    pid: Any,
    g: pd.DataFrame,
    *,
    id_col: str,
    date_col: str,
    gender_col: str,
    test_cols: list[str],
    # treat_cols: list[str],
    target_col: str = "FEV1_FVC_Meas",
) -> tuple[Any, dict[str, Any] | None]:
    """
    Returns (pid, record_dict_or_None)
    g is the mini-DataFrame for a single patient with wide columns per visit.
    """
    # Parse dates & sort
    g = g.copy()
    g = g.dropna(subset=[date_col])
    if g.empty:
        return pid, None
    g = g.sort_values(
        by=[
            date_col,
        ]
    ).reset_index(drop=True)

    # Coerce numeric in test_result columns (best-effort)
    _coerce_numeric_inplace(g, test_cols)

    # Pull stable demographics
    gender_val = None
    if gender_col in g.columns:
        first_valid = g[gender_col].first_valid_index()
        gender_val = None if first_valid is None else g.loc[first_valid, gender_col]

    # Collect distinct visit dates (ascending)
    visit_dates = g[date_col].dropna().sort_values().unique().tolist()
    num_visits = len(visit_dates)
    if num_visits == 0:
        return pid, None

    # Split group into per-visit mini-dataframes
    visits: list[pd.DataFrame] = [g[g[date_col] == d] for d in visit_dates]

    # Target
    # y_index, y_value, y_source_col, y_date = _find_y_backward_wide(g, date_col, target)
    y_index, y_value, y_source_col, y_date = _find_y_backward(visits, visit_dates, target_col)
    if y_index is None or y_value is None or y_date is None:
        return pid, None

    # Build history timeseries [0 : y_index)
    ts_list: list[dict[str, Any]] = []
    for idx in range(0, y_index):
        # row = g.iloc[idx]
        # cur_date = visit_dates[idx]
        # days_to_target = int((y_date - cur_date).days)

        vdf = visits[idx]
        vdate = visit_dates[idx]
        dt_gap = int(
            (visit_dates[y_index] - vdate).days
        )  # dt_gap relative to y_date (target visit)

        single = _make_visit_feature_record(
            vdf,
            gender_val=gender_val,
            dt_gap=dt_gap,
            test_cols=test_cols,
        )
        ts_list.append(single)

    # Basic info
    num_visits = int(g[date_col].nunique())
    basic_info = {
        "H_no": pid,
        "ht": g["ht"].dropna().iloc[0]
        if "ht" in g.columns and not g["ht"].dropna().empty
        else None,
        "bw": g["bw"].dropna().iloc[0]
        if "bw" in g.columns and not g["bw"].dropna().empty
        else None,
        "gender": gender_val,
        "dob": dob_val,
        "number_of_visit": num_visits,
    }

    out = {
        "patient_id": pid,
        "basic_info": basic_info,
        "y": float(y_value),
        "y_index": int(y_index),
        "y_source": str(y_source_col),
        "y_date": pd.Timestamp(y_date),
        "patient_timeseries": ts_list,
    }
    return pid, out


# ----------------------------
# 5) Public API: build patients_data (wide)
# ----------------------------
def build_patients_data_wide(
    df_path: str | Path = INTERIM_DATA_DIR / "All Inhaler Filtered.csv",
    output_path: str | Path = PROCESSED_DATA_DIR / "NEW_COPD_PATIENTS_DATA.pkl",
    *,
    id_col: str = "H_no",
    date_col: str = "PFT_date",
    gender_col: str = "Sex",
    dob_col: str = "Date of Birth",  #! not present
    # Column maps
    patient_map: dict[str, list[str]] = None,
    ignore_map: dict[str, list[str]] = None,
    target_col: str = "FEV1_FVC_Meas",
) -> dict[Any, dict[str, Any]]:
    """
    Build patients_data from a wide-format dataframe where each row = one visit.
    """
    if patient_map is None:
        patient_map = PATIENT_MAP
        # raise ValueError("patient_map (PATIENT_MAP) must be provided.")
    if ignore_map is None:
        ignore_map = PATIENT_MAP_CONTROLLER["ignore"]

    df = pd.read_csv(df_path)
    # Parse dates for key columns early (robust)
    _coerce_dates_inplace(  #! problem
        df,
        [
            date_col,
            dob_col,
            "Inhaler_Start_Date",
            "Inhaler_End_Date",
            "Exacerbation_Date",
            "Exacerbation_Date_2",
        ],
    )
    df = df.dropna(subset=[date_col])  # Drop rows without a valid visit date

    # Decide which columns to keep per section
    def _section_cols(section: str) -> list[str]:
        cols = list(patient_map.get(section, []))
        # Drop ignored
        for c in ignore_map.get(section, []):
            if c in cols:
                cols.remove(c)
        # Keep only those present in df
        return [c for c in cols if c in df.columns]

    # basic_cols = _section_cols("basic_info")
    test_cols = _section_cols("test_result")
    _coerce_numeric_inplace(df, test_cols)
    df[test_cols] = df[test_cols].apply(pd.to_numeric, errors="coerce")

    # treat_cols = _section_cols("treatment_info")

    # Group per patient
    groups: list[tuple[Any, pd.DataFrame]] = [
        (pid, g.copy(deep=True)) for pid, g in df.groupby(id_col, sort=False) if not g.empty
    ]

    max_workers = max(1, min(48, (os.cpu_count() or 8) - 2))

    # Deterministic aggregation: preserve groups order
    index_by_pid = {pid: i for i, (pid, _) in enumerate(groups)}
    tmp: dict[Any, dict[str, Any]] = {}

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        fut2pid = {
            ex.submit(
                _build_single_patient_record_wide,
                pid,
                g,
                id_col=id_col,
                date_col=date_col,
                gender_col=gender_col,
                test_cols=test_cols,
                target_col=target_col,
            ): pid
            for pid, g in groups
        }
        for fut in tqdm(as_completed(fut2pid), total=len(fut2pid), desc="Collecting patients data"):
            pid_out, rec = fut.result()
            if rec is not None:
                tmp[pid_out] = rec

    results: OrderedDict[Any, dict[str, Any]] = OrderedDict()
    for pid, _ in groups:
        if pid in tmp:
            results[pid] = tmp[pid]

    with open(output_path, "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved {len(results)} patients to {output_path}")

    return results


# ----------------------------
# 6) Example usage
# ----------------------------
if __name__ == "__main__":
    # Example:
    # build_patients_data_wide(
    #     df_path=INTERIM_DATA_DIR / "ALL_VISITS_WIDE.csv",
    #     output_path=PROCESSED_DATA_DIR / "COPD_PATIENTS_DATA_WIDE.pkl",
    #     id_col="H_no",
    #     date_col="PFT_date",
    #     gender_col="gender",
    #     dob_col="dob",
    #     patient_map=PATIENT_MAP,
    #     ignore_map=IGNORE_PATIENT_MAP,
    #     target=TargetSpecWide(target_priority=("postFEV1_FVC_Meas", "FEV1_FVC_Meas")),  # or FEV1
    # )
    pass
