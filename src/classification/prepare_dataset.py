# prepare_dataset.py
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

# ----------------------------
# 0) Clinical feature template
# ----------------------------
PATIENT_MAP: dict[str, list[str]] = OrderedDict(
    {
        "basic_info": ["H_no", "number_of_visit"],
        "test_result": [
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
        "target_info": ["y", "y_index", "y_source", "y_date"],
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
# Helpers
# ----------------------------


def _norm(s: str) -> str:
    return (s or "").strip().lower()


def _coerce_numeric_inplace(df: pd.DataFrame, cols: Iterable[str]) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")


def _parse_date_any(x) -> pd.Timestamp | None:
    if pd.isna(x):
        return pd.NaT
    s = str(x).strip()
    ts = pd.to_datetime(s, errors="coerce")
    if pd.isna(ts):
        ts = pd.to_datetime(s, format="%Y%m%d", errors="coerce")
    return ts


def _coerce_dates_inplace(df: pd.DataFrame, cols: Iterable[str]) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = df[c].apply(_parse_date_any)


# def _coerce_dates_inplace_v2(df: pd.DataFrame, cols: Iterable[str]) -> None:
#     """Vectorized: try %Y%m%d first, then general parse on remaining NaT rows."""
#     for c in cols:
#         if c not in df.columns:
#             continue
#         s = pd.to_datetime(df[c], format="%Y%m%d", errors="coerce")
#         mask = s.isna()
#         if mask.any():
#             s.loc[mask] = pd.to_datetime(df.loc[mask, c], errors="coerce")
#         df[c] = s


# ----------------------------
# Per-visit feature extraction
# ----------------------------


def _make_visit_feature_record(
    visit_df: pd.DataFrame,
    *,
    gender_val: Any,
    dt_gap: int,
    test_cols: list[str],
    tol: float = 1e-6,
) -> dict[str, Any]:
    """Build a per-visit dict with latest values for each test col, plus treatment/exacerbation snapshot.
    Output keys are **flat** feature names (no nested clinical_values). Extras:
      - out['gender'] (from patient-level gender)
      - out['dt_gap'] (days until target)
    """
    out: dict[str, Any] = {"dt_gap": int(dt_gap), "gender": gender_val}

    # Latest usable value per test column
    for col in test_cols:
        if col not in visit_df.columns:
            out[col] = None
            continue
        vals = pd.to_numeric(visit_df[col], errors="coerce").dropna()
        if tol is not None:
            vals = vals[vals > tol]
        out[col] = None if vals.empty else float(vals.iloc[-1])

    # Inhaler (latest by End date; fallback to Start)
    if "Inhaler_Name" in visit_df.columns:
        df_inh = visit_df.dropna(subset=["Inhaler_Name"]).copy()
    else:
        df_inh = pd.DataFrame()

    if df_inh.empty:
        out.update(
            {
                "Inhaler_Name": None,
                "Drug_Class": None,
                "Device_Type": None,
                "Inhaler_Duration": None,
            }
        )
    else:
        for c in ("Inhaler_Start_Date", "Inhaler_End_Date"):
            if c in df_inh.columns:
                df_inh[c] = pd.to_datetime(df_inh[c], errors="coerce")
        if "Inhaler_End_Date" in df_inh.columns:
            df_inh = df_inh.sort_values(
                ["Inhaler_End_Date", "Inhaler_Start_Date"],
                ascending=[True, True],
                na_position="first",
            )
        elif "Inhaler_Start_Date" in df_inh.columns:
            df_inh = df_inh.sort_values("Inhaler_Start_Date", ascending=True, na_position="first")
        row = df_inh.iloc[-1]
        start_dt = row.get("Inhaler_Start_Date", pd.NaT)
        end_dt = row.get("Inhaler_End_Date", pd.NaT)
        duration_days = (
            (end_dt - start_dt).days if (pd.notna(start_dt) and pd.notna(end_dt)) else None
        )
        out.update(
            {
                "Inhaler_Name": row.get("Inhaler_Name", None),
                "Drug_Class": row.get("Drug_Class", None),
                "Device_Type": row.get("Device_Type", None),
                "Inhaler_Duration": duration_days,
            }
        )

    # Exacerbation (latest by date if present)
    if "Exacerbation" in visit_df.columns:
        df_ex = visit_df.dropna(subset=["Exacerbation"]).copy()
    else:
        df_ex = pd.DataFrame()

    if df_ex.empty:
        out.update(
            {
                "Exacerbation": None,
                "Exacerbation_Type": None,
                "Exacerbation_Treatment": None,
            }
        )
    else:
        if "Exacerbation_Date" in df_ex.columns:
            df_ex["Exacerbation_Date"] = pd.to_datetime(df_ex["Exacerbation_Date"], errors="coerce")
            df_ex = df_ex.sort_values("Exacerbation_Date", ascending=True, na_position="first")
        row = df_ex.iloc[-1]
        out.update(
            {
                "Exacerbation": row.get("Exacerbation", None),
                "Exacerbation_Type": row.get("Exacerbation_Type", None),
                "Exacerbation_Treatment": row.get("Exacerbation_Treatment", None),
            }
        )

    return out


# ----------------------------
# Target selection
# ----------------------------


def _find_y_backward(
    visit_groups: list[pd.DataFrame],
    date_order: list[pd.Timestamp],
    target_col: str = "FEV1_FVC_Meas",
) -> tuple[int | None, float | None, str | None, pd.Timestamp | None]:
    for idx in range(len(visit_groups) - 1, -1, -1):
        visit_df = visit_groups[idx]
        if target_col not in visit_df.columns:
            continue
        vals = pd.to_numeric(visit_df[target_col], errors="coerce").dropna()
        if vals.empty:
            continue
        y_val = float(vals.iloc[-1])
        if target_col.lower() == "fev1_fvc_meas":
            y_val = y_val / 100.0
        return (idx, y_val, target_col, date_order[idx])
    return (None, None, None, None)


def compute_y_age(
    g: pd.DataFrame,
    y_date: Any,
    *,
    date_col: str = "PFT_date",
    age_col: str = "Age",
) -> float:
    """
    Age at y_date.

    Rules:
      1) If there is an age on y_date, use it.
      2) If all ages are NaN, return NaN.
      3) Otherwise, take the closest dated age and adjust by the day gap / 365.2425.
    """
    if age_col not in g.columns or date_col not in g.columns:
        return float("nan")

    if pd.isna(y_date):
        return float("nan")
    y_date = pd.Timestamp(y_date).normalize()

    ages = pd.to_numeric(g[age_col], errors="coerce")
    dates = pd.to_datetime(g[date_col], errors="coerce").dt.normalize()

    # 1) Exact match on y_date
    exact = ages[dates == y_date].dropna()
    if not exact.empty:
        return float(exact.iloc[-1])

    # 2) If all ages are NaN
    if ages.dropna().empty:
        return float("nan")

    # 3) Project from closest dated age
    df_age = pd.DataFrame({"age": ages, "date": dates}).dropna()
    if df_age.empty:
        return float("nan")

    df_age["delta_days"] = (y_date - df_age["date"]).dt.days
    i = df_age["delta_days"].abs().idxmin()
    base_age = float(df_age.loc[i, "age"])
    delta_years = float(df_age.loc[i, "delta_days"]) / 365.2425
    return base_age + delta_years


# ----------------------------
# Build single patient record (wide)
# ----------------------------


def _build_single_patient_record_wide(
    pid: Any,
    g: pd.DataFrame,
    *,
    id_col: str,
    date_col: str,
    gender_col: str,
    test_cols: list[str],
    target_col: str = "FEV1_FVC_Meas",
) -> tuple[Any, dict[str, Any] | None]:
    g = g.copy()
    g = g.dropna(subset=[date_col])
    if g.empty:
        return pid, None

    g = g.sort_values(by=[date_col]).reset_index(drop=True)

    _coerce_numeric_inplace(g, test_cols)

    gender_val = None
    if gender_col in g.columns:
        idx0 = g[gender_col].first_valid_index()
        gender_val = None if idx0 is None else g.loc[idx0, gender_col]

    visit_dates = (
        pd.to_datetime(g[date_col], errors="coerce").dropna().sort_values().unique().tolist()
    )
    if not visit_dates:
        return pid, None

    visits: list[pd.DataFrame] = [g[g[date_col] == d] for d in visit_dates]

    y_index, y_value, y_source_col, y_date = _find_y_backward(visits, visit_dates, target_col)
    if y_index is None or y_value is None or y_date is None:
        return pid, None

    ts_list: list[dict[str, Any]] = []
    for idx, vdate in enumerate(visit_dates[:y_index]):
        vdf = visits[idx]
        dt_gap = int((visit_dates[y_index] - vdate).days)
        single = _make_visit_feature_record(
            vdf,
            gender_val=gender_val,
            dt_gap=dt_gap,
            test_cols=test_cols,
        )
        ts_list.append(single)

        y_age = compute_y_age(g, y_date, date_col=date_col, age_col="Age")

    out = {
        "patient_id": pid,
        "number_of_visit": len(visit_dates),
        "y": float(y_value),
        "y_index": int(y_index),
        "y_source": str(y_source_col),
        "y_date": pd.Timestamp(y_date),
        "y_age": y_age,  # Ahe of the patient on the y_date
        "patient_timeseries": ts_list,
    }
    return pid, out


# ----------------------------
# Public API: build patients_data
# ----------------------------


def build_patients_data(
    df_path: str | Path = INTERIM_DATA_DIR / "All Inhaler MERGED.csv",
    output_path: str | Path = PROCESSED_DATA_DIR / "NEW_COPD_PATIENTS_DATA.pkl",
    *,
    id_col: str = "H_no",
    date_col: str = "PFT_date",
    gender_col: str = "Sex",
    dob_col: str = "Date of Birth",
    patient_map: dict[str, list[str]] | None = None,
    ignore_map: dict[str, list[str]] | None = None,
    target_col: str = "FEV1_FVC_Meas",
    max_workers: int | None = None,
) -> dict[Any, dict[str, Any]]:
    if patient_map is None:
        patient_map = PATIENT_MAP
    if ignore_map is None:
        ignore_map = PATIENT_MAP_CONTROLLER["ignore"]

    df = pd.read_csv(df_path)

    _coerce_dates_inplace(
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
    df = df.dropna(subset=[date_col])

    def _section_cols(section: str) -> list[str]:
        cols = list(patient_map.get(section, []))
        for c in ignore_map.get(section, []):
            if c in cols:
                cols.remove(c)
        return [c for c in cols if c in df.columns]

    test_cols = _section_cols("test_result")
    _coerce_numeric_inplace(df, test_cols)
    treat_cols = _section_cols("treatment_info")

    keep_cols = sorted(
        set([id_col, date_col, gender_col, dob_col])
        | set(test_cols)
        | set(treat_cols)
        | {target_col}
    )
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].copy()

    groups: list[tuple[Any, pd.DataFrame]] = [
        (pid, g.copy(deep=True)) for pid, g in df.groupby(id_col, sort=False) if not g.empty
    ]

    print(f"Total input patients: {len(groups)}")

    if max_workers is None:
        max_workers = max(1, min(48, (os.cpu_count() or 8) - 2))

    results_tmp: dict[Any, dict[str, Any]] = {}

    if max_workers == 1:
        for pid, g in tqdm(groups, total=len(groups), desc="Building patients data"):
            try:
                _, rec = _build_single_patient_record_wide(
                    pid,
                    g,
                    id_col=id_col,
                    date_col=date_col,
                    gender_col=gender_col,
                    test_cols=test_cols,
                    target_col=target_col,
                )
                if rec is not None:
                    results_tmp[pid] = rec
            except Exception as e:
                print(f"[WARN] Failed pid={pid}: {e}")
    else:
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
            for fut in tqdm(
                as_completed(fut2pid), total=len(fut2pid), desc="Collecting patients data"
            ):
                pid_out = fut2pid[fut]
                try:
                    pid_ret, rec = fut.result()
                    if rec is not None:
                        results_tmp[pid_ret] = rec
                except Exception as e:
                    print(f"[WARN] Failed pid={pid_out}: {e}")

    ordered: OrderedDict[Any, dict[str, Any]] = OrderedDict()
    for pid, _ in groups:
        if pid in results_tmp:
            ordered[pid] = results_tmp[pid]

    with open(output_path, "wb") as f:
        pickle.dump(ordered, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved {len(ordered)} patients to {output_path}")

    return ordered


if __name__ == "__main__":
    # Example:
    build_patients_data(
        df_path=INTERIM_DATA_DIR / "All Inhaler MERGED.csv",
        output_path=PROCESSED_DATA_DIR / "NEW_COPD_PATIENTS_DATA.pkl",
        id_col="H_no",
        date_col="PFT_date",
        gender_col="Sex",
        target_col="FEV1_FVC_Meas",
        max_workers=1,
    )
    pass
