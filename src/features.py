# # from pathlib import Path

# # from loguru import logger
# # from tqdm import tqdm
# # import typer

# # from src.config import PROCESSED_DATA_DIR

# # app = typer.Typer()


# # @app.command()
# # def main(
# #     # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
# #     input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
# #     output_path: Path = PROCESSED_DATA_DIR / "features.csv",
# #     # -----------------------------------------
# # ):
# #     # ---- REPLACE THIS WITH YOUR OWN CODE ----
# #     logger.info("Generating features from dataset...")
# #     for i in tqdm(range(10), total=10):
# #         if i == 5:
# #             logger.info("Something happened for iteration 5.")
# #     logger.success("Features generation complete.")
# #     # -----------------------------------------


# src/features.py
from __future__ import annotations

import os
import pickle
import re
from collections import OrderedDict
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from src.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR

# ----------------------------
# 0) Clinical feature template
# ----------------------------
# Use OrderedDict to lock a deterministic traversal order (important for models).
CLINICAL_MAP: dict[str, dict[str, Iterable[str]]] = OrderedDict(
    {
        "COD": OrderedDict(
            {
                "DLCO": {"Meas", "%Pred"},
                "DLCO/VA": {"Meas", "%Pred"},
                "VA": {"Meas"},
            }
        ),
        "Post_BD": OrderedDict(
            {
                "FEF/FIF50": {"Meas", "%Chg."},
                "FEF25%": {"Meas", "%Chg."},
                "FEF50%": {"Meas", "%Pred", "%Chg."},
                "FEF25~75%": {"Meas", "%Pred", "%Chg."},
                "FEF75%": {"Meas", "%Pred", "%Chg."},
                "FET100%": {"Meas", "%Chg."},
                "FEV1": {"Meas", "%Pred", "%Chg."},
                "FEV1/FVC": {"Meas", "%Pred", "%Chg."},
                "FIV1": {"Meas", "%Chg."},
                "FIVC": {"Meas", "%Pred", "%Chg."},
                "FVC": {"Meas", "%Pred", "%Chg."},
                "PEF": {"Meas", "%Pred", "%Chg."},
                "PIF": {"Meas", "%Pred", "%Chg."},
            }
        ),
        "Pre_PFT": OrderedDict(
            {
                "FEF/FIF50": {"Meas"},
                "FEF25%": {"Meas"},
                "FEF25~75%": {"Meas", "%Pred", "%Chg."},
                "FEF50%": {"Meas", "%Pred", "%Chg."},
                "FEF75%": {"Meas", "%Pred", "%Chg."},
                "FET100%": {"Meas"},
                "FEV1": {"Meas", "%Pred", "%Chg."},
                "FEV1/FVC": {"Meas", "%Chg."},
                "FIV1": {"Meas"},
                "FIVC": {"Meas", "%Pred", "%Chg."},
                "FVC": {"Meas", "%Pred", "%Chg."},
                "PEF": {"Meas", "%Pred", "%Chg."},
                "PIF": {"Meas"},
            }
        ),
    }
)


# ------------------------------------
# 1) Lightweight normalization helpers
# ------------------------------------
def _norm_alnum(s: str) -> str:
    """Lowercase and keep only letters/digits (robust cross-vendor matching)."""
    # return re.sub(r"[^A-Za-z0-9]+", "", str(s)).lower()
    return re.sub(r"[^A-Za-z0-9%]+", "", str(s)).lower()


def _norm_contains(hay: str, needle: str) -> bool:
    # ! problem hay is hay = "DLCO/VA test" and needle = "DLCO"
    """True if normalized 'needle' is a substring of normalized 'hay'."""
    return _norm_alnum(needle) in _norm_alnum(hay)


def _age_years_from(dob: pd.Timestamp | None, on_date: pd.Timestamp | None) -> int | None:
    if pd.isna(dob) or pd.isna(on_date):
        return None
    # integer age in years
    return int((on_date.date() - dob.date()).days // 365)


# -------------------------------------------------
# 2) Find y (target) visit by walking backward
# -------------------------------------------------
@dataclass  # * Need to Learn later [Without @dataclass, need to manually implement __init__.]
class TargetSpec:
    # test_priority: tuple[str, str] = ("Post_BD", "Pre_PFT")  # prefer Post_BD, fallback Pre_PFT
    test_priority: tuple[str, ...] = ("Pre_PFT",)  # prefer Pre_PFT
    measurement: str = "FEV1"
    variable: str = "Meas"  # FEV1 Meas


def _extract_value_for_visit(
    visit_df: pd.DataFrame,
    test_col: str,
    meas_col: str,
    var_col: str,
    value_col: str,
    want_test: str,
    want_meas: str,
    want_var: str,
) -> float | None:
    """Return the numeric 'value_col' for (test, measurement, variable) within a single visit."""
    # Filter rows in the visit that match test (substring, robust), and exact meas+var (normalized alnum)
    cand = visit_df[
        visit_df[test_col].apply(lambda x: _norm_contains(str(x), want_test))
        & (visit_df[meas_col].apply(_norm_alnum) == _norm_alnum(want_meas))
        & (visit_df[var_col].apply(_norm_alnum) == _norm_alnum(want_var))
    ]
    if cand.empty:
        return None
    val = cand[value_col].dropna()
    # TODO CKECK if multiple rows exist i.e Pre_PFT and Post_BD occure in the same date
    # TODO take the last non-NA value (or first; consistency is key)
    return float(val.iloc[-1]) if not val.empty else None


def _find_y_backward(
    visit_groups: list[pd.DataFrame],
    date_order: list[pd.Timestamp],
    test_col: str,
    meas_col: str,
    var_col: str,
    value_col: str,
    target: TargetSpec,
) -> tuple[int | None, float | None, str | None]:
    """
    Walk backward over visits. Return (y_index, y_value, y_source_test) where y is FEV1 Meas,
    preferring Post_BD then Pre_PFT. If not found or y_index==0, return (None, None, None).
    """
    for idx in range(len(visit_groups) - 1, -1, -1):
        visit_df = visit_groups[idx]
        # Try preferred tests in order
        for tname in target.test_priority:
            val = _extract_value_for_visit(
                visit_df,
                test_col,
                meas_col,
                var_col,
                value_col,
                tname,
                target.measurement,
                target.variable,
            )
            if val is not None:
                # Must have at least one history step before y
                if idx == 0:
                    return (None, None, None)
                return (idx, val, tname)
    return (None, None, None)


# -------------------------------------------------
# 3) Build single-timestep feature dict for a visit
# -------------------------------------------------
def _empty_clinical_block() -> dict[str, dict[str, dict[str, float | None]]]:
    """Create an empty nested structure matching CLINICAL_MAP, filled with None."""
    block: dict[str, dict[str, dict[str, float | None]]] = OrderedDict()
    for test, meas_map in CLINICAL_MAP.items():
        block[test] = OrderedDict()
        for meas, vars_set in meas_map.items():
            block[test][meas] = OrderedDict((var, None) for var in vars_set)
    return block


def _fill_clinical_block_for_visit(
    visit_df: pd.DataFrame,
    test_col: str,
    meas_col: str,
    var_col: str,
    value_col: str,
) -> dict[str, dict[str, dict[str, float | None]]]:
    """Populate clinical values for one visit according to CLINICAL_MAP."""
    out = _empty_clinical_block()
    # For speed, pre-normalize columns for matching
    visit_df = visit_df.copy()
    visit_df["_meas_norm"] = visit_df[meas_col].apply(_norm_alnum)
    visit_df["_var_norm"] = visit_df[var_col].apply(_norm_alnum)

    for test, meas_map in CLINICAL_MAP.items():
        # Subset rows whose 'Test' contains the test key (robust to vendor strings)
        test_mask = visit_df[test_col].apply(lambda x: _norm_contains(str(x), test))
        if not test_mask.any():
            continue
        sub = visit_df[test_mask]
        for meas, vars_set in meas_map.items():
            mnorm = _norm_alnum(meas)
            m_sub = sub[sub["_meas_norm"] == mnorm]
            if m_sub.empty:
                continue
            for var in vars_set:
                vnorm = _norm_alnum(var)
                v_sub = m_sub[m_sub["_var_norm"] == vnorm]
                if v_sub.empty:
                    continue
                # TODO take last non-NA value in case of duplicates
                val = v_sub[value_col].dropna()
                if not val.empty:
                    out[test][meas][var] = float(val.iloc[-1])
    return out


def _make_single_timestamp_features(
    visit_df: pd.DataFrame,
    gender: Any,
    age_years: int | None,
    dt_gap_days: int | None,
    test_col: str,
    meas_col: str,
    var_col: str,
    value_col: str,
) -> dict[str, Any]:
    """Compose the per-visit feature dictionary."""
    clinical_values = _fill_clinical_block_for_visit(
        visit_df, test_col, meas_col, var_col, value_col
    )
    single = {
        "Gender": gender,
        "Age": age_years,
        "dt_gap_days": dt_gap_days if dt_gap_days is not None else 0,
        "clinical_values": clinical_values,
    }
    return single


# -------------------------------------------------
# 4) Public API: build patients_data
# -------------------------------------------------


def _build_single_patient_record(
    pid: Any,
    g: pd.DataFrame,
    *,
    date_col: str,
    gender_col: str,
    dob_col: str,
    test_col: str,
    meas_col: str,
    var_col: str,
    value_col: str,
    target: TargetSpec,
) -> tuple[Any, dict[str, Any] | None]:
    """
    Build the patients_data entry for a single patient.
    Returns (pid, record_dict_or_None).
    """
    # Collect distinct visit dates (ascending)
    visit_dates = g[date_col].dropna().sort_values().unique().tolist()
    num_visits = len(visit_dates)
    if num_visits == 0:
        return pid, None

    # Split group into per-visit mini-dataframes
    visits: list[pd.DataFrame] = [g[g[date_col] == d] for d in visit_dates]

    # Pull stable demographics (first non-NA)
    gender_val = (
        s.at[idx]
        if (s := g.get(gender_col)) is not None and (idx := s.first_valid_index()) is not None
        else None
    )
    dob_val = (
        # pd.to_datetime(g[dob_col].dropna().iloc[0], errors="coerce")
        pd.to_datetime(g[dob_col].dropna().iloc[0], format="%Y%m%d", errors="coerce")
        if dob_col in g.columns and not g[dob_col].dropna().empty
        else pd.NaT
    )

    # Find target y by scanning backward
    y_index, y_value, y_source = _find_y_backward(
        visits, visit_dates, test_col, meas_col, var_col, value_col, target
    )
    # Discard if missing
    if y_index is None or y_value is None:
        return pid, None

    # Build history X = visits [0 : y_index)
    ts_list: list[dict[str, Any]] = []
    # prev_date = None
    for idx in range(0, y_index):
        vdf = visits[idx]
        vdate = visit_dates[idx]
        # dt_gap = (vdate - prev_date).days if prev_date is not None else 0 # dt_gap relative to previous visit
        dt_gap = (visit_dates[y_index] - vdate).days  # dt_gap relative to y_date (target visit)
        # prev_date = vdate

        age_years = _age_years_from(dob_val, vdate)
        single = _make_single_timestamp_features(
            vdf, gender_val, age_years, dt_gap, test_col, meas_col, var_col, value_col
        )
        ts_list.append(single)

    out = {
        "patient_timeseries": ts_list,
        "y": float(y_value),
        "number_of_visit": int(num_visits),
        "y_index": int(y_index),
        "y_source": y_source,
        "y_date": pd.Timestamp(visit_dates[y_index]),
    }
    return pid, out


def build_patients_data(
    # df: pd.DataFrame,
    df_path: str | Path = INTERIM_DATA_DIR / "ALL_PRESCRIPTION_DATA_TMV.csv",
    output_path: str | Path = PROCESSED_DATA_DIR / "COPD_PATIENTS_DATA.pkl",
    *,
    id_col: str = "Patient Number",
    date_col: str = "Prescription Date",
    gender_col: str = "Gender",
    dob_col: str = "Date of Birth",
    test_col: str = "Test",
    meas_col: str = "Measurement",
    var_col: str = "Variable",
    value_col: str = "Result Numerical Value",  # "Value",
    target: TargetSpec = TargetSpec(),
) -> dict[Any, dict[str, Any]]:
    """
    Parallel-safe builder. If n_jobs > 1, uses processes.
    Ensures the main DataFrame is not shared with workers by sending per-patient copies only.
    """
    df = pd.read_csv(df_path)
    df = df.dropna(subset=[date_col])  # Drop rows without a valid visit date
    df[date_col] = pd.to_datetime(df[date_col], format="%Y%m%d", errors="coerce")
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df[df[value_col].notna() & (df[value_col] > 0)]  # Remove NA and Zero result values

    want = df[id_col].unique().tolist()  # all patients
    # want = [886482, 1207865, 1452945, 611957, 965594, 5665, 7429, 29903, 42405]

    # Materialize patient groups as (pid, mini_df_copy)
    # NOTE: g.copy(deep=True) ensures no shared views
    groups: list[tuple[Any, pd.DataFrame]] = [
        (pid, g.copy(deep=True))
        for pid, g in df.groupby(id_col, sort=False)
        if pid in want and not g.empty
    ]
    # groups: list[tuple[Any, pd.DataFrame]] = [
    #     (pid, g.copy(deep=True)) for pid, g in df.groupby(id_col, sort=False)
    # ]

    # Parallel with processes (no DF sharing)
    max_workers = max(1, min(48, (os.cpu_count() or 8) - 2))
    results: dict[Any, dict[str, Any]] = OrderedDict()
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [
            ex.submit(
                _build_single_patient_record,
                pid,
                g,  # mini DF copy sent to worker
                date_col=date_col,
                gender_col=gender_col,
                dob_col=dob_col,
                test_col=test_col,
                meas_col=meas_col,
                var_col=var_col,
                value_col=value_col,
                target=target,
            )
            for pid, g in groups
        ]

        # Collect in submission order to keep group order deterministic
        # for fut in futures:
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Collecting patients data"):
            pid_out, rec = fut.result()
            if rec is not None:
                results[pid_out] = rec
    with open(output_path, "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved {len(results)} patients to {output_path}")

    # return results


# -------------------------------------------------
# 5) (Optional) helper to derive lengths vector
# -------------------------------------------------
# code def sequence_lengths_from(patients_data: dict[Any, dict[str, Any]]) -> tuple[list[Any], list[int]]:
# code     """Return (patient_ids, lengths) for packed sequences."""
# code     pids: list[Any] = []
# code     lens: list[int] = []
# code     for pid, rec in patients_data.items():
# code         pids.append(pid)
# code         lens.append(len(rec["patient_timeseries"]))
# code     return pids, lens


# How this matches your spec

# Target rule & y_index: walks backward through visits, picks FEV1 → Meas from Post_BD else Pre_PFT; drops patient if target missing or y_index == 0.

# patients_data[patient_id] = {
#   "patient_timeseries": [ single_timestamp_features ... ],
#   "y": <FEV1 Meas at y_index>,
#   "number_of_visit": <distinct dates>,
#   "y_index": <int>,
#   "y_source": "Post_BD" | "Pre_PFT",
#   "y_date": <Timestamp>
# }

# single_timestamp_features contains exactly:

# Gender

# Age (DOB → visit date, integer years)

# dt_gap_days (days since previous visit; 0 for first in history)

# clinical_values nested as your clinical_values mapping (test → measurement → {variables}).

# Inputs expected in your dataframe

# One row per observation with at least:

# Patient Number, Prescription Date,

# Test, Measurement, Variable, Value,

# optional: Gender, Date of Birth.

# If your column names differ, pass them via the function parameters.

# Next step (packing)

# When you batch for LSTM with packed sequences, derive the lengths with:

# code pids, lengths = sequence_lengths_from(patients_data)

# …and feed the per-patient patient_timeseries (ragged) plus lengths to your collate function / packer


if __name__ == "__main__":
    pass
