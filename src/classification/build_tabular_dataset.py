# src/classification/build_tabular_dataset.py
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# XGBoost
from src.classification.prepare_dataset import PATIENT_MAP  # for stable feature order


# --------------------------
# 1) Sequence → Tabular features
# --------------------------
def feature_order(
    include_gender: bool = True,
    include_dt: bool = True,
    include_treatment: bool = True,
    include_exac: bool = True,
) -> list[str]:
    order = list(PATIENT_MAP.get("test_result", []))
    if include_gender:
        order.append("gender")
    if include_dt:
        order.append("dt_gap")
    if include_treatment:
        order.extend(["Inhaler", "Inhaler_Duration"])
    if include_exac:
        order.append("Exacerbation")
    return order


def _as_float_or_none(v) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
        if pd.isna(f):
            return None
        return f
    except Exception:
        return None


def _present(v) -> bool:
    if v is None:
        return False
    try:
        if pd.isna(v):
            return False
    except Exception:
        pass
    if isinstance(v, str):
        return v.strip() != ""
    return True


def _flatten_visit_numpy(
    visit: dict[str, Any], columns: list[str]
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (x_row, m_row, dt_gap_value) for one visit."""
    x_vals, m_vals = [], []
    dt_gap_val = _as_float_or_none(visit.get("dt_gap", visit.get("dt_gap_days", None)))

    for col in columns:
        if col == "gender":
            g = str(visit.get("gender", visit.get("Gender", ""))).strip().upper()
            if g in {"M", "MALE", "1"}:
                x_vals.append(1.0)
                m_vals.append(1.0)
            elif g in {"F", "FEMALE", "0"}:
                x_vals.append(0.0)
                m_vals.append(1.0)
            else:
                x_vals.append(0.0)
                m_vals.append(0.0)
            continue

        if col == "dt_gap":
            f = dt_gap_val
            x_vals.append(0.0 if f is None else f)
            m_vals.append(0.0 if f is None else 1.0)
            continue

        if col == "Inhaler":
            inh_present = any(
                _present(visit.get(k))
                for k in ("Inhaler_Name", "Inhaler_Start_Date", "Inhaler_End_Date")
            )
            if not inh_present:
                inh_present = _as_float_or_none(visit.get("Inhaler_Duration")) is not None
            x_vals.append(1.0 if inh_present else 0.0)
            m_vals.append(1.0 if inh_present else 0.0)
            continue

        if col == "Inhaler_Duration":
            f = _as_float_or_none(visit.get("Inhaler_Duration"))
            x_vals.append(0.0 if f is None else f)
            m_vals.append(0.0 if f is None else 1.0)
            continue

        if col == "Exacerbation":
            ex_present = _present(visit.get("Exacerbation")) or _present(
                visit.get("Exacerbation_Date")
            )
            x_vals.append(1.0 if ex_present else 0.0)
            m_vals.append(1.0 if ex_present else 0.0)
            continue

        f = _as_float_or_none(visit.get(col))
        x_vals.append(0.0 if f is None else f)
        m_vals.append(0.0 if f is None else 1.0)

    return (
        np.asarray(x_vals, dtype=float),
        np.asarray(m_vals, dtype=float),
        (dt_gap_val if dt_gap_val is not None else np.nan),
    )


def _last_observed_per_feature(X: np.ndarray, M: np.ndarray) -> np.ndarray:
    T, D = X.shape
    out = np.full(D, np.nan, dtype=float)
    for d in range(D):
        obs = np.where(M[:, d] == 1)[0]
        if obs.size:
            out[d] = X[obs[-1], d]
    return out


def _mean_observed_per_feature(X: np.ndarray, M: np.ndarray) -> np.ndarray:
    num = (X * M).sum(axis=0)
    den = M.sum(axis=0)
    mu = np.divide(num, den, out=np.full_like(num, np.nan, dtype=float), where=den != 0)
    return mu


def _count_observed_per_feature(M: np.ndarray) -> np.ndarray:
    return M.sum(axis=0).astype(float)


# --------------------------
# 2) Build final dataset
# --------------------------
def build_tabular_data_from_pkl(
    pkl_path: str | Path,
    *,
    include_gender: bool = True,
    include_last: bool = True,
    min_visits: int = 1,
    threshold: float = 0.7,
    # Treatment
    # with
    include_treatment: bool = True,
    include_exac: bool = True,
    # without
    # include_treatment: bool = False,
    # include_exac: bool = False,
    # longitudinal info
    # with
    include_mean: bool = True,
    include_count: bool = True,
    add_length: bool = True,
    add_span_days: bool = True,
    include_dt: bool = True,
    # # without
    # include_mean: bool = False,
    # include_count: bool = False,
    # add_length: bool = False,
    # add_span_days: bool = False,
    # include_dt: bool = False,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, list[str], list[Any]]:
    """Build XGB-ready tabular features from patient .pkl data."""
    with open(pkl_path, "rb") as f:
        data: dict[Any, dict[str, Any]] = pickle.load(f)

    columns = feature_order(
        include_gender=include_gender,
        include_dt=include_dt,
        include_treatment=include_treatment,
        include_exac=include_exac,
    )

    rows, ys_reg, ys_cls, pids = [], [], [], []
    D = len(columns)

    # ✅ Human-readable feature names directly here
    feat_names: list[str] = []
    if include_last:
        feat_names += [f"last_{col}" for col in columns]
    if include_mean:
        feat_names += [f"mean_{col}" for col in columns]
    if include_count:
        feat_names += [f"count_{col}" for col in columns]
    if add_length:
        feat_names.append("seq_length")
    if add_span_days:
        feat_names.append("span_days")

    print(f"{len(data.keys())=} patients in dataset")

    for pid, rec in data.items():
        visits = rec.get("patient_timeseries", [])
        y = rec.get("y", None)
        if (y is None) or (len(visits) < min_visits):
            continue

        X_rows, M_rows, gaps = [], [], []
        for v in visits:
            x_row, m_row, gap = _flatten_visit_numpy(v, columns)
            X_rows.append(x_row)
            M_rows.append(m_row)
            gaps.append(gap)

        X = np.vstack(X_rows)
        M = np.vstack(M_rows)

        gaps_arr = np.asarray(gaps, dtype=float)
        DT = np.zeros(len(gaps_arr), dtype=float)
        if len(gaps_arr) > 1:
            g = gaps_arr.copy()
            for i in range(len(g)):
                if np.isnan(g[i]):
                    g[i] = g[i - 1] if i > 0 else g[i]
            DT[1:] = g[:-1] - g[1:]
        span_days = float(np.nansum(DT))

        feats: list[float] = []
        if include_last:
            feats.extend(_last_observed_per_feature(X, M))
        if include_mean:
            feats.extend(_mean_observed_per_feature(X, M))
        if include_count:
            feats.extend(_count_observed_per_feature(M))
        if add_length:
            feats.append(float(len(visits)))
        if add_span_days:
            feats.append(span_days)

        rows.append(feats)
        ys_reg.append(float(y))
        ys_cls.append(1 if y < threshold else 0)
        pids.append(pid)

    X_df = pd.DataFrame(rows, columns=feat_names, dtype=float)
    y_reg = pd.Series(ys_reg, name="y_reg", dtype=float)
    y_cls = pd.Series(ys_cls, name="y_cls", dtype=int)
    return X_df, y_reg, y_cls, feat_names, pids
