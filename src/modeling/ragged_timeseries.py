# src/modeling/ragged_timeseries.py
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import Dataset

from src.config import PROCESSED_DATA_DIR
from src.features import PATIENT_MAP  # for stable feature order

# ----------------------------
# Feature ordering
# ----------------------------


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


# ----------------------------
# Flatten one visit (flat dict → x, m)
# ----------------------------


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


def flatten_visit(visit: dict[str, Any], columns: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    x_vals: list[float] = []
    m_vals: list[float] = []

    for col in columns:
        # ---- gender (binary) ----
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

        # ---- dt_gap (days to target) ----
        if col == "dt_gap":
            dt = visit.get("dt_gap", visit.get("dt_gap_days", None))
            f = _as_float_or_none(dt)
            if f is None:
                x_vals.append(0.0)
                m_vals.append(0.0)
            else:
                x_vals.append(f)
                m_vals.append(1.0)
            continue

        # ---- Inhaler (indicator) ----
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

        # ---- Inhaler_Duration (days) ----
        if col == "Inhaler_Duration":
            f = _as_float_or_none(visit.get("Inhaler_Duration"))
            if f is None:
                x_vals.append(0.0)
                m_vals.append(0.0)
            else:
                x_vals.append(f)
                m_vals.append(1.0)
            continue

        # ---- Exacerbation (indicator) ----
        if col == "Exacerbation":
            ex_present = _present(visit.get("Exacerbation"))
            # if your per-visit dict includes Exacerbation_Date later, this will pick it up too:
            if not ex_present:
                ex_present = _present(visit.get("Exacerbation_Date"))
            x_vals.append(1.0 if ex_present else 0.0)
            m_vals.append(1.0 if ex_present else 0.0)
            continue

        # ---- regular numeric feature ----
        f = _as_float_or_none(visit.get(col, None))
        if f is None:
            x_vals.append(0.0)
            m_vals.append(0.0)
        else:
            x_vals.append(f)
            m_vals.append(1.0)

    x = torch.tensor(x_vals, dtype=torch.float32)
    m = torch.tensor(m_vals, dtype=torch.float32)
    return x, m


# ----------------------------
# Collate for GRU-D style models
# ----------------------------


def collate_grud(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    B = len(batch)
    lengths = torch.tensor([b["length"].item() for b in batch], dtype=torch.long)
    maxT = int(lengths.max().item())
    D = batch[0]["X"].size(1)

    X = torch.zeros(B, maxT, D, dtype=torch.float32)
    M = torch.zeros(B, maxT, D, dtype=torch.float32)
    DT = torch.zeros(B, maxT, dtype=torch.float32)
    y = torch.stack([b["y"] for b in batch], dim=0)

    for i, b in enumerate(batch):
        T = b["length"].item()
        X[i, :T] = b["X"]
        M[i, :T] = b["M"]
        DT[i, :T] = b["DT"]

    return dict(X=X, M=M, DT=DT, lengths=lengths, y=y)


# ----------------------------
# Dataset
# ----------------------------


class TimeSeriesDataset(Dataset):
    """Ragged time-series dataset for NEW_COPD_PATIENTS_DATA.pkl

    Each patient record has:
      - 'patient_timeseries': List[dict[str, Any]] with flat keys (test features + 'gender', 'dt_gap').
      - 'y': float target.
    We build tensors X, M, DT and cache them.
    """

    def __init__(
        self,
        pkl_path: Path | str = PROCESSED_DATA_DIR / "NEW_COPD_PATIENTS_DATA.pkl",
        compute: bool = False,
        timeseries_path: Path | str | None = PROCESSED_DATA_DIR / "COPD_TIMESERIES_CACHE.pt",
        include_gender: bool = True,
        include_dt: bool = True,
    ):
        super().__init__()
        include_gender = True
        include_dt = True
        include_treatment = True
        include_exac = True

        self.columns = feature_order(
            include_gender=include_gender,
            include_dt=include_dt,
            include_treatment=include_treatment,
            include_exac=include_exac,
        )
        self._cache: list[dict[str, torch.Tensor]] = []
        self._pids: list[Any] = []
        self.index: list[Any] = []

        if compute:
            # print(f"{pkl_path=}")
            with open(pkl_path, "rb") as f:
                self.data: dict[Any, dict[str, Any]] = pickle.load(f)

            # print(f"{len(self.data.keys())=}")

            for pid, rec in self.data.items():
                T = len(rec.get("patient_timeseries", []))
                if T >= 1 and rec.get("y") is not None:
                    self.index.append(pid)

            for pid in self.index:
                rec = self.data[pid]
                visits: list[dict[str, Any]] = rec["patient_timeseries"]

                X_list, M_list, gaps = [], [], []
                for v in visits:
                    x, m = flatten_visit(v, self.columns)
                    X_list.append(x)
                    M_list.append(m)
                    gaps.append(float(v.get("dt_gap", v.get("dt_gap_days", 0)) or 0))

                X = torch.stack(X_list, dim=0)
                M = torch.stack(M_list, dim=0)

                # Build DT from decreasing dt_gap to target
                Tlen = X.size(0)
                DT = torch.zeros(Tlen, dtype=torch.float32)
                if Tlen > 1:
                    g = torch.tensor(gaps, dtype=torch.float32)
                    DT[1:] = g[:-1] - g[1:]

                sample = dict(
                    X=X,
                    M=M,
                    DT=DT,
                    length=torch.tensor(Tlen, dtype=torch.long),
                    y=torch.tensor(float(rec["y"]), dtype=torch.float32),
                    pid=pid,
                )
                self._cache.append(sample)
                self._pids.append(pid)

            if timeseries_path is not None:
                payload = {
                    "cache": self._cache,
                    "pids": self._pids,
                    "columns": self.columns,
                }
                torch.save(payload, timeseries_path)

            if hasattr(self, "data"):
                self.data.clear()
                del self.data
                import gc

                gc.collect()

            print(
                f"[TimeSeriesDataset] Computed and cached {len(self._cache)} patients (D={self._cache[0]['X'].shape[-1] if self._cache else 'NA'})"
            )

        else:
            if not timeseries_path:
                raise ValueError("timeseries_path must be provided when compute=False.")
            blob = torch.load(timeseries_path, map_location="cpu")
            self._cache = blob["cache"]
            self._pids = blob.get("pids", list(range(len(self._cache))))
            self.index = list(range(len(self._cache)))
            self.columns = blob.get("columns", self.columns)
            print(
                f"[TimeSeriesDataset] Loaded cached dataset from {timeseries_path} ({len(self._cache)} patients)"
            )

    def __len__(self) -> int:
        return len(self._cache)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        return self._cache[i]


# Notes

# * Your previous dataset imported `CLINICAL_MAP` and expected `visit['clinical_values']`. That no longer matches your producer; the updated loader above consumes the new flat structure safely.
# * If you later want to prioritize `postFEV1_FVC_Meas` over `FEV1_FVC_Meas` as target, we can extend `_find_y_backward` to accept a priority list.
# * If you need Age as a special channel instead of a normal feature, we can append it at the end similarly to `gender`/`dt_gap`.
