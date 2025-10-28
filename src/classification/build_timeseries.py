# src/classification/build_timeserie.py
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import Dataset

from src.classification.prepare_dataset import PATIENT_MAP  # for stable feature list
from src.config import PROCESSED_DATA_DIR


# ----------------------------
# Feature ordering
# ----------------------------
def feature_order(
    include_gender: bool = True,
    include_dt: bool = True,
    include_treatment: bool = True,
    include_exac: bool = True,
) -> list[str]:
    """
    Defines the canonical column order used by flatten_visit().
    """
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
    """
    Build X (values) and M (mask: 1 if present else 0) tensors for a single visit
    following 'columns' order.
    """
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
            ex_present = _present(visit.get("Exacerbation")) or _present(
                visit.get("Exacerbation_Date")
            )
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
    """
    Collate a ragged batch of patients into (B, T_max, D) padded tensors (X, M),
    inter-visit gaps DT, lengths, and binary y for classification (default threshold=0.7).
    """
    B = len(batch)
    lengths = torch.tensor([int(b["length"].item()) for b in batch], dtype=torch.long)
    maxT = int(lengths.max().item())
    D = batch[0]["X"].size(1)

    X = torch.zeros(B, maxT, D, dtype=torch.float32)
    M = torch.zeros(B, maxT, D, dtype=torch.float32)
    DT = torch.zeros(B, maxT, dtype=torch.float32)

    threshold: float = 0.7
    y = torch.tensor(
        [1 if float(b["y"].item()) < threshold else 0 for b in batch],
        dtype=torch.float32,
    )
    pids = [b.get("pid", None) for b in batch]

    for i, b in enumerate(batch):
        T = int(b["length"].item())
        X[i, :T] = b["X"]
        M[i, :T] = b["M"]
        DT[i, :T] = b["DT"]

    # span: sum of inter-visit gaps (days) per patient
    span = DT.sum(dim=1)  # (B,)

    return dict(X=X, M=M, DT=DT, lengths=lengths, y=y, pid=pids, span=span)


# ----------------------------
# Dataset
# ----------------------------
class TimeSeriesDataset(Dataset):
    """Ragged time-series dataset for NEW_COPD_PATIENTS_DATA.pkl"""

    def __init__(
        self,
        pkl_path: Path | str = PROCESSED_DATA_DIR / "NEW_COPD_PATIENTS_DATA.pkl",
        compute: bool = False,
        timeseries_path: Path | str | None = PROCESSED_DATA_DIR / "COPD_TIMESERIES_CACHE.pt",
        include_gender: bool = True,
        include_dt: bool = True,
        include_treatment: bool = True,
        include_exac: bool = True,
    ):
        super().__init__()

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
            with open(pkl_path, "rb") as f:
                self.data: dict[Any, dict[str, Any]] = pickle.load(f)

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

                X = torch.stack(X_list, dim=0)  # (T, D)
                M = torch.stack(M_list, dim=0)  # (T, D)

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
                f"[TimeSeriesDataset] Computed and cached {len(self._cache)} patients "
                f"(D={self._cache[0]['X'].shape[-1] if self._cache else 'NA'})"
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
                f"[TimeSeriesDataset] Loaded cached dataset from {timeseries_path} "
                f"({len(self._cache)} patients)"
            )

    def __len__(self) -> int:
        return len(self._cache)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        return self._cache[i]


# =====================OLD==================
# src/classification/build_timeserie.py
# from __future__ import annotations

# import pickle
# from pathlib import Path
# from typing import Any

# import pandas as pd
# import torch
# from torch.utils.data import Dataset

# from src.classification.prepare_dataset import PATIENT_MAP  # for stable feature list
# from src.config import PROCESSED_DATA_DIR


# # ----------------------------
# # Feature ordering
# # ----------------------------
# def feature_order(
#     include_gender: bool = True,
#     include_dt: bool = True,
#     include_treatment: bool = True,
#     include_exac: bool = True,
# ) -> list[str]:
#     """
#     Defines the canonical column order used by flatten_visit().
#     """
#     order = list(PATIENT_MAP.get("test_result", []))
#     if include_gender:
#         order.append("gender")
#     if include_dt:
#         order.append("dt_gap")
#     if include_treatment:
#         order.extend(["Inhaler", "Inhaler_Duration"])
#     if include_exac:
#         order.append("Exacerbation")
#     return order


# # ----------------------------
# # Flatten one visit (flat dict → x, m)
# # ----------------------------
# def _as_float_or_none(v) -> float | None:
#     if v is None:
#         return None
#     try:
#         f = float(v)
#         if pd.isna(f):
#             return None
#         return f
#     except Exception:
#         return None


# def _present(v) -> bool:
#     if v is None:
#         return False
#     try:
#         if pd.isna(v):
#             return False
#     except Exception:
#         pass
#     if isinstance(v, str):
#         return v.strip() != ""
#     return True


# def flatten_visit(visit: dict[str, Any], columns: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
#     """
#     Build X (values) and M (mask: 1 if present else 0) tensors for a single visit
#     following 'columns' order.
#     """
#     x_vals: list[float] = []
#     m_vals: list[float] = []

#     for col in columns:
#         # ---- gender (binary) ----
#         if col == "gender":
#             g = str(visit.get("gender", visit.get("Gender", ""))).strip().upper()
#             if g in {"M", "MALE", "1"}:
#                 x_vals.append(1.0)
#                 m_vals.append(1.0)
#             elif g in {"F", "FEMALE", "0"}:
#                 x_vals.append(0.0)
#                 m_vals.append(1.0)
#             else:
#                 x_vals.append(0.0)
#                 m_vals.append(0.0)
#             continue

#         # ---- dt_gap (days to target) ----
#         if col == "dt_gap":
#             dt = visit.get("dt_gap", visit.get("dt_gap_days", None))
#             f = _as_float_or_none(dt)
#             if f is None:
#                 x_vals.append(0.0)
#                 m_vals.append(0.0)
#             else:
#                 x_vals.append(f)
#                 m_vals.append(1.0)
#             continue

#         # ---- Inhaler (indicator) ----
#         if col == "Inhaler":
#             inh_present = any(
#                 _present(visit.get(k))
#                 for k in ("Inhaler_Name", "Inhaler_Start_Date", "Inhaler_End_Date")
#             )
#             if not inh_present:
#                 inh_present = _as_float_or_none(visit.get("Inhaler_Duration")) is not None
#             x_vals.append(1.0 if inh_present else 0.0)
#             m_vals.append(1.0 if inh_present else 0.0)
#             continue

#         # ---- Inhaler_Duration (days) ----
#         if col == "Inhaler_Duration":
#             f = _as_float_or_none(visit.get("Inhaler_Duration"))
#             if f is None:
#                 x_vals.append(0.0)
#                 m_vals.append(0.0)
#             else:
#                 x_vals.append(f)
#                 m_vals.append(1.0)
#             continue

#         # ---- Exacerbation (indicator) ----
#         if col == "Exacerbation":
#             ex_present = _present(visit.get("Exacerbation"))
#             if not ex_present:
#                 ex_present = _present(visit.get("Exacerbation_Date"))
#             x_vals.append(1.0 if ex_present else 0.0)
#             m_vals.append(1.0 if ex_present else 0.0)
#             continue

#         # ---- regular numeric feature ----
#         f = _as_float_or_none(visit.get(col, None))
#         if f is None:
#             x_vals.append(0.0)
#             m_vals.append(0.0)
#         else:
#             x_vals.append(f)
#             m_vals.append(1.0)

#     x = torch.tensor(x_vals, dtype=torch.float32)
#     m = torch.tensor(m_vals, dtype=torch.float32)
#     return x, m


# # ----------------------------
# # Collate for GRU-D style models
# # ----------------------------
# def collate_grud(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
#     """
#     Collate a ragged batch of patients into (B, T_max, D) padded tensors (X, M),
#     inter-visit gaps DT, lengths, and binary y for classification (default threshold=0.7).

#     To change the threshold from the DataLoader, wrap with functools.partial:
#         from functools import partial
#         collate = partial(collate_grud_with_threshold, threshold=0.65)
#     """
#     B = len(batch)
#     lengths = torch.tensor([int(b["length"].item()) for b in batch], dtype=torch.long)
#     maxT = int(lengths.max().item())
#     D = batch[0]["X"].size(1)

#     X = torch.zeros(B, maxT, D, dtype=torch.float32)
#     M = torch.zeros(B, maxT, D, dtype=torch.float32)
#     DT = torch.zeros(B, maxT, dtype=torch.float32)

#     # ----- build y as 0/1 using a fixed threshold (default 0.7) -----
#     threshold: float = 0.7
#     # For BCEWithLogitsLoss (float targets 0/1):
#     y = torch.tensor(
#         [1 if float(b["y"].item()) < threshold else 0 for b in batch],
#         dtype=torch.float32,
#     )
#     # If you use CrossEntropyLoss instead, use long targets:
#     # y = torch.tensor(
#     #     [1 if float(b["y"].item()) < threshold else 0 for b in batch],
#     #     dtype=torch.long,
#     # )

#     # (optional) keep pids for downstream analysis/saving predictions
#     pids = [b.get("pid", None) for b in batch]

#     for i, b in enumerate(batch):
#         T = int(b["length"].item())
#         X[i, :T] = b["X"]
#         M[i, :T] = b["M"]
#         DT[i, :T] = b["DT"]

#     return dict(X=X, M=M, DT=DT, lengths=lengths, y=y, pid=pids)


# # ----------------------------
# # Dataset
# # ----------------------------
# class TimeSeriesDataset(Dataset):
#     """Ragged time-series dataset for NEW_COPD_PATIENTS_DATA.pkl

#     Each patient record has:
#       - 'patient_timeseries': List[dict[str, Any]] with flat keys (test features + 'gender', 'dt_gap', etc.).
#       - 'y': float target (e.g., FEV1/FVC or similar) — thresholded in collate into binary for classification.
#     We build tensors X, M, DT and cache them for fast loading.
#     """

#     def __init__(
#         self,
#         pkl_path: Path | str = PROCESSED_DATA_DIR / "NEW_COPD_PATIENTS_DATA.pkl",
#         compute: bool = False,
#         timeseries_path: Path | str | None = PROCESSED_DATA_DIR / "COPD_TIMESERIES_CACHE.pt",
#         include_gender: bool = True,
#         include_dt: bool = True,
#         include_treatment: bool = True,
#         include_exac: bool = True,
#     ):
#         super().__init__()

#         # Respect caller flags
#         self.columns = feature_order(
#             include_gender=include_gender,
#             include_dt=include_dt,
#             include_treatment=include_treatment,
#             include_exac=include_exac,
#         )

#         self._cache: list[dict[str, torch.Tensor]] = []
#         self._pids: list[Any] = []
#         self.index: list[Any] = []

#         if compute:
#             with open(pkl_path, "rb") as f:
#                 self.data: dict[Any, dict[str, Any]] = pickle.load(f)

#             for pid, rec in self.data.items():
#                 T = len(rec.get("patient_timeseries", []))
#                 if T >= 1 and rec.get("y") is not None:
#                     self.index.append(pid)

#             for pid in self.index:
#                 rec = self.data[pid]
#                 visits: list[dict[str, Any]] = rec["patient_timeseries"]

#                 X_list, M_list, gaps = [], [], []
#                 for v in visits:
#                     x, m = flatten_visit(v, self.columns)
#                     X_list.append(x)
#                     M_list.append(m)
#                     gaps.append(float(v.get("dt_gap", v.get("dt_gap_days", 0)) or 0))

#                 X = torch.stack(X_list, dim=0)  # (T, D)
#                 M = torch.stack(M_list, dim=0)  # (T, D)

#                 # Build DT from decreasing dt_gap to target
#                 Tlen = X.size(0)
#                 DT = torch.zeros(Tlen, dtype=torch.float32)
#                 if Tlen > 1:
#                     g = torch.tensor(gaps, dtype=torch.float32)
#                     DT[1:] = g[:-1] - g[1:]

#                 sample = dict(
#                     X=X,
#                     M=M,
#                     DT=DT,
#                     length=torch.tensor(Tlen, dtype=torch.long),
#                     y=torch.tensor(float(rec["y"]), dtype=torch.float32),
#                     pid=pid,
#                 )
#                 self._cache.append(sample)
#                 self._pids.append(pid)

#             if timeseries_path is not None:
#                 payload = {
#                     "cache": self._cache,
#                     "pids": self._pids,
#                     "columns": self.columns,
#                 }
#                 torch.save(payload, timeseries_path)

#             # free raw dict to save memory
#             if hasattr(self, "data"):
#                 self.data.clear()
#                 del self.data
#                 import gc

#                 gc.collect()

#             print(
#                 f"[TimeSeriesDataset] Computed and cached {len(self._cache)} patients "
#                 f"(D={self._cache[0]['X'].shape[-1] if self._cache else 'NA'})"
#             )

#         else:
#             if not timeseries_path:
#                 raise ValueError("timeseries_path must be provided when compute=False.")
#             blob = torch.load(timeseries_path, map_location="cpu")
#             self._cache = blob["cache"]
#             self._pids = blob.get("pids", list(range(len(self._cache))))
#             self.index = list(range(len(self._cache)))
#             self.columns = blob.get("columns", self.columns)
#             print(
#                 f"[TimeSeriesDataset] Loaded cached dataset from {timeseries_path} "
#                 f"({len(self._cache)} patients)"
#             )

#     def __len__(self) -> int:
#         return len(self._cache)

#     def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
#         return self._cache[i]


# # Notes
# # - Labels for classification are produced in collate_grud via thresholding (default 0.7).
# #   If you want a different threshold without editing this file, wrap collate_grud with functools.partial.
# # - collate_grud returns 'pid' as a Python list (not a tensor) to keep string IDs intact.
# # - If you later switch to CrossEntropyLoss, uncomment the CE target line in collate_grud and comment the BCE line.
