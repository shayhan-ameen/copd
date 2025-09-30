# src/copd/data/grud_dataset.py
from __future__ import annotations

import pickle
from typing import Any

import torch
from torch.utils.data import Dataset

# import your CLINICAL_MAP from wherever it is defined
from src.features import CLINICAL_MAP  # adjust path if different

# def materialize_and_save(ds, out_pt="data/processed/grud_materialized.pt"):
#     Xs, Ms, DTs, lengths, ys, offsets = [], [], [], [], [], [0]
#     for i in range(len(ds)):
#         s = ds[i] # {"X","M","DT","length","y"}
#         Xs.append(s["X"])
#         Ms.append(s["M"])
#         DTs.append(s["DT"])
#         lengths.append(s["length"].item())
#         ys.append(s["y"].item())
#         offsets.append(offsets[-1] + s["length"].item())

#     payload = {
#         "X": torch.cat(Xs, dim=0),  # (sumT, D)
#         "M": torch.cat(Ms, dim=0),  # (sumT, D)
#         "DT": torch.cat(DTs, dim=0),  # (sumT,)
#         "lengths": torch.tensor(lengths),
#         "offsets": torch.tensor(offsets),  # prefix sums to slice sequences
#         "y": torch.tensor(ys, dtype=torch.float32),
#     }
#     torch.save(payload, out_pt)
#     return out_pt


def clinical_feature_order() -> list[tuple[str, str, str]]:
    """Deterministic flatten order: (Test, Measurement, Variable)."""
    keys: list[tuple[str, str, str]] = []
    for test, meas_map in CLINICAL_MAP.items():
        for meas, vars_set in meas_map.items():
            for var in sorted(vars_set):  # set → sorted list for stable order
                keys.append((test, meas, var))
    return keys


def flatten_visit(
    visit: dict[str, Any],
    keys: list[tuple[str, str, str]],
    include_age: bool = True,
    include_gender: bool = True,
    include_dt_feature: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
        x: (D,) float tensor
        m: (D,) mask 1 if observed else 0
    """
    cv = visit["clinical_values"]
    x_vals: list[float] = []
    m_vals: list[float] = []

    # optional extras
    if include_age:
        age = visit.get("Age", None)
        if age is None:
            x_vals.append(0.0)
            m_vals.append(0.0)
        else:
            x_vals.append(float(age))
            m_vals.append(1.0)

    if include_gender:
        g = str(visit.get("Gender", "")).strip().upper()
        g_num = 1.0 if g in {"M", "MALE", "1"} else (0.0 if g in {"F", "FEMALE", "0"} else 0.0)
        m_g = 1.0 if g in {"M", "MALE", "1", "F", "FEMALE", "0"} else 0.0
        x_vals.append(g_num)
        m_vals.append(m_g)

    if include_dt_feature:
        dtg = visit.get("dt_gap_days", None)
        if dtg is None:
            x_vals.append(0.0)
            m_vals.append(0.0)
        else:
            x_vals.append(float(dtg))
            m_vals.append(1.0)

    # clinical values
    for tst, meas, var in keys:
        v = ((cv.get(tst) or {}).get(meas) or {}).get(var)
        if v is None:
            x_vals.append(0.0)
            m_vals.append(0.0)
        else:
            x_vals.append(float(v))
            m_vals.append(1.0)

    x = torch.tensor(x_vals, dtype=torch.float32)
    m = torch.tensor(m_vals, dtype=torch.float32)
    return x, m


class COPDGRUDDataset(Dataset):
    """
    Loads COPD_PATIENTS_DATA.pkl and creates per-patient sequences:
      X: (T, D), M: (T, D), DT: (T,), length: int, y: float
    Assumes patient_timeseries are ascending by date and dt_gap_days = (y_date - visit_date).
    """

    def __init__(
        self,
        pkl_path: str,
        include_age: bool = True,
        include_gender: bool = True,
        include_dt_feature: bool = True,
    ):
        super().__init__()
        with open(pkl_path, "rb") as f:
            self.data: dict[Any, dict[str, Any]] = pickle.load(f)

        self.keys = clinical_feature_order()
        self.include_age = include_age
        self.include_gender = include_gender
        self.include_dt_feature = include_dt_feature

        # prebuild index of valid patients (at least 1 history step)
        self.index: list[Any] = []
        for pid, rec in self.data.items():
            T = len(rec.get("patient_timeseries", []))  # T = number of visits
            if T >= 1 and rec.get("y") is not None:
                self.index.append(pid)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        pid = self.index[i]
        rec = self.data[pid]
        visits: list[dict[str, Any]] = rec["patient_timeseries"]  # history only, ascending

        X_list: list[torch.Tensor] = []
        M_list: list[torch.Tensor] = []
        g_list: list[int] = []  # dt_gap_days (distance to y_date)

        for v in visits:
            x, m = flatten_visit(
                v, self.keys, self.include_age, self.include_gender, self.include_dt_feature
            )
            X_list.append(x)
            M_list.append(m)
            g_list.append(int(v.get("dt_gap_days", 0)))

        X = torch.stack(X_list, dim=0)  # (T, D)
        M = torch.stack(M_list, dim=0)  # (T, D)

        # Per-step gap between consecutive visits (positive days):
        # g[t] = distance to y_date, decreasing; Δ_t = g[t-1] - g[t], Δ_0=0
        T = X.size(0)
        DT = torch.zeros(T, dtype=torch.float32)
        if T > 1:
            g = torch.tensor(g_list, dtype=torch.float32)
            DT[1:] = g[:-1] - g[1:]  # positive

        return dict(
            X=X,  # (T, D) # feature values
            M=M,  # (T, D) # mask 1 if observed else 0
            DT=DT,  # (T,) # time gap since last visit (0 for first visit)
            length=torch.tensor(T, dtype=torch.long),  # sequence length T
            y=torch.tensor(float(rec["y"]), dtype=torch.float32),  # scalar target
            pid=pid,
        )


def collate_grud(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Pad ragged sequences; return X, M, DT, lengths, y (batch-first)."""
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
