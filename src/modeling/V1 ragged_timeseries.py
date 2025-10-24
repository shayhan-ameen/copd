# src/copd/data/grud_dataset.py
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

# import your CLINICAL_MAP from wherever it is defined
from src.config import PROCESSED_DATA_DIR
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


class TimeSeriesDataset(Dataset):
    """
    Dual-mode cached dataset.

    - compute=True  → precompute once in __init__, optionally save to timeseries_path.
    - compute=False → load precomputed cache from timeseries_path.
    Always uses include_age=True, include_gender=True, include_dt_feature=True.
    """

    def __init__(
        self,
        pkl_path: Path | str = PROCESSED_DATA_DIR / "COPD_PATIENTS_DATA.pkl",
        compute: bool = False,
        timeseries_path: Path | str | None = PROCESSED_DATA_DIR / "COPD_TIMESERIES_CACHE.pt",
    ):
        super().__init__()

        # Fixed feature configuration
        self.keys = clinical_feature_order()
        self.include_age = True
        self.include_gender = True
        self.include_dt_feature = True

        self._cache: list[dict[str, torch.Tensor]] = []
        self._pids: list[Any] = []
        self.index: list[Any] = []

        if compute:
            # --- Build from raw patients pkl ---
            with open(pkl_path, "rb") as f:
                self.data: dict[Any, dict[str, Any]] = pickle.load(f)

            # Build stable index
            for pid, rec in self.data.items():
                T = len(rec.get("patient_timeseries", []))
                if T >= 1 and rec.get("y") is not None:
                    self.index.append(pid)

            # Precompute & cache
            for pid in self.index:
                rec = self.data[pid]
                visits: list[dict[str, Any]] = rec["patient_timeseries"]

                X_list: list[torch.Tensor] = []
                M_list: list[torch.Tensor] = []
                g_list: list[int] = []

                for v in visits:
                    x, m = flatten_visit(
                        v,
                        self.keys,
                        include_age=self.include_age,
                        include_gender=self.include_gender,
                        include_dt_feature=self.include_dt_feature,
                    )
                    X_list.append(x)
                    M_list.append(m)
                    g_list.append(int(v.get("dt_gap_days", 0)))

                X = torch.stack(X_list, dim=0)  # (T, D)
                M = torch.stack(M_list, dim=0)  # (T, D)

                Tlen = X.size(0)
                DT = torch.zeros(Tlen, dtype=torch.float32)
                if Tlen > 1:
                    g = torch.tensor(g_list, dtype=torch.float32)
                    DT[1:] = g[:-1] - g[1:]

                sample = dict(
                    X=X,
                    M=M,
                    DT=DT,
                    length=torch.tensor(Tlen, dtype=torch.long),
                    y=torch.tensor(float(rec["y"]), dtype=torch.float32),
                    pid=torch.tensor(pid) if isinstance(pid, (int, float)) else pid,
                )
                self._cache.append(sample)
                self._pids.append(pid)

            # Optionally persist cache
            if timeseries_path is not None:
                payload = {
                    "cache": self._cache,
                    "pids": self._pids,
                    "keys": self.keys,  # helpful for downstream interpretation
                    "include_age": self.include_age,
                    "include_gender": self.include_gender,
                    "include_dt_feature": self.include_dt_feature,
                }
                torch.save(payload, timeseries_path)

            # === Free the raw patient dict to reclaim RAM ===
            if hasattr(self, "data"):  # only exists in compute=True path
                self.data.clear()  # drop inner dicts quickly
                del self.data  # remove the attribute
                import gc

                gc.collect()  # prompt Python to release memory

            print(
                f"[TimeSeriesDataset] Computed and cached {len(self._cache)} patients "
                f"(D={self._cache[0]['X'].shape[-1] if self._cache else 'NA'})"
            )

        else:
            # --- Load precomputed cache ---
            if not timeseries_path:
                raise ValueError("timeseries_path must be provided when compute=False.")
            blob = torch.load(timeseries_path, map_location="cpu")
            self._cache = blob["cache"]
            self._pids = blob.get("pids", list(range(len(self._cache))))
            # (optional) sanity: keys & include_* should match; we trust the file
            self.index = list(range(len(self._cache)))
            print(
                f"[TimeSeriesDataset] Loaded cached dataset from {timeseries_path} "
                f"({len(self._cache)} patients)"
            )

    def __len__(self) -> int:
        return len(self._cache)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        return self._cache[i]
