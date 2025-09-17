# -*- coding: utf-8 -*-
from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools
import os
from pathlib import Path
import re
from typing import Dict, Iterable, List, Tuple

from loguru import logger
import matplotlib
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pptx import Presentation
from pptx.util import Inches, Pt
from tqdm import tqdm

from src import dataset
from src.config import FIGURES_DIR, INTERIM_DATA_DIR


def plot_variable_statistics(
    df_path: str | Path,
    output_dir: Path,
    *,
    log_mode: str = "none",  # "none" | "logy" | "logx" | "logdata"
    bins: int = 30,
) -> None:
    """
    Make per-(Test, Measurement, Variable) histograms with optional log scaling.

    log_mode:
      - "none"    : normal histogram.
      - "logy"    : log scale on y-axis (counts). Works with any values.
      - "logx"    : log scale on x-axis (values). Requires values > 0.
      - "logdata" : histogram of log10(values). Requires values > 0.

    Advantages:
      • logy: reveals rare tail counts without losing the bulk near zero.
      • logx: spreads small values, compresses long right tail (orders of magnitude).
      • logdata: if data are log-normal, distribution becomes closer to normal.
    """
    # fonts (small, consistent)
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 10,
            "axes.labelsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
        }
    )

    df = pd.read_csv(df_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory created at: {output_dir}")

    df["Prescription Date"] = pd.to_datetime(
        df["Prescription Date"], format="%Y%m%d", errors="coerce"
    )
    df["Result Numerical Value"] = pd.to_numeric(df["Result Numerical Value"], errors="coerce")

    # fixed variable order (adjust as needed)
    wanted_vars = ["Meas", "%Pred", "%Chg.", "Post_Meas", "Post_%Chg", "Post_%Pred"]

    for test in df["Test"].dropna().unique():
        for meas in df["Measurement"].dropna().unique():
            for var in wanted_vars:
                df_filtered = df[
                    (df["Test"] == test) & (df["Measurement"] == meas) & (df["Variable"] == var)
                ]
                if df_filtered.empty:
                    continue

                # Clean file-safe parts
                safe = lambda s: re.sub(r"[\\/]", "_", str(s))
                safe_test, safe_meas, safe_var = safe(test), safe(meas), safe(var)

                vals = df_filtered["Result Numerical Value"].dropna()
                if vals.empty:
                    continue

                # Stats (on original scale)
                n_pat = df_filtered["Patient Number"].nunique()
                vmin, vmax, vmean = vals.min(), vals.max(), vals.mean()

                # ---- pick plot mode ----
                title_suffix = ""
                zeros_or_neg = int((vals <= 0).sum())

                if log_mode == "logy":
                    # log counts
                    plt.figure(figsize=(4.5, 2.7))
                    plt.hist(vals, bins=bins, alpha=0.7, log=True)
                    plt.ylabel("Frequency (log scale)")
                    title_suffix = " [log-y]"

                elif log_mode == "logx":
                    # log x-axis → need positive values
                    pos = vals[vals > 0]
                    if pos.empty:
                        print(f"[skip logx] {test}-{meas}-{var}: no positive values.")
                        continue
                    # log-spaced bins
                    lo, hi = pos.min(), pos.max()
                    if lo == hi:  # avoid identical bin edges
                        lo *= 0.9
                        hi *= 1.1
                    log_bins = np.logspace(np.log10(lo), np.log10(hi), bins)
                    plt.figure(figsize=(4.5, 2.7))
                    plt.hist(pos, bins=log_bins, alpha=0.7)
                    plt.xscale("log")
                    plt.xlabel("Result Numerical Value (log scale)")
                    if zeros_or_neg:
                        title_suffix = f" [log-x | excl ≤0: {zeros_or_neg}]"
                    else:
                        title_suffix = " [log-x]"

                elif log_mode == "logdata":
                    # histogram of log10(values) → need positive values
                    pos = vals[vals > 0]
                    if pos.empty:
                        print(f"[skip logdata] {test}-{meas}-{var}: no positive values.")
                        continue
                    plt.figure(figsize=(4.5, 2.7))
                    plt.hist(np.log10(pos), bins=bins, alpha=0.7)
                    plt.xlabel("log10(Result Numerical Value)")
                    title_suffix = " [log10(data)]"

                else:  # "none"
                    plt.figure(figsize=(4.5, 2.7))
                    plt.hist(vals, bins=bins, alpha=0.7)

                # Common labels
                if log_mode != "logx":  # logx already set a custom x label
                    plt.xlabel("Result Numerical Value")
                plt.ylabel("Frequency")
                plt.title(
                    f"Distribution of '{var}' for '{test}' - '{meas}'{title_suffix}\n"
                    f"Unique Patients: {n_pat}\n"
                    f"Min: {vmin:.2f}, Max: {vmax:.2f}, Mean: {vmean:.2f}"
                )

                plt.tight_layout()

                # Save with mode tag
                mode_tag = {"none": "lin", "logy": "logy", "logx": "logx", "logdata": "logdata"}[
                    log_mode
                ]
                out_name = f"{safe_var}_{safe_test}_{safe_meas}_distribution_{mode_tag}.png"
                plt.savefig(Path(output_dir) / out_name, dpi=300)
                plt.close()


# plot_variable_statistics(INTERIM_DATA_DIR / "ALL_PRESCRIPTION_DATA_FILTERED.csv",
#                          FIGURES_DIR / "Variant_statistics", log_mode="logy")


def plot_patient_variables_grid(
    dfx: pd.DataFrame,
    patient_id: int,
    out_dir: Path = Path(FIGURES_DIR / "pft_plots_all_variants"),
    plot_variable_order: List[str] = None,
    wanted_measure_indexes: List[str] = None,
    measure_index_colors: Dict[str, str] = None,
    marker: str = "o",
    linewidth: float = 1.6,
    plot_anomaly: bool = True,
    *,
    title_suffix: str = "",
) -> None:
    """
    Create one figure with vertical subplots (one per Variable).
    - Plots selected Measurements over time (per subplot).
    - Uses full YYYY-MM-DD dates on x-axis; only bottom subplot has x-label.
    - Overlays anomaly markers as hollow squares (no text labels).
    - Adds measurement legend (single row) and stacked anomaly legend lines.

    Expects anomaly columns if available:
      'Anomaly_Missing', 'Anomaly_Range', 'Outlier_MAD', 'Outlier_Jump'
    The plot renders fine even if they are absent.
    """

    if dfx.empty:
        return

    # ---- constants / basic prep ----
    DATE_COL = "Prescription Date"
    VALUE_COL = "Result Numerical Value"
    VAR_COL = "Variable"
    MEAS_COL = "Measurement"

    # dfx = dfx.copy()
    dfx[DATE_COL] = pd.to_datetime(dfx[DATE_COL], errors="coerce")
    dfx = dfx.sort_values(DATE_COL)

    if plot_variable_order is None:
        # plot_variable_order = ["Meas", "%Pred", "%Chg.", "Post_Meas", "Post_%Pred", "Post_%Chg"]
        plot_variable_order = ["Meas", "Post_Meas"]
    if wanted_measure_indexes is None:
        wanted_measure_indexes = sorted(dfx[MEAS_COL].dropna().unique().tolist())
    if measure_index_colors is None:
        # base = ["C0","C1","C2","C3","C4","C5","C6","C7","C8","C9"]
        # '#9467bd'
        base = [
            "#440B79",
            "#7f7f7f",
            "#ff7f0e",
            "#1f77b4",
            "#aec7e8",
            "#ffbb78",
            "#2ca02c",
            "#98df8a",
            "#d62728",
            "#ff9896",
            "#c5b0d5",
            "#8c564b",
            "#c49c94",
            "#e377c2",
            "#f7b6d2",
            "#c7c7c7",
            "#bcbd22",
            "#dbdb8d",
            "#17becf",
            "#9edae5",
        ]
        measure_index_colors = {
            mi: c for mi, c in zip(wanted_measure_indexes, itertools.cycle(base))
        }

    # anomaly columns we might have
    anomaly_cols = [
        "Anomaly_Missing",
        "Anomaly_Range",
        "Outlier_MAD",
        "Outlier_Jump",
        "Outlier_MAD_JUMP",
    ]

    # pre-check: only Meas/Post_Meas
    subset = dfx[dfx["Variable"].isin(["Meas", "Post_Meas"])]
    has_anomaly = any(col in subset.columns and subset[col].any() for col in anomaly_cols)
    if not has_anomaly:
        return  # skip plotting entirely

    # Define marker mapping per Test
    MARKERS_BY_TEST = {"Pre_PFT": ".", "Post_BD": "1", "COD": "^", None: "|"}
    MARKER_COLORS_BY_TEST = {"Pre_PFT": "red", "Post_BD": "black", "COD": "green", None: "blue"}

    # anomaly colors (distinct from typical Matplotlib defaults)
    ANOM_COLORS = {
        "Missing": "#FFB300",  # amber
        "Range": "#8B0000",  # dark red
        "MAD+Jump": "#036825",  # purple
        "MAD": "#000000",  # black
        "Jump": "#00BFA6",  # teal
    }
    have_anom_cols = all(
        col in dfx.columns
        for col in ["Anomaly_Missing", "Anomaly_Range", "Outlier_MAD", "Outlier_Jump"]
    )

    # ---- figure & axes ----
    nrows = len(plot_variable_order)
    nr = nrows + 1  # extra room for legends
    sub_plot_height = 2.5
    plot_height = sub_plot_height * nr
    plot_width = (16 * sub_plot_height * nr) / 9

    fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(plot_width, plot_height), sharex=False)
    if nrows == 1:
        axes = [axes]

    handles_all, labels_all = [], []

    # ---- plotting per variable ----
    for ax, var in zip(axes, plot_variable_order):
        dft_var = dfx[dfx[VAR_COL] == var]
        ax.set_ylabel(var)
        ax.grid(True, linestyle="--", alpha=0.3)

        any_line = False
        for mi in wanted_measure_indexes:
            dft_sub = dft_var[dft_var[MEAS_COL] == mi]
            if dft_sub.empty:
                continue

            # group by Test so each subgroup has its own marker
            for test_val, dft in dft_sub.groupby("Test"):
                (h,) = ax.plot(
                    dft[DATE_COL],
                    dft[VALUE_COL],
                    marker=MARKERS_BY_TEST.get(test_val, "|"),
                    markerfacecolor=MARKER_COLORS_BY_TEST.get(test_val, "blue"),
                    markeredgecolor=MARKER_COLORS_BY_TEST.get(test_val, "blue"),
                    linewidth=linewidth,
                    color=measure_index_colors.get(mi, "black"),
                    label=f"{test_val}-{mi}",
                    zorder=2,
                )
                any_line = True
                if not any(lbl.get_label() == f"{test_val}-{mi}" for lbl in handles_all):
                    handles_all.append(h)
                    labels_all.append(f"{test_val}-{mi}")

                # anomaly overlays (squares) — only if anomaly cols exist
                if plot_anomaly and have_anom_cols:
                    anom_specs = [
                        ("MAD", "Outlier_MAD"),
                        ("Jump", "Outlier_Jump"),
                        ("MAD+Jump", "Outlier_MAD_JUMP"),
                        ("Range", "Anomaly_Range"),
                        ("Missing", "Anomaly_Missing"),
                    ]
                    for key, col in anom_specs:
                        if col in dft.columns and dft[col].any():
                            bad = dft[dft[col]]
                            ax.scatter(
                                bad[DATE_COL],
                                bad[VALUE_COL],
                                s=80,
                                marker="s",
                                facecolors="none",
                                edgecolors=ANOM_COLORS[key],
                                linewidths=1.8,
                                zorder=4,
                            )

        if not any_line:
            ax.text(
                0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, alpha=0.6
            )

        # x-axis as full date per subplot
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        for tick in ax.get_xticklabels():
            tick.set_rotation(30)
            tick.set_ha("right")

    # x-label only on bottom subplot
    axes[-1].set_xlabel("Prescription Date")

    # title (with optional suffix)
    fig.suptitle(
        f"Patient {patient_id} — All Variables{(' — ' + title_suffix) if title_suffix else ''}",
        y=0.995,
        fontsize=14,
    )

    # -------- LEGENDS (measurement row, then one anomaly per line) --------
    # 1) One line per anomaly (stacked)
    anom_specs = [
        ("Missing Value: Value is zero.", ANOM_COLORS["Missing"]),
        (
            "Out of Valid Range: FEV1 & FVC (Meas/Post_Meas) 0.2-10.0; DLCO 0.3-50; "
            "FEV1/FVC 0.1-120; %Pred 0-200.",
            ANOM_COLORS["Range"],
        ),
        ("MAD: |robust_z| > z_thresh (3.5)", ANOM_COLORS["MAD"]),
        (
            "Jump (Δ/month) — Meas: (FEV1 0.3, FVC 0.4, DLCO 3.0, FEV1/FVC 8%, DLCO/VA 0.6) | "
            "%Pred: (FEV1 8%, FVC 8%, DLCO 8%, FEV1/FVC 8%, DLCO/VA 8%) | "
            "%Chg.: (FEV1 7%, FVC 4%, DLCO 0.3%, FEV1/FVC 4%, DLCO/VA 0.6%)",
            ANOM_COLORS["Jump"],
        ),
        ("MAD+Jump: Both MAD and Jump anomalies.", ANOM_COLORS["MAD+Jump"]),
    ]

    # 2) Measurement legend: single row

    lengend_start_y = 0.18

    if handles_all:
        fig.legend(
            handles_all,
            labels_all,
            loc="lower center",
            # bbox_to_anchor=(0.5, 0.08),
            bbox_to_anchor=(0.5, lengend_start_y),
            ncol=len(labels_all),
            fontsize=9,
            frameon=False,
            handlelength=2.0,
            handletextpad=0.6,
            columnspacing=1.2,
        )

    y0, dy = lengend_start_y - 0.02, 0.02  # starting y and spacing between lines
    for i, (lab, col) in enumerate(anom_specs):
        h = Line2D(
            [0],
            [0],
            marker="s",
            linestyle="None",
            markersize=8,
            markerfacecolor="none",
            markeredgecolor=col,
            label=lab,
        )
        fig.legend(
            [h],
            [lab],
            loc="lower center",
            bbox_to_anchor=(0.5, y0 - i * dy),
            ncol=1,
            fontsize=9,
            frameon=False,
            handlelength=1.2,
        )
    fig.subplots_adjust(bottom=sub_plot_height / plot_height, top=0.95, hspace=0.5)

    # ---- save ----
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = f"patient_{patient_id}_variables_grid.png"
    fig.savefig(out_dir / out_name, dpi=300, bbox_inches="tight")
    plt.close(fig)


matplotlib.use("Agg")  # headless, safe in child processes


def _plot_one(args):
    pid, dfx = args

    # Prevent thread over-subscription inside each process
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    # plot_patient_variables_grid(dfx, pid)
    dfx_flagged, title_tag, anomaly_flag = dataset.detect_anomalies(dfx)  # ← one function
    if anomaly_flag:
        plot_patient_variables_grid(dfx_flagged, pid, title_suffix=title_tag)
    return pid


def plot_all_patients_in_parallel():
    df = pd.read_csv(Path(INTERIM_DATA_DIR / "ALL_PRESCRIPTION_DATA_TMV.csv"))
    df["Prescription Date"] = pd.to_datetime(
        df["Prescription Date"], format="%Y%m%d", errors="coerce"
    )
    df["Result Numerical Value"] = pd.to_numeric(df["Result Numerical Value"], errors="coerce")

    # want = df["Patient Number"].unique().tolist()  # all patients
    # want = [886482, 1207865, 1452945, 611957, 965594, 5665, 7429, 29903, 42405]
    want = [886482]

    tasks = [
        (pid, g.copy()) for pid, g in df.groupby("Patient Number") if pid in want and not g.empty
    ]

    max_workers = max(1, min(48, (os.cpu_count() or 8) - 2))
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_plot_one, t) for t in tasks]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Plotting patients"):
            pid_done = fut.result()  # raises if error inside worker
            # logger.info(f"✓ plotted patient {pid_done}")

    # with ProcessPoolExecutor(max_workers=max_workers) as ex:
    #     futures = [ex.submit(_plot_one, t) for t in tasks]
    #     for fut in as_completed(futures):
    #         logger.info(f"✓ plotted patient {fut.result()}")


if __name__ == "__main__":
    df = pd.read_csv(
        Path(r"D:\Research\Project_COPD\COPD\data\interim\ALL_PRESCRIPTION_DATA_FILTERED.csv")
    )
    df["Prescription Date"] = pd.to_datetime(
        df["Prescription Date"], format="%Y%m%d", errors="coerce"
    )
    df["Result Numerical Value"] = pd.to_numeric(df["Result Numerical Value"], errors="coerce")
    patient_ids = [886482, 1207865, 1452945, 611957, 965594]
    for pid in patient_ids:
        dfx = df[(df["Patient Number"] == pid)]
        plot_patient_variables_grid(dfx, pid)
