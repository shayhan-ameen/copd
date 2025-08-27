from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path
from typing import Dict, List
import itertools
import pandas as pd
import matplotlib.pyplot as plt

FIGURES_DIR = Path("reports/figures")

df = pd.read_csv(Path(r"D:\Research\Project_COPD\COPD\data\interim\ALL_PRESCRIPTION_DATA_FILTERED.csv"))
dfx = df[(df["Patient Number"] == 5805)]

def plot_patient_variables_grid(
    dfx: pd.DataFrame,
    patient_id: int,
    out_dir: Path = Path(FIGURES_DIR / "pft_plots_all_variants"),
    plot_variable_order: List[str] = None,              # fixed order of subplots
    wanted_measure_indexes: List[str] = None,           # which measurements (e.g., ["FVC","FEV1","FEV1/FVC","DLCO"])
    measure_index_colors: Dict[str, str] = None,        # color per measurement
    marker: str = "o",
    linewidth: float = 1.6,
) -> None:
    """
    One figure with seven vertical subplots, one per Variable.
    Each subplot shows all selected Measurements over time.
    No twin axis (all values on same y for that subplot).
    """
    if dfx.empty:
        return

    # sort by time
    dfx = dfx.copy().sort_values("Prescription Date")

    # defaults
    if plot_variable_order is None:
        plot_variable_order = ["Meas","Pred","%Pred","%Chg.","Post_Meas","Post_%Pred","Post_%Chg"]
    if wanted_measure_indexes is None:
        wanted_measure_indexes = sorted(dfx["Measurement"].dropna().unique().tolist())
    if measure_index_colors is None:
        base = ["C0","C1","C2","C3","C4","C5","C6","C7","C8","C9"]
        measure_index_colors = {mi: c for mi, c in zip(wanted_measure_indexes, itertools.cycle(base))}

    # build figure with 7 rows
    nrows = len(plot_variable_order)
    fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(14, 18), sharex=True)
    if nrows == 1:
        axes = [axes]

    handles_all, labels_all = [], []

    for ax, var in zip(axes, plot_variable_order):
        dft_var = dfx[dfx["Variable"] == var]
        ax.set_title(f"Variable: {var}")
        ax.grid(True, linestyle="--", alpha=0.3)

        any_line = False
        for mi in wanted_measure_indexes:
            dft = dft_var[dft_var["Measurement"] == mi]
            if dft.empty:
                continue

            h, = ax.plot(
                dft["Prescription Date"],
                dft["Result Numerical Value"],
                marker=marker,
                linewidth=linewidth,
                color=measure_index_colors.get(mi, "black"),
                label=mi,
            )
            any_line = True

            # collect one handle per measurement for a global legend
            if not any(lbl.get_label() == mi for lbl in handles_all):
                handles_all.append(h)
                labels_all.append(mi)

        if not any_line:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, alpha=0.6)

    # cosmetics
    axes[-1].set_xlabel("Prescription Date")
    fig.suptitle(f"Patient {patient_id} — All Variables", y=0.995, fontsize=14)

    # global legend (measurements) at bottom
    if handles_all:
        fig.legend(
            handles_all, labels_all,
            loc="lower center", bbox_to_anchor=(0.5, 0.0),
            ncol=min(5, len(labels_all)), fontsize=10, frameon=False
        )
        fig.subplots_adjust(bottom=0.08, top=0.96, hspace=0.22)
    else:
        fig.subplots_adjust(top=0.96, hspace=0.22)

    # save
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = f"patient_{patient_id}_variables_grid.png"
    fig.savefig(out_dir / out_name, dpi=300, bbox_inches="tight")
    plt.close(fig)

plot_patient_variables_grid(dfx, 5805)