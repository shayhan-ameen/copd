# from pathlib import Path

# from loguru import logger
# from tqdm import tqdm
# import typer

# from src.config import PROCESSED_DATA_DIR

# app = typer.Typer()


# @app.command()
# def main(
#     # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
#     input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
#     output_path: Path = PROCESSED_DATA_DIR / "features.csv",
#     # -----------------------------------------
# ):
#     # ---- REPLACE THIS WITH YOUR OWN CODE ----
#     logger.info("Generating features from dataset...")
#     for i in tqdm(range(10), total=10):
#         if i == 5:
#             logger.info("Something happened for iteration 5.")
#     logger.success("Features generation complete.")
#     # -----------------------------------------


# if __name__ == "__main__":
#     app()


from ast import For
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib
matplotlib.use("Agg")  # headless, safe in child processes

def _plot_one(args):
    pid, dfx = args
    # dfx is already filtered for this pid
    plot_patient_variables_grid(dfx, pid)
    return pid

if __name__ == "__main__":
    # Prepare small per-patient slices to avoid pickling the whole df to every worker
    want = set([886482, 1207865, 1452945, 611957, 965594, 5665, 7429, 29903])
    tasks = [(pid, g.copy()) for pid, g in df.groupby("Patient Number") if pid in want and not g.empty]

    # Choose worker count
    max_workers = max(1, (os.cpu_count() or 2) - 1)

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_plot_one, t) for t in tasks]
        for fut in as_completed(futures):
            pid_done = fut.result()  # raises if an error happened
            print(f"✓ plotted patient {pid_done}")


# Why this pattern:

#   Processes (not threads) → avoids Matplotlib thread-safety issues.

#   Pre-sliced DataFrames → only small chunks are pickled to workers (much faster).

#   Agg backend → safe, no GUI needed when saving PNGs.

#   Windows‑safe → everything inside if __name__ == "__main__":.

#   That’s it—your figures for all patients will generate in parallel.


# --------------------- my processor is intel(r) xeon(r) silver 4216 cpu @ 2.10ghz

# Recommended worker count

#     For this kind of CPU‑bound + a bit of I/O (saving PNGs), use:

#     max_workers = 14–24
#     (leave 2–4 threads free for the OS; oversubscribing all 32 threads can slow things down)

# Avoid BLAS over‑subscription (very important)

#     Pandas/NumPy may spawn their own threads (MKL/OpenBLAS). Set these to 1 in workers so your 14–24 processes don’t each spin up 32 threads:

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib
matplotlib.use("Agg")  # headless, safe in child processes

def _plot_one(args):
    pid, dfx = args

    # Prevent thread over-subscription inside each process
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    plot_patient_variables_grid(dfx, pid)
    return pid

if __name__ == "__main__":
    want = [886482, 1207865, 1452945, 611957, 965594, 5665, 7429, 29903]
    tasks = [(pid, g.copy()) for pid, g in df.groupby("Patient Number") if pid in want and not g.empty]

    max_workers = max(1, min(24, (os.cpu_count() or 8) - 2))
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_plot_one, t) for t in tasks]
        for fut in as_completed(futures):
            print(f"✓ plotted patient {fut.result()}")
