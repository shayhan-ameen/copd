from src.logging_config import setup_logger
from src import dataset  # import your dataset.py module

# imports may be removed
import pandas as pd
from pathlib import Path
from src.config import INTERIM_DATA_DIR
import os
from src.plots import plot_all_patients_in_parallel

logger = setup_logger(verbose=True)

def main():
    logger.info("🚀 Starting main pipeline")
    # dataset.process_prescription_files()   # call function from dataset.py
    plot_all_patients_in_parallel()
    logger.success("✅ Pipeline finished successfully")

if __name__ == "__main__":
    main()
