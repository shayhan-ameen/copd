# from pathlib import Path

# from loguru import logger
# from tqdm import tqdm
# import typer

# from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

# app = typer.Typer()


# @app.command()
# def main(
#     # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
#     input_path: Path = RAW_DATA_DIR / "dataset.csv",
#     output_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
#     # ----------------------------------------------
# ):
#     # ---- REPLACE THIS WITH YOUR OWN CODE ----
#     logger.info("Processing dataset...")
#     for i in tqdm(range(10), total=10):
#         if i == 5:
#             logger.info("Something happened for iteration 5.")
#     logger.success("Processing dataset complete.")
#     # -----------------------------------------


# if __name__ == "__main__":
#     app()

# from pathlib import Path

# # from loguru import logger
# import pandas as pd
# # from src.logging_config import logger
# # , setup_logger
# # logger = setup_logger(verbose=True)


from pathlib import Path
from loguru import logger
import pandas as pd

from src.config import RAW_DATA_DIR, INTERIM_DATA_DIR  # import your globals

from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np


def summarize_pft_df(df: pd.DataFrame, show_values: bool = True) -> Dict[str, Any]:
    """
    Prints and returns:
      1) total unique patients
      2) total unique tests (count + values)
      3) total unique measurements (count + values)
      4) total unique variables (count + values)
      5) total unique (measurement, test) combinations (count + value_counts table)
    """
    required = ['Patient Number', 'Test', 'Measurement', 'Variable']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    # Patients
    patients = df['Patient Number'].dropna().unique()

    # Helper for cleaned uniques
    def uniq(col: str):
        s = df.loc[df[col].notna(), col].astype(str).str.strip()
        return sorted(s.unique())

    tests_u = uniq('Test')
    meas_u  = uniq('Measurement')
    vars_u  = uniq('Variable')

    # ---- (Measurement, Test) value counts (cleaned) ----
    mt_base = df.loc[df['Measurement'].notna() & df['Test'].notna(), ['Measurement', 'Test']].copy()
    mt_base['Measurement'] = mt_base['Measurement'].astype(str).str.strip()
    mt_base['Test'] = mt_base['Test'].astype(str).str.strip()

    combo_counts = (mt_base
                    .value_counts(['Measurement', 'Test'])
                    .reset_index(name='count')
                    .sort_values(['Measurement', 'Test'])
                    .reset_index(drop=True))

    result = {
        'total_unique_patients': int(len(patients)),

        'total_unique_tests': len(tests_u),
        'unique_tests': tests_u,

        'total_unique_measurements': len(meas_u),
        'unique_measurements': meas_u,

        'total_unique_variables': len(vars_u),
        'unique_variables': vars_u,

        # based on the rows in combo_counts
        'total_unique_(measurement,test)_combinations': int(combo_counts.shape[0]),
        'combo_counts_df': combo_counts,  # DataFrame with Measurement, Test, count
    }

    # ---- Print nicely ----
    print(f"1) Total unique patients: {result['total_unique_patients']}")
    print(f"2) Total unique tests: {result['total_unique_tests']}")
    if show_values:
        print("   Unique tests: " + ", ".join(tests_u))
    print(f"3) Total unique measurements: {result['total_unique_measurements']}")
    if show_values:
        print("   Unique measurements: " + ", ".join(meas_u))
    print(f"4) Total unique variables: {result['total_unique_variables']}")
    if show_values:
        print("   Unique variables: " + ", ".join(vars_u))
    print(f"5) Total unique (measurement, test) combinations: {result['total_unique_(measurement,test)_combinations']}")
    if show_values:
        print("   Combination counts (Measurement, Test, count):")
        # print(combo_counts.to_string(index=False))
        display(combo_counts)

    return result


def detect_anomalies(dfx: pd.DataFrame,
                     date_col: str = "Prescription Date",
                     value_col: str = "Result Numerical Value",
                     measure_col: str = "Measurement",
                     variable_col: str = "Variable",
                     z_thresh: float = 3.5,
                     jump_thresh_per_month: Dict[str, float] | None = None
                    ) -> Tuple[pd.DataFrame, str]:
    """
    Returns:
      flagged_df (same rows as dfx, with anomaly columns)
      title_tag  (e.g., 'Missing value+Out of valid range+MAD+Jump' for this patient)

    Flags:
      - Anomaly_Missing: value == 0 or NaN
      - Anomaly_Range: outside clinical ranges
          FEV1 (Meas/Post_Meas): 0.2–10.0
          FVC  (Meas/Post_Meas): 0.3–10.0
          DLCO (Meas/Post_Meas): 0.3–50
          FEV1/FVC ratio (Meas/Post_Meas): 0.2–1.2
          %Pred ( %Pred/Post_%Pred ): 0–200
      - Outlier_MAD: |robust_z| > z_thresh (per-measurement series)
      - Outlier_Jump: month-normalized step exceeds measurement threshold
    """
    jump_thresh_per_month = jump_thresh_per_month or {
        "FEV1": 0.30, "FVC": 0.40, "DLCO": 3.0, "FEV1/FVC": 0.08, "DLCO/VA": 0.60,
    }
    abs_vars   = {"Meas", "Post_Meas"}
    perc_vars  = {"%Pred", "Post_%Pred"}

    df2 = dfx.copy()

    # Ensure numeric + datetime for calculations
    v = pd.to_numeric(df2[value_col], errors="coerce")
    dt = pd.to_datetime(df2[date_col], errors="coerce")
    df2["_val_"] = v
    df2["_date_"] = dt

    # 1) Missing (as requested: treat 0 as missing; also NaN is missing)
    df2["Anomaly_Missing"] = v.isna() | (v == 0)

    # 2) Valid ranges
    rng_flag = pd.Series(False, index=df2.index)

    def _violate(series_mask, low, high):
        if not series_mask.any(): 
            return pd.Series(False, index=df2.index)
        s = v.where(series_mask)
        return (s < low) | (s > high)

    # Absolute measurements (Meas/Post_Meas)
    m = df2[measure_col].astype(str)
    var = df2[variable_col].astype(str)

    mask_abs = var.isin(abs_vars)
    rng_flag |= _violate(mask_abs & (m == "FEV1"), 0.2, 10.0)
    rng_flag |= _violate(mask_abs & (m == "FVC"),  0.3, 10.0)
    rng_flag |= _violate(mask_abs & (m == "DLCO"), 0.3, 50.0)

    # Ratio FEV1/FVC (absolute)
    rng_flag |= _violate(mask_abs & (m == "FEV1/FVC"), 0.2, 1.2)

    # Percent predicted
    mask_perc = var.isin(perc_vars)
    rng_flag |= _violate(mask_perc, 0.0, 200.0)

    df2["Anomaly_Range"] = rng_flag.fillna(False)

    # 3) Outliers (MAD + Jump) per measurement series
    out_mad  = pd.Series(False, index=df2.index)
    out_jump = pd.Series(False, index=df2.index)

    for mi, g in df2.groupby(measure_col, dropna=False):
        g = g.sort_values("_date_")
        vv = g["_val_"]
        dd = g["_date_"]

        # MAD
        med = vv.median()
        mad = float(np.median(np.abs(vv - med))) if len(vv) else 0.0
        mad = mad if mad > 0 else 1e-9
        robust_z = 0.6745 * (vv - med) / mad
        out_mad.loc[g.index] = robust_z.abs() > z_thresh

        # Jump per month
        dv = vv.diff().abs()
        dt_days = dd.diff().dt.days
        dt_days = dt_days.where(dt_days > 0, 1)  # avoid 0/NaN/<=0
        months = dt_days / 30.0
        thr = jump_thresh_per_month.get(str(mi), np.inf)
        out_jump.loc[g.index] = (dv / months) > thr

    df2["Outlier_MAD"]  = out_mad.fillna(False)
    df2["Outlier_Jump"] = out_jump.fillna(False)
    df2["Outlier"]      = df2["Outlier_MAD"] | df2["Outlier_Jump"]

    # Compose row-wise tags
    def _row_tags(row):
        tags = []
        if row["Anomaly_Missing"]: tags.append("Missing value")
        if row["Anomaly_Range"]:   tags.append("Out of valid range")
        if row["Outlier_MAD"]:     tags.append("MAD")
        if row["Outlier_Jump"]:    tags.append("Jump")
        return " | ".join(tags)

    df2["Anomaly_Tags"] = df2.apply(_row_tags, axis=1)
    # Short tag for tight annotations
    df2["Anomaly_Tags_Short"] = (df2["Anomaly_Tags"]
                                 .str.replace("Missing value", "Miss", regex=False)
                                 .str.replace("Out of valid range", "Range", regex=False))

    df2["Anomaly_Any"] = df2[["Anomaly_Missing","Anomaly_Range","Outlier_MAD","Outlier_Jump"]].any(axis=1)

    # Build patient-level title tag
    present = []
    if df2["Anomaly_Missing"].any(): present.append("Missing value")
    if df2["Anomaly_Range"].any():   present.append("Out of valid range")
    if df2["Outlier_MAD"].any():     present.append("MAD")
    if df2["Outlier_Jump"].any():    present.append("Jump")
    title_tag = "+".join(present)

    # Clean temp cols for plotting (keep dates numeric as original)
    df2 = df2.drop(columns=["_val_", "_date_"])

    return df2, title_tag





PFT_COL_MAP = {
    # PFT_2009--
    "환자번호": "Patient Number",
    "접수일자": "Prescription Date",  # "Reception Date",
    "처방코드": "Prescription Code",
    "수가처방명": "Prescription Name",
    "검사실": "Laboratory",
    "실시검사실": "Implementation laboratory",
    "부위": "Region",
    "결과항목명": "Result item name",
    "결과항목결과값": "Result Value",
    "결과항목결과값 수치": "Result Numerical Value",
    "Pacs번호": "Pacs Number",

    # __2403 01_240710
    "등록번호": "Patient Number",  # "Registration Number",
    "성별": "Gender",
    "생년월일": "Date of Birth",
    "내원구분": "Visit Type",  # O - Outdoor | I - Indoor | E - Emergency
    "진료일자": "Treatment Date",
    "처방명": "Prescription Name",
    "처방코드": "Prescription Code",
    "처방일자": "Prescription Date",
    "시행일자": "Implementation Date",
    "결과항목": "Result item name",  # "Result Item",
    "수치": "Result Numerical Value",  # "Value"
}


def rename_prescription_files(
    raw_dir: str | Path = RAW_DATA_DIR, #"data/raw",
    output_dir: str | Path | None = INTERIM_DATA_DIR / "pft_renamed_columns",
    audit_csv: str | Path | None = INTERIM_DATA_DIR / "pft_renamed_columns_audit.csv",
    sheet_name=0,
) -> pd.DataFrame:
    """
    Read all 'PFT*.xlsx' in raw_dir, rename Korean columns to English using PFT_COL_MAP,
    save cleaned CSVs, and write an audit CSV listing unknown/missing columns per file.

    Returns the audit DataFrame.
    """
    raw_dir = Path(raw_dir)
    if output_dir is None:
        output_dir = raw_dir.parent / "interim" / "pft_renamed_columns"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if audit_csv is None:
        audit_csv = raw_dir.parent / "interim" / "pft_renamed_columns_audit.csv"
    audit_csv = Path(audit_csv)
    audit_csv.parent.mkdir(parents=True, exist_ok=True)

    mapping_keys = set(PFT_COL_MAP.keys())
    audit_rows = []

    files = sorted(raw_dir.glob("*.xlsx"))
    # Exclude drug-related files
    if any("drug" in f.name.lower() for f in files):  
        files = [f for f in files if "drug" not in f.name.lower()]
    if not files:
        logger.warning(f"No files matched '*.xlsx' under {raw_dir.resolve()}")
        return pd.DataFrame()
    logger.success(f"Total prescription files: {len(files)}")

    for f in files:
        try:
            df = pd.read_excel(f, sheet_name=sheet_name, engine="openpyxl")
            if isinstance(df, dict):  # multiple sheets
                first_key = list(df.keys())[0]
                df = df[first_key]

            original_cols = list(df.columns)
            original_set = set(original_cols)

            unknown_cols = sorted(original_set - mapping_keys)
            missing_cols = sorted(mapping_keys - original_set)

            # df_renamed = df.rename(columns=PFT_COL_MAP)
            # df_renamed = df_renamed[[c for c in PFT_COL_MAP.values() if c in df_renamed.columns]]

            valid_map = {k: v for k, v in PFT_COL_MAP.items() if k in df.columns}
            df_renamed = df.rename(columns=valid_map)
            if valid_map:
                df_renamed = df_renamed[list(valid_map.values())]
            else:
                logger.warning(f"No known columns found in {f.name}; saving original columns.")
                df_renamed = df  # or skip saving this file



            out_path = output_dir / f"{f.stem}_renamed.csv"
            df_renamed.to_csv(out_path, index=False, encoding="utf-8-sig")

            logger.info(f"Processed file: {f.name} → {out_path.name}")

            if unknown_cols:
                logger.debug(f"Unknown columns in {f.name}: {unknown_cols}")
            if missing_cols:
                logger.debug(f"Missing mapping columns in {f.name}: {missing_cols}")

            audit_rows.append({
                "file": f.name,
                "rows": len(df),
                "cols": len(df.columns),
                "renamed_count": sum(c in PFT_COL_MAP for c in original_cols),
                "unknown_columns_not_in_mapping": " | ".join(map(str, unknown_cols)) if unknown_cols else "",
                "mapping_columns_missing_in_file": " | ".join(map(str, missing_cols)) if missing_cols else "",
                "saved_to": str(out_path),
                "status": "ok",
                "error": "",
            })

        except Exception as e:
            logger.error(f"Failed to process file: {f.name} → {repr(e)}")
            audit_rows.append({
                "file": f.name,
                "rows": "",
                "cols": "",
                "renamed_count": "",
                "unknown_columns_not_in_mapping": "",
                "mapping_columns_missing_in_file": "",
                "saved_to": "",
                "status": "error",
                "error": repr(e),
            })

    audit_df = pd.DataFrame(audit_rows)
    audit_df.to_csv(audit_csv, index=False, encoding="utf-8-sig")
    logger.info(f"Audit saved to: {audit_csv.resolve()}")
    logger.info(f"Cleaned files saved to: {output_dir.resolve()}")
    return audit_df


def merge_prescription_files(
    pft_dir: str | Path = INTERIM_DATA_DIR / "pft_renamed_columns",
    output_path: str | Path | None = None,
    save_parquet: bool = False
) -> pd.DataFrame:
    """
    Merge all cleaned PFT CSV files into one DataFrame and save it.

    Parameters
    ----------
    pft_dir : str | Path
        Directory containing cleaned PFT CSV files.
    output_path : str | Path | None
        Where to save the merged file (CSV by default).
        If None, saves to data/processed/pft_merged.csv.
    save_parquet : bool
        If True, also save a Parquet version for faster loading later.

    Returns
    -------
    pd.DataFrame
        The merged DataFrame.
    """
    pft_dir = Path(pft_dir)
    if not pft_dir.exists():
        raise FileNotFoundError(f"PFT directory not found: {pft_dir}")

    files = sorted(pft_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {pft_dir}")
    
    logger.success(f"Total prescription files: {len(files)}")

    df_list = []
    for f in files:
        try:
            df = pd.read_csv(f, encoding="utf-8-sig")
            # df = pd.read_csv(f, encoding="utf-8-sig", dtype=str)
            df_list.append(df)
        except Exception as e:
            print(f"Error reading {f.name}: {e}")

    if not df_list:
        raise ValueError("No valid CSVs could be read.")

    merged_df = pd.concat(df_list, ignore_index=True)

    if output_path is None:
        output_path = INTERIM_DATA_DIR  / "ALL_PRESCRIPTION_DATA.csv"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    merged_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Merged PFT saved to: {output_path.resolve()}")

    if save_parquet:
        pq_path = output_path.with_suffix(".parquet")
        merged_df.to_parquet(pq_path, index=False)
        print(f"Parquet version saved to: {pq_path.resolve()}")

    return merged_df

def log_uniques(col: str, uniques: list) -> None:
    """
    Log the number of unique values and list them all (one per line).
    """
    logger.info(f"Number of unique {col}: {len(uniques)}")
    logger.info(f"Unique {col}:\n" + "\n".join(map(str, uniques)))


def extract_test_measurement_variable(
    df_path: str | Path = INTERIM_DATA_DIR / "ALL_PRESCRIPTION_DATA.csv",
    output_path: str | Path = INTERIM_DATA_DIR / "ALL_PRESCRIPTION_DATA_TMV.csv",
    save_parquet: bool = False
) -> pd.DataFrame:
    """
    Build Test / Measurement / Variable from 'Result item name'.
    - Test: text before the measurement token (may be NA; later filled from Prescription Name)
    - Measurement: physiologic variable (e.g., FVC, DLCO, FEF/FIF50)
    - Variable: last token (Measurement, %Pred, %Chg., Pred); keeps 'Post_' if present
    """

    # --- paths ---
    df = pd.read_csv(Path(df_path))

    # ---- 2) Quick stats ----
    logger.info(f"Number of unique patients: {df['Patient Number'].nunique()}")
    df["Result item name"] = df["Result item name"].astype("string")
    logger.info(f"Number of unique Result item name: {df['Result item name'].nunique()}" )
    # logger.info(f"Uique Result item name: {df['Result item name'].unique()}")
    log_uniques("Result item name", df['Result item name'].dropna().unique().tolist())


    # ---- 3) Normalize source strings ----
    s = df['Result item name'].fillna('')

    pat = r'((?:Post_)?[^ _]+)$'        # last chunk; keep leading Post_ if present

    # Put the last token in Variable
    df['Variable']    = s.str.extract(pat, expand=False)

    # prefix Test+Measurement / Measurement:
    prefix = s.str.replace(pat, '', regex=True).str.rstrip(' _')  # e.g. "CO Diffusing DLCO", "FVC"

    # split from the RIGHT once → [Test, Measurement]
    parts = prefix.str.rsplit(' ', n=1, expand=True)

    # parts has columns 0 (Test prefix) and 1 (Measurement), with some 1 = None
    mask = parts[1].isna()          # True where parts[1] is None/NaN

    # move 0 -> 1, and clear 0
    parts.loc[mask, 1] = parts.loc[mask, 0]
    parts.loc[mask, 0] = pd.NA

    # now build final columns
    df['Measurement'] = parts[1]
    df['Test'] = parts[0]

    # If Test is empty, copy from Prescription Name
    df['Test'] = df['Test'].fillna(df['Prescription Name'])

    # turn blanks back to NA (optional)
    df.loc[s.eq(''), ['Measurement','Variable']] = pd.NA    


    log_uniques("Test", df['Test'].dropna().unique().tolist())
    log_uniques("Measurement", df['Measurement'].dropna().unique().tolist())
    log_uniques("Variable", df['Variable'].dropna().unique().tolist())


    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Saved to: {output_path.resolve()}")


    if save_parquet:
        pq_path = output_path.with_suffix(".parquet")
        df.to_parquet(pq_path, index=False)
        print(f"Parquet version saved to: {pq_path.resolve()}")

    return df


    

def filter_relevant_measurements(
    df_path: str | Path = INTERIM_DATA_DIR / "ALL_PRESCRIPTION_DATA_TMV.csv",
    desired_items: list[str] | None = None,         # default set below
    match: str = "exact",                           # "exact" or "contains"
    output_path: str | Path = INTERIM_DATA_DIR / "ALL_PRESCRIPTION_DATA_FILTERED.csv",
    save_parquet: bool = False,
) -> pd.DataFrame:
    """
    #TODO: Filter patients with only one record
    #TODO: Filter patients whose total duration is less than three years
    Keep rows where Measurement matches desired_items.
    match="exact"     -> exact labels (uses .isin)
    match="contains"  -> substring match (uses .str.contains)
    """
    df_path = Path(df_path)
    output_path = Path(output_path)

    if desired_items is None:
        desired_items = ["FVC", "FEV1", "DLCO", "FEV1/FVC"]  # sensible default

    # read as text to avoid dtype warnings / keep leading zeros
    df = pd.read_csv(df_path, dtype=str, encoding="utf-8-sig", low_memory=False)

    logger.info(f"Desired items: {desired_items}")
    logger.info(f"Unique patients before filtering: {df['Patient Number'].nunique()}")

    # build mask
    if match == "exact":
        mask = df["Measurement"].isin(desired_items)
    elif match == "contains":
        pattern = "|".join(map(re.escape, desired_items))  # safe OR-pattern
        mask = df["Measurement"].astype(str).str.contains(pattern, na=False)
    else:
        raise ValueError("match must be 'exact' or 'contains'")

    df_filtered = df[mask].copy()

    logger.info(f"Unique patients after filtering: {df_filtered['Patient Number'].nunique()}")

    log_uniques("Measurement after filtering", df_filtered['Measurement'].dropna().unique().tolist())
    

    # save filtered
    df_filtered.to_csv(output_path, index=False, encoding="utf-8-sig")
    logger.success(f"Saved filtered CSV to: {output_path.resolve()}")

    if save_parquet:
        pq_path = output_path.with_suffix(".parquet")
        df_filtered.to_parquet(pq_path, index=False)
        logger.success(f"Saved Parquet to: {pq_path.resolve()}")

    return df_filtered

    # item_list = df['Result item name'].unique()
    # desired_items = ["FVC", "FEV1", "DLCO"]

    # filter_items = [
    #     x for x in item_list
    #     if isinstance(x, str) and any(k in x for k in desired_items)
    # ]
    # logger.info(f"Desired items: {desired_items}")
    # logger.info(f"Number of relevant items: {len(filter_items)}")
    # logger.info(f"Relevant items: {filter_items}")
    # def last_token(s: str) -> str:
    # s = s.strip()
    # m = re.search(r'(?:Post_)?[^ _]+$', s)   # last chunk, optionally prefixed by Post_
    # return m.group(0) if m else ""
    # variables = [last_token(x) for x in filter_items]
    # uniq = list(set(variables))
    # logger.info(f"Unique variables: {uniq}")

    return df


def process_prescription_files(
    raw_dir: str | Path = "data/raw",
    output_dir: str | Path | None = None,
    audit_csv: str | Path | None = None,
    sheet_name=0,
) -> pd.DataFrame:
    # rename_prescription_files(raw_dir=raw_dir,output_dir=output_dir,audit_csv=audit_csv,sheet_name=sheet_name)
    # merge_prescription_files()
    # extract_test_measurement_variable()
    filter_relevant_measurements()
    # summarize_pft_df(df)




if __name__ == "__main__":
    logger.info("Starting prescription file processing...")
    process_prescription_files()
    logger.success("All prescription files processed successfully.")
