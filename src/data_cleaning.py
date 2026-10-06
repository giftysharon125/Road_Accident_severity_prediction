"""
Data Cleaning Module
Handles loading, missing value imputation, duplicate removal,
inconsistent value standardization, and outlier analysis.
"""

import pandas as pd
import numpy as np


def load_raw_data(file_path: str = "data/raw/road_accidents.csv") -> pd.DataFrame:
    """Loads the raw road accidents CSV file."""
    df = pd.read_csv(file_path)
    return df


def inspect_data(df: pd.DataFrame) -> dict:
    """
    Returns structural information about the dataset:
    dimensions, columns, dtypes, missing values, duplicates.
    """
    missing_counts = df.isnull().sum()
    missing_pct = (missing_counts / len(df) * 100).round(2)
    missing_df = pd.DataFrame({"Missing_Count": missing_counts, "Missing_Percentage": missing_pct})
    missing_summary = missing_df[missing_df["Missing_Count"] > 0].sort_values(
        by="Missing_Count", ascending=False
    )

    summary = {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "duplicates": int(df.duplicated().sum()),
        "missing_summary": missing_summary,
        "column_names": list(df.columns),
        "dtypes": df.dtypes.to_dict(),
    }
    return summary


def clean_inconsistent_categories(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes inconsistent strings, encoding glitches, and casing.
    For example:
    - Normalizes accident severity labels ('Fatal injury' -> 'Fatal Injury')
    - Fixes corrupted characters like '41?100Q' -> '41-100Q'
    - Normalizes driver experience and age bands ('unknown' -> 'Unknown')
    - Strips whitespace across all object columns
    """
    df = df.copy()

    # Strip whitespaces from string columns
    str_cols = df.select_dtypes(include=["object"]).columns
    for c in str_cols:
        df[c] = df[c].astype(str).str.strip()
        # Replace string literal 'nan' or 'na' with real NaN
        df[c] = df[c].replace({"nan": np.nan, "na": np.nan, "Unknown": "Unknown", "unknown": "Unknown"})

    # Standardize target Accident_severity
    if "Accident_severity" in df.columns:
        df["Accident_severity"] = df["Accident_severity"].replace({
            "Fatal injury": "Fatal Injury",
            "fatal injury": "Fatal Injury",
            "Slight injury": "Slight Injury",
            "Serious injury": "Serious Injury"
        })

    # Standardize vehicle type artifact characters (e.g., '?' instead of '-')
    if "Type_of_vehicle" in df.columns:
        df["Type_of_vehicle"] = df["Type_of_vehicle"].str.replace("?", "-", regex=False)

    # Standardize Driving_experience
    if "Driving_experience" in df.columns:
        df["Driving_experience"] = df["Driving_experience"].replace({
            "unknown": "Unknown",
            "No Licence": "No Licence"
        })

    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Imputes missing values:
    - Numerical columns: Median imputation (preserves distribution against extreme skew)
    - Categorical columns: Mode imputation or logical default ('Unknown' / 'Other')
    """
    df = df.copy()

    # Numerical imputation
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)

    # Categorical imputation
    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        if df[col].isnull().sum() > 0:
            mode_series = df[col].mode()
            mode_val = mode_series[0] if not mode_series.empty else "Unknown"
            df[col] = df[col].fillna(mode_val)

    return df


def remove_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Detects and removes duplicate records from the dataset."""
    initial_len = len(df)
    df_clean = df.drop_duplicates().copy()
    duplicates_removed = initial_len - len(df_clean)
    return df_clean, duplicates_removed


def detect_outliers_iqr(df: pd.DataFrame, column: str) -> dict:
    """
    Calculates IQR bounds and detects outliers for a given numerical column.
    Explains the statistical thresholds.
    """
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]

    return {
        "column": column,
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(iqr),
        "lower_bound": float(lower_bound),
        "upper_bound": float(upper_bound),
        "outlier_count": len(outliers),
        "outlier_percentage": round((len(outliers) / len(df)) * 100, 2),
    }


def clean_dataset(raw_path: str = "data/raw/road_accidents.csv") -> pd.DataFrame:
    """Complete data cleaning pipeline."""
    df = load_raw_data(raw_path)
    df = clean_inconsistent_categories(df)
    df, _ = remove_duplicates(df)
    df = handle_missing_values(df)
    return df


if __name__ == "__main__":
    cleaned_df = clean_dataset()
    print("Data cleaning completed successfully. Cleaned shape:", cleaned_df.shape)
    print("Null count:", cleaned_df.isnull().sum().sum())
