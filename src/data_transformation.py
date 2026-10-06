"""
Data Transformation Module
Derives meaningful features from cleaned dataset:
- Temporal features (Hour, Minute, Time_of_day, Rush_hour)
- Calendar features (Day_type: Weekday vs Weekend)
- Risk and operational indices (Collision_scale, Casualty_scale)
- Saves the fully engineered dataset to data/processed/
"""

import pandas as pd
import numpy as np
import sys
import os

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_cleaning import clean_dataset



def assign_time_of_day(hour: int) -> str:
    """Categorizes an hour into Morning, Afternoon, Evening, or Night."""
    if 6 <= hour < 12:
        return "Morning"
    elif 12 <= hour < 17:
        return "Afternoon"
    elif 17 <= hour < 22:
        return "Evening"
    else:
        return "Night"


def assign_rush_hour(hour: int) -> str:
    """Classifies peak commuting rush hours versus non-rush hours."""
    if (7 <= hour <= 9) or (16 <= hour <= 19):
        return "Rush Hour"
    return "Non-Rush Hour"


def assign_day_type(day: str) -> str:
    """Categorizes day of week into Weekday or Weekend."""
    if str(day).strip() in ["Saturday", "Sunday"]:
        return "Weekend"
    return "Weekday"


def assign_collision_scale(vehicles: int) -> str:
    """Classifies collision severity by vehicles involved."""
    if vehicles <= 1:
        return "Single Vehicle"
    elif vehicles == 2:
        return "Two Vehicles"
    else:
        return "Multi-Vehicle Pile-up"


def assign_casualty_scale(casualties: int) -> str:
    """Classifies incident scale by total casualties."""
    if casualties <= 1:
        return "Low Casualty (1)"
    elif casualties == 2:
        return "Medium Casualty (2)"
    else:
        return "High Casualty (3+)"


def transform_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies all feature transformations and creates derived columns.
    """
    df = df.copy()

    # Temporal feature extraction from 'Time' (HH:MM:SS)
    if "Time" in df.columns:
        # Convert Time column to string and parse Hour & Minute
        time_parsed = pd.to_datetime(df["Time"].astype(str), format="%H:%M:%S", errors="coerce").dt
        df["Hour"] = time_parsed.hour.fillna(12).astype(int)
        df["Minute"] = time_parsed.minute.fillna(0).astype(int)

        # Derived temporal categorizations
        df["Time_of_day"] = df["Hour"].apply(assign_time_of_day)
        df["Rush_hour"] = df["Hour"].apply(assign_rush_hour)

    # Calendar categorization
    if "Day_of_week" in df.columns:
        df["Day_type"] = df["Day_of_week"].apply(assign_day_type)

    # Operational scales
    if "Number_of_vehicles_involved" in df.columns:
        df["Collision_scale"] = df["Number_of_vehicles_involved"].apply(assign_collision_scale)

    if "Number_of_casualties" in df.columns:
        df["Casualty_scale"] = df["Number_of_casualties"].apply(assign_casualty_scale)

    return df


def run_full_pipeline(
    raw_path: str = "data/raw/road_accidents.csv",
    processed_path: str = "data/processed/road_accidents_cleaned.csv"
) -> pd.DataFrame:
    """
    Executes full pipeline:
    Raw Data -> Cleaned Data -> Transformed Features -> Processed CSV
    """
    df_clean = clean_dataset(raw_path)
    df_transformed = transform_data(df_clean)
    df_transformed.to_csv(processed_path, index=False)
    return df_transformed


if __name__ == "__main__":
    df_final = run_full_pipeline()
    print("Data transformation pipeline complete.")
    print("Processed shape:", df_final.shape)
    print("New columns:", [c for c in df_final.columns if c not in clean_dataset().columns])
