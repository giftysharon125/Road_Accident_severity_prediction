"""
Data Analysis & Wrangling Module
Performs data wrangling, descriptive statistics, grouped analysis,
and addresses core exploratory research questions.
"""

import sys
import os
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def load_processed_data(file_path: str = "data/processed/road_accidents_cleaned.csv") -> pd.DataFrame:
    """Loads the preprocessed and feature-engineered road accidents dataset."""
    return pd.read_csv(file_path)


# ==========================================
# 1. DESCRIPTIVE & GROUPED STATISTICS
# ==========================================

def get_descriptive_statistics(df: pd.DataFrame) -> dict:
    """
    Computes standard statistical measures:
    Mean, Median, Mode, Std, Min, Max, and Quartiles for numerical columns.
    """
    num_cols = ["Number_of_vehicles_involved", "Number_of_casualties", "Hour"]
    stats_dict = {}

    for col in num_cols:
        if col in df.columns:
            stats_dict[col] = {
                "Mean": float(df[col].mean()),
                "Median": float(df[col].median()),
                "Mode": float(df[col].mode()[0]),
                "Std_Dev": float(df[col].std()),
                "Min": float(df[col].min()),
                "Q1 (25%)": float(df[col].quantile(0.25)),
                "Q2 (50%)": float(df[col].quantile(0.50)),
                "Q3 (75%)": float(df[col].quantile(0.75)),
                "Max": float(df[col].max()),
            }

    stats_df = pd.DataFrame(stats_dict).T
    return {
        "summary_table": stats_df,
        "raw_describe": df.describe(),
        "categorical_describe": df.describe(include=["object"]),
    }


def get_grouped_statistics(df: pd.DataFrame) -> dict:
    """
    Computes multi-level aggregations and grouped insights:
    - Average and total casualties by severity
    - Average vehicles involved by severity
    - Accident distribution by weather and road surface
    """
    severity_grouped = df.groupby("Accident_severity").agg(
        Total_Accidents=("Accident_severity", "count"),
        Avg_Casualties=("Number_of_casualties", "mean"),
        Max_Casualties=("Number_of_casualties", "max"),
        Total_Casualties=("Number_of_casualties", "sum"),
        Avg_Vehicles=("Number_of_vehicles_involved", "mean"),
    ).reset_index()

    weather_grouped = df.groupby("Weather_conditions").agg(
        Accident_Count=("Weather_conditions", "count"),
        Total_Casualties=("Number_of_casualties", "sum"),
        Avg_Casualties=("Number_of_casualties", "mean")
    ).sort_values(by="Accident_Count", ascending=False).reset_index()

    road_grouped = df.groupby("Road_surface_conditions").agg(
        Accident_Count=("Road_surface_conditions", "count"),
        Avg_Casualties=("Number_of_casualties", "mean")
    ).sort_values(by="Accident_Count", ascending=False).reset_index()

    return {
        "severity_summary": severity_grouped,
        "weather_summary": weather_grouped,
        "road_summary": road_grouped
    }


# ==========================================
# 2. DATA WRANGLING EXAMPLES
# ==========================================

def perform_wrangling_operations(df: pd.DataFrame) -> dict:
    """
    Demonstrates key Pandas data wrangling operations:
    - Filtering
    - Sorting
    - Multi-aggregation GroupBy
    - Pivot Tables
    - Cross-tabulation
    """
    # 1. Filtering: High-risk accidents (Casualties >= 3 or Fatal Injury)
    high_risk_filtered = df[(df["Number_of_casualties"] >= 3) | (df["Accident_severity"] == "Fatal Injury")]

    # 2. Sorting: Top 10 worst collisions by casualties and vehicles involved
    sorted_severe = df.sort_values(
        by=["Number_of_casualties", "Number_of_vehicles_involved"],
        ascending=[False, False]
    ).head(10)[["Time", "Day_of_week", "Weather_conditions", "Number_of_casualties", "Number_of_vehicles_involved", "Accident_severity"]]

    # 3. Pivot Table: Day of Week vs Time of Day accident frequency
    pivot_day_time = pd.pivot_table(
        df,
        values="Number_of_casualties",
        index="Day_of_week",
        columns="Time_of_day",
        aggfunc="count",
        fill_value=0
    )

    # 4. Cross-tabulation: Weather Condition vs Accident Severity
    crosstab_weather_severity = pd.crosstab(
        df["Weather_conditions"],
        df["Accident_severity"],
        margins=True,
        margins_name="Total"
    )

    # 5. Multi-aggregation: Casualties by Light Conditions
    light_multi_agg = df.groupby("Light_conditions")["Number_of_casualties"].agg(
        ["count", "mean", "median", "std", "min", "max"]
    ).reset_index()

    return {
        "high_risk_count": len(high_risk_filtered),
        "sorted_severe_sample": sorted_severe,
        "pivot_day_time": pivot_day_time,
        "crosstab_weather_severity": crosstab_weather_severity,
        "light_multi_agg": light_multi_agg,
    }


# ==========================================
# 3. RESEARCH EDA ANSWERS
# ==========================================

def answer_eda_questions(df: pd.DataFrame) -> dict:
    """
    Systematically computes dynamic empirical answers to all 11 core questions.
    """
    # Q1: Total accidents
    total_accidents = len(df)

    # Q2: Most frequent severity
    severity_counts = df["Accident_severity"].value_counts()
    most_freq_severity = severity_counts.index[0]
    most_freq_severity_count = int(severity_counts.iloc[0])

    # Q3: Weather condition with highest accidents
    weather_counts = df["Weather_conditions"].value_counts()
    top_weather = weather_counts.index[0]
    top_weather_count = int(weather_counts.iloc[0])

    # Q4: Road surface with highest accidents
    road_counts = df["Road_surface_conditions"].value_counts()
    top_road = road_counts.index[0]
    top_road_count = int(road_counts.iloc[0])

    # Q5: Day of week with highest accidents
    day_counts = df["Day_of_week"].value_counts()
    top_day = day_counts.index[0]
    top_day_count = int(day_counts.iloc[0])

    # Q6: Peak hours
    hour_counts = df["Hour"].value_counts()
    top_hour = int(hour_counts.index[0])
    top_hour_count = int(hour_counts.iloc[0])

    # Q7: Time of day with highest frequency
    tod_counts = df["Time_of_day"].value_counts()
    top_tod = tod_counts.index[0]
    top_tod_count = int(tod_counts.iloc[0])

    # Q8: Speed / collision scale vs severity
    collision_severity = pd.crosstab(df["Collision_scale"], df["Accident_severity"], normalize="index") * 100

    # Q9: Correlation / relation between vehicles and casualties
    vehicle_casualty_corr = float(df["Number_of_vehicles_involved"].corr(df["Number_of_casualties"]))

    # Q10: Numerical correlations
    corr_matrix = df[["Number_of_vehicles_involved", "Number_of_casualties", "Hour"]].corr()

    # Q11: Differences across severity categories
    sev_diff = df.groupby("Accident_severity")[["Number_of_casualties", "Number_of_vehicles_involved"]].mean()

    return {
        "q1_total_accidents": total_accidents,
        "q2_most_freq_severity": (most_freq_severity, most_freq_severity_count),
        "q3_top_weather": (top_weather, top_weather_count),
        "q4_top_road": (top_road, top_road_count),
        "q5_top_day": (top_day, top_day_count),
        "q6_top_hour": (top_hour, top_hour_count),
        "q7_top_tod": (top_tod, top_tod_count),
        "q8_collision_severity_pct": collision_severity,
        "q9_vehicle_casualty_corr": vehicle_casualty_corr,
        "q10_numerical_correlation": corr_matrix,
        "q11_severity_differences": sev_diff,
    }


if __name__ == "__main__":
    df = load_processed_data()
    eda_res = answer_eda_questions(df)
    print("=== EDA RESEARCH QUESTIONS SUMMARY ===")
    print(f"Total Accidents: {eda_res['q1_total_accidents']}")
    print(f"Most Frequent Severity: {eda_res['q2_most_freq_severity'][0]} ({eda_res['q2_most_freq_severity'][1]})")
    print(f"Top Weather Condition: {eda_res['q3_top_weather'][0]} ({eda_res['q3_top_weather'][1]})")
    print(f"Top Road Surface: {eda_res['q4_top_road'][0]} ({eda_res['q4_top_road'][1]})")
    print(f"Peak Day of Week: {eda_res['q5_top_day'][0]} ({eda_res['q5_top_day'][1]})")
    print(f"Peak Hour: {eda_res['q6_top_hour'][0]}:00 ({eda_res['q6_top_hour'][1]} accidents)")
    print(f"Peak Time of Day: {eda_res['q7_top_tod'][0]} ({eda_res['q7_top_tod'][1]} accidents)")
    print(f"Vehicle vs Casualty Correlation: {eda_res['q9_vehicle_casualty_corr']:.4f}")

