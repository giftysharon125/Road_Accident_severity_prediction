"""
Preprocessing Module
Defines feature schema, ColumnTransformer pipeline,
One-Hot Encoding, StandardScaler, and Train/Test dataset splitting.
"""

import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Core feature schema used for severity prediction
CATEGORICAL_FEATURES = [
    "Day_of_week",
    "Age_band_of_driver",
    "Sex_of_driver",
    "Driving_experience",
    "Type_of_vehicle",
    "Area_accident_occured",
    "Types_of_Junction",
    "Road_surface_conditions",
    "Light_conditions",
    "Weather_conditions",
    "Type_of_collision",
    "Vehicle_movement",
    "Cause_of_accident",
    "Time_of_day"
]

NUMERICAL_FEATURES = [
    "Number_of_vehicles_involved",
    "Number_of_casualties",
    "Hour"
]

TARGET_COLUMN = "Accident_severity"


def build_preprocessor() -> ColumnTransformer:
    """
    Constructs a ColumnTransformer that standardizes numerical features
    and One-Hot encodes categorical features with robust unknown handling.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                NUMERICAL_FEATURES
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES
            )
        ],
        remainder="drop"
    )
    return preprocessor


def prepare_train_test_data(
    df: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 42
) -> tuple:
    """
    Extracts features (X) and target (y), splits into train and test sets,
    and fits the ColumnTransformer on X_train only to prevent data leakage.
    Returns:
        X_train_proc, X_test_proc, y_train, y_test, preprocessor, feature_names
    """
    # Filter features that exist in dataframe
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    num_cols = [c for c in NUMERICAL_FEATURES if c in df.columns]

    X = df[cat_cols + num_cols].copy()
    y = df[TARGET_COLUMN].copy()

    # Stratified train/test split to preserve severity class balance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # Initialize and fit preprocessor on training partition ONLY
    preprocessor = build_preprocessor()
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    # Extract transformed feature names
    cat_encoder = preprocessor.named_transformers_["cat"]
    cat_feature_names = list(cat_encoder.get_feature_names_out(cat_cols))
    all_feature_names = num_cols + cat_feature_names

    return (
        X_train_proc,
        X_test_proc,
        y_train,
        y_test,
        preprocessor,
        all_feature_names,
        X_train,
        X_test
    )
