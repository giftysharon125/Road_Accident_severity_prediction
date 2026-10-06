"""
Machine Learning Models Module
Trains, evaluates, compares, and saves exactly two models:
1. Logistic Regression
2. Random Forest Classifier
Also provides live dynamic inference for single-incident user predictions.
"""

import sys
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_transformation import run_full_pipeline
from src.preprocessing import prepare_train_test_data, CATEGORICAL_FEATURES, NUMERICAL_FEATURES, TARGET_COLUMN


def train_and_evaluate_models(
    data_path: str = "data/processed/road_accidents_cleaned.csv",
    models_dir: str = "models"
) -> dict:
    """
    Executes full ML training pipeline:
    1. Loads dataset (or runs pipeline if needed)
    2. Splits data and fits preprocessor
    3. Trains Logistic Regression & Random Forest
    4. Evaluates dynamic metrics (Accuracy, Precision, Recall, F1, Confusion Matrix)
    5. Saves models and metrics to models_dir
    """
    os.makedirs(models_dir, exist_ok=True)

    if not os.path.exists(data_path):
        df = run_full_pipeline()
    else:
        df = pd.read_csv(data_path)

    (
        X_train_proc,
        X_test_proc,
        y_train,
        y_test,
        preprocessor,
        feature_names,
        X_train,
        X_test
    ) = prepare_train_test_data(df)

    class_names = sorted(list(y_train.unique()))

    # ==========================================
    # MODEL 1 — LOGISTIC REGRESSION
    # ==========================================
    print("Training Model 1: Logistic Regression...")
    lr_model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight="balanced",
        solver="lbfgs"
    )
    lr_model.fit(X_train_proc, y_train)
    y_pred_lr = lr_model.predict(X_test_proc)

    # Dynamic metrics for Logistic Regression
    lr_acc = float(accuracy_score(y_test, y_pred_lr))
    lr_prec = float(precision_score(y_test, y_pred_lr, average="weighted", zero_division=0))
    lr_rec = float(recall_score(y_test, y_pred_lr, average="weighted", zero_division=0))
    lr_f1 = float(f1_score(y_test, y_pred_lr, average="weighted", zero_division=0))
    lr_cm = confusion_matrix(y_test, y_pred_lr, labels=class_names)
    lr_report_dict = classification_report(y_test, y_pred_lr, output_dict=True, zero_division=0)
    lr_report_text = classification_report(y_test, y_pred_lr, zero_division=0)

    # ==========================================
    # MODEL 2 — RANDOM FOREST CLASSIFIER
    # ==========================================
    print("Training Model 2: Random Forest Classifier...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=16,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )
    rf_model.fit(X_train_proc, y_train)
    y_pred_rf = rf_model.predict(X_test_proc)

    # Dynamic metrics for Random Forest
    rf_acc = float(accuracy_score(y_test, y_pred_rf))
    rf_prec = float(precision_score(y_test, y_pred_rf, average="weighted", zero_division=0))
    rf_rec = float(recall_score(y_test, y_pred_rf, average="weighted", zero_division=0))
    rf_f1 = float(f1_score(y_test, y_pred_rf, average="weighted", zero_division=0))
    rf_cm = confusion_matrix(y_test, y_pred_rf, labels=class_names)
    rf_report_dict = classification_report(y_test, y_pred_rf, output_dict=True, zero_division=0)
    rf_report_text = classification_report(y_test, y_pred_rf, zero_division=0)

    # Dynamic feature importances
    rf_importances = rf_model.feature_importances_

    # ==========================================
    # DYNAMIC MODEL COMPARISON TABLE
    # ==========================================
    comparison_df = pd.DataFrame([
        {
            "Model": "Logistic Regression",
            "Accuracy": round(lr_acc, 4),
            "Precision": round(lr_prec, 4),
            "Recall": round(lr_rec, 4),
            "F1 Score": round(lr_f1, 4)
        },
        {
            "Model": "Random Forest",
            "Accuracy": round(rf_acc, 4),
            "Precision": round(rf_prec, 4),
            "Recall": round(rf_rec, 4),
            "F1 Score": round(rf_f1, 4)
        }
    ])

    results = {
        "class_names": class_names,
        "feature_names": feature_names,
        "comparison_df": comparison_df,
        "logistic_regression": {
            "accuracy": lr_acc,
            "precision": lr_prec,
            "recall": lr_rec,
            "f1_score": lr_f1,
            "confusion_matrix": lr_cm,
            "report_dict": lr_report_dict,
            "report_text": lr_report_text,
        },
        "random_forest": {
            "accuracy": rf_acc,
            "precision": rf_prec,
            "recall": rf_rec,
            "f1_score": rf_f1,
            "confusion_matrix": rf_cm,
            "report_dict": rf_report_dict,
            "report_text": rf_report_text,
            "feature_importances": rf_importances,
        }
    }

    # Save artifacts using joblib
    print("Saving models and evaluation artifacts...")
    joblib.dump(lr_model, os.path.join(models_dir, "logistic_regression_model.joblib"))
    joblib.dump(rf_model, os.path.join(models_dir, "random_forest_model.joblib"))
    joblib.dump(preprocessor, os.path.join(models_dir, "preprocessor.joblib"))
    joblib.dump(results, os.path.join(models_dir, "evaluation_metrics.joblib"))

    print("All models and metrics successfully trained and persisted.")
    return results


def predict_severity(
    model_name: str,
    input_data: dict,
    models_dir: str = "models"
) -> dict:
    """
    Executes live model inference for a single user-specified incident.
    Loads the saved preprocessor and specified model, runs prediction,
    and returns predicted class name and probability distribution.
    """
    preprocessor_path = os.path.join(models_dir, "preprocessor.joblib")
    if model_name.lower().startswith("log"):
        model_path = os.path.join(models_dir, "logistic_regression_model.joblib")
        name_display = "Logistic Regression"
    else:
        model_path = os.path.join(models_dir, "random_forest_model.joblib")
        name_display = "Random Forest Classifier"

    if not os.path.exists(model_path) or not os.path.exists(preprocessor_path):
        train_and_evaluate_models(models_dir=models_dir)

    preprocessor = joblib.load(preprocessor_path)
    model = joblib.load(model_path)

    # Convert single dictionary into single-row DataFrame
    input_df = pd.DataFrame([input_data])

    # Transform input using training pipeline
    input_proc = preprocessor.transform(input_df)

    # Perform inference
    predicted_class = model.predict(input_proc)[0]

    probabilities = {}
    if hasattr(model, "predict_proba"):
        prob_values = model.predict_proba(input_proc)[0]
        for cls_name, prob in zip(model.classes_, prob_values):
            probabilities[cls_name] = round(float(prob), 4)

    return {
        "model_used": name_display,
        "prediction": str(predicted_class),
        "probabilities": probabilities
    }


if __name__ == "__main__":
    res = train_and_evaluate_models()
    print("\n=== MODEL COMPARISON TABLE ===")
    print(res["comparison_df"])
