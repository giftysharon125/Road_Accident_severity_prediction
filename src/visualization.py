"""
Visualization Module
Generates academic-quality, publication-ready charts using Matplotlib and Seaborn:
- Bar charts, Line charts, Pie/Donut charts, Histograms,
- Scatter plots, Box plots, Correlation Heatmaps, Count plots,
- Machine Learning Confusion Matrices, Feature Importance, and Model Comparison charts.
"""

import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Use clean academic white style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10.5,
    "figure.facecolor": "#ffffff",
    "axes.facecolor": "#ffffff",
    "axes.edgecolor": "#cbd5e1",
    "grid.color": "#f1f5f9",
    "text.color": "#0f172a",
    "axes.labelcolor": "#1e293b",
    "xtick.color": "#334155",
    "ytick.color": "#334155",
    "axes.titlesize": 12.5,
    "axes.titleweight": "bold",
    "axes.titlecolor": "#1e3a8a",
    "axes.labelsize": 11,
    "axes.labelweight": "semibold",
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "figure.titlesize": 13.5,
    "figure.titleweight": "bold"
})



def plot_severity_distribution(df: pd.DataFrame) -> plt.Figure:
    """Pie/Donut chart illustrating proportion of Accident Severity categories."""
    fig, ax = plt.subplots(figsize=(7, 6))
    counts = df["Accident_severity"].value_counts()
    colors = ["#3498db", "#e67e22", "#e74c3c"]  # Blue for slight, Orange for serious, Red for fatal

    wedges, texts, autotexts = ax.pie(
        counts.values,
        labels=counts.index,
        autopct="%1.1f%%",
        startangle=140,
        colors=colors[:len(counts)],
        pctdistance=0.75,
        explode=[0.02, 0.05, 0.12][:len(counts)],
        wedgeprops=dict(width=0.45, edgecolor="white", linewidth=2)
    )
    for at in autotexts:
        at.set_color("black")
        at.set_fontsize(10)
        at.set_weight("bold")

    ax.set_title("Distribution of Road Accident Severity", pad=15)
    plt.tight_layout()
    return fig


def plot_accidents_by_weather(df: pd.DataFrame) -> plt.Figure:
    """Bar chart displaying accidents categorized by weather conditions."""
    fig, ax = plt.subplots(figsize=(9, 5))
    weather_counts = df["Weather_conditions"].value_counts().head(8)

    sns.barplot(
        x=weather_counts.values,
        y=weather_counts.index,
        hue=weather_counts.index,
        palette="Blues_r",
        legend=False,
        ax=ax
    )

    for i, v in enumerate(weather_counts.values):
        ax.text(v + 50, i, f"{v:,} ({v/len(df):.1%})", va="center", fontsize=9, fontweight="semibold")

    ax.set_title("Accident Frequency by Weather Conditions", pad=12)
    ax.set_xlabel("Number of Incidents")
    ax.set_ylabel("Weather Condition")
    ax.set_xlim(0, max(weather_counts.values) * 1.15)
    plt.tight_layout()
    return fig


def plot_accidents_by_road_surface(df: pd.DataFrame) -> plt.Figure:
    """Bar chart showing accident distribution across road surface conditions."""
    fig, ax = plt.subplots(figsize=(8, 5))
    road_counts = df["Road_surface_conditions"].value_counts()

    sns.barplot(
        x=road_counts.index,
        y=road_counts.values,
        hue=road_counts.index,
        palette="mako",
        legend=False,
        ax=ax
    )

    for i, v in enumerate(road_counts.values):
        ax.text(i, v + 80, f"{v:,}", ha="center", fontsize=10, fontweight="bold")

    ax.set_title("Accident Frequency by Road Surface Condition", pad=12)
    ax.set_xlabel("Road Surface Condition")
    ax.set_ylabel("Accident Count")
    ax.tick_params(axis="x", rotation=15)
    ax.set_ylim(0, max(road_counts.values) * 1.12)
    plt.tight_layout()
    return fig


def plot_accidents_by_hour(df: pd.DataFrame) -> plt.Figure:
    """Line chart showing accident distribution over the 24 hours of the day."""
    fig, ax = plt.subplots(figsize=(10, 5))
    hourly_counts = df["Hour"].value_counts().sort_index()

    ax.plot(hourly_counts.index, hourly_counts.values, marker="o", color="#2c3e50", linewidth=2.5, markersize=6)
    ax.fill_between(hourly_counts.index, hourly_counts.values, color="#3498db", alpha=0.25)

    # Highlight peak hour
    peak_hr = hourly_counts.idxmax()
    peak_val = hourly_counts.max()
    ax.annotate(
        f"Peak: {peak_hr}:00\n({peak_val} crashes)",
        xy=(peak_hr, peak_val),
        xytext=(peak_hr - 2, peak_val + 70),
        arrowprops=dict(facecolor="#e74c3c", shrink=0.08, width=1.5, headwidth=7),
        fontsize=9,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", fc="#fadbd8", ec="#e74c3c", lw=1)
    )

    ax.set_title("Hourly Distribution of Road Accidents (24-Hour Cycle)", pad=12)
    ax.set_xlabel("Hour of Day (00:00 - 23:00)")
    ax.set_ylabel("Number of Accidents")
    ax.set_xticks(range(0, 24))
    ax.set_ylim(0, peak_val * 1.2)
    plt.tight_layout()
    return fig


def plot_casualty_distribution(df: pd.DataFrame) -> plt.Figure:
    """Histogram with KDE displaying distribution of casualties per accident."""
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(
        df["Number_of_casualties"],
        bins=range(1, df["Number_of_casualties"].max() + 2),
        kde=True,
        color="#27ae60",
        discrete=True,
        ax=ax
    )

    ax.set_title("Distribution of Casualties per Incident", pad=12)
    ax.set_xlabel("Number of Casualties")
    ax.set_ylabel("Frequency of Incidents")
    ax.set_xticks(range(1, df["Number_of_casualties"].max() + 1))
    plt.tight_layout()
    return fig


def plot_vehicles_vs_casualties_scatter(df: pd.DataFrame) -> plt.Figure:
    """Scatter plot with jitter illustrating correlation between vehicles and casualties."""
    fig, ax = plt.subplots(figsize=(8, 5))

    # Add slight random jitter for visual clarity on discrete counts
    rng = np.random.default_rng(42)
    x_jitter = df["Number_of_vehicles_involved"] + rng.uniform(-0.15, 0.15, size=len(df))
    y_jitter = df["Number_of_casualties"] + rng.uniform(-0.15, 0.15, size=len(df))

    scatter = ax.scatter(
        x_jitter,
        y_jitter,
        alpha=0.35,
        c="#8e44ad",
        edgecolors="none",
        s=30,
        label="Accident records"
    )

    # Linear trendline
    sns.regplot(
        x="Number_of_vehicles_involved",
        y="Number_of_casualties",
        data=df,
        scatter=False,
        ax=ax,
        color="#e74c3c",
        line_kws={"linewidth": 2, "label": "Linear Fit Trend"}
    )

    ax.set_title("Correlation: Number of Vehicles Involved vs. Casualties", pad=12)
    ax.set_xlabel("Vehicles Involved")
    ax.set_ylabel("Casualties Incurred")
    ax.legend(loc="upper left")
    plt.tight_layout()
    return fig


def plot_box_casualties_by_severity(df: pd.DataFrame) -> plt.Figure:
    """Box plot demonstrating casualty distributions stratified across severity levels."""
    fig, ax = plt.subplots(figsize=(8, 5))
    order = ["Slight Injury", "Serious Injury", "Fatal Injury"]
    palette = {"Slight Injury": "#2ecc71", "Serious Injury": "#f39c12", "Fatal Injury": "#e74c3c"}

    sns.boxplot(
        x="Accident_severity",
        y="Number_of_casualties",
        data=df,
        order=[o for o in order if o in df["Accident_severity"].unique()],
        palette=palette,
        hue="Accident_severity",
        legend=False,
        showmeans=True,
        meanprops={"marker": "D", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": 6},
        ax=ax
    )

    ax.set_title("Casualty Distribution Across Accident Severity Classes", pad=12)
    ax.set_xlabel("Accident Severity Category")
    ax.set_ylabel("Number of Casualties")
    plt.tight_layout()
    return fig


def plot_correlation_heatmap(df: pd.DataFrame) -> plt.Figure:
    """Correlation matrix heatmap for all numerical features."""
    fig, ax = plt.subplots(figsize=(7, 5))
    num_cols = ["Number_of_vehicles_involved", "Number_of_casualties", "Hour"]
    corr = df[num_cols].corr()

    # Friendly labels
    label_map = {
        "Number_of_vehicles_involved": "Vehicles Involved",
        "Number_of_casualties": "Total Casualties",
        "Hour": "Crash Hour"
    }
    corr_labeled = corr.rename(index=label_map, columns=label_map)

    sns.heatmap(
        corr_labeled,
        annot=True,
        fmt=".3f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=1,
        cbar_kws={"shrink": 0.8},
        ax=ax
    )

    ax.set_title("Correlation Matrix of Numerical Variables", pad=12)
    plt.tight_layout()
    return fig


def plot_day_of_week_countplot(df: pd.DataFrame) -> plt.Figure:
    """Count plot showing frequency of accidents across days of the week."""
    fig, ax = plt.subplots(figsize=(9, 5))
    days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    sns.countplot(
        x="Day_of_week",
        data=df,
        order=[d for d in days_order if d in df["Day_of_week"].unique()],
        hue="Day_of_week",
        palette="viridis",
        legend=False,
        ax=ax
    )

    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(
                f"{int(height):,}",
                (p.get_x() + p.get_width() / 2.0, height + 30),
                ha="center",
                fontsize=9,
                fontweight="bold"
            )

    ax.set_title("Accident Volume Across Days of the Week", pad=12)
    ax.set_xlabel("Day of Week")
    ax.set_ylabel("Total Recorded Accidents")
    ax.set_ylim(0, df["Day_of_week"].value_counts().max() * 1.15)
    plt.tight_layout()
    return fig


def plot_light_conditions_distribution(df: pd.DataFrame) -> plt.Figure:
    """Bar plot highlighting accident counts by lighting conditions."""
    fig, ax = plt.subplots(figsize=(9, 5))
    light_counts = df["Light_conditions"].value_counts()

    sns.barplot(
        x=light_counts.values,
        y=light_counts.index,
        hue=light_counts.index,
        palette="magma",
        legend=False,
        ax=ax
    )

    for i, v in enumerate(light_counts.values):
        ax.text(v + 50, i, f"{v:,} ({v/len(df):.1%})", va="center", fontsize=9, fontweight="semibold")

    ax.set_title("Accident Frequency by Lighting Conditions", pad=12)
    ax.set_xlabel("Number of Incidents")
    ax.set_ylabel("Light Condition")
    ax.set_xlim(0, max(light_counts.values) * 1.15)
    plt.tight_layout()
    return fig


# ==========================================
# MACHINE LEARNING EVALUATION VISUALIZATIONS
# ==========================================

def plot_confusion_matrix(cm: np.ndarray, class_names: list, model_name: str) -> plt.Figure:
    """
    Renders a clear confusion matrix heatmap displaying raw counts and percentages.
    """
    fig, ax = plt.subplots(figsize=(7, 6))

    cm_percent = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100

    annot = np.empty_like(cm).astype(str)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm[i, j]:,}\n({cm_percent[i, j]:.1f}%)"

    sns.heatmap(
        cm,
        annot=annot,
        fmt="",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        linewidths=1,
        linecolor="#bdc3c7",
        ax=ax
    )

    ax.set_title(f"Confusion Matrix — {model_name}", pad=14)
    ax.set_xlabel("Predicted Label", fontweight="bold")
    ax.set_ylabel("True Ground Truth Label", fontweight="bold")
    plt.tight_layout()
    return fig


def plot_feature_importance(importances: np.ndarray, feature_names: list, top_n: int = 15) -> plt.Figure:
    """
    Renders horizontal bar chart of the top N feature importances
    extracted directly from the trained Random Forest model.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    feat_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    feat_df = feat_df.sort_values(by="Importance", ascending=False).head(top_n)

    sns.barplot(
        x="Importance",
        y="Feature",
        data=feat_df,
        hue="Feature",
        palette="crest",
        legend=False,
        ax=ax
    )

    for i, v in enumerate(feat_df["Importance"]):
        ax.text(v + 0.001, i, f"{v:.4f}", va="center", fontsize=9, fontweight="semibold")

    ax.set_title(f"Top {top_n} Predictive Feature Importances (Random Forest)", pad=14)
    ax.set_xlabel("Gini Feature Importance Score")
    ax.set_ylabel("Feature Name")
    ax.set_xlim(0, feat_df["Importance"].max() * 1.15)
    plt.tight_layout()
    return fig


def plot_model_comparison(comparison_df: pd.DataFrame) -> plt.Figure:
    """
    Renders grouped bar chart comparing performance metrics
    between Logistic Regression and Random Forest.
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    # Reshape for seaborn barplot
    metrics = [c for c in ["Accuracy", "Precision", "Recall", "F1 Score"] if c in comparison_df.columns]
    melted = comparison_df.melt(id_vars="Model", value_vars=metrics, var_name="Metric", value_name="Score")

    sns.barplot(
        x="Metric",
        y="Score",
        hue="Model",
        data=melted,
        palette=["#3498db", "#2ecc71"],
        ax=ax
    )

    for p in ax.patches:
        height = p.get_height()
        if not np.isnan(height) and height > 0:
            ax.annotate(
                f"{height:.3f}",
                (p.get_x() + p.get_width() / 2.0, height + 0.015),
                ha="center",
                fontsize=9,
                fontweight="bold"
            )

    ax.set_title("Empirical Model Performance Metric Comparison", pad=14)
    ax.set_xlabel("Evaluation Metric")
    ax.set_ylabel("Metric Score (0.0 to 1.0)")
    ax.set_ylim(0, 1.15)
    ax.legend(title="Algorithm", loc="upper right")
    plt.tight_layout()
    return fig
