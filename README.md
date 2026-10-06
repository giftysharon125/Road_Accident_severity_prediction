# Road Accident Data Analysis, Visualization and Accident Severity Prediction Using Machine Learning

An end-to-end academic data science and machine learning project exploring traffic accident trends and predicting accident severity (**Slight Injury**, **Serious Injury**, **Fatal Injury**) using empirical highway police accident records.

---

## 📌 Project Overview
Traffic accidents are a leading cause of fatalities and economic losses worldwide. This project provides an academic-grade data science workflow:
$$\text{Data Ingestion} \rightarrow \text{Data Cleaning} \rightarrow \text{Transformation} \rightarrow \text{Wrangling} \rightarrow \text{EDA} \rightarrow \text{Statistical Analysis} \rightarrow \text{Data Visualization} \rightarrow \text{Machine Learning} \rightarrow \text{Model Evaluation} \rightarrow \text{Interactive Dashboard}$$

Everything displayed across the system—from data statistics and charts to classification reports and prediction forms—is **dynamically calculated** from the actual dataset and trained models (no hardcoded values).

---

## 🚀 Key Features
- **Authentic Dataset Ingestion:** 12,316 authentic traffic collision records with 32 raw attributes.
- **Robust Cleaning Pipeline:** Median imputation for numerical features, mode/token imputation for categorical features, whitespace stripping, corrupted string normalization, and IQR outlier boundaries.
- **Derived Feature Engineering:** Extracts crash hour, minute, time of day (Morning/Afternoon/Evening/Night), rush-hour commute indicators, and weekday/weekend classification.
- **Systematic EDA & Statistics:** Formally answers 11 core transportation research questions and provides multi-level grouped aggregations.
- **Rich Visualization Suite:** Pie/donut charts, 24-hour cycle line plots, weather/surface bar charts, casualty histograms, jittered scatter plots, severity boxplots, and numerical correlation heatmaps.
- **Exactly Two ML Algorithms:**
  1. **Logistic Regression** (Multinomial with balanced class weights)
  2. **Random Forest Classifier** (100 decision trees with balanced subsampling)
- **Live User Prediction:** Preprocessing pipeline (`ColumnTransformer`, `StandardScaler`, `OneHotEncoder`) serialized with `joblib` for live severity inference.
- **Interactive Streamlit Web Dashboard:** Features **two clearly visible navigation buttons** for instant access to each ML algorithm's dedicated analysis and interactive prediction page.
- **20-Section Jupyter / Google Colab Notebook:** Complete notebook with Markdown documentation and automated cloud dataset download fallback, fully executable on Google Colab.
- **Interactive Google Colab Buttons:** Integrated buttons in both the Streamlit Home page and the expanded Left Sidebar to launch Google Colab or download [`notebooks/road_accident_analysis.ipynb`](file:///c:/Users/Gifty/OneDrive/Desktop/DAV_Project/notebooks/road_accident_analysis.ipynb) with one click.


---

## 📊 Dataset Information
- **Source:** Benchmark Road Traffic Accident (RTA) Severity Dataset
- **Records:** 12,316 rows
- **Columns:** 32 original + 7 derived = 39 total columns
- **Target Variable:** `Accident_severity`
  - `Slight Injury` (84.56%)
  - `Serious Injury` (14.15%)
  - `Fatal Injury` (1.28%)

---

## 🛠️ Technology Stack
- **Language:** Python 3.10+
- **Data Manipulation:** Pandas, NumPy
- **Visualizations:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn
- **Model Persistence:** Joblib
- **Web Interface:** Streamlit

---

## 📁 Project Directory Structure
```
road_accident_analysis/
│
├── data/
│   ├── raw/
│   │   └── road_accidents.csv               # Raw traffic accident dataset
│   └── processed/
│       └── road_accidents_cleaned.csv       # Cleaned & transformed dataset
│
├── notebooks/
│   └── road_accident_analysis.ipynb         # 20-section sequential notebook
│
├── src/
│   ├── __init__.py                          # Package initializer
│   ├── data_cleaning.py                     # Missing values, duplicates, outliers, string clean
│   ├── data_transformation.py               # Feature derivation & pipeline orchestration
│   ├── data_analysis.py                     # Wrangling, statistics, and 11 EDA questions
│   ├── visualization.py                     # Publication-ready Matplotlib & Seaborn plots
│   ├── preprocessing.py                     # ColumnTransformer, train/test split
│   └── models.py                            # Logistic Regression & Random Forest training & inference
│
├── models/
│   ├── logistic_regression_model.joblib     # Serialized Logistic Regression
│   ├── random_forest_model.joblib           # Serialized Random Forest
│   ├── preprocessor.joblib                  # Serialized ColumnTransformer pipeline
│   └── evaluation_metrics.joblib            # Dynamically evaluated test metrics & results
│
├── report/
│   └── project_report.md                    # 23-section academic project report
│
├── app.py                                   # Interactive Streamlit dashboard
├── requirements.txt                         # Minimal required dependencies
└── README.md                                # Project documentation
```

---

## ⚙️ Installation & Setup

1. **Clone or navigate to the project directory:**
   ```bash
   cd c:\Users\Gifty\OneDrive\Desktop\DAV_Project
   ```

2. **Install required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 💻 How to Run the Project

### 1. Execute the Data Pipeline & Train ML Models
To run data cleaning, feature transformation, model training, and artifact persistence:
```bash
python src/data_transformation.py
python src/data_analysis.py
python src/models.py
```

### 2. Launch the Streamlit Interactive Dashboard
Run the Streamlit application:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### 3. Open in Google Colab (Directly in Your Account)
To inspect and run all data cleaning, preprocessing, and machine learning methods in Google Colab:
1. **Direct Upload to Your Account:** Click the **"🚀 Open Google Colab (Your Account)"** button in the Streamlit app (or open `https://colab.research.google.com/#upload=true`).
2. **Select Notebook:** Upload [`notebooks/road_accident_analysis.ipynb`](file:///c:/Users/Gifty/OneDrive/Desktop/DAV_Project/notebooks/road_accident_analysis.ipynb).
3. **Save to Your Google Drive:** In Google Colab, select **File → Save a copy in Drive**. This permanently saves the notebook under your account in `Colab Notebooks`.
4. **Pre-computed Results:** All 20 sections are already executed with complete outputs (data cleaning tables, IQR outlier boundaries, 8 publication charts, classification reports, confusion matrix heatmaps, and model comparisons).
5. **(Optional) Direct Link:** Copy your Google Drive Colab URL and paste it into the dashboard expander to open your personal notebook in 1 click!

---

## 🤖 Machine Learning Algorithms & Performance

Both algorithms were trained on an 80/20 stratified split (9,852 train / 2,464 test) using balanced class weights:

| Model | Accuracy | Weighted Precision | Weighted Recall | Weighted F1 Score |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 48.90% | 76.87% | 48.90% | 0.5776 |
| **Random Forest Classifier** | **81.53%** | **80.00%** | **81.53%** | **0.8035** |

- **Random Forest** captured non-linear interactions across driver demographics, time of day, and environmental hazards, outperforming linear boundaries.
- **Top Predictive Features:** Crash Hour, Number of Casualties, Vehicles Involved, Friday Commute, Driver Age (18–30), and Novice Driving Experience (1–2yr).

---

## 🖥️ Dashboard Functionality
- **🏠 Home / Overview:** Live KPI cards (Total Accidents, Fatalities, Serious Injuries, Casualties), dataset preview, and two prominent ML navigation buttons.
- **🧹 Data Cleaning Summary:** Visual audit of missing value distributions, IQR outlier bounds, and text standardization.
- **📊 Exploratory Analysis & Statistics:** Answers to all 11 core traffic safety questions and grouped descriptive tables.
- **📈 Data Visualizations:** 4 categorized galleries displaying distributions, trends, and correlations.
- **🤖 Logistic Regression Page:** Live evaluation metrics, confusion matrix heatmap, classification report, and live severity prediction form.
- **🌲 Random Forest Page:** Live evaluation metrics, confusion matrix, dynamic Gini feature importance chart, and live prediction form.
- **⚖️ Model Comparison:** Side-by-side performance table and grouped comparative bar chart.

---

## 📜 Academic Submission Details
- **Project Title:** Road Accident Data Analysis, Visualization and Accident Severity Prediction Using Machine Learning
- **Course:** Data Analytics and Visualization (DAV)
