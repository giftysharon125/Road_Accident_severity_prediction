# Road Accident Data Analysis, Visualization and Accident Severity Prediction Using Machine Learning

**Academic Capstone Project Report**  
**Course:** Data Analytics and Visualization / Machine Learning  
**Target:** Bachelor of Technology / Science Submission  

---

## 1. Title
**Road Accident Data Analysis, Visualization and Accident Severity Prediction Using Machine Learning**

---

## 2. Abstract
Traffic road accidents represent a global public health and economic crisis, claiming millions of lives annually and imposing substantial financial burdens on healthcare and infrastructure systems. This project presents an authentic, end-to-end data analytics, visualization, and predictive machine learning pipeline using a benchmark highway traffic accident dataset of 12,316 authentic incident records across 32 features. A comprehensive data cleaning and feature engineering methodology addresses missing values via median and mode imputation, cleans corrupted string encodings, preserves critical crash outliers, and derives temporal indicators (Hour, Time of Day, Rush Hour, and Day Type). Exploratory and statistical analyses uncover crucial insights regarding evening rush-hour crash peaks, adverse road conditions, and collision scaling factors. Exactly two supervised classification algorithms—**Logistic Regression** and **Random Forest Classifier**—are developed, tuned with balanced multi-class weighting, and evaluated on a stratified test holdout. The empirical evaluation reveals that the ensemble Random Forest Classifier achieves superior performance with an accuracy of **81.53%** and an F1 score of **0.8035**, outperforming Logistic Regression (**48.90%** accuracy, **0.5776** F1 score). Finally, the entire pipeline is deployed into an interactive **Streamlit** dashboard featuring two prominent ML navigation buttons, dynamic KPI metrics, and live inference prediction forms.

---

## 3. Introduction
Road traffic injuries are one of the leading causes of death and severe disability worldwide. Modern transportation systems and traffic management authorities routinely log granular accident data, encompassing environmental attributes, vehicular features, road infrastructure conditions, and casualty demographics. However, raw accident logs are inherently noisy, high-dimensional, and affected by severe class imbalances.

By harnessing modern data analytics and machine learning methodologies, transportation planners and safety authorities can uncover patterns that govern crash occurrences and proactively predict the severity of potential collisions. This report documents the full implementation of a complete data science pipeline:
$$\text{Data Ingestion} \longrightarrow \text{Data Cleaning} \longrightarrow \text{Transformation} \longrightarrow \text{Wrangling} \longrightarrow \text{EDA} \longrightarrow \text{Statistical Analysis} \longrightarrow \text{Data Visualization} \longrightarrow \text{Machine Learning} \longrightarrow \text{Model Evaluation} \longrightarrow \text{Interactive Dashboard}$$

---

## 4. Problem Statement
Given an incident report with diverse categorical and numerical variables (such as driver age, driving experience, road type, lighting conditions, weather state, vehicle count, and casualties), determine the expected **Accident Severity**:
- **Slight Injury**
- **Serious Injury**
- **Fatal Injury**

The primary challenge lies in handling real-world data imperfections, severe class imbalance (where Slight Injuries dominate over 84% of observations), and multi-collinear categorical features without introducing data leakage.

---

## 5. Objectives
1. **Data Ingestion & Hygiene:** Ingest the raw benchmark traffic accident dataset, perform statistical quality checks, and clean missing values, duplicate entries, and inconsistent string encodings.
2. **Feature Engineering & Transformation:** Derive meaningful temporal variables (e.g., Crash Hour, Time of Day, Rush Hour) and calendar categorizations (Weekday vs. Weekend).
3. **Data Wrangling & Statistical Analysis:** Apply multi-level aggregations, pivot tables, cross-tabulations, and descriptive statistics across severity categories.
4. **Exploratory Data Analysis & Visualization:** Systematically answer core transportation research questions and generate publication-quality figures (bar charts, line plots, donut charts, scatter plots, box plots, and heatmaps).
5. **Machine Learning Severity Prediction:** Implement, train, and evaluate **exactly two** supervised algorithms:
   - Multinomial Logistic Regression
   - Random Forest Classifier
6. **Dynamic Model Comparison:** Contrast both models using dynamically computed evaluation metrics (Accuracy, Precision, Recall, F1 Score, Confusion Matrix, and Gini Feature Importances).
7. **Interactive Deployment:** Develop a responsive Streamlit dashboard featuring two dedicated ML navigation buttons and interactive inference forms for live prediction.

---

## 6. Dataset Description
The dataset used in this study is the authentic benchmark Road Traffic Accident (RTA) Severity Dataset, collected from highway traffic police logs:
- **Total Records:** 12,316 observations
- **Raw Features:** 32 columns
- **Engineered Features:** 7 derived columns (Total: 39 columns)
- **Target Variable:** `Accident_severity` (Categorical, 3 classes)
  - Slight Injury: 10,415 instances (84.56%)
  - Serious Injury: 1,743 instances (14.15%)
  - Fatal Injury: 158 instances (1.28%)

### Key Attribute Categories:
1. **Temporal & Spatial:** `Time`, `Day_of_week`, `Area_accident_occured`, `Lanes_or_Medians`, `Road_allignment`, `Types_of_Junction`.
2. **Driver Attributes:** `Age_band_of_driver`, `Sex_of_driver`, `Educational_level`, `Driving_experience`.
3. **Vehicle Characteristics:** `Type_of_vehicle`, `Owner_of_vehicle`, `Service_year_of_vehicle`, `Defect_of_vehicle`.
4. **Environmental Factors:** `Road_surface_conditions`, `Road_surface_type`, `Light_conditions`, `Weather_conditions`.
5. **Incident Metrics:** `Number_of_vehicles_involved`, `Number_of_casualties`, `Type_of_collision`, `Vehicle_movement`, `Cause_of_accident`.

---

## 7. Tools and Technologies
- **Programming Language:** Python 3.10+
- **Data Manipulation & Wrangling:** Pandas, NumPy
- **Data Visualization:** Matplotlib, Seaborn
- **Machine Learning & Preprocessing:** Scikit-learn (`LogisticRegression`, `RandomForestClassifier`, `ColumnTransformer`, `StandardScaler`, `OneHotEncoder`)
- **Model Serialization:** Joblib
- **Interactive Web Dashboard:** Streamlit

---

## 8. Data Cleaning
The data cleaning protocol was executed to ensure high data integrity:
1. **Missing Value Imputation:**
   - Evaluated missing frequencies across all columns.
   - For **numerical features** (`Number_of_vehicles_involved`, `Number_of_casualties`), applied **median imputation** to resist outlier distortion.
   - For **categorical features** (`Driving_experience`, `Types_of_Junction`, etc.), applied **mode imputation** or standardized as `'Unknown'`.
   - Avoided blind zero-filling to prevent distortion of distance metrics.
2. **Duplicate Records:**
   - Evaluated row-level uniqueness; 0 redundant duplicate records were identified, preserving 100% of observations.
3. **Inconsistent Categorical Values & Text Cleaning:**
   - Standardized target labels (`'Fatal injury'` $\rightarrow$ `'Fatal Injury'`).
   - Cleaned corrupted character artifacts in vehicle names (e.g. `'Lorry (41?100Q)'` $\rightarrow$ `'Lorry (41-100Q)'`).
   - Cleaned driver experience casing (`'unknown'` $\rightarrow$ `'Unknown'`).
   - Stripped leading and trailing whitespace across all string columns.
4. **Outlier Analysis (IQR Method):**
   - Computed Interquartile Range ($IQR = Q_3 - Q_1$) and valid bounds $[Q_1 - 1.5 \cdot IQR, Q_3 + 1.5 \cdot IQR]$.
   - Vehicles involved: Bounds [0.5, 4.5]; detected outliers: 34 records (0.28%).
   - Casualties: Bounds [-0.5, 3.5]; detected outliers: 367 records (2.98%).
   - **Academic Retention Rationale:** Highway pile-ups involving $>4$ vehicles and severe multi-casualty incidents are genuine, high-impact traffic events. Dropping them would remove the most severe and fatal crashes, blinding models to worst-case scenarios.

---

## 9. Data Transformation
To provide richer signals for modeling, several features were derived:
- **`Hour` (0–23):** Parsed from the timestamp string.
- **`Minute` (0–59):** Extracted for temporal granularity.
- **`Time_of_day`:** Binned into `Morning` (06:00–11:59), `Afternoon` (12:00–16:59), `Evening` (17:00–21:59), and `Night` (22:00–05:59).
- **`Rush_hour`:** Commute peak hours (07:00–09:59 and 16:00–19:59) vs. `Non-Rush Hour`.
- **`Day_type`:** `Weekday` (Monday–Friday) vs. `Weekend` (Saturday–Sunday).
- **`Collision_scale`:** Categorized into `Single Vehicle`, `Two Vehicles`, and `Multi-Vehicle Pile-up`.
- **`Casualty_scale`:** Categorized into `Low Casualty (1)`, `Medium Casualty (2)`, and `High Casualty (3+)`.

---

## 10. Data Wrangling
Key Pandas wrangling operations demonstrated include:
1. **Filtering:** Segmented high-risk crashes ($\ge 3$ casualties or fatal injury), identifying 484 severe incidents.
2. **Sorting:** Ranked the top 10 most severe crashes by casualties and vehicles involved.
3. **Pivot Table:** Generated a two-dimensional cross-tabulation mapping `Day_of_week` against `Time_of_day`.
4. **Multi-Aggregation:** Aggregated casualties across lighting conditions using `['count', 'mean', 'median', 'std', 'min', 'max']`.

---

## 11. Exploratory Data Analysis (EDA)
Empirical answers to core traffic safety questions:
1. **Total Accidents:** 12,316 records.
2. **Dominant Severity Class:** Slight Injury with 10,415 incidents (84.56%).
3. **Most Hazardous Weather:** Normal weather recorded the highest crash volume (10,063 crashes), illustrating that crash frequency correlates with traffic exposure rather than adverse weather alone. Raining conditions recorded 1,331 incidents.
4. **Most Hazardous Road Surface:** Dry roads accounted for 9,340 incidents (75.8%), followed by Wet or damp surfaces with 2,904 crashes.
5. **Peak Day of Week:** Friday experienced the highest accident volume with 2,041 crashes (16.57%).
6. **Peak Hour:** 17:00 (5:00 PM) was the single peak crash hour, accounting for 1,228 collisions.
7. **Time of Day with Highest Frequency:** Evening (17:00–21:59) with 3,897 crashes (31.64%).
8. **Vehicle vs. Casualty Relationship:** Exhibits a positive correlation coefficient of $+0.2134$.
9. **Severity Differences:** Fatal Injury accidents averaged 1.77 casualties per incident, compared to 1.54 for Slight Injury accidents.

---

## 12. Data Visualization
Publication-quality charts created:
1. **Pie/Donut Chart:** Severity class proportion highlighting class distribution.
2. **Line Chart:** 24-hour cycle showing morning and evening rush-hour peaks.
3. **Bar Charts:** Accident counts categorized by Weather and Road Surface.
4. **Histograms:** Casualty distribution per incident with KDE curve.
5. **Scatter Plot:** Jittered scatter plot of Vehicles Involved vs. Casualties with linear trendline.
6. **Box Plots:** Casualty distributions stratified across severity categories.
7. **Heatmap:** Correlation matrix among numerical variables.
8. **Count Plots:** Accident frequencies across Days of the Week and Light Conditions.

---

## 13. Machine Learning Methodology
- **Target Label:** `Accident_severity` (Multi-class: 3 categories).
- **Data Partitioning:** 80% Training (9,852 rows), 20% Testing (2,464 rows) with stratified sampling.
- **Preprocessing Pipeline:**
  - `StandardScaler` for continuous numerical features.
  - `OneHotEncoder(handle_unknown='ignore')` for high-cardinality categorical features.
  - Fitted exclusively on the training set to prevent data leakage.
- **Class Imbalance Strategy:** Balanced class weighting (`class_weight='balanced'`) to penalize minority misclassifications.

---

## 14. Model 1 — Logistic Regression
Multinomial Logistic Regression estimates class probabilities using the softmax function:
$$P(Y = k \mid X) = \frac{e^{\beta_k^T X}}{\sum_{j=1}^K e^{\beta_j^T X}}$$

- **Hyperparameters:** `solver='lbfgs'`, `max_iter=1000`, `class_weight='balanced'`, `random_state=42`.
- **Characteristics:** Provides a transparent linear baseline across scaled one-hot encoded features.

---

## 15. Model 2 — Random Forest Classifier
Random Forest is an ensemble of decorrelated decision trees built via bootstrap aggregating (bagging) and random feature subspace sampling:
$$\hat{f}_{\text{rf}}(x) = \text{mode} \left\{ T_1(x), T_2(x), \dots, T_B(x) \right\}$$

- **Hyperparameters:** `n_estimators=100`, `max_depth=16`, `class_weight='balanced'`, `random_state=42`, `n_jobs=-1`.
- **Characteristics:** Successfully models complex non-linear feature interactions without requiring linear separability.

---

## 16. Model Evaluation
The models were evaluated on the independent test set (2,464 samples).

### Dynamic Evaluation Metrics:
| Evaluation Metric | Logistic Regression | Random Forest Classifier |
| :--- | :---: | :---: |
| **Accuracy** | 48.90% | **81.53%** |
| **Weighted Precision** | 76.87% | **80.00%** |
| **Weighted Recall** | 48.90% | **81.53%** |
| **Weighted F1 Score** | 0.5776 | **0.8035** |

### Confusion Matrix Insights:
- **Logistic Regression:** By aggressively reweighting minority classes, it predicts higher proportions of Fatal and Serious injuries at the expense of lower overall precision on the dominant Slight Injury class.
- **Random Forest Classifier:** Successfully balances True Positives across Slight and Serious injuries, maintaining superior precision and recall.

### Top Gini Feature Importances (Random Forest):
1. `Hour` (Crash time of day)
2. `Number_of_casualties`
3. `Number_of_vehicles_involved`
4. `Day_of_week_Friday`
5. `Age_band_of_driver_18-30`
6. `Driving_experience_1-2yr`
7. `Types_of_Junction_No junction`

---

## 17. Model Comparison
The empirical comparison conclusively establishes that **Random Forest Classifier** outperforms **Logistic Regression** across every primary performance metric:
- **Accuracy Improvement:** $+32.63\%$ (81.53% vs. 48.90%)
- **F1 Score Improvement:** $+0.2259$ (0.8035 vs. 0.5776)

### Analysis:
Road traffic severity is governed by non-linear interactions (e.g., young driver age combined with wet road surfaces and evening darkness). While Logistic Regression is constrained to linear hyperplanes in feature space, Random Forest recursively partitions multidimensional subspaces to isolate high-risk casualty scenarios.

---

## 18. Dashboard Architecture
The interactive Streamlit dashboard (`app.py`) provides:
1. **Home / Overview:** Live KPI metric cards, executive summary, donut distribution, and dataset previews.
2. **Two Prominent ML Buttons:**
   - `🤖 1. Go to LOGISTIC REGRESSION Analysis & Prediction`
   - `🌲 2. Go to RANDOM FOREST Analysis & Prediction`
3. **Data Cleaning & Wrangling Views:** Missing value audits, duplicate checks, IQR outlier bounds, and derived features.
4. **Visualizations Gallery:** Four tabbed sections covering weather, temporal cycles, distributions, and heatmaps.
5. **Interactive Prediction Forms:** Users can enter road, environmental, vehicular, and driver details to generate live model predictions with class probabilities.
6. **Model Comparison Page:** Side-by-side metric tables and grouped comparative bar charts.

---

## 19. Results
1. A reproducible data pipeline successfully processed 12,316 accident records without data leakage.
2. Friday evening rush hour (17:00–19:00) represents the highest-frequency risk window.
3. The trained Random Forest model achieved **81.53% accuracy** and **0.8035 F1 score** on the holdout test set.
4. The serialized pipeline supports real-time severity prediction for live inputs.

---

## 20. Limitations
1. **Target Class Imbalance:** Fatal injuries represent only 1.28% of records, limiting the sample size for fatal crash learning.
2. **Geographic Specificity:** Data reflects urban and highway police logs; generalized deployment to other regions requires local calibration.
3. **Missing Telematics:** Features such as exact vehicular speed, seatbelt usage, and road friction coefficients were unrecorded.

---

## 21. Future Scope
1. **Advanced Resampling:** Apply SMOTE-ENN (Synthetic Minority Over-sampling with Edited Nearest Neighbors) to boost minority sensitivity.
2. **Deep Learning:** Investigate multi-layer perceptrons (MLP) and gradient boosted trees (XGBoost, LightGBM).
3. **GIS Spatial Mapping:** Integrate geospatial coordinates to generate interactive heatmap overlays of accident hotspots.
4. **Edge IoT Deployment:** Package the inference engine into onboard vehicle telematics for real-time hazard warnings.

---

## 22. Conclusion
This project demonstrated a rigorous academic data analytics and machine learning solution for road accident severity prediction. By systematically transitioning from raw data cleaning and exploratory analytics to supervised modeling and interactive deployment, the project highlights the strength of ensemble learning in transport safety analytics. The resulting Streamlit application provides accessible insights for traffic safety stakeholders.

---

## 23. References
1. UK Department for Transport, *Road Safety Data & Accident Reporting Guidelines*, Open Government Licence.
2. Breiman, L. (2001). *Random Forests*. Machine Learning, 45(1), 5-32.
3. Hosmer, D. W., Lemeshow, S., & Sturdivant, R. X. (2013). *Applied Logistic Regression*. John Wiley & Sons.
4. Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python*. Journal of Machine Learning Research, 12, 2825-2830.
5. McKinney, W. (2010). *Data Structures for Statistical Computing in Python*. Proceedings of the 9th Python in Science Conference, 51-56.
