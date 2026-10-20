# Week 4: Supervised Learning Model Implementation

## Project Overview
This repository contains the deliverables for **Week 4** of the Data Science virtual internship at **Yuva Intern**[span_1](start_span)[span_1](end_span). The objective of this project is to design, implement, and benchmark supervised predictive learning pipelines to solve a binary classification problem on the **Titanic passenger manifest dataset** (predicting survival: 0 = Perished, 1 = Survived)[span_2](start_span)[span_2](end_span).

---

## Objectives
* Perform structured domain-specific feature engineering (title extraction, family structure aggregation, and log transformation of financial features)[span_3](start_span)[span_3](end_span).
* Handle missingness through conditional grouped median imputation.
* Train and benchmark two complementary supervised classification models:
  * **Baseline Model:** Regularized Logistic Regression (L2 / L-BFGS)
  * **Ensemble Model:** Random Forest Classifier (Bagging with depth regularization)
* Perform cross-validation using **Stratified 5-Fold Cross-Validation** to evaluate generalization stability[span_4](start_span)[span_4](end_span).
* Conduct diagnostic metric auditing via Confusion Matrix, Precision, Recall, F1-Score, and Gini Impurity feature importance[span_5](start_span)[span_5](end_span).

---

## Tech Stack & Libraries
* **Language:** Python
* **Environment:** VS Code / Jupyter Notebook
* **Libraries:**
  * `pandas` & `numpy` — Data transformation and numerical vector processing
  * `matplotlib` & `seaborn` — Diagnostic metric heatmaps and feature contribution charts
  * `scikit-learn` — Model architectures, stratified cross-validation, standard scaling, and classification diagnostics[span_6](start_span)[span_6](end_span)

---

## Methodology & Pipeline Workflow

### 1. Feature Engineering & Preprocessing
* **Honorific Title Extraction:** Parsed passenger titles from the `Name` column; mapped rare/aristocratic titles to a single `Rare` category and normalized French titles (`Mlle`, `Ms` $\rightarrow$ `Miss`, `Mme` $\rightarrow$ `Mrs`).
* **Conditional Median Imputation:** Imputed missing values for `Age` using grouped medians conditioned on passenger class (`Pclass`) and `Title`.
* **Family Topology:** Engineered `FamilySize` (`SibSp + Parch + 1`) and a binary indicator `IsAlone` (`1` if solitary traveler, `0` otherwise).
* **Fare Scaling:** Applied natural logarithmic transformation (`np.log1p`) to ticket fares to normalize heavy right-skewness.
* **Stratified Split:** Executed an 80-20 stratified train-test split (179 test records) to maintain exact class proportions.

### 2. Model Training & Cross-Validation
* **Logistic Regression:** Scaled features using `StandardScaler` to optimize convergence over 1,000 L-BFGS iterations.
* **Random Forest Classifier:** Configured with `n_estimators=100` and `max_depth=5` to prevent overfitting on complex non-linear feature interactions.

---

## Performance & Diagnostic Results

### Model Comparison
| Metric | Logistic Regression | Random Forest Classifier |
| :--- | :--- | :--- |
| **Holdout Test Accuracy** | 81.01% | 81.01% |
| **5-Fold Cross-Validation Mean** | 80.62% | **82.30%** |
| **Cross-Validation Std Dev** | $\pm$ 0.021 | **$\pm$ 0.018** |

### Random Forest Classification Report (Test Set)
| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **0 (Perished)** | 0.82 | 0.89 | 0.85 | 110 |
| **1 (Survived)** | 0.80 | 0.68 | 0.73 | 69 |
| **Accuracy** | | | **0.81** | 179 |
| **Macro Average** | 0.81 | 0.79 | 0.79 | 179 |
| **Weighted Average** | 0.81 | 0.81 | 0.81 | 179 |

### Confusion Matrix Breakdown
* **True Negatives (TN):** 98 (Correctly predicted Perished)
* **False Positives (FP):** 12 (Perished misclassified as Survived)
* **False Negatives (FN):** 22 (Survived misclassified as Perished)
* **True Positives (TP):** 47 (Correctly predicted Survived)

### Feature Importance Drivers
1. `Title_Mr` (~21.0% relative importance)
2. `Sex_male` (~20.5% relative importance)
3. `Fare_Log` (~15.0% relative importance)
4. `Pclass` (~10.5% relative importance)
5. `Age` (~8.0% relative importance)

---

## Repository Structure
```text
├── supervised_learning.ipynb      # Complete Python code with end-to-end pipeline
├── model_evaluation_plots.png     # Confusion matrix & feature importances figure
└── README.md                      # Detailed technical documentation
