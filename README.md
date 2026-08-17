# Gastric Cancer Prediction Pipeline

An end-to-end machine learning pipeline for predicting the risk of gastric cancer from clinical findings, demographic features, and miRNA binding consensus scores.

---

## 🧬 Project Overview
Gastric cancer is a complex disease where early diagnostics are critical. This repository implements a modular machine learning pipeline to:
1. **Clean and preprocess** raw synthetic/clinical diagnostic records.
2. **Engineer domain-specific features** (non-linear age groupings, risk factor interactions, miRNA database consensus aggregates).
3. **Discover patient subgroups** using density-based clustering algorithms (DBSCAN/HDBSCAN).
4. **Train and evaluate classification models** (Standard and Class-Balanced Logistic Regression baseline models).
5. **Verify prediction metrics** and return dictionary formatting through programmatic unit testing.
6. **Interact and visualize** findings and diagnostic risks using a rich Streamlit Web Dashboard.

---

## 📂 Directory Structure

```directory
gastric_cancer_prediction/
├── Dataset/
│   ├── gastric_cancer_detection_dataset.csv # Raw input data
│   ├── gastric_cancer_cleaned.csv           # Output from data cleaning
│   └── gastric_cancer_clustered.csv         # Output from clustering
├── app/
│   └── dashboard.py                         # Streamlit Dashboard application
├── clustering/
│   └── dbscan_clustering.py                 # DBSCAN & HDBSCAN clustering script
├── data_cleaning/
│   └── data_cleaning.py                     # Initial data imputation and encoding
├── feature_engineering/
│   └── feature_engineering.py               # Age binning, interactions, & miRNA aggregates
├── models/
│   ├── accuracy.py                          # Programmatic unit tests & evaluation checker
│   ├── train_logistic_regression.py         # Logistic Regression training pipeline
│   └── saved/                               # Trained weights (joblib format)
│       ├── logistic_regression_balanced.joblib
│       ├── logistic_regression_standard.joblib
│       └── scaler.joblib
├── gastric_cancer_engineered.csv            # Fully engineered dataset
├── main.py                                  # Root entrypoint
├── pyproject.toml                           # Package dependencies
└── README.md                                # Project documentation
```

---

## ⚡ Setup & Installation

This project utilizes the `uv` Python package manager for fast, reproducible dependency resolution. 

### 1. Initialize Virtual Environment & Install Dependencies
Run the following in the root directory:
```bash
# Create virtual environment and install packages
uv venv
uv sync
```

### 2. Activate Virtual Environment
* **macOS / Linux**:
  ```bash
  source .venv/bin/activate
  ```
* **Windows**:
  ```bash
  .venv\Scripts\activate
  ```

---

## ⚙️ Running the Pipeline

You can run each stage of the pipeline sequentially:

### 1. Data Cleaning
Imputes missing categorical strings (e.g. `existing_conditions`), maps binary states (e.g. gender, lab findings) to `0/1`, one-hot encodes multi-category columns (e.g. target symbols), and drops high-cardinality ID features.
```bash
python data_cleaning/data_cleaning.py
```

### 2. Feature Engineering
Creates interactions (e.g. `smoking_habits` * `alcohol_consumption`), bins patient ages into clinical groups, and generates statistical consensus values (mean, std, max, min, consensus count) across the 8 miRNA search databases.
```bash
python feature_engineering/feature_engineering.py
```

### 3. Subgroup Clustering
Clusters patients based on continuous variables (`age` + miRNA scores) using DBSCAN/HDBSCAN. This script also contains a K-Nearest-Neighbor distance tool to identify the "elbow" point for optimal `eps` parameter selection.
```bash
# Run KNN-distance analysis to tune eps
python clustering/dbscan_clustering.py --k_distance --plot_dir clustering/plots

# Perform DBSCAN clustering on a sample of 20,000 points
python clustering/dbscan_clustering.py --method dbscan --eps 1.8 --min_samples 18 --plot_dir clustering/plots
```

### 4. Train Baseline Classification Models
Splits the engineered dataset (50/50 stratified train/test split), fits a `StandardScaler` to the training set (preventing data leakage), and trains both **Unweighted (Standard)** and **Class-Balanced** Logistic Regression models.
```bash
python models/train_logistic_regression.py
```

### 5. Programmatic Metric Testing & Verification
A test file containing a unit test suite to verify the return signature of `evaluate_model` (requiring `accuracy`, `precision`, `recall`, `f1`, and `roc_auc`) using mocks, and prints actual evaluation dictionaries for saved model weights.
```bash
python models/accuracy.py
```

### 6. Start the Web Dashboard
Launches the Streamlit application containing data summaries, correlation heatmaps, interactive 2D PCA cluster visualizations, and a live patient diagnostic risk calculator.
```bash
streamlit run app/dashboard.py
```

---

## 📊 Baseline Model Performance

Due to the synthetic nature and uniform distribution of miRNA features, the baseline linear classification yields the following results on the test split:

| Model Variant | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Description |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Standard Logistic Regression** | 90.13% | 0.00% | 0.00% | 0.00 | 0.4922 | Overfits to majority class (No Cancer) due to ~10% positive rate. |
| **Balanced Logistic Regression** | 50.79% | 9.70% | 47.97% | 0.1614 | 0.4917 | Penalizes minority errors; improves recall but operates close to random chance. |

*These metrics highlight the need to move beyond simple linear models and introduce advanced non-linear architectures (e.g. Random Forest, XGBoost) and deeper feature selection.*
