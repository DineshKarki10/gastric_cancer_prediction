# 🧬 Gastric Cancer Risk Prediction

A machine learning framework and interactive Streamlit application designed to evaluate patient gastric cancer risk using clinical data, diagnostic imaging indicators, and genomic biomarkers.

---

## 📌 Problem Statement & Core Challenges

Diagnosing gastric cancer early is critical, but machine learning on this dataset presents three key challenges:

1. **Severe Class Imbalance (~9:1 Ratio)**  
   The dataset contains ~90.1% negative and ~9.9% positive gastric cancer cases. Standard unweighted models (Logistic Regression, default XGBoost) default to predicting 0 for all patients, yielding **0% cancer recall**.

2. **Weak Linear Signals**  
   Individual clinical and miRNA features exhibit very low linear correlation ($|r| \le 0.04$) with cancer diagnosis. Linear classifiers fail to find meaningful linear separation.

3. **High Overfitting Risk**  
   Unpruned decision trees memorize noisy training patterns (100% train accuracy) but fail to generalize to unseen test patients.

---

## 💡 Solution & Model Approach

We engineered a **regularized, class-balanced Decision Tree** optimized specifically for high sensitivity (recall) in medical screening:

* **Class Balancing**: `class_weight='balanced'` adjusts loss weights to penalize false negatives heavily.
* **Regularization**: `max_depth=8` and `min_samples_leaf=50` prevent noisy leaf splits.
* **Evaluation Metric**: Prioritized **Recall** to minimize missed cancer diagnoses.

### Model Performance

| Metric | Score | Clinical Meaning |
| :--- | :--- | :--- |
| **Recall (Cancer)** | **67.9%** | Correctly identifies ~68% of actual cancer cases |
| **Accuracy** | **81.7%** | Overall dataset classification accuracy |
| **Precision** | **9.9%** | Reflects high false-positive rate from class balancing |
| **Balanced Accuracy** | **50.1%** | Average accuracy across both classes |

---

## 🔬 Dataset & Feature Breakdown

* **Demographics & Risk Factors**: Age, Gender, Family History, Smoking/Alcohol, Dietary Salt.
* **Clinical Diagnostic Tests**: *H. Pylori* infection status, Endoscopic imaging, Biopsy results, CT scan, Chronic Gastritis.
* **Genomic & Biomarker Profile**:
  * **MicroRNAs**: `MIR123_1`, `MIR234_2`, `MIR345_3`
  * **Target Genes**: `CDH1` (E-Cadherin), `KRAS`, `TP53` (p53 tumor suppressor)
  * **Bioinformatics Scores**: Algorithm consensus prediction scores (TargetScan, miRanda, PITA, etc.)

---

## 🖥️ Streamlit Web Application

The interactive web application provides:
* **Live Patient Risk Predictor**: Instant dynamic risk calculation with a visual **Speedometer Gauge Chart** (Low, Moderate, High Risk tiers).
* **Dataset Overview**: Patient population breakdown, age distribution, and missing value verification.
* **Feature Analysis**: Chi-Square significance rankings and Pearson correlation charts.

---

## 🚀 Quick Start

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/DineshKarki10/gastric_cancer_prediction.git
cd gastric_cancer_prediction
python3 -m venv .venv
source .venv/bin/activate
pip install -r app/Gastric_Cancer_Prediction/requirements.txt
```

### 2. Run Streamlit Application
```bash
cd app/Gastric_Cancer_Prediction
streamlit run gastric_cancer_prediction.py
```
Open `http://localhost:8501` in your browser.
