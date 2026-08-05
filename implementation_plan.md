# Implementation Plan: Feature Engineering for Logistic Regression

This plan outlines the proposed feature engineering strategies to prepare the cleaned gastric cancer dataset for training a **Logistic Regression** classifier.

## Context & Background
Logistic regression models a linear relationship between input features and the log-odds of the binary target label (`label` represents gastric cancer presence, ~10% positive rate). Because it is a linear model, it does not automatically capture non-linear relationships (like age progression) or interactions between risk factors (like smoking and alcohol) unless we explicitly engineer them. Additionally, it is highly sensitive to the scale of numeric inputs when regularization (L1/L2) is applied.

---

## Proposed Feature Engineering Strategies

We propose to create a Python pipeline in [feature_engineering.py](file:///Users/Sanskar/Documents/gastric_cancer_prediction/feature_engineering/feature_engineering.py) implementing the following:

### 1. Remove High-Cardinality ID features
* **Action:** Drop `target_ensembl` (which has 191,184 unique values).
* **Rationale:** Numeric IDs imply an ordering that is medically meaningless. One-hot encoding this feature would create ~191k sparse columns, leading to severe overfitting and out-of-memory errors. The higher-level categorical groupings are already represented by `target_symbol_KRAS` and `target_symbol_TP53` (with `CDH1` as the reference class).

### 2. Feature Scaling (Standardization)
* **Action:** Apply standard scaling ($Z$-score normalization: mean=0, std=1) to all continuous numerical features.
* **Rationale:** Scikit-learn's `LogisticRegression` applies L1 (Lasso) or L2 (Ridge) regularization by default, which penalizes the magnitude of the model coefficients. If features are on different scales (e.g. `age` ranges from 18 to 90, while database scores range from 0 to 1), features with smaller scales will be penalized disproportionately, degrading model performance.

### 3. Non-Linear Age Transformations
* **Action:** Risk of gastric cancer increases non-linearly with age. We propose two approaches (to be evaluated):
  * **Option A (Polynomial Age):** Add `age_squared` ($age^2$) to capture accelerating risk.
  * **Option B (Age Binning):** Discretize age into bins (e.g., `<35`, `35-50`, `50-65`, `65+`) and one-hot encode them. This allows the model to learn specific step-wise risk coefficients for different stages of life.

### 4. Domain-Specific Interaction Features
* **Action:** Explicitly create multiplicative interaction terms for known synergistic risk factors:
  * `smoking_alcohol_interaction` = `smoking_habits * alcohol_consumption` (well-documented synergism in gastrointestinal cancers).
  * `hpylori_salt_interaction` = `helicobacter_pylori_infection * dietary_habits` (H. pylori infection coupled with a high-salt diet has a compounding effect on stomach inflammation).
  * `family_history_age` = `family_history * age` (to capture potential early-onset hereditary cancer risk).

### 5. miRNA Binding Consensus Aggregates
* **Action:** The dataset contains 8 continuous prediction score columns from different miRNA-target search tools (`diana_microt`, `elmmo`, `microcosm`, `miranda`, `mirdb`, `pictar`, `pita`, `targetscan`). We will generate cross-database aggregates:
  * `mirna_mean_score`: Mean prediction probability across the 8 tools.
  * `mirna_std_score`: Standard deviation of the 8 scores (lower standard deviation represents higher consensus among the databases).
  * `mirna_max_score`: Maximum score achieved across all tools.
  * `mirna_min_score`: Minimum score achieved across all tools.
  * `mirna_consensus_count`: Count of prediction scores that exceed a confidence threshold (e.g., $> 0.70$).

---

## User Review Required

> [!IMPORTANT]
> **Key Decisions Needed:**
> 1. **Age Representation:** Do you prefer keeping age as continuous with polynomial terms (Option A) or binning it into categories (Option B)? Binning is highly interpretable, whereas polynomial terms preserve resolution.
> 2. **Target Ensembl ID:** Confirming that we should drop `target_ensembl` completely as planned, rather than trying complex target-encoding which might cause leakage or overfitting.

---

## Proposed Changes

### Feature Engineering Pipeline

#### [MODIFY] [feature_engineering.py](file:///Users/Sanskar/Documents/gastric_cancer_prediction/feature_engineering/feature_engineering.py)
We will implement a python script containing:
* A function `engineer_features(df)` that applies the dropping, scaling, interaction creation, age transformation, and miRNA aggregation logic.
* Main execution blocks to load `gastric_cancer_cleaned.csv`, apply the transformation, and save the resulting dataset to a new file `Dataset/gastric_cancer_engineered.csv`.

---

## Verification Plan

### Automated/Code Verification
* Check shape, column list, and verify no missing/infinite values are introduced in the engineered output.
* Confirm that $Z$-score standardized numerical columns have a mean of approximately 0 and standard deviation of approximately 1.
* Check correlation of engineered features with the label using a quick correlation check script.
