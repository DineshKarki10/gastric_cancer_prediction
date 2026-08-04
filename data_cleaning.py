import pandas as pd
import numpy as np


def section(title):
    """Prints a clean section header so the output reads like a report,
    not a raw dump of every command's output."""
    print(f"\n{'='*60}\n{title}\n{'='*60}")


# ============================================================
# STEP 1: Load & inspect
# ============================================================
section("STEP 1: LOAD DATA")

df = pd.read_csv('/Users/Sanskar/Documents/gastric_cancer_prediction/Dataset/gastric_cancer_detection_dataset.csv')

print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
print(f"\nColumns:\n{list(df.columns)}")


# ============================================================
# STEP 2: Missing values
# ------------------------------------------------------------
# WHY: missing data can affect model performance, so we check
# before deciding whether to drop or fill.
# ============================================================
section("STEP 2: MISSING VALUES")

missing = df.isnull().sum()
print("Columns with missing values:")
print(missing[missing > 0] if missing.sum() > 0 else "None")

# existing_conditions' NaNs are actually the string "None" from the
# raw CSV, auto-converted by pandas — restore it as a real category
# rather than dropping rows or imputing statistically.
df['existing_conditions'] = df['existing_conditions'].fillna('None')
print(f"\nAfter fix -> total missing values: {df.isnull().sum().sum()}")


# ============================================================
# STEP 3: Duplicates
# ============================================================
section("STEP 3: DUPLICATES")

dupes = df.duplicated().sum()
df = df.drop_duplicates()
print(f"Exact duplicate rows found & removed: {dupes}")


# ============================================================
# STEP 4: Sanity checks (ranges & valid values)
# ============================================================
section("STEP 4: SANITY CHECKS")

print(f"Age range: {df['age'].min()} - {df['age'].max()}")

binary_cols = ['family_history', 'smoking_habits', 'alcohol_consumption',
               'helicobacter_pylori_infection', 'label']
for col in binary_cols:
    assert set(df[col].unique()).issubset({0, 1}), f"Unexpected values in {col}"
print("Binary flag columns confirmed to contain only 0/1.")

# Strip stray whitespace defensively (protects against silent
# encoding bugs later even though this dataset checked out clean)
cat_cols = df.select_dtypes(include=['object', 'str']).columns
for col in cat_cols:
    df[col] = df[col].str.strip()


# ============================================================
# STEP 5: Binary-encode 2-value columns
# ------------------------------------------------------------
# WHY: models need numbers, not text. Mapped so 1 = higher-risk /
# condition-present state, to keep coefficients interpretable.
# ============================================================
section("STEP 5: BINARY ENCODING")

binary_map = {
    'gender':                 {'Male': 1, 'Female': 0},
    'geographical_location':  {'California': 1, 'Other': 0},
    'dietary_habits':         {'High_Salt': 1, 'Low_Salt': 0},
    'endoscopic_images':      {'Abnormal': 1, 'Normal': 0},
    'biopsy_results':         {'Positive': 1, 'Negative': 0},
    'ct_scan':                {'Positive': 1, 'Negative': 0},
}
for col, mapping in binary_map.items():
    df[col] = df[col].map(mapping)

nan_check = df[list(binary_map.keys())].isnull().sum().sum()
print(f"Encoded columns: {list(binary_map.keys())}")
print(f"NaNs introduced by mapping: {nan_check} (should be 0)")


# ============================================================
# STEP 6: One-hot encode multi-category columns
# ------------------------------------------------------------
# WHY: ethnicity and existing_conditions have 3+ categories, so a
# single 0/1 column would falsely imply an order between them.
# drop_first=True avoids the dummy variable trap.
# ============================================================
section("STEP 6: ONE-HOT ENCODING")

print(f"ethnicity categories: {df['ethnicity'].unique()}")
print(f"existing_conditions categories: {df['existing_conditions'].unique()}")

df = pd.get_dummies(df, columns=['ethnicity', 'existing_conditions'], drop_first=True)

new_dummy_cols = [c for c in df.columns
                   if c.startswith('ethnicity_') or c.startswith('existing_conditions_')]
df[new_dummy_cols] = df[new_dummy_cols].astype(int)
print(f"New dummy columns created: {new_dummy_cols}")


# ============================================================
# STEP 7: Drop high-cardinality ID columns
# ------------------------------------------------------------
# WHY: miRNA/gene identifiers are not clinical measurements.
# One-hot encoding them would create thousands of sparse, mostly
# useless columns and cause severe overfitting.
# ============================================================
section("STEP 7: DROP ID COLUMNS")

id_cols_to_drop = ['mature_mirna_acc', 'mature_mirna_id', 'target_symbol',
                    'target_entrez', 'target_ensembl']
df = df.drop(columns=id_cols_to_drop)
print(f"Dropped: {id_cols_to_drop}")


# ============================================================
# STEP 8: Final summary
# ============================================================
section("STEP 8: FINAL SUMMARY")

remaining_text_cols = df.select_dtypes(include=['object', 'str']).columns.tolist()
print(f"Final shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
print(f"Remaining non-numeric columns: {remaining_text_cols if remaining_text_cols else 'None — all numeric ✓'}")

df.to_csv('/Users/Sanskar/Documents/gastric_cancer_prediction/Dataset/gastric_cancer_cleaned.csv', index=False)
print("\nSaved cleaned dataset -> gastric_cancer_cleaned.csv")