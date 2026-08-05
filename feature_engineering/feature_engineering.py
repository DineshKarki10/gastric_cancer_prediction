import pandas as pd
import numpy as np


def section(title):
    print(f"\n{'='*60}\n{title}\n{'='*60}")


def engineer_features(df):
    """
    Applies feature engineering to the cleaned gastric cancer dataset.
    NOTE: Does NOT scale features here. Scaling must happen after the
    train/test split (fit on train, transform on both) to avoid leaking
    test-set statistics into training. See train_test_and_scale.py.
    """
    df = df.copy()

    # ========================================================
    # 1. Age binning
    # --------------------------------------------------------
    # WHY: gastric cancer risk isn't linear with age, and logistic
    # regression can only learn a single straight-line effect per
    # feature. Binning into life stages lets the model assign a
    # separate, interpretable coefficient to each age group instead
    # of assuming risk increases at a constant rate every year.
    # drop_first=True on the dummies avoids the dummy variable trap.
    # ========================================================
    age_bins = [0, 35, 50, 65, 150]
    age_labels = ['under_35', '35_to_50', '50_to_65', '65_plus']
    df['age_group'] = pd.cut(df['age'], bins=age_bins, labels=age_labels, right=False)
    age_dummies = pd.get_dummies(df['age_group'], prefix='age_group', drop_first=True).astype(int)
    df = pd.concat([df, age_dummies], axis=1)
    df = df.drop(columns=['age_group'])

    # ========================================================
    # 2. Domain-specific interaction features
    # --------------------------------------------------------
    # WHY: logistic regression only models each feature's individual
    # effect unless we explicitly multiply features together. These
    # three interactions are based on documented synergistic risk
    # relationships in gastric/GI cancer literature, not arbitrary
    # combinations — each tests a specific hypothesis.
    # ========================================================
    df['smoking_alcohol_interaction'] = df['smoking_habits'] * df['alcohol_consumption']
    df['hpylori_salt_interaction'] = df['helicobacter_pylori_infection'] * df['dietary_habits']
    df['family_history_age'] = df['family_history'] * df['age']

    # ========================================================
    # 3. miRNA binding consensus aggregates
    # --------------------------------------------------------
    # WHY: 8 different tools independently score the same miRNA-gene
    # interaction. Rather than feeding all 8 in individually (which
    # our EDA showed are individually uncorrelated with the label),
    # we test whether agreement/summary statistics across tools carry
    # signal the individual scores don't. This is a reasonable thing
    # to test given the domain, but our EDA already suggests these
    # scores carry little relationship with the label — treat this as
    # a hypothesis to confirm or reject with the correlation check
    # below, not an assumption that it will help.
    # ========================================================
    mirna_cols = ['diana_microt', 'elmmo', 'microcosm', 'miranda',
                  'mirdb', 'pictar', 'pita', 'targetscan']

    df['mirna_mean_score'] = df[mirna_cols].mean(axis=1)
    df['mirna_std_score'] = df[mirna_cols].std(axis=1)
    df['mirna_max_score'] = df[mirna_cols].max(axis=1)
    df['mirna_min_score'] = df[mirna_cols].min(axis=1)
    df['mirna_consensus_count'] = (df[mirna_cols] > 0.70).sum(axis=1)

    return df


def check_correlation_with_label(df, new_feature_cols):
    """Quick correlation check specifically for the newly engineered features."""
    section("CORRELATION CHECK: ENGINEERED FEATURES vs LABEL")
    corr = df[new_feature_cols + ['label']].corr(numeric_only=True)['label'].drop('label')
    corr = corr.sort_values(key=abs, ascending=False)
    print(corr)


if __name__ == '__main__':
    section("LOAD CLEANED DATA")
    df = pd.read_csv('/Users/Sanskar/Documents/gastric_cancer_prediction/Dataset/gastric_cancer_cleaned.csv')
    print(f"Shape before feature engineering: {df.shape}")

    section("ENGINEER FEATURES")
    df_engineered = engineer_features(df)

    new_cols = ['age_group_35_to_50', 'age_group_50_to_65', 'age_group_65_plus',
                'smoking_alcohol_interaction', 'hpylori_salt_interaction', 'family_history_age',
                'mirna_mean_score', 'mirna_std_score', 'mirna_max_score',
                'mirna_min_score', 'mirna_consensus_count']

    print(f"Shape after feature engineering: {df_engineered.shape}")
    print(f"New columns added: {new_cols}")

    section("VALIDATION")
    print("Missing values introduced:", df_engineered[new_cols].isnull().sum().sum())
    print("Infinite values introduced:", np.isinf(df_engineered[new_cols].select_dtypes(include=[np.number])).sum().sum())

    # Correlation check is defined above (check_correlation_with_label) but
    # not run here by default. Call it manually when you're ready to see
    # engineered-feature relationships to the label.

    df_engineered.to_csv('gastric_cancer_engineered.csv', index=False)
    print("\nSaved -> gastric_cancer_engineered.csv")
    print("\nReminder: scaling is NOT applied yet. Fit StandardScaler on the")
    print("training split only, in the next step, to avoid data leakage.")