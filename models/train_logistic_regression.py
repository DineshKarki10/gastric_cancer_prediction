import os
import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)


def section(title):
    print(f"\n{'='*60}\n{title}\n{'='*60}")


def load_and_prepare_data(data_path):
    """Loads dataset and separates features and target."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at: {data_path}")
        
    print(f"Loading engineered dataset from: {data_path} ...")
    df = pd.read_csv(data_path)
    print(f"Loaded shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    
    # 1. Separate target label
    if 'label' not in df.columns:
        raise ValueError("Dataset does not contain target column 'label'")
    y = df['label']
    
    # 2. Extract features
    # Drop target and non-predictive high-cardinality ID column
    cols_to_drop = ['label', 'target_ensembl']
    cols_to_drop = [c for c in cols_to_drop if c in df.columns]
    X = df.drop(columns=cols_to_drop)
    
    print(f"Target distribution (class 1 rate): {y.mean()*100:.2f}% ({y.sum()} / {len(y)})")
    print(f"Number of predictor features: {X.shape[1]}")
    print(f"Features: {X.columns.tolist()[:10]} ... [and {X.shape[1]-10} more]")
    
    return X, y


def evaluate_model(model, X_test_scaled, y_test, model_name="Logistic Regression"):
    """Evaluates the model on test split and prints results."""
    # Predict labels
    y_pred = model.predict(X_test_scaled)
    # Predict probabilities (for ROC-AUC)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    print(f"\n### Results for: {model_name} ###")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"  True Negatives (No Cancer):  {cm[0, 0]:6d} | False Positives: {cm[0, 1]:6d}")
    print(f"  False Negatives:             {cm[1, 0]:6d} | True Positives:  {cm[1, 1]:6d}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc
    }


def main():
    parser = argparse.ArgumentParser(description="Logistic Regression Training & Evaluation Pipeline")
    parser.add_argument("--data_path", type=str, default="/Users/Sanskar/Documents/gastric_cancer_prediction/gastric_cancer_engineered.csv",
                        help="Path to the engineered CSV file")
    parser.add_argument("--save_dir", type=str, default="/Users/Sanskar/Documents/gastric_cancer_prediction/models/saved",
                        help="Directory to save trained models and scaler")
    parser.add_argument("--random_state", type=int, default=42,
                        help="Seed for train/test split reproducibility")
    
    args = parser.parse_args()
    
    # 1. Load data
    X, y = load_and_prepare_data(args.data_path)
    
    # 2. 50/50 Train/Test Split
    section("STEP 1: 50/50 TRAIN/TEST SPLIT")
    print(f"Splitting dataset: 50% training, 50% prediction (test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.50, random_state=args.random_state, stratify=y
    )
    print(f"Training split shape:   {X_train.shape[0]:,} samples")
    print(f"Prediction split shape: {X_test.shape[0]:,} samples")
    
    # 3. Scaling (Prevent Leakage: Fit on train only!)
    section("STEP 2: FEATURE STANDARDIZATION (SCALING)")
    print("Fitting StandardScaler on training split...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    print("Normalizing prediction (test) split using training statistics...")
    X_test_scaled = scaler.transform(X_test)
    
    # 4. Train standard model
    section("STEP 3: TRAIN DEFAULT LOGISTIC REGRESSION (UNWEIGHTED)")
    print("Training standard Logistic Regression model...")
    model_std = LogisticRegression(max_iter=1000, random_state=args.random_state)
    model_std.fit(X_train_scaled, y_train)
    evaluate_model(model_std, X_test_scaled, y_test, "Standard Logistic Regression (Unweighted)")
    
    # 5. Train class-balanced model
    section("STEP 4: TRAIN BALANCED LOGISTIC REGRESSION (CLASS WEIGHTS)")
    print("Training Logistic Regression model with class_weight='balanced'...")
    model_bal = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=args.random_state)
    model_bal.fit(X_train_scaled, y_train)
    evaluate_model(model_bal, X_test_scaled, y_test, "Balanced Logistic Regression (Weighted)")
    
    # 6. Save artifacts
    section("STEP 5: SAVING ARTIFACTS")
    os.makedirs(args.save_dir, exist_ok=True)
    
    scaler_path = os.path.join(args.save_dir, "scaler.joblib")
    model_std_path = os.path.join(args.save_dir, "logistic_regression_standard.joblib")
    model_bal_path = os.path.join(args.save_dir, "logistic_regression_balanced.joblib")
    
    joblib.dump(scaler, scaler_path)
    joblib.dump(model_std, model_std_path)
    joblib.dump(model_bal, model_bal_path)
    
    print(f"Saved preprocessing scaler to:  {scaler_path}")
    print(f"Saved standard model to:        {model_std_path}")
    print(f"Saved class-balanced model to:  {model_bal_path}")
    
    print("\nTraining workflow completed successfully!")


if __name__ == "__main__":
    main()
