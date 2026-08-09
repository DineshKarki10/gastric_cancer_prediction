import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)
from xgboost import XGBClassifier


def load_data(data_path: str) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    return df


def prepare_data(df: pd.DataFrame):
    X = df.drop(columns=["label"])
    y = df["label"]
    feature_columns = list(X.columns)
    return X, y, feature_columns


def train_model(X, y, test_size: float = 0.2, random_state: int = 42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        eval_metric="logloss",
        use_label_encoder=False,
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, output_dict=True, zero_division=0),
    }

    return model, metrics, X_test, y_test, y_pred


def save_model(model, feature_columns, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "xgboost_model.pkl")
    columns_path = os.path.join(output_dir, "feature_columns.pkl")

    joblib.dump(model, model_path)
    joblib.dump(feature_columns, columns_path)

    print(f"Model saved to: {model_path}")
    print(f"Feature columns saved to: {columns_path}")


def print_metrics(metrics: dict):
    print("=" * 60)
    print("MODEL PERFORMANCE METRICS")
    print("=" * 60)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    print(f"ROC AUC:   {metrics['roc_auc']:.4f}")
    print("\nConfusion Matrix:")
    cm = metrics["confusion_matrix"]
    print(f"  True Negatives:  {cm[0][0]}")
    print(f"  False Positives: {cm[0][1]}")
    print(f"  False Negatives: {cm[1][0]}")
    print(f"  True Positives:  {cm[1][1]}")
    print("\nClassification Report:")
    report = metrics["classification_report"]
    for label, scores in report.items():
        if isinstance(scores, dict):
            print(f"  {label}: precision={scores['precision']:.4f}, recall={scores['recall']:.4f}, f1={scores['f1-score']:.4f}")
    print("=" * 60)


def main():
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "final_data.csv")
    output_dir = os.path.join(os.path.dirname(__file__), "..", "models")

    print("Loading data...")
    df = load_data(data_path)
    print(f"Data shape: {df.shape}")

    X, y, feature_columns = prepare_data(df)
    print(f"Features: {len(feature_columns)}")
    print(f"Target distribution:\n{y.value_counts()}")

    print("\nTraining XGBoost model...")
    model, metrics, X_test, y_test, y_pred = train_model(X, y)

    print_metrics(metrics)

    save_model(model, feature_columns, output_dir)

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
