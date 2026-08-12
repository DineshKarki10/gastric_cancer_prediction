import os
import sys
import argparse
import joblib
import unittest
from unittest.mock import MagicMock
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Ensure the parent and current directory are in the import path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from train_logistic_regression import evaluate_model, load_and_prepare_data, section


class TestEvaluateModel(unittest.TestCase):
    """Unit tests for the evaluate_model function."""

    def test_evaluate_model_returns_correct_keys(self):
        """Verifies that evaluate_model returns the expected metrics dictionary structure."""
        # Create a mock model
        mock_model = MagicMock()
        mock_model.predict = MagicMock(return_value=np.array([0, 1, 0, 1]))
        mock_model.predict_proba = MagicMock(return_value=np.array([
            [0.9, 0.1],
            [0.1, 0.9],
            [0.8, 0.2],
            [0.2, 0.8]
        ]))

        # Test inputs
        X_test_dummy = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y_test = np.array([0, 0, 1, 1])

        # Execute
        metrics = evaluate_model(mock_model, X_test_dummy, y_test, model_name="Mock Model Test")

        # Assertions
        self.assertIsInstance(metrics, dict, "evaluate_model must return a dictionary")
        
        expected_keys = {"accuracy", "precision", "recall", "f1", "roc_auc"}
        self.assertTrue(expected_keys.issubset(metrics.keys()), 
                        f"Returned dict missing keys. Expected {expected_keys}, got {metrics.keys()}")
        
        # Verify manual values:
        # Accuracy = 2 / 4 = 0.5
        # Precision = 1 / 2 = 0.5 (TP=1, FP=1)
        # Recall = 1 / 2 = 0.5 (TP=1, FN=1)
        # F1 = 0.5
        # ROC-AUC = 0.5
        self.assertAlmostEqual(metrics["accuracy"], 0.5)
        self.assertAlmostEqual(metrics["precision"], 0.5)
        self.assertAlmostEqual(metrics["recall"], 0.5)
        self.assertAlmostEqual(metrics["f1"], 0.5)
        self.assertAlmostEqual(metrics["roc_auc"], 0.5)


def run_unit_tests():
    """Runs the unit test suite and returns True if all tests pass."""
    section("RUNNING UNIT TESTS FOR EVALUATE_MODEL")
    suite = unittest.TestLoader().loadTestsFromTestCase(TestEvaluateModel)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return result.wasSuccessful()


def evaluate_saved_models(data_path, save_dir, random_state):
    """Loads the saved models/scaler and evaluates them on the test split."""
    section("CHECKING SAVED MODELS AND EVALUATING ACCURACY")
    
    # 1. Load the data
    X, y = load_and_prepare_data(data_path)
    
    # 2. Re-create the 50/50 Train/Test Split using same settings
    print(f"\nRe-creating train/test split (random_state={random_state}, 50% test)...")
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.50, random_state=random_state, stratify=y
    )
    print(f"Test split shape: {X_test.shape[0]:,} samples")
    
    # 3. Load preprocessing scaler
    scaler_path = os.path.join(save_dir, "scaler.joblib")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler not found at: {scaler_path}")
    print(f"Loading scaler from: {scaler_path}")
    scaler = joblib.load(scaler_path)
    
    # Scale test set
    X_test_scaled = scaler.transform(X_test)
    
    # 4. Load standard model
    model_std_path = os.path.join(save_dir, "logistic_regression_standard.joblib")
    if not os.path.exists(model_std_path):
        raise FileNotFoundError(f"Standard model not found at: {model_std_path}")
    print(f"Loading standard model from: {model_std_path}")
    model_std = joblib.load(model_std_path)
    
    # Evaluate Standard model
    metrics_std = evaluate_model(model_std, X_test_scaled, y_test, "Standard Logistic Regression")
    print(f"\nReturned metrics dict (Standard): {metrics_std}")
    
    # 5. Load balanced model
    model_bal_path = os.path.join(save_dir, "logistic_regression_balanced.joblib")
    if not os.path.exists(model_bal_path):
        raise FileNotFoundError(f"Balanced model not found at: {model_bal_path}")
    print(f"Loading balanced model from: {model_bal_path}")
    model_bal = joblib.load(model_bal_path)
    
    # Evaluate Balanced model
    metrics_bal = evaluate_model(model_bal, X_test_scaled, y_test, "Balanced Logistic Regression")
    print(f"\nReturned metrics dict (Balanced): {metrics_bal}")
    
    # 6. Sanity check validations on dictionary structures
    for name, metrics in [("Standard", metrics_std), ("Balanced", metrics_bal)]:
        assert isinstance(metrics, dict), f"{name} metrics should be a dictionary"
        for key in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            assert key in metrics, f"{key} missing in {name} metrics"
            assert 0.0 <= metrics[key] <= 1.0, f"{key} value ({metrics[key]}) is out of bounds [0, 1]"
            
    print("\nAll accuracy checks completed successfully! Saved models returned valid metrics.")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Accuracy & Test Metrics Return Format")
    parser.add_argument("--data_path", type=str, default="/Users/Sanskar/Documents/gastric_cancer_prediction/gastric_cancer_engineered.csv",
                        help="Path to the engineered CSV file")
    parser.add_argument("--save_dir", type=str, default="/Users/Sanskar/Documents/gastric_cancer_prediction/models/saved",
                        help="Directory where trained models/scaler are saved")
    parser.add_argument("--random_state", type=int, default=42,
                        help="Seed for train/test split reproducibility")
    parser.add_argument("--skip_tests", action="store_true",
                        help="Skip unit testing and only check the saved models")
    
    args = parser.parse_args()
    
    # Step 1: Run unit tests
    if not args.skip_tests:
        tests_passed = run_unit_tests()
        if not tests_passed:
            print("Unit tests failed. Aborting model evaluation.", file=sys.stderr)
            sys.exit(1)
            
    # Step 2: Evaluate actual saved models
    try:
        evaluate_saved_models(args.data_path, args.save_dir, args.random_state)
    except Exception as e:
        print(f"Error during model evaluation: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
