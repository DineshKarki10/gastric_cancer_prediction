{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "7dc65edf_2",
   "metadata": {},
   "source": [
    "## Model Training with Class Imbalance Handling (Decision Trees, XGBoost, Random Forests, Ensemble Methods) ##\n",
    "\n",
    "The dataset has a **~9:1 class imbalance** (90.1% negative, 9.9% positive). Accuracy is misleading here \u2014 we optimize **F1-score, recall, precision, ROC-AUC, and PR-AUC** for the minority (cancer) class."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2c086c0a_2",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "import xgboost as xgb\n",
    "\n",
    "from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score\n",
    "from sklearn.tree import DecisionTreeClassifier\n",
    "from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.metrics import (\n",
    "    classification_report, confusion_matrix, roc_auc_score, roc_curve,\n",
    "    precision_recall_curve, f1_score, precision_score, recall_score,\n",
    "    average_precision_score, make_scorer, accuracy_score, balanced_accuracy_score,\n",
    ")\n",
    "from sklearn.utils import resample\n",
    "\n",
    "from imblearn.under_sampling import RandomUnderSampler\n",
    "from imblearn.over_sampling import SMOTE, RandomOverSampler\n",
    "from imblearn.pipeline import Pipeline as ImbPipeline\n",
    "from imblearn.ensemble import BalancedRandomForestClassifier"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "9ce47fb1_2",
   "metadata": {},
   "source": [
    "**Loading Dataset**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d969e5b5_2",
   "metadata": {},
   "outputs": [],
   "source": [
    "df = pd.read_csv(\"../Dataset/cleaned_gcs_kushal.csv\")\n",
    "\n",
    "# Convert boolean one-hot columns to integers for sklearn / xgboost\n",
    "for col in df.select_dtypes(include=\"bool\").columns:\n",
    "    df[col] = df[col].astype(int)\n",
    "\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "features_title",
   "metadata": {},
   "source": [
    "### Features and Target ###"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "43a9b011_2",
   "metadata": {},
   "outputs": [],
   "source": [
    "X = df.drop(\"label\", axis=1)\n",
    "y = df[\"label\"]"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "cb9ecb80",
   "metadata": {},
   "source": [
    "### Class Imbalance Overview ###"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "267b8ee7",
   "metadata": {},
   "outputs": [],
   "source": [
    "class_counts = y.value_counts().sort_index()\n",
    "imbalance_ratio = class_counts[0] / class_counts[1]\n",
    "\n",
    "print(\"Class distribution:\")\n",
    "print(class_counts)\n",
    "print(f\"\\nImbalance ratio (majority:minority): {imbalance_ratio:.1f}:1\")\n",
    "print(f\"Minority class prevalence: {100 * class_counts[1] / len(y):.1f}%\")\n",
    "\n",
    "fig, axes = plt.subplots(1, 2, figsize=(10, 4))\n",
    "\n",
    "sns.barplot(x=class_counts.index.map({0: \"No Cancer (0)\", 1: \"Cancer (1)\"}), y=class_counts.values, ax=axes[0])\n",
    "axes[0].set_title(\"Class Counts\")\n",
    "axes[0].set_ylabel(\"Count\")\n",
    "\n",
    "axes[1].pie(class_counts, labels=[\"No Cancer (0)\", \"Cancer (1)\"], autopct=\"%1.1f%%\", startangle=90)\n",
    "axes[1].set_title(\"Class Proportions\")\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "split_title",
   "metadata": {},
   "source": [
    "### Training, Validation, and Testing Data Split ###"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "27d9b910_2",
   "metadata": {},
   "source": [
    "# Hold out 20% for final test; carve 20% of train for validation (threshold tuning)\n",
    "X_train_full, X_test, y_train_full, y_test = train_test_split(\n",
    "    X, y, test_size=0.20, random_state=42, stratify=y\n",
    ")\n",
    "X_train, X_val, y_train, y_val = train_test_split(\n",
    "    X_train_full, y_train_full, test_size=0.20, random_state=42, stratify=y_train_full\n",
    ")\n",
    "\n",
    "print(f\"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}\")\n",
    "print(f\"Train class ratio \u2014 0: {(y_train == 0).sum()}, 1: {(y_train == 1).sum()}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "default_title",
   "metadata": {},
   "source": [
    "### 1. Default Decision Tree Classifier (Baseline) ###"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "default_run",
   "metadata": {},
   "outputs": [],
   "source": [
    "dt_default = DecisionTreeClassifier(random_state=42)\n",
    "dt_default.fit(X_train, y_train)\n",
    "\n",
    "print(\"--- Train Classification Report (Default) ---\")\n",
    "print(classification_report(y_train, dt_default.predict(X_train)))\n",
    "print(\"--- Test Classification Report (Default) ---\")\n",
    "print(classification_report(y_test, dt_default.predict(X_test)))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "mitigated_title",
   "metadata": {},
   "source": [
    "### 2. Class Imbalance Strategies \u2014 Model Comparison (Including XGBoost) ###"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "strategies_run",
   "metadata": {},
   "outputs": [],
   "source": [
    "scale_pos_w = (y_train == 0).sum() / (y_train == 1).sum()\n",
    "\n",
    "strategies = {\n",
    "    \"DT + class_weight='balanced'\": (\n",
    "        DecisionTreeClassifier(class_weight=\"balanced\", max_depth=8, min_samples_leaf=50, random_state=42),\n",
    "        False,\n",
    "    ),\n",
    "    \"RUS + Decision Tree\": (\n",
    "        ImbPipeline([\n",
    "            (\"rus\", RandomUnderSampler(random_state=42)),\n",
    "            (\"clf\", DecisionTreeClassifier(max_depth=8, min_samples_leaf=50, random_state=42)),\n",
    "        ]),\n",
    "        False,\n",
    "    ),\n",
    "    \"XGBoost + scale_pos_weight\": (\n",
    "        xgb.XGBClassifier(\n",
    "            scale_pos_weight=scale_pos_w,\n",
    "            max_depth=5,\n",
    "            learning_rate=0.05,\n",
    "            n_estimators=150,\n",
    "            subsample=0.8,\n",
    "            colsample_bytree=0.8,\n",
    "            random_state=42,\n",
    "            eval_metric=\"logloss\"\n",
    "        ),\n",
    "        False,\n",
    "    ),\n",
    "    \"RUS + XGBoost\": (\n",
    "        ImbPipeline([\n",
    "            (\"rus\", RandomUnderSampler(random_state=42)),\n",
    "            (\"clf\", xgb.XGBClassifier(\n",
    "                max_depth=5,\n",
    "                learning_rate=0.05,\n",
    "                n_estimators=150,\n",
    "                subsample=0.8,\n",
    "                colsample_bytree=0.8,\n",
    "                random_state=42,\n",
    "                eval_metric=\"logloss\"\n",
    "            )),\n",
    "        ]),\n",
    "        False,\n",
    "    ),\n",
    "    \"RF + class_weight='balanced'\": (\n",
    "        RandomForestClassifier(class_weight=\"balanced\", n_estimators=100, max_depth=12, min_samples_leaf=20, random_state=42),\n",
    "        False,\n",
    "    ),\n",
    "    \"Balanced Random Forest\": (\n",
    "        BalancedRandomForestClassifier(n_estimators=100, max_depth=12, min_samples_leaf=20, random_state=42),\n",
    "        False,\n",
    "    ),\n",
    "    \"GBM + sample_weight\": (\n",
    "        GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42),\n",
    "        \"sample_weight\",\n",
    "    ),\n",
    "    \"LR + class_weight='balanced'\": (\n",
    "        LogisticRegression(class_weight=\"balanced\", max_iter=1000, random_state=42),\n",
    "        False,\n",
    "    ),\n",
    "    \"SMOTE + RF (30k subsample)\": (\n",
    "        ImbPipeline([\n",
    "            (\"smote\", SMOTE(random_state=42)),\n",
    "            (\"clf\", RandomForestClassifier(n_estimators=100, max_depth=12, min_samples_leaf=20, random_state=42)),\n",
    "        ]),\n",
    "        \"smote_subsample\",\n",
    "    ),\n",
    "}\n",
    "\n",
    "comparison_rows = []\n",
    "fitted_models = {}\n",
    "\n",
    "for name, (model, mode) in strategies.items():\n",
    "    if mode == \"smote_subsample\":\n",
    "        idx = np.random.RandomState(42).choice(len(X_train), size=30_000, replace=False)\n",
    "        model.fit(X_train.iloc[idx], y_train.iloc[idx])\n",
    "    elif mode == \"sample_weight\":\n",
    "        sample_w = len(y_train) / (2 * np.bincount(y_train))\n",
    "        model.fit(X_train, y_train, sample_weight=sample_w[y_train.values])\n",
    "    else:\n",
    "        model.fit(X_train, y_train)\n",
    "\n",
    "    val_pred = model.predict(X_val)\n",
    "    test_pred = model.predict(X_test)\n",
    "    test_prob = model.predict_proba(X_test)[:, 1]\n",
    "\n",
    "    comparison_rows.append({\n",
    "        \"Strategy\": name,\n",
    "        \"Val F1 (cancer)\": f1_score(y_val, val_pred, pos_label=1),\n",
    "        \"Test F1 (cancer)\": f1_score(y_test, test_pred, pos_label=1),\n",
    "        \"Test Recall\": recall_score(y_test, test_pred),\n",
    "        \"Test Precision\": precision_score(y_test, test_pred, zero_division=0),\n",
    "        \"Test ROC-AUC\": roc_auc_score(y_test, test_prob),\n",
    "        \"Test PR-AUC\": average_precision_score(y_test, test_prob),\n",
    "    })\n",
    "    fitted_models[name] = model\n",
    "\n",
    "comparison_df = pd.DataFrame(comparison_rows).sort_values(\"Test F1 (cancer)\", ascending=False)\n",
    "comparison_df.reset_index(drop=True, inplace=True)\n",
    "comparison_df"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "3efff644",
   "metadata": {},
   "source": [
    "### 2c. Final Model Evaluation & Confusion Matrix ###\n",
    "\n",
    "The top performing model is evaluated on the holdout test set."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4615099b",
   "metadata": {},
   "outputs": [],
   "source": [
    "best_strategy = comparison_df.iloc[0][\"Strategy\"]\n",
    "best_model = fitted_models[best_strategy]\n",
    "y_pred_final = best_model.predict(X_test)\n",
    "y_prob_final = best_model.predict_proba(X_test)[:, 1]\n",
    "\n",
    "print(f\"Selected strategy: {best_strategy}\\n\")\n",
    "print(\"=\" * 55)\n",
    "print(\"FINAL TEST SET METRICS\")\n",
    "print(\"=\" * 55)\n",
    "print(f\"Accuracy:           {accuracy_score(y_test, y_pred_final):.4f}\")\n",
    "print(f\"Balanced Accuracy:  {balanced_accuracy_score(y_test, y_pred_final):.4f}\")\n",
    "print(f\"Precision (cancer): {precision_score(y_test, y_pred_final, zero_division=0):.4f}\")\n",
    "print(f\"Recall (cancer):    {recall_score(y_test, y_pred_final):.4f}\")\n",
    "print(f\"F1-Score (cancer):  {f1_score(y_test, y_pred_final, pos_label=1):.4f}\")\n",
    "print(f\"ROC-AUC:            {roc_auc_score(y_test, y_prob_final):.4f}\")\n",
    "print(f\"PR-AUC (AP):        {average_precision_score(y_test, y_prob_final):.4f}\")\n",
    "print(\"=\" * 55)\n",
    "\n",
    "print(\"\\nClassification Report:\\n\")\n",
    "print(classification_report(\n",
    "    y_test, y_pred_final,\n",
    "    target_names=[\"No Cancer (0)\", \"Cancer (1)\"],\n",
    "    zero_division=0,\n",
    "))\n",
    "\n",
    "cm = confusion_matrix(y_test, y_pred_final)\n",
    "print(\"Confusion Matrix (raw counts):\")\n",
    "print(cm)\n",
    "\n",
    "fig, ax = plt.subplots(figsize=(6, 5))\n",
    "sns.heatmap(\n",
    "    cm,\n",
    "    annot=True,\n",
    "    fmt=\"d\",\n",
    "    cmap=\"Blues\",\n",
    "    xticklabels=[\"Pred: No Cancer\", \"Pred: Cancer\"],\n",
    "    yticklabels=[\"Actual: No Cancer\", \"Actual: Cancer\"],\n",
    "    ax=ax,\n",
    ")\n",
    "ax.set_title(f\"Confusion Matrix \u2014 {best_strategy}\")\n",
    "ax.set_ylabel(\"Actual\")\n",
    "ax.set_xlabel(\"Predicted\")\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "curves_title",
   "metadata": {},
   "source": [
    "### 3. Classification Threshold Curves ###"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "curves_run",
   "metadata": {},
   "outputs": [],
   "source": [
    "y_prob = best_model.predict_proba(X_test)[:, 1]\n",
    "\n",
    "fpr, tpr, roc_thresholds = roc_curve(y_test, y_prob)\n",
    "precision, recall, pr_thresholds = precision_recall_curve(y_test, y_prob)\n",
    "\n",
    "plt.figure(figsize=(12, 5))\n",
    "plt.subplot(1, 2, 1)\n",
    "plt.plot(fpr, tpr, label=f\"ROC (AUC = {roc_auc_score(y_test, y_prob):.3f})\")\n",
    "plt.plot([0, 1], [0, 1], 'k--')\n",
    "plt.xlabel('False Positive Rate')\n",
    "plt.ylabel('True Positive Rate')\n",
    "plt.title('ROC Curve')\n",
    "plt.legend()\n",
    "\n",
    "plt.subplot(1, 2, 2)\n",
    "plt.plot(recall, precision, label=\"Precision-Recall Curve\")\n",
    "plt.xlabel('Recall')\n",
    "plt.ylabel('Precision')\n",
    "plt.title('Precision-Recall Curve')\n",
    "plt.legend()\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "tuning_title",
   "metadata": {},
   "source": [
    "### 4. Tuning Decision Threshold on Validation Set for F1-Score ###"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "tuning_run",
   "metadata": {},
   "outputs": [],
   "source": [
    "val_prob = best_model.predict_proba(X_val)[:, 1]\n",
    "best_threshold = 0.5\n",
    "best_f1_val = 0\n",
    "for threshold in np.linspace(0.01, 0.99, 100):\n",
    "    preds_v = (val_prob >= threshold).astype(int)\n",
    "    report_v = classification_report(y_val, preds_v, output_dict=True, zero_division=0)\n",
    "    f1_v = report_v['1']['f1-score']\n",
    "    if f1_v > best_f1_val:\n",
    "        best_f1_val = f1_v\n",
    "        best_threshold = threshold\n",
    "\n",
    "print(f\"Validation-Tuned Optimal Threshold for F1-Score: {best_threshold:.3f}\")\n",
    "print(f\"Validation F1-score achieved: {best_f1_val:.3f}\")\n",
    "\n",
    "y_pred_tuned = (y_prob >= best_threshold).astype(int)\n",
    "print(\"\\n--- Test Classification Report with Validation-Tuned Threshold ---\")\n",
    "print(classification_report(y_test, y_pred_tuned, zero_division=0))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "importance_title",
   "metadata": {},
   "source": [
    "### 5. Feature Importance ###"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "importance_run",
   "metadata": {},
   "outputs": [],
   "source": [
    "if hasattr(best_model, \"feature_importances_\"):\n",
    "    importances = best_model.feature_importances_\n",
    "elif hasattr(best_model, \"named_steps\") and hasattr(best_model.named_steps[\"clf\"], \"feature_importances_\"):\n",
    "    importances = best_model.named_steps[\"clf\"].feature_importances_\n",
    "else:\n",
    "    importances = np.zeros(X.shape[1])\n",
    "\n",
    "importance_df = pd.DataFrame({\n",
    "    \"Feature\": X.columns,\n",
    "    \"Importance\": importances\n",
    "}).sort_values(\"Importance\", ascending=False)\n",
    "\n",
    "plt.figure(figsize=(10, 6))\n",
    "sns.barplot(x=\"Importance\", y=\"Feature\", data=importance_df.head(15))\n",
    "plt.title(f\"Top 15 Feature Importances ({best_strategy})\")\n",
    "plt.show()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": ".venv (3.13.5)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}