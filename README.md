# Gastric Cancer Prediction

Machine learning project that predicts the likelihood of gastric cancer using patient clinical data and microRNA target prediction scores. The project includes data preprocessing, exploratory data analysis, XGBoost model training, and an interactive Streamlit web application for real-time risk prediction.

## Features

- End-to-end pipeline from raw data to deployment-ready model
- Handles missing values, removes non-predictive identifiers, and encodes categorical features
- XGBoost classifier optimized for imbalanced binary classification
- Model evaluation with accuracy, precision, recall, F1 score, ROC AUC, and confusion matrix
- Interactive Streamlit app for single-patient prediction with probability output

## Project Structure

```
gastric_cancer_prediction/
├── data/
│   ├── raw_data.csv
│   └── final_data.csv
├── notebooks/
│   └── eda_analysis.ipynb
├── src/
│   ├── __init__.py
│   ├── model_training.py
│   ├── data_preprocessing.py
│   └── utils.py
├── app/
│   └── streamlit_app.py
├── models/
│   ├── xgboost_model.pkl
│   └── feature_columns.pkl
├── requirements.txt
└── README.md
```

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/<your-username>/gastric_cancer_prediction.git
cd gastric_cancer_prediction
pip install -r requirements.txt
```

## Usage

### Train the Model

```bash
python src/model_training.py
```

The script will:
- Load the processed dataset from `data/final_data.csv`
- Split the data into training and test sets (80/20 stratified)
- Train an XGBoost classifier with class-weight balancing
- Print evaluation metrics
- Save the trained model and feature schema to the `models/` directory

### Run the Streamlit App

```bash
streamlit run app/streamlit_app.py
```

The web app will open in your browser at `http://localhost:8501`. Enter patient information across demographic, lifestyle, clinical, and microRNA target prediction fields, then click **Predict** to see the cancer risk classification and probability.

## Model Details

- **Algorithm**: XGBoost (`XGBClassifier`)
- **Task**: Binary classification (0 = No Cancer, 1 = Cancer)
- **Class balancing**: `scale_pos_weight` applied to address the ~9:1 class imbalance
- **Features**: 24 columns including age, gender, family history, lifestyle factors, clinical markers, and microRNA target prediction scores (miRDB, TargetScan, PicTar, etc.)
- **Target variable**: `label`

### Evaluation Metrics

Typical results on the held-out test set:

| Metric | Value |
|--------|-------|
| Accuracy | ~0.90 (baseline) |
| Precision | ~0.10 |
| Recall | ~0.30 |
| F1 Score | ~0.15 |
| ROC AUC | ~0.50 |

Note: Because the dataset is heavily imbalanced, accuracy alone is misleading. The model is tuned to improve recall on the minority (cancer) class while controlling false positives.

## Data Pipeline

1. **Raw data ingestion** from `data/raw_data.csv`
2. **Notebook preprocessing** (`notebooks/eda_analysis.ipynb`):
   - Remove duplicate rows
   - Fill missing `existing_conditions` with `"None"`
   - Drop non-predictive identifiers (`ethnicity`, `geographical_location`, microRNA IDs, Entrez/Ensembl target IDs)
   - One-hot encode multi-category categorical columns
   - Label-encode binary categorical columns
3. **Export** to `data/final_data.csv`
4. **Model training** reads the final CSV directly (no additional preprocessing needed)

## Input Features

The Streamlit app accepts the following patient data:

- **Demographics**: age, gender
- **Lifestyle**: family history, smoking habits, alcohol consumption, dietary habits
- **Clinical**: helicobacter pylori infection, endoscopic images, biopsy results, CT scan
- **Existing conditions**: diabetes, none
- **MicroRNA predictions**: DIANA-microT, ElMMo, MicroCosm, Miranda, miRDB, PicTar, PITA, TargetScan
- **Aggregated scores**: predicted.sum, all.sum
- **Gene targets**: KRAS, TP53

## Future Improvements

- Hyperparameter tuning with cross-validation (GridSearchCV / Optuna)
- Try additional algorithms (Random Forest, LightGBM, CatBoost) and ensemble methods
- Add SHAP/LIME explanations for model interpretability
- Persist preprocessing pipeline (ColumnTransformer / Pipeline) to ensure consistent encoding at inference
- Collect or augment data to better represent the minority class
- Add user-friendly input forms with validation and default ranges
- Deploy to cloud (Streamlit Community, Hugging Face Spaces, or AWS)

## License

This project is intended for educational and research purposes.
