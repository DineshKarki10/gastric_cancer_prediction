import os
import sys
import joblib
import pandas as pd
import streamlit as st

# Ensure src is importable
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "xgboost_model.pkl")
COLUMNS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "feature_columns.pkl")


@st.cache_resource
def load_model_and_columns():
    model = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(COLUMNS_PATH)
    return model, feature_columns


def main():
    st.set_page_config(page_title="Gastric Cancer Prediction", page_icon="🏥", layout="centered")
    st.title("Gastric Cancer Prediction")
    st.markdown("This application predicts the likelihood of gastric cancer based on patient data using XGBoost.")

    if not os.path.exists(MODEL_PATH) or not os.path.exists(COLUMNS_PATH):
        st.error("Model files not found. Please run `python src/model_training.py` first to train and save the model.")
        st.stop()

    model, feature_columns = load_model_and_columns()

    st.header("Patient Information")

    input_data = {}

    col1, col2, col3 = st.columns(3)
    with col1:
        input_data["age"] = st.number_input("Age", min_value=0, max_value=120, value=45)
        input_data["gender"] = st.selectbox("Gender", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
        input_data["family_history"] = st.selectbox("Family History", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        input_data["smoking_habits"] = st.selectbox("Smoking Habits", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        input_data["alcohol_consumption"] = st.selectbox("Alcohol Consumption", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        input_data["helicobacter_pylori_infection"] = st.selectbox("Helicobacter Pylori Infection", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        input_data["dietary_habits"] = st.selectbox("Dietary Habits", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

    with col2:
        input_data["endoscopic_images"] = st.selectbox("Endoscopic Images", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        input_data["biopsy_results"] = st.selectbox("Biopsy Results", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        input_data["ct_scan"] = st.selectbox("CT Scan", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        input_data["diana_microt"] = st.number_input("Diana MicroT", value=0.5, format="%.4f")
        input_data["elmmo"] = st.number_input("ElMMo", value=0.5, format="%.4f")
        input_data["microcosm"] = st.number_input("MicroCosm", value=0.5, format="%.4f")
        input_data["miranda"] = st.number_input("Miranda", value=0.5, format="%.4f")

    with col3:
        input_data["mirdb"] = st.number_input("miRDB", value=0.5, format="%.4f")
        input_data["pictar"] = st.number_input("PicTar", value=0.5, format="%.4f")
        input_data["pita"] = st.number_input("PITA", value=0.5, format="%.4f")
        input_data["targetscan"] = st.number_input("TargetScan", value=0.5, format="%.4f")
        input_data["predicted.sum"] = st.number_input("Predicted Sum", value=5.0, format="%.4f")
        input_data["all.sum"] = st.number_input("All Sum", value=5.0, format="%.4f")
        input_data["existing_conditions_Diabetes"] = st.selectbox("Existing Conditions: Diabetes", [False, True])
        input_data["existing_conditions_None"] = st.selectbox("Existing Conditions: None", [False, True])
        input_data["target_symbol_KRAS"] = st.selectbox("Target Symbol: KRAS", [False, True])
        input_data["target_symbol_TP53"] = st.selectbox("Target Symbol: TP53", [False, True])

    if st.button("Predict", type="primary"):
        input_df = pd.DataFrame([input_data])

        for col in feature_columns:
            if col not in input_df.columns:
                input_df[col] = 0

        input_df = input_df[feature_columns]

        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        st.divider()
        st.subheader("Prediction Result")

        if prediction == 1:
            st.error(f"**High Risk of Gastric Cancer**")
        else:
            st.success(f"**Low Risk of Gastric Cancer**")

        st.metric(label="Cancer Probability", value=f"{probability:.2%}")
        st.metric(label="Predicted Class", value="Cancer" if prediction == 1 else "No Cancer")


if __name__ == "__main__":
    main()
