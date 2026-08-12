import os
import sys
import subprocess
import streamlit as st

# Auto-launch with 'streamlit run' if executed directly with python3
if not st.runtime.exists():
    print("\n" + "="*60)
    print("WARNING: Running dashboard.py directly with python3 is not supported.")
    print("Relaunching automatically with 'streamlit run'...")
    print("="*60 + "\n")
    try:
        subprocess.run(["streamlit", "run", sys.argv[0]] + sys.argv[1:])
    except Exception as e:
        print(f"Auto-launch failed: {e}")
        print("Please run manually: streamlit run dashboard.py")
    sys.exit(0)

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.decomposition import PCA



# Set Streamlit Page Configuration
st.set_page_config(
    page_title="Gastric Cancer Diagnostics Dashboard",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for Dark & Modern Theme Look
st.markdown("""
<style>
    .main {
        background-color: #0f1116;
        color: #e2e8f0;
    }
    .stApp {
        background-color: #0f1116;
    }
    .metric-card {
        background-color: #1e293b;
        border-radius: 10px;
        padding: 20px;
        border: 1px solid #334155;
        text-align: center;
        margin-bottom: 20px;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: bold;
        color: #38bdf8;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }
    .card-header {
        font-size: 1.25rem;
        font-weight: bold;
        color: #f8fafc;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# Caching Data Loading Functions
# ============================================================
@st.cache_data
def load_cleaned_data():
    """Loads cleaned dataset for general visualization."""
    data_path = "/Users/Sanskar/Documents/gastric_cancer_prediction/Dataset/gastric_cancer_cleaned.csv"
    if not os.path.exists(data_path):
        return None
    return pd.read_csv(data_path)


@st.cache_data
def load_clustered_data():
    """Loads clustered dataset for PCA and cluster analysis."""
    data_path = "/Users/Sanskar/Documents/gastric_cancer_prediction/Dataset/gastric_cancer_clustered.csv"
    if not os.path.exists(data_path):
        return None
    return pd.read_csv(data_path)


# Load datasets
df_cleaned = load_cleaned_data()
df_clustered = load_clustered_data()


# ============================================================
# Page 1: Dataset Overview
# ============================================================
def render_overview():
    st.title("🧬 Gastric Cancer Dataset Overview")
    st.markdown("Explore high-level statistical summaries and patient demographics from our cleaned detection database.")
    
    if df_cleaned is None:
        st.error("Cleaned dataset not found. Please verify `Dataset/gastric_cancer_cleaned.csv` exists.")
        return
        
    # KPIs Row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(df_cleaned):,}</div>
            <div class="metric-label">Total Patients</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        cancer_rate = df_cleaned['label'].mean() * 100
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{cancer_rate:.2f}%</div>
            <div class="metric-label">Cancer Incidence Rate</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        median_age = df_cleaned['age'].median()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{median_age:.0f}</div>
            <div class="metric-label">Median Age (Years)</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        hpylori_rate = df_cleaned['helicobacter_pylori_infection'].mean() * 100
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{hpylori_rate:.1f}%</div>
            <div class="metric-label">H. Pylori Infection Rate</div>
        </div>
        """, unsafe_allow_html=True)
        
    # Demographics and Clinical plots
    left_col, right_col = st.columns(2)
    
    with left_col:
        st.markdown('<div class="card-header">Age Distribution by Cancer Status</div>', unsafe_allow_html=True)
        # Create interactive Plotly histogram
        # Map label to readable categories for plot legend
        df_plot = df_cleaned.copy()
        df_plot['Cancer Status'] = df_plot['label'].map({0: "No Cancer (0)", 1: "Cancer (1)"})
        
        fig = px.histogram(
            df_plot, x="age", color="Cancer Status",
            barmode="overlay", marginal="box",
            color_discrete_map={"No Cancer (0)": "#38bdf8", "Cancer (1)": "#ef4444"},
            labels={"age": "Age (Years)", "count": "Number of Patients"},
            template="plotly_dark"
        )
        fig.update_layout(
            plot_bgcolor="#1e293b",
            paper_bgcolor="#0f1116",
            font_color="#e2e8f0",
            margin=dict(l=40, r=40, t=40, b=40)
        )
        st.plotly_chart(fig, width="stretch")
        
    with right_col:
        st.markdown('<div class="card-header">Clinical Risks Breakdown</div>', unsafe_allow_html=True)
        
        # Categorical columns summary
        risk_factors = {
            "Gender": df_cleaned['gender'].map({1: "Male", 0: "Female"}),
            "Dietary Habit": df_cleaned['dietary_habits'].map({1: "High Salt", 0: "Low Salt"}),
            "Smoking habits": df_cleaned['smoking_habits'].map({1: "Smoker", 0: "Non-Smoker"}),
            "Alcohol usage": df_cleaned['alcohol_consumption'].map({1: "Consumer", 0: "Non-Consumer"})
        }
        
        selected_risk = st.selectbox("Select risk factor to display distribution:", list(risk_factors.keys()))
        
        temp_df = pd.DataFrame({
            "Factor": risk_factors[selected_risk], 
            "Cancer Status": df_cleaned['label'].map({1: "Cancer", 0: "No Cancer"})
        })
        
        # Plot interactive grouped histogram
        fig = px.histogram(
            temp_df, x="Factor", color="Cancer Status",
            barmode="group",
            color_discrete_map={"No Cancer": "#38bdf8", "Cancer": "#ef4444"},
            labels={"Factor": selected_risk, "count": "Patient Count"},
            template="plotly_dark"
        )
        fig.update_layout(
            plot_bgcolor="#1e293b",
            paper_bgcolor="#0f1116",
            font_color="#e2e8f0",
            margin=dict(l=40, r=40, t=40, b=40)
        )
        st.plotly_chart(fig, width="stretch")


# ============================================================
# Page 2: Feature Distributions
# ============================================================
def render_features():
    st.title("📊 Continuous miRNA Scores & Correlations")
    st.markdown("Explore numerical distributions and linear relationships of continuous variables in the dataset.")
    
    if df_cleaned is None:
        st.error("Cleaned dataset not found.")
        return
        
    mirna_cols = ["diana_microt", "elmmo", "microcosm", "miranda", "mirdb", "pictar", "pita", "targetscan"]
    
    # Left Column: Individual Distribution
    # Right Column: Correlation Matrix
    left_col, right_col = st.columns(2)
    
    with left_col:
        st.markdown('<div class="card-header">miRNA Database Scores Distribution</div>', unsafe_allow_html=True)
        selected_mirna = st.selectbox("Select miRNA tool score column:", mirna_cols)
        
        # Plot interactive Plotly histogram
        fig = px.histogram(
            df_cleaned, x=selected_mirna,
            nbins=30,
            color_discrete_sequence=["#38bdf8"],
            labels={selected_mirna: f"{selected_mirna} Score"},
            template="plotly_dark"
        )
        fig.update_layout(
            plot_bgcolor="#1e293b",
            paper_bgcolor="#0f1116",
            font_color="#e2e8f0",
            margin=dict(l=40, r=40, t=40, b=40)
        )
        st.plotly_chart(fig, width="stretch")
        
        st.info("💡 **Notice:** The miRNA prediction scores exhibit a perfectly flat, uniform distribution in the range [0, 1]. This is highly characteristic of synthetic databases and directly impacts density-based clustering.")
        
    with right_col:
        st.markdown('<div class="card-header">Feature Correlation Heatmap</div>', unsafe_allow_html=True)
        
        # Calculate correlation matrix
        corr_cols = ["age"] + mirna_cols + ["predicted.sum", "all.sum", "label"]
        corr_matrix = df_cleaned[corr_cols].corr()
        
        # Create interactive Plotly heatmap (imshow)
        fig = px.imshow(
            corr_matrix,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            zmin=-1.0, zmax=1.0,
            aspect="auto",
            template="plotly_dark"
        )
        fig.update_layout(
            plot_bgcolor="#1e293b",
            paper_bgcolor="#0f1116",
            font_color="#e2e8f0",
            margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig, width="stretch")


# ============================================================
# Page 3: Clustering Explorer
# ============================================================
def render_clustering():
    st.title("🧮 DBSCAN Clustering Explorer")
    st.markdown("Density-based clustering partitions multi-dimensional patient profiles into subgroups.")
    
    if df_clustered is None:
        st.warning("Clustered dataset (gastric_cancer_clustered.csv) not found. Run DBSCAN script to output this file, or view the pre-computed PCA projection below.")
        # Load a default plot if present in the artifact folder
        artifact_plot_path = "/Users/Sanskar/.gemini/antigravity-ide/brain/7a0f58ec-8a55-4195-8085-8ca6770f5311/clustering_plot.png"
        if os.path.exists(artifact_plot_path):
            st.image(artifact_plot_path, caption="Pre-computed 2D PCA Projection of DBSCAN Clusters")
        return
        
    left_col, right_col = st.columns([3, 2])
    
    with left_col:
        st.markdown('<div class="card-header">2D PCA Cluster Projection</div>', unsafe_allow_html=True)
        
        # Sample for plotting speed
        df_sample = df_clustered.sample(n=min(10000, len(df_clustered)), random_state=42).copy()
        
        # Extract features and fit PCA
        mirna_cols = ["diana_microt", "elmmo", "microcosm", "miranda", "mirdb", "pictar", "pita", "targetscan"]
        continuous_cols = ["age"] + mirna_cols + ["predicted.sum", "all.sum"]
        continuous_cols = [c for c in continuous_cols if c in df_sample.columns]
        
        from sklearn.preprocessing import StandardScaler
        X_scaled = StandardScaler().fit_transform(df_sample[continuous_cols])
        
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        df_sample['PCA1'] = X_pca[:, 0]
        df_sample['PCA2'] = X_pca[:, 1]
        
        # Plot PCA colored by cluster using Plotly
        df_sample['Cluster ID'] = df_sample['dbscan_cluster'].astype(str)
        
        # Sort values so noise (-1) is plotted behind clusters
        df_sample = df_sample.sort_values(by='dbscan_cluster', ascending=True)
        
        fig = px.scatter(
            df_sample, x='PCA1', y='PCA2', color='Cluster ID',
            hover_data=['age', 'dbscan_cluster', 'label'],
            opacity=0.7,
            labels={"PCA1": "PCA Component 1", "PCA2": "PCA Component 2"},
            template="plotly_dark",
            color_discrete_sequence=px.colors.qualitative.Alphabet
        )
        fig.update_traces(marker=dict(size=5))
        fig.update_layout(
            plot_bgcolor="#1e293b",
            paper_bgcolor="#0f1116",
            font_color="#e2e8f0",
            margin=dict(l=40, r=40, t=40, b=40)
        )
        st.plotly_chart(fig, width="stretch")
        
    with right_col:
        st.markdown('<div class="card-header">Cluster Incidence breakdown</div>', unsafe_allow_html=True)
        
        # Display cluster counts and target rates
        cluster_summary = df_clustered.groupby("dbscan_cluster")["label"].agg(["count", "mean"])
        cluster_summary.columns = ["Size", "Cancer Rate"]
        cluster_summary["Incidence %"] = cluster_summary["Cancer Rate"] * 100
        cluster_summary.index = [f"Cluster {i}" if i != -1 else "Noise (-1)" for i in cluster_summary.index]
        
        st.dataframe(
            cluster_summary[["Size", "Incidence %"]].style.format({"Incidence %": "{:.2f}%"}),
            use_container_width=True
        )
        
        st.markdown("""
        **Clustering Diagnostics:**
        - **Noise (-1):** Contains outliers in low-density space.
        - **Cluster 0:** The main density subgroup.
        - **Uniform Space:** Because the continuous inputs are uncorrelated, density is relatively flat, and the algorithm splits the hypersphere into one massive cluster and sparse noise points.
        """)


# ============================================================
# Page 4: Diagnostic Predictor
# ============================================================
def render_predictor():
    st.title("🔮 Interactive Diagnostic Predictor")
    st.markdown("Input patient demographics, clinical flags, and miRNA prediction scores to predict Gastric Cancer probability.")
    
    # Paths for model loading
    scaler_path = "/Users/Sanskar/Documents/gastric_cancer_prediction/models/saved/scaler.joblib"
    model_bal_path = "/Users/Sanskar/Documents/gastric_cancer_prediction/models/saved/logistic_regression_balanced.joblib"
    model_std_path = "/Users/Sanskar/Documents/gastric_cancer_prediction/models/saved/logistic_regression_standard.joblib"
    
    # Verify models exist
    if not (os.path.exists(scaler_path) and os.path.exists(model_bal_path)):
        st.error("Trained classification models or scaler not found. Please run the training script: `python models/train_logistic_regression.py` first to generate them.")
        return
        
    # Load model and scaler
    scaler = joblib.load(scaler_path)
    model_bal = joblib.load(model_bal_path)
    model_std = joblib.load(model_std_path)
    
    # Clinical Usage Note
    st.info("ℹ️ **Practical Usage Note:** In real-world clinical testing, obtaining exact binding scores from all 8 miRNA prediction databases simultaneously is often impossible or impractical. The sliders below are pre-set to a neutral default of `0.5`. If specific database scores are unknown, they can be safely left at their defaults, allowing you to predict based on the demographic, lifestyle, and lab imaging parameters.")
    
    # Create tabs to organize inputs cleanly
    tab_demo, tab_lab, tab_mirna = st.tabs([
        "📋 Demographics & Lifestyle", 
        "🔬 Clinical Findings & Gene Markers", 
        "🧬 miRNA Scores & Model Choice"
    ])
    
    with tab_demo:
        col1, col2 = st.columns(2)
        with col1:
            age = st.slider("Patient Age (Years):", 18, 90, 50)
            gender = st.selectbox("Patient Gender:", ["Male", "Female"])
            family_history = st.selectbox("Family History of Gastric Cancer:", ["No", "Yes"])
        with col2:
            smoking = st.selectbox("Smoking Habits:", ["Non-Smoker", "Smoker"])
            alcohol = st.selectbox("Alcohol Consumption:", ["No / Occasional", "Regular Consumer"])
            diet = st.selectbox("Dietary Habits:", ["Low Salt Diet", "High Salt Diet"])
            existing_cond = st.selectbox("Select Existing Condition:", ["Diabetes", "None", "Other (e.g., Hypertension)"])
            
    with tab_lab:
        col1, col2 = st.columns(2)
        with col1:
            hpylori = st.selectbox("Helicobacter Pylori Infection:", ["Negative", "Positive"])
            endoscopy = st.selectbox("Endoscopic Imaging:", ["Normal Findings", "Abnormal Findings"])
            biopsy = st.selectbox("Biopsy Result:", ["Negative", "Positive"])
        with col2:
            ct_scan = st.selectbox("CT Scan Finding:", ["Negative", "Positive"])
            mature_mirna = st.selectbox("Select Mature miRNA Target ID:", ["MIR234_2", "MIR345_3", "Other / None"])
            target_symbol = st.selectbox("Select target Genic Marker:", ["KRAS", "TP53", "Other (e.g. CDH1)"])
            
    with tab_mirna:
        col1, col2 = st.columns(2)
        with col1:
            diana_microt = st.slider("DIANA-microT Score:", 0.0, 1.0, 0.5, 0.05)
            elmmo = st.slider("ElMMo Score:", 0.0, 1.0, 0.5, 0.05)
            microcosm = st.slider("Microcosm Score:", 0.0, 1.0, 0.5, 0.05)
            miranda = st.slider("Miranda Score:", 0.0, 1.0, 0.5, 0.05)
            mirdb = st.slider("miRDB Score:", 0.0, 1.0, 0.5, 0.05)
        with col2:
            pictar = st.slider("PicTar Score:", 0.0, 1.0, 0.5, 0.05)
            pita = st.slider("PITA Score:", 0.0, 1.0, 0.5, 0.05)
            targetscan = st.slider("TargetScan Score:", 0.0, 1.0, 0.5, 0.05)
            predicted_sum = st.slider("Predicted miRNA Binding Sum:", 0.0, 10.0, 5.0, 0.5)
            all_sum = st.slider("All miRNA Binding Sum:", 0.0, 10.0, 5.0, 0.5)
            
        model_choice = st.radio(
            "Select Logistic Regression Variant to Predict Risk:",
            ["Class-Balanced Model (Weighted - Recommended for Diagnostics)", "Standard Model (Unweighted - Biased towards Negative class)"]
        )
        
    # Convert input values to model feature format
    # Map raw entries to binaries
    gender_val = 1 if gender == "Male" else 0
    fam_hist_val = 1 if family_history == "Yes" else 0
    smoking_val = 1 if smoking == "Smoker" else 0
    alcohol_val = 1 if alcohol == "Regular Consumer" else 0
    hpylori_val = 1 if hpylori == "Positive" else 0
    diet_val = 1 if diet == "High Salt Diet" else 0
    endoscopy_val = 1 if endoscopy == "Abnormal Findings" else 0
    biopsy_val = 1 if biopsy == "Positive" else 0
    ct_scan_val = 1 if ct_scan == "Positive" else 0
    
    # One-hot categorical mappings
    mature_mirna_MIR234 = 1 if mature_mirna == "MIR234_2" else 0
    mature_mirna_MIR345 = 1 if mature_mirna == "MIR345_3" else 0
    
    existing_cond_Diabetes = 1 if existing_cond == "Diabetes" else 0
    existing_cond_None = 1 if existing_cond == "None" else 0
    
    target_symbol_KRAS = 1 if target_symbol == "KRAS" else 0
    target_symbol_TP53 = 1 if target_symbol == "TP53" else 0
    
    # Compute engineered features
    age_group_35_to_50 = 1 if (35 <= age < 50) else 0
    age_group_50_to_65 = 1 if (50 <= age < 65) else 0
    age_group_65_plus = 1 if (age >= 65) else 0
    
    smoking_alcohol_interaction = smoking_val * alcohol_val
    hpylori_salt_interaction = hpylori_val * diet_val
    family_history_age = fam_hist_val * age
    
    mirna_list = [diana_microt, elmmo, microcosm, miranda, mirdb, pictar, pita, targetscan]
    mirna_mean_score = np.mean(mirna_list)
    mirna_std_score = np.std(mirna_list)
    mirna_max_score = np.max(mirna_list)
    mirna_min_score = np.min(mirna_list)
    mirna_consensus_count = np.sum([x > 0.70 for x in mirna_list])
    
    # Construct complete input vector (in the exact trained feature_order)
    input_vector = [
        age, gender_val, fam_hist_val, smoking_val, alcohol_val,
        hpylori_val, diet_val, endoscopy_val, biopsy_val, ct_scan_val,
        diana_microt, elmmo, microcosm, miranda, mirdb, pictar, pita, targetscan,
        predicted_sum, all_sum,
        mature_mirna_MIR234, mature_mirna_MIR345,
        existing_cond_Diabetes, existing_cond_None,
        target_symbol_KRAS, target_symbol_TP53,
        age_group_35_to_50, age_group_50_to_65, age_group_65_plus,
        smoking_alcohol_interaction, hpylori_salt_interaction,
        family_history_age, mirna_mean_score, mirna_std_score,
        mirna_max_score, mirna_min_score, mirna_consensus_count
    ]
    
    # Predict button
    st.markdown("---")
    if st.button("🚀 Calculate Patient Diagnostic Risk"):
        # Scale the inputs
        input_array = np.array([input_vector])
        try:
            # Map input array according to the scaler
            input_scaled = scaler.transform(input_array)
            
            # Run prediction
            selected_model = model_bal if "Balanced" in model_choice else model_std
            prob = selected_model.predict_proba(input_scaled)[0, 1]
            pred_class = selected_model.predict(input_scaled)[0]
            
            # Display result
            st.subheader("📊 Diagnostic Summary")
            
            col_res1, col_res2 = st.columns(2)
            with col_res1:
                st.markdown(f"**Predicted Gastric Cancer Risk Probability:**")
                st.markdown(f"<div style='font-size: 3rem; font-weight: bold; color: {'#ef4444' if prob > 0.50 else '#10b981'};'>{prob*100:.2f}%</div>", unsafe_allow_html=True)
                
                # Progress Bar
                st.progress(float(prob))
                
            with col_res2:
                st.markdown(f"**Diagnostic Decision:**")
                if pred_class == 1:
                    st.markdown("<div style='background-color: #fef2f2; color: #ef4444; border-left: 6px solid #ef4444; padding: 15px; border-radius: 5px;'>⚠️ <b>High Diagnostic Risk:</b> Patient characteristics strongly match the cancer-positive cohort profile. Further clinical assessment recommended.</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div style='background-color: #f0fdf4; color: #16a34a; border-left: 6px solid #16a34a; padding: 15px; border-radius: 5px;'>✓ <b>Low Diagnostic Risk:</b> Patient characteristics align with the baseline control/negative cohort profile.</div>", unsafe_allow_html=True)
                    
        except Exception as e:
            st.error(f"Error executing prediction: {e}")


# ============================================================
# Main Page Navigation Router
# ============================================================
def main():
    st.sidebar.markdown("""
    <div style='text-align: center; margin-bottom: 20px;'>
        <h2 style='color: #38bdf8;'>🧬 Diagnostic App</h2>
        <p style='color: #94a3b8; font-size: 0.8rem;'>Gastric Cancer Analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    page = st.sidebar.radio(
        "Select Dashboard Page:",
        ["Dataset Overview", "Feature Distributions", "Clustering Explorer", "Diagnostic Predictor"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    **Developer Diagnostics Console**
    - Platform: macOS
    - Python environment: uv venv
    - Classification: Logistic Regression
    - Clustering: DBSCAN / HDBSCAN
    """)
    
    if page == "Dataset Overview":
        render_overview()
    elif page == "Feature Distributions":
        render_features()
    elif page == "Clustering Explorer":
        render_clustering()
    elif page == "Diagnostic Predictor":
        render_predictor()


if __name__ == "__main__":
    main()
