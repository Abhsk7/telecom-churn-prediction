"""
Telecom Customer Churn Prediction Dashboard
Theme: Classic Burgundy & Beige | High Contrast Black Text & Game of Thrones (Cinzel)
Verveox Technologies AI/ML Internship - Project 5
"""

import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import tensorflow as tf

# Configure Streamlit page
st.set_page_config(
    page_title="Telecom Churn Prediction System",
    page_icon="👑",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply High-Contrast Classic Burgundy & Beige + Solid Black Text
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@500;700;900&family=Cinzel+Decorative:wght@700;900&family=Cormorant+Garamond:wght@400;600;700&family=Special+Elite&display=swap');

/* Main Canvas - Warm Beige */
.stApp {
    background-color: #F7F3EB !important;
    color: #000000 !important;
    font-family: 'Cormorant Garamond', Georgia, serif !important;
}

/* Headings */
h1, h2, h3, h4, h5, h6, .stTitle {
    font-family: 'Cinzel', serif !important;
    color: #4A0E17 !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    font-weight: 900 !important;
}

/* General Body Paragraphs & Descriptions - Solid Black */
p, li {
    color: #111111 !important;
    font-family: 'Cormorant Garamond', Georgia, serif !important;
    font-size: 1.15rem !important;
}

/* Form Input Labels - Crisp Bold Black */
label, .stSelectbox label, .stSlider label, .stNumberInput label {
    font-family: 'Cinzel', serif !important;
    font-weight: 800 !important;
    color: #000000 !important;
    letter-spacing: 1px !important;
    font-size: 1rem !important;
}

/* ==========================================
   NAVIGATION TABS - HIGH CONTRAST FIX
   ========================================== */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    border-bottom: 2.5px solid #4A0E17 !important;
    background-color: transparent !important;
    padding-bottom: 2px;
}

/* INACTIVE TABS - SOLID BOLD BLACK TEXT */
.stTabs [data-baseweb="tab"] {
    background-color: #E5DAC8 !important;
    border: 1.5px solid #C4A47C !important;
    border-bottom: none !important;
    border-radius: 6px 6px 0 0 !important;
    padding: 10px 22px !important;
    transition: all 0.2s ease !important;
}
.stTabs [data-baseweb="tab"] p, 
.stTabs [data-baseweb="tab"] span, 
.stTabs [data-baseweb="tab"] div {
    font-family: 'Cinzel', serif !important;
    font-size: 1.05rem !important;
    font-weight: 900 !important;
    color: #000000 !important; /* FIXED: Solid black for clear visibility */
    letter-spacing: 1px !important;
}

/* ACTIVE TAB - DEEP BURGUNDY WITH WHITE TEXT */
.stTabs [aria-selected="true"] {
    background-color: #540D1A !important;
    border: 2px solid #540D1A !important;
    border-bottom: none !important;
}
.stTabs [aria-selected="true"] p, 
.stTabs [aria-selected="true"] span, 
.stTabs [aria-selected="true"] div {
    color: #FFFFFF !important; /* Crisp white on active burgundy */
    font-weight: 900 !important;
}

/* Sidebar - Deep Imperial Burgundy */
[data-testid="stSidebar"] {
    background-color: #3B0A11 !important;
    border-right: 2px solid #C4A47C !important;
}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    color: #EBD9B4 !important;
    font-family: 'Cinzel', serif !important;
}
[data-testid="stSidebar"] p, [data-testid="stSidebar"] label, [data-testid="stSidebar"] span, [data-testid="stSidebar"] div {
    color: #F7F3EB !important;
    font-family: 'Cormorant Garamond', Georgia, serif !important;
    font-size: 1.15rem !important;
}

/* Regal Burgundy & Gold Button with Bright White Text */
.stButton > button {
    background: linear-gradient(135deg, #6B1D2F 0%, #440A12 100%) !important;
    border: 2px solid #C4A47C !important;
    border-radius: 4px !important;
    box-shadow: 0 4px 12px rgba(62, 10, 17, 0.35) !important;
    padding: 12px 24px !important;
    transition: all 0.3s ease !important;
}
.stButton > button, 
.stButton > button p, 
.stButton > button span, 
.stButton > button div {
    font-family: 'Cinzel', serif !important;
    font-weight: 800 !important;
    font-size: 1.1rem !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    color: #FFFFFF !important; /* FIXED: Crisp white button text */
}
.stButton > button:hover {
    background: linear-gradient(135deg, #800020 0%, #5A111E 100%) !important;
    border-color: #E2C799 !important;
    transform: translateY(-2px);
    box-shadow: 0 6px 18px rgba(62, 10, 17, 0.5) !important;
}

/* Tables */
table {
    border: 2px solid #C4A47C !important;
    border-collapse: collapse !important;
    width: 100% !important;
    background-color: #FFFDF9 !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05) !important;
    margin: 15px 0 !important;
}
th {
    background-color: #4A0E17 !important;
    color: #F7F3EB !important;
    font-family: 'Cinzel', serif !important;
    font-size: 1rem !important;
    letter-spacing: 1px !important;
    padding: 12px !important;
    text-align: center !important;
    border: 1px solid #C4A47C !important;
}
td {
    color: #000000 !important;
    font-family: 'Cormorant Garamond', Georgia, serif !important;
    font-size: 1.25rem !important;
    font-weight: 700 !important;
    padding: 10px !important;
    border: 1px solid #E5D6C0 !important;
    text-align: center !important;
}

/* ==========================================
   HERO BOX - GLOWING GOLD & WARM CREAM TEXT
   ========================================== */
.hero-box {
    background: linear-gradient(135deg, #4A0E17 0%, #2A050B 100%) !important;
    border: 2px solid #C4A47C !important;
    border-radius: 6px !important;
    padding: 24px 30px !important;
    margin-bottom: 25px !important;
    box-shadow: 0 6px 18px rgba(59, 10, 17, 0.25) !important;
}
.hero-box .hero-title, .hero-title {
    font-family: 'Cinzel Decorative', 'Cinzel', serif !important;
    font-size: 2.3rem !important;
    font-weight: 900 !important;
    color: #EBD9B4 !important; /* Elegant Royal Gold */
    letter-spacing: 2px !important;
    margin-bottom: 6px !important;
}
.hero-box .hero-subtitle, .hero-subtitle {
    font-family: 'Cormorant Garamond', Georgia, serif !important;
    font-size: 1.35rem !important;
    color: #F7F3EB !important; /* Bright Warm Cream */
    font-style: italic !important;
}

/* Typewriter Result Cards */
.result-stay {
    background-color: #FFFFFF !important;
    border: 2.5px solid #2D6A4F !important;
    border-radius: 6px !important;
    padding: 22px 26px !important;
    box-shadow: 0 6px 16px rgba(45, 106, 79, 0.12) !important;
}
.result-stay h2 {
    font-family: 'Special Elite', 'Courier New', monospace !important;
    color: #1B4332 !important;
    font-size: 1.6rem !important;
    margin: 0 !important;
    letter-spacing: 1px !important;
}
.result-stay p {
    font-family: 'Special Elite', 'Courier New', monospace !important;
    color: #000000 !important;
    font-size: 1.2rem !important;
    margin-top: 10px !important;
    line-height: 1.5 !important;
}

.result-churn {
    background-color: #FFFFFF !important;
    border: 2.5px solid #A82035 !important;
    border-radius: 6px !important;
    padding: 22px 26px !important;
    box-shadow: 0 6px 16px rgba(168, 32, 53, 0.12) !important;
}
.result-churn h2 {
    font-family: 'Special Elite', 'Courier New', monospace !important;
    color: #A82035 !important;
    font-size: 1.6rem !important;
    margin: 0 !important;
    letter-spacing: 1px !important;
}
.result-churn p {
    font-family: 'Special Elite', 'Courier New', monospace !important;
    color: #000000 !important;
    font-size: 1.2rem !important;
    margin-top: 10px !important;
    line-height: 1.5 !important;
}
</style>
""", unsafe_allow_html=True)

# Paths
MODELS_DIR = "models"
REPORTS_DIR = "reports"
FIGURES_DIR = "reports/figures"

@st.cache_resource
def load_artifacts():
    """Loads saved preprocessor pipeline and all trained models."""
    preprocessor = joblib.load(os.path.join(MODELS_DIR, "preprocessor.joblib"))
    best_model = joblib.load(os.path.join(MODELS_DIR, "best_model.joblib"))
    lr_model = joblib.load(os.path.join(MODELS_DIR, "logistic_regression.joblib"))
    knn_model = joblib.load(os.path.join(MODELS_DIR, "knn.joblib"))
    rf_model = joblib.load(os.path.join(MODELS_DIR, "random_forest.joblib"))
    ann_model = tf.keras.models.load_model(os.path.join(MODELS_DIR, "ann_model.keras"))
    
    return {
        "preprocessor": preprocessor,
        "models": {
            "Champion Model (Logistic Regression)": best_model,
            "Artificial Neural Network (ANN)": ann_model,
            "Random Forest": rf_model,
            "K-Nearest Neighbors (KNN)": knn_model
        }
    }

artifacts = load_artifacts()
preprocessor = artifacts["preprocessor"]
available_models = artifacts["models"]

# Hero Banner
st.markdown("""
<div class="hero-box">
    <div class="hero-title">Telecom Customer Churn Prediction</div>
    <div class="hero-subtitle">Verveox Technologies AI/ML Internship | Project 5 &bull; Machine Learning & Neural Network Classification</div>
</div>
""", unsafe_allow_html=True)

# Navigation Tabs
tab_predict, tab_comparison, tab_eda, tab_about = st.tabs([
    "⚔️ Churn Predictor",
    "🛡️ Model Benchmark & Comparison",
    "📜 Exploratory Data Analysis",
    "🏛️ Project Architecture"
])

# ==========================================
# TAB 1: CHURN PREDICTOR
# ==========================================
with tab_predict:
    with st.sidebar:
        st.markdown("### ⚙️ Royal Council (Model)")
        selected_model_name = st.selectbox(
            "Select Evaluation Algorithm:",
            list(available_models.keys()),
            index=0
        )
        st.markdown(f"**Active Model:** `{selected_model_name}`")
        st.markdown("---")
        st.markdown("### 📌 Vital Customer Indicators")
        st.caption("Month-to-month contracts, electronic checks, and fiber optic service are heavy drivers of customer departure.")

    st.markdown("### Customer Subscription & Billing Profile")
    st.write("Input subscriber details below to compute churn probability and retention recommendations.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 👤 Demographics")
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior_citizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["No", "Yes"])
        tenure = st.slider("Tenure (Months with Provider)", min_value=0, max_value=72, value=12)
        
    with col2:
        st.markdown("#### 🌐 Telecom Services")
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        
    with col3:
        st.markdown("#### 💳 Billing & Contract")
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
        )
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=18.0, max_value=130.0, value=70.0, step=0.5)
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=9000.0, value=float(monthly_charges * tenure), step=10.0)

    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("🔮 Calculate Churn Risk Verdict", use_container_width=True):
        raw_input = {
            'gender': gender,
            'SeniorCitizen': senior_citizen,
            'Partner': partner,
            'Dependents': dependents,
            'tenure': tenure,
            'PhoneService': phone_service,
            'MultipleLines': multiple_lines,
            'InternetService': internet_service,
            'OnlineSecurity': online_security,
            'OnlineBackup': online_backup,
            'DeviceProtection': device_protection,
            'TechSupport': tech_support,
            'StreamingTV': streaming_tv,
            'StreamingMovies': streaming_movies,
            'Contract': contract,
            'PaperlessBilling': paperless_billing,
            'PaymentMethod': payment_method,
            'MonthlyCharges': monthly_charges,
            'TotalCharges': total_charges
        }
        
        input_df = pd.DataFrame([raw_input])
        
        # Robust one-hot encoding for single inference row:
        # Generate dummy indicators and align exactly with the preprocessor feature space
        input_encoded = pd.get_dummies(input_df, columns=preprocessor['categorical_cols'], dtype=int)
        input_encoded = input_encoded.reindex(columns=preprocessor['all_feature_names'], fill_value=0)
        
        input_scaled = input_encoded.copy()
        input_scaled[preprocessor['numerical_cols']] = preprocessor['scaler'].transform(
            input_encoded[preprocessor['numerical_cols']]
        )
        
        input_selected = preprocessor['selector'].transform(input_scaled)
        chosen_model = available_models[selected_model_name]
        
        if "ANN" in selected_model_name:
            churn_prob = float(chosen_model.predict(input_selected, verbose=0)[0][0])
            churn_pred = int(churn_prob >= 0.5)
        else:
            churn_prob = float(chosen_model.predict_proba(input_selected)[0][1])
            churn_pred = int(chosen_model.predict(input_selected)[0])
            
        res_col1, res_col2 = st.columns([1.3, 0.7])
        
        with res_col1:
            if churn_pred == 1:
                st.markdown(f"""
                <div class="result-churn">
                    <h2>🚨 VERDICT: LIKELY TO CHURN</h2>
                    <p><b>[DISPATCH STATUS]:</b> HIGH RISK OF SERVICE CANCELLATION</p>
                    <p><b>[RISK PROBABILITY]:</b> {churn_prob * 100:.1f}% CHURN RISK</p>
                    <p><b>[RECOMMENDED ACTION]:</b> Immediate retention protocol advised. Offer annual contract incentives and dedicated support to prevent subscriber departure.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-stay">
                    <h2>🛡️ VERDICT: LIKELY TO STAY</h2>
                    <p><b>[DISPATCH STATUS]:</b> LOYAL SUBSCRIBER CONFIRMED</p>
                    <p><b>[LOYALTY CONFIDENCE]:</b> {(1 - churn_prob) * 100:.1f}% RETENTION SCORE</p>
                    <p><b>[RECOMMENDED ACTION]:</b> Healthy subscriber relationship detected. Maintain standard service quality and periodic satisfaction checks.</p>
                </div>
                """, unsafe_allow_html=True)
                
        with res_col2:
            st.markdown("### 📊 Churn Probability")
            st.metric(label="Calculated Risk", value=f"{churn_prob * 100:.2f}%")
            st.progress(churn_prob)
            st.caption(f"Evaluated via: **{selected_model_name}**")

# ==========================================
# TAB 2: MODEL PERFORMANCE & COMPARISON
# ==========================================
with tab_comparison:
    st.subheader("Benchmark Comparison: All 4 Models")
    
    master_csv_path = os.path.join(REPORTS_DIR, "master_model_comparison.csv")
    if os.path.exists(master_csv_path):
        comparison_df = pd.read_csv(master_csv_path)
        st.markdown(
            comparison_df.to_html(index=False),
            unsafe_allow_html=True
        )
    
    st.markdown("---")
    col_comp1, col_comp2 = st.columns(2)
    
    with col_comp1:
        st.markdown("#### 📈 Multi-Model ROC Curves")
        roc_img = os.path.join(FIGURES_DIR, "roc_curves.png")
        if os.path.exists(roc_img):
            st.image(roc_img, caption="ROC Curves comparison across all models", use_column_width=True)
            
    with col_comp2:
        st.markdown("#### 📊 Comparative Metrics Chart")
        bar_img = os.path.join(FIGURES_DIR, "model_comparison_bar_chart.png")
        if os.path.exists(bar_img):
            st.image(bar_img, caption="Accuracy, Precision, Recall, F1, and ROC-AUC scores", use_column_width=True)
            
    st.markdown("---")
    st.markdown("#### 🔍 Test Set Confusion Matrices")
    cm_img = os.path.join(FIGURES_DIR, "confusion_matrices.png")
    if os.path.exists(cm_img):
        st.image(cm_img, caption="Confusion matrices for all four trained algorithms", use_column_width=True)

# ==========================================
# TAB 3: EXPLORATORY DATA ANALYSIS (EDA)
# ==========================================
with tab_eda:
    st.subheader("Exploratory Data Analysis Findings")
    st.write("Visualizations derived from 7,043 genuine IBM Telco subscriber records.")
    
    eda_col1, eda_col2 = st.columns(2)
    with eda_col1:
        p1 = os.path.join(FIGURES_DIR, "churn_distribution.png")
        if os.path.exists(p1):
            st.image(p1, caption="Target Class Imbalance (~73% Stay vs ~27% Churn)", use_column_width=True)
            
        p3 = os.path.join(FIGURES_DIR, "monthly_charges_vs_churn.png")
        if os.path.exists(p3):
            st.image(p3, caption="Elevated monthly bills correlate strongly with customer departures", use_column_width=True)
            
    with eda_col2:
        p2 = os.path.join(FIGURES_DIR, "tenure_vs_churn.png")
        if os.path.exists(p2):
            st.image(p2, caption="New subscribers (< 6 months) experience peak churn vulnerability", use_column_width=True)
            
        p4 = os.path.join(FIGURES_DIR, "contract_vs_churn.png")
        if os.path.exists(p4):
            st.image(p4, caption="Month-to-month contracts demonstrate the highest departure rate", use_column_width=True)

# ==========================================
# TAB 4: ARCHITECTURE & ABOUT
# ==========================================
with tab_about:
    st.subheader("Internship Project Architecture & Specifications")
    st.markdown("""
    **Project Title:** Telecom Customer Churn Prediction System Using Machine Learning and Artificial Neural Networks  
    **Institution:** Verveox Technologies AI/ML Internship &bull; Project 5  
    
    #### 🛠️ Technical Pipeline:
    - **Data Preprocessing:** Imputed missing values in `TotalCharges`, applied `OneHotEncoder` with dummy variable trap prevention, standardized features via `StandardScaler`.
    - **Feature Selection:** Statistical `SelectKBest` with ANOVA F-statistic (`f_classif`) isolating the top 18 predictive features without data leakage.
    - **Traditional ML Models:**
      1. *Logistic Regression:* Champion Model with highest test Accuracy (`80.20%`) and F1-score (`0.5939`).
      2. *K-Nearest Neighbors (KNN):* Distance-based classifier achieving `78.78%` Accuracy and `0.5535` Recall.
      3. *Random Forest Classifier:* Ensemble tree bagging achieving high Precision (`67.21%`) and ROC-AUC (`0.8402`).
    - **Deep Learning Model:**
      4. *Artificial Neural Network (ANN):* 3-Layer Sequential Dense network with BatchNormalization, Dropout, and Sigmoid output achieving highest Recall (`56.15%`).
    """)