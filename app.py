import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set Page Config
st.set_page_config(
    page_title="SmartCare - Disease Risk Predictor",
    layout="wide",
    page_icon="🏥"
)

# Custom Palette CSS Injection
CUSTOM_CSS = """
<style>
    /* Gradient Header */
    .smartcare-header {
        background: linear-gradient(135deg, #000851 0%, #003366 30%, #0077B6 70%, #00B4D8 100%);
        padding: 24px;
        border-radius: 12px;
        color: #FFFFFF;
        text-align: center;
        margin-bottom: 25px;
        box-shadow: 0 4px 14px rgba(0, 51, 102, 0.15);
    }
    .smartcare-header h1 {
        color: #FFFFFF !important;
        font-weight: 700;
        margin: 0;
        font-size: 2.2rem;
    }
    .smartcare-header p {
        color: #CAF0F8 !important;
        margin-top: 8px;
        font-size: 1.05rem;
    }

    /* Section Subheaders */
    .section-title {
        color: #003366;
        font-weight: 700;
        font-size: 1.15rem;
        padding-bottom: 6px;
        border-bottom: 2px solid #90E0EF;
        margin-top: 15px;
        margin-bottom: 18px;
    }

    /* Streamlit Primary Button Styling */
    div.stButton > button:first-child {
        background: linear-gradient(90deg, #005A9E 0%, #0077B6 50%, #0096C7 100%) !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
        font-size: 1.05rem !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 12px 28px !important;
        width: 100% !important;
        box-shadow: 0 4px 10px rgba(0, 119, 182, 0.3) !important;
        transition: all 0.3s ease-in-out !important;
    }
    div.stButton > button:first-child:hover {
        background: linear-gradient(90deg, #003366 0%, #005A9E 50%, #0077B6 100%) !important;
        box-shadow: 0 6px 14px rgba(0, 51, 102, 0.4) !important;
        transform: translateY(-1px);
    }

    /* Prediction Result Cards */
    .result-card-low {
        background: #E8F8FC;
        border-left: 6px solid #48CAE4;
        border-radius: 8px;
        padding: 16px 20px;
        color: #003366;
        margin-top: 15px;
    }
    .result-card-medium {
        background: #FFF9E6;
        border-left: 6px solid #0077B6;
        border-radius: 8px;
        padding: 16px 20px;
        color: #003366;
        margin-top: 15px;
    }
    .result-card-high {
        background: #FDF0F0;
        border-left: 6px solid #000851;
        border-radius: 8px;
        padding: 16px 20px;
        color: #000851;
        margin-top: 15px;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Load trained artifacts
@st.cache_resource
def load_artifacts():
    model = joblib.load("models/best_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    columns = joblib.load("models/model_columns.pkl")
    return model, scaler, columns

try:
    model, scaler, model_columns = load_artifacts()
    ready = True
except Exception as e:
    st.error(f"Error loading model artifacts: {e}")
    st.info("Ensure best_model.pkl, scaler.pkl, and model_columns.pkl are located inside the 'models/' folder.")
    ready = False

# Allowed Input Value Ranges (Clinical Validation Boundary)
VALID_RANGES = {
    "Age": (1, 100, "years"),
    "Systolic Blood Pressure": (70, 220, "mmHg"),
    "Diastolic Blood Pressure": (40, 140, "mmHg"),
    "Blood Sugar": (50, 400, "mg/dL"),
    "Cholesterol": (100, 400, "mg/dL"),
    "BMI": (12.0, 50.0, "kg/m²"),
    "Previous Admissions": (0, 20, "admissions"),
    "Lab Tests Count": (0, 15, "tests"),
    "Treatments Count": (0, 15, "treatments")
}

# Branded App Header
st.markdown("""
<div class="smartcare-header">
    <h1>SmartCare AI: Disease Risk Assessment</h1>
    <p>Clinical Decision Support System for Multi-Class Risk Stratification</p>
</div>
""", unsafe_allow_html=True)

if ready:
    with st.form("risk_assessment_form"):
        # Section 1: Primary Biomarkers
        st.markdown('<div class="section-title">1. Primary Clinical Biomarkers & Vitals (High Importance)</div>', unsafe_allow_html=True)
        col1_1, col1_2, col1_3 = st.columns(3)
        
        with col1_1:
            blood_sugar = st.number_input(
                "Blood Sugar (mg/dL)",
                value=110.0,
                step=1.0,
                help="Valid Range: 50 – 400 mg/dL"
            )
            systolic_bp = st.number_input(
                "Systolic Blood Pressure (mmHg)",
                value=120.0,
                step=1.0,
                help="Valid Range: 70 – 220 mmHg"
            )
            
        with col1_2:
            cholesterol = st.number_input(
                "Cholesterol (mg/dL)",
                value=190.0,
                step=1.0,
                help="Valid Range: 100 – 400 mg/dL"
            )
            diastolic_bp = st.number_input(
                "Diastolic Blood Pressure (mmHg)",
                value=80.0,
                step=1.0,
                help="Valid Range: 40 – 140 mmHg"
            )
            
        with col1_3:
            bmi = st.number_input(
                "Body Mass Index (BMI in kg/m²)",
                value=24.5,
                step=0.1,
                help="Valid Range: 12.0 – 50.0 kg/m²"
            )

        # Section 2: Medical History
        st.markdown('<div class="section-title">2. Diagnosis & Medical History (Medium Importance)</div>', unsafe_allow_html=True)
        col2_1, col2_2, col2_3 = st.columns(3)
        
        with col2_1:
            diagnosis = st.selectbox(
                "Diagnosis",
                ["Diabetes", "Hypertension", "Heart Disease", "Migraine", "Asthma", "Normal"]
            )
            department = st.selectbox(
                "Hospital Department",
                ["General Medicine", "Cardiology", "Neurology", "Orthopedics", "Pediatrics"]
            )
            
        with col2_2:
            previous_admissions = st.number_input(
                "Previous Admissions Count",
                value=0,
                step=1,
                help="Valid Range: 0 – 20"
            )
            treatments_count = st.number_input(
                "Treatments Count",
                value=1,
                step=1,
                help="Valid Range: 0 – 15"
            )
            
        with col2_3:
            lab_tests_count = st.number_input(
                "Lab Tests Count",
                value=1,
                step=1,
                help="Valid Range: 0 – 15"
            )

        # Section 3: Patient Demographics
        st.markdown('<div class="section-title">3. Patient Demographics</div>', unsafe_allow_html=True)
        col3_1, col3_2, col3_3 = st.columns(3)
        
        with col3_1:
            age = st.number_input(
                "Age (Years)",
                value=45,
                step=1,
                help="Valid Range: 1 – 100 years"
            )
            
        with col3_2:
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            
        with col3_3:
            blood_group = st.selectbox(
                "Blood Group",
                ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]
            )

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("Predict Disease Risk Level")

    if submitted:
        # Validate input ranges
        user_inputs = {
            "Age": age,
            "Systolic Blood Pressure": systolic_bp,
            "Diastolic Blood Pressure": diastolic_bp,
            "Blood Sugar": blood_sugar,
            "Cholesterol": cholesterol,
            "BMI": bmi,
            "Previous Admissions": previous_admissions,
            "Lab Tests Count": lab_tests_count,
            "Treatments Count": treatments_count
        }

        invalid_inputs = []
        for feature_name, value in user_inputs.items():
            min_val, max_val, unit = VALID_RANGES[feature_name]
            if value < min_val:
                invalid_inputs.append(
                    f"**{feature_name}**: Entered value `{value}` is below the minimum allowed value `{min_val} {unit}` (Acceptable Range: {min_val} – {max_val} {unit})"
                )
            elif value > max_val:
                invalid_inputs.append(
                    f"**{feature_name}**: Entered value `{value}` exceeds the maximum allowed value `{max_val} {unit}` (Acceptable Range: {min_val} – {max_val} {unit})"
                )

        if invalid_inputs:
            st.error("❌ **Prediction Halted: Invalid Input Values Detected**")
            st.warning("Please adjust the following parameters to remain within valid physiological boundaries:")
            for msg in invalid_inputs:
                st.markdown(f"- {msg}")
        else:
            # Build input DataFrame
            input_data = pd.DataFrame([{
                'age': age,
                'gender': gender,
                'blood_group': blood_group,
                'department': department,
                'diagnosis': diagnosis,
                'previous_admissions': previous_admissions,
                'systolic_bp': systolic_bp,
                'diastolic_bp': diastolic_bp,
                'blood_sugar_mg_dl': blood_sugar,
                'cholesterol_mg_dl': cholesterol,
                'bmi': bmi,
                'lab_tests_count': lab_tests_count,
                'treatments_count': treatments_count
            }])

            # Encoding & Scaler Alignment
            input_encoded = pd.get_dummies(input_data)
            input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)

            num_cols = [
                'age', 'previous_admissions', 'systolic_bp', 'diastolic_bp',
                'blood_sugar_mg_dl', 'cholesterol_mg_dl', 'bmi',
                'lab_tests_count', 'treatments_count'
            ]
            input_encoded[num_cols] = scaler.transform(input_encoded[num_cols])

            # Prediction
            prediction = model.predict(input_encoded)[0]

            # Render Styled Output Cards
            st.markdown('<div class="section-title">Assessment Result</div>', unsafe_allow_html=True)
            if prediction == "High":
                st.markdown("""
                <div class="result-card-high">
                    <h3 style="margin:0; color:#000851;">🚨 Predicted Risk Level: HIGH RISK</h3>
                    <p style="margin:5px 0 0 0;">Patient parameters indicate significantly elevated clinical biomarkers. Immediate preventive specialist review recommended.</p>
                </div>
                """, unsafe_allow_html=True)
            elif prediction == "Medium":
                st.markdown("""
                <div class="result-card-medium">
                    <h3 style="margin:0; color:#005A9E;">⚠️ Predicted Risk Level: MEDIUM RISK</h3>
                    <p style="margin:5px 0 0 0;">Patient displays borderline clinical risk indicators. Routine follow-up and lifestyle intervention advised.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="result-card-low">
                    <h3 style="margin:0; color:#0077B6;">✅ Predicted Risk Level: LOW RISK</h3>
                    <p style="margin:5px 0 0 0;">Patient vitals and clinical metrics fall comfortably within normal operational thresholds.</p>
                </div>
                """, unsafe_allow_html=True)
