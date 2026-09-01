import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page configuration
st.set_page_config(
    page_title="SmartCare - Disease Risk Predictor",
    layout="wide",
    page_icon="🏥"
)

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
    st.info("Ensure best_model.pkl, scaler.pkl, and model_columns.pkl are inside the 'models/' folder.")
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

st.title("🏥 SmartCare AI: Disease Risk Assessment")
st.markdown("Multi-class disease risk classification decision support system.")

if ready:
    with st.form("risk_assessment_form"):
        # Section 1: Primary Clinical Vitals & Lab Measurements (Highest Importance)
        st.subheader("1. Primary Clinical Biomarkers & Vitals (High Importance)")
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

        st.divider()

        # Section 2: Clinical Diagnosis & Medical History (Medium Importance)
        st.subheader("2. Diagnosis & Medical History (Medium Importance)")
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

        st.divider()

        # Section 3: Patient Demographics (Standard Context)
        st.subheader("3. Patient Demographics")
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

        submitted = st.form_submit_button("Predict Disease Risk Level")

    if submitted:
        # Validate all inputs against defined ranges
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
                    f"**{feature_name}**: Entered value `{value}` is below minimum acceptable value `{min_val} {unit}` (Acceptable Range: {min_val} – {max_val} {unit})"
                )
            elif value > max_val:
                invalid_inputs.append(
                    f"**{feature_name}**: Entered value `{value}` exceeds maximum acceptable value `{max_val} {unit}` (Acceptable Range: {min_val} – {max_val} {unit})"
                )

        # Halt execution and show validation report if out of bounds
        if invalid_inputs:
            st.error("❌ **Prediction Halted: Invalid Input Values Detected**")
            st.warning("Please enter valid clinical values within the defined physiological ranges before generating a prediction:")
            for msg in invalid_inputs:
                st.markdown(f"- {msg}")
        else:
            # Prepare tabular data for inference
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

            # One-Hot Encoding alignment
            input_encoded = pd.get_dummies(input_data)
            input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)

            # Feature Scaling
            num_cols = [
                'age', 'previous_admissions', 'systolic_bp', 'diastolic_bp',
                'blood_sugar_mg_dl', 'cholesterol_mg_dl', 'bmi',
                'lab_tests_count', 'treatments_count'
            ]
            input_encoded[num_cols] = scaler.transform(input_encoded[num_cols])

            # Prediction
            prediction = model.predict(input_encoded)[0]

            # Display Output Result
            st.subheader("Assessment Result")
            if prediction == "High":
                st.error("🚨 Predicted Disease Risk Level: **HIGH RISK**")
            elif prediction == "Medium":
                st.warning("⚠️ Predicted Disease Risk Level: **MEDIUM RISK**")
            else:
                st.success("✅ Predicted Disease Risk Level: **LOW RISK**")
