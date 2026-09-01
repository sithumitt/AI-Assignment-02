import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set application layout and header
st.set_page_config(
    page_title="SmartCare - Disease Risk Predictor",
    layout="wide",
    page_icon="🏥"
)

# Load trained model artifacts
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
    st.info("Ensure best_model.pkl, scaler.pkl, and model_columns.pkl are present in the 'models/' directory.")
    ready = False

st.title("🏥 SmartCare AI: Disease Risk Assessment")
st.markdown("Early identification and multi-class disease risk classification with strict input bounds and clinical guardrails.")

if ready:
    with st.form("risk_form"):
        col1, col2, col3 = st.columns(3)
        
        # Column 1: Patient Demographics & Department
        with col1:
            st.caption("📌 Range: 1 – 100 years")
            age = st.slider("Age (Years)", min_value=1, max_value=100, value=45)
            
            st.caption("📌 Standard demographic selection")
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            
            st.caption("📌 ABO & Rh blood classification")
            blood_group = st.selectbox("Blood Group", ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"])
            
            st.caption("📌 Clinical department")
            department = st.selectbox("Department", ["General Medicine", "Cardiology", "Neurology", "Orthopedics", "Pediatrics"])
            
        # Column 2: Clinical Diagnosis & Vitals
        with col2:
            st.caption("📌 Primary clinical diagnosis")
            diagnosis = st.selectbox("Diagnosis", ["Diabetes", "Hypertension", "Heart Disease", "Migraine", "Asthma", "Normal"])
            
            st.caption("📌 Allowed: 70 – 220 mmHg (Normal: 90 – 120)")
            systolic_bp = st.number_input(
                "Systolic Blood Pressure (mmHg)", 
                min_value=70, 
                max_value=220, 
                value=120, 
                step=1
            )
            
            st.caption("📌 Allowed: 40 – 140 mmHg (Normal: 60 – 80)")
            diastolic_bp = st.number_input(
                "Diastolic Blood Pressure (mmHg)", 
                min_value=40, 
                max_value=140, 
                value=80, 
                step=1
            )
            
            st.caption("📌 Allowed: 12.0 – 50.0 kg/m² (Normal: 18.5 – 24.9)")
            bmi = st.number_input(
                "Body Mass Index (BMI)", 
                min_value=12.0, 
                max_value=50.0, 
                value=24.5, 
                step=0.1
            )

        # Column 3: Metabolic & Operational Metrics
        with col3:
            st.caption("📌 Allowed: 50 – 400 mg/dL (Normal Fasting: 70 – 99)")
            blood_sugar = st.number_input(
                "Blood Sugar (mg/dL)", 
                min_value=50, 
                max_value=400, 
                value=110, 
                step=1
            )
            
            st.caption("📌 Allowed: 100 – 400 mg/dL (Desirable: < 200)")
            cholesterol = st.number_input(
                "Cholesterol (mg/dL)", 
                min_value=100, 
                max_value=400, 
                value=190, 
                step=1
            )
            
            st.caption("📌 Allowed: 0 – 20 recorded admissions")
            previous_admissions = st.number_input(
                "Previous Admissions", 
                min_value=0, 
                max_value=20, 
                value=0, 
                step=1
            )
            
            st.caption("📌 Allowed: 0 – 15 tests")
            lab_tests_count = st.number_input(
                "Lab Tests Count", 
                min_value=0, 
                max_value=15, 
                value=1, 
                step=1
            )
            
            st.caption("📌 Allowed: 0 – 15 treatments")
            treatments_count = st.number_input(
                "Treatments Count", 
                min_value=0, 
                max_value=15, 
                value=1, 
                step=1
            )
            
        submitted = st.form_submit_button("Predict Disease Risk Level")
        
    if submitted:
        st.subheader("Assessment Result")
        
        # Clinical Guardrail Check
        critical_alerts = []
        
        if blood_sugar < 60:
            critical_alerts.append("Severe Hypoglycemia (Blood Sugar < 60 mg/dL)")
        elif blood_sugar > 350:
            critical_alerts.append("Severe Hyperglycemia (Blood Sugar > 350 mg/dL)")
            
        if systolic_bp < 80 or diastolic_bp < 50:
            critical_alerts.append("Severe Hypotension (Critically low blood pressure)")
        elif systolic_bp >= 180 or diastolic_bp >= 120:
            critical_alerts.append("Hypertensive Crisis (Systolic ≥ 180 or Diastolic ≥ 120 mmHg)")

        if critical_alerts:
            st.error("🚨 **CRITICAL CLINICAL ALERT: IMMEDIATE MEDICAL INTERVENTION REQUIRED**")
            for alert in critical_alerts:
                st.markdown(f"- **{alert}**")
            st.warning("⚠️ Critical physiological boundary reached. Machine learning risk score bypassed for patient safety.")
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
            
            # One-Hot Encoding Alignment
            input_encoded = pd.get_dummies(input_data)
            input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)
            
            # Scale numerical features
            num_cols = [
                'age', 'previous_admissions', 'systolic_bp', 'diastolic_bp',
                'blood_sugar_mg_dl', 'cholesterol_mg_dl', 'bmi',
                'lab_tests_count', 'treatments_count'
            ]
            input_encoded[num_cols] = scaler.transform(input_encoded[num_cols])
            
            # Inference
            prediction = model.predict(input_encoded)[0]
            
            if prediction == "High":
                st.error("🚨 Predicted Disease Risk Level: **HIGH RISK**")
            elif prediction == "Medium":
                st.warning("⚠️ Predicted Disease Risk Level: **MEDIUM RISK**")
            else:
                st.success("✅ Predicted Disease Risk Level: **LOW RISK**")
