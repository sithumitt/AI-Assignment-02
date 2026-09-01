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
    st.info("Ensure best_model.pkl, scaler.pkl, and model_columns.pkl are located inside the 'models/' folder.")
    ready = False

st.title("🏥 SmartCare AI: Disease Risk Assessment")
st.markdown("Early identification and multi-class disease risk classification with clinical safety checks.")

if ready:
    with st.form("risk_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.slider("Age (Years)", min_value=1, max_value=100, value=45)
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            blood_group = st.selectbox("Blood Group", ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"])
            department = st.selectbox("Department", ["General Medicine", "Cardiology", "Neurology", "Orthopedics", "Pediatrics"])
            
        with col2:
            diagnosis = st.selectbox("Diagnosis", ["Diabetes", "Hypertension", "Heart Disease", "Migraine", "Asthma", "Normal"])
            systolic_bp = st.number_input("Systolic Blood Pressure (mmHg)", min_value=50, max_value=250, value=120, help="Normal range: 90 - 120 mmHg")
            diastolic_bp = st.number_input("Diastolic Blood Pressure (mmHg)", min_value=30, max_value=150, value=80, help="Normal range: 60 - 80 mmHg")
            bmi = st.number_input("BMI (kg/m²)", min_value=10.0, max_value=60.0, value=24.5, step=0.1, help="Normal range: 18.5 - 24.9 kg/m²")

        with col3:
            blood_sugar = st.number_input("Blood Sugar (mg/dL)", min_value=20, max_value=500, value=110, help="Normal fasting range: 70 - 99 mg/dL")
            cholesterol = st.number_input("Cholesterol (mg/dL)", min_value=50, max_value=500, value=190, help="Desirable: < 200 mg/dL")
            previous_admissions = st.number_input("Previous Admissions", min_value=0, max_value=20, value=0)
            lab_tests_count = st.number_input("Lab Tests Count", min_value=0, max_value=15, value=1)
            treatments_count = st.number_input("Treatments Count", min_value=0, max_value=15, value=1)
            
        submitted = st.form_submit_button("Predict Disease Risk Level")
        
    if submitted:
        st.subheader("Assessment Result")
        
        # Clinical Guardrails & Boundary Verification
        critical_alerts = []
        
        if blood_sugar < 60:
            critical_alerts.append("Severe Hypoglycemia (Blood Sugar < 60 mg/dL)")
        elif blood_sugar > 350:
            critical_alerts.append("Severe Hyperglycemia (Blood Sugar > 350 mg/dL)")
            
        if systolic_bp < 70 or diastolic_bp < 40:
            critical_alerts.append("Severe Hypotension (Critically Low Blood Pressure)")
        elif systolic_bp >= 180 or diastolic_bp >= 120:
            critical_alerts.append("Hypertensive Crisis (Systolic ≥ 180 or Diastolic ≥ 120 mmHg)")

        if critical_alerts:
            st.error("🚨 **CRITICAL CLINICAL ALERT: IMMEDIATE INTERVENTION REQUIRED**")
            for alert in critical_alerts:
                st.markdown(f"- **{alert}**")
            st.warning("⚠️ Machine Learning baseline indicates an extreme physiological state outside standard operational bounds.")
        else:
            # Prepare tabular input for ML Model
            input_data = pd.DataFrame([{
                'age': age, 'gender': gender, 'blood_group': blood_group,
                'department': department, 'diagnosis': diagnosis,
                'previous_admissions': previous_admissions,
                'systolic_bp': systolic_bp, 'diastolic_bp': diastolic_bp,
                'blood_sugar_mg_dl': blood_sugar,
                'cholesterol_mg_dl': cholesterol,
                'bmi': bmi,
                'lab_tests_count': lab_tests_count,
                'treatments_count': treatments_count
            }])
            
            # One-Hot Encoding Alignment
            input_encoded = pd.get_dummies(input_data)
            input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)
            
            # Feature Scaling
            num_cols = ['age', 'previous_admissions', 'systolic_bp', 'diastolic_bp',
                        'blood_sugar_mg_dl', 'cholesterol_mg_dl', 'bmi',
                        'lab_tests_count', 'treatments_count']
            input_encoded[num_cols] = scaler.transform(input_encoded[num_cols])
            
            # Model Inference
            prediction = model.predict(input_encoded)[0]
            
            if prediction == "High":
                st.error("🚨 Predicted Disease Risk Level: **HIGH RISK**")
            elif prediction == "Medium":
                st.warning("⚠️ Predicted Disease Risk Level: **MEDIUM RISK**")
            else:
                st.success("✅ Predicted Disease Risk Level: **LOW RISK**")
