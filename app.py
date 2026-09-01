import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set Page Config
st.set_page_config(page_title="SmartCare - Disease Risk Predictor", layout="wide", page_icon="🏥")

# Load artifacts
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
    st.error(f"Model artifacts not found: {e}")
    ready = False

st.title("🏥 SmartCare AI: Disease Risk Assessment")
st.markdown("Early identification and multi-class disease risk classification.")

if ready:
    with st.form("risk_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.slider("Age", 1, 100, 45)
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            blood_group = st.selectbox("Blood Group", ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"])
            department = st.selectbox("Department", ["General Medicine", "Cardiology", "Neurology", "Orthopedics", "Pediatrics"])
            
        with col2:
            diagnosis = st.selectbox("Diagnosis", ["Diabetes", "Hypertension", "Heart Disease", "Migraine", "Asthma", "Normal"])
            systolic_bp = st.number_input("Systolic Blood Pressure (mmHg)", 80, 200, 120)
            diastolic_bp = st.number_input("Diastolic Blood Pressure (mmHg)", 50, 130, 80)
            bmi = st.number_input("BMI (kg/m²)", 10.0, 50.0, 24.5, step=0.1)

        with col3:
            blood_sugar = st.number_input("Blood Sugar (mg/dL)", 50, 400, 110)
            cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 400, 190)
            previous_admissions = st.number_input("Previous Admissions", 0, 20, 0)
            lab_tests_count = st.number_input("Lab Tests Count", 0, 15, 1)
            treatments_count = st.number_input("Treatments Count", 0, 15, 1)
            
        submitted = st.form_submit_button("Predict Disease Risk Level")
        
    if submitted:
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
        
        # Encoding
        input_encoded = pd.get_dummies(input_data)
        input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)
        
        # Scaling numerical features
        num_cols = ['age', 'previous_admissions', 'systolic_bp', 'diastolic_bp',
                    'blood_sugar_mg_dl', 'cholesterol_mg_dl', 'bmi',
                    'lab_tests_count', 'treatments_count']
        input_encoded[num_cols] = scaler.transform(input_encoded[num_cols])
        
        # Prediction
        prediction = model.predict(input_encoded)[0]
        
        # Display Result
        st.subheader("Assessment Result")
        if prediction == "High":
            st.error(f"🚨 Predicted Disease Risk Level: **HIGH RISK**")
        elif prediction == "Medium":
            st.warning(f"⚠️ Predicted Disease Risk Level: **MEDIUM RISK**")
        else:
            st.success(f"✅ Predicted Disease Risk Level: **LOW RISK**")
