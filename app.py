
import streamlit as st
import pandas as pd
import joblib

# --------------------------------
# 1. Application title
# --------------------------------

st.title("Heart Disease Prediction System")

st.write(
    "Enter patient details to predict the possibility of heart disease."
)

# --------------------------------
# 2. Load trained model and scaler
# --------------------------------

model = joblib.load("heart_disease_model.pkl")
scaler = joblib.load("scaler.pkl")

st.success("Machine Learning Model Loaded Successfully!")

# --------------------------------
# 3. Patient information
# --------------------------------

st.header("Patient Information")

# Basic details
st.subheader("Basic Details")

age = st.number_input(
    "Age (years)",
    min_value=1,
    max_value=120,
    value=45
)

sex = st.selectbox(
    "Sex",
    options=[0, 1],
    format_func=lambda x: "Female" if x == 0 else "Male"
)

# Chest pain
st.subheader("Chest Pain Details")

cp = st.selectbox(
    "Chest Pain Type",
    options=[1, 2, 3, 4],
    format_func=lambda x: {
        1: "Typical Angina",
        2: "Atypical Angina",
        3: "Non-anginal Pain",
        4: "Asymptomatic"
    }[x]
)

# Resting blood pressure
trestbps = st.number_input(
    "Resting Blood Pressure (mm Hg)",
    min_value=50,
    max_value=250,
    value=120
)

# Cholesterol
chol = st.number_input(
    "Serum Cholesterol (mg/dl)",
    min_value=50,
    max_value=600,
    value=200
)

# Fasting blood sugar
fbs = st.selectbox(
    "Fasting Blood Sugar",
    options=[0, 1],
    format_func=lambda x: "No (≤ 120 mg/dl)" if x == 0 else "Yes (> 120 mg/dl)"
)

# Resting ECG
restecg = st.selectbox(
    "Resting ECG Results",
    options=[0, 1, 2],
    format_func=lambda x: {
        0: "Normal",
        1: "ST-T Wave Abnormality",
        2: "Left Ventricular Hypertrophy"
    }[x]
)

# Maximum heart rate
thalach = st.number_input(
    "Maximum Heart Rate Achieved",
    min_value=50,
    max_value=250,
    value=150
)

# Exercise-induced angina
exang = st.selectbox(
    "Exercise-Induced Angina",
    options=[0, 1],
    format_func=lambda x: "No" if x == 0 else "Yes"
)

# ST depression
oldpeak = st.number_input(
    "ST Depression (Oldpeak)",
    min_value=0.0,
    max_value=10.0,
    value=1.0,
    step=0.1
)

# Slope
slope = st.selectbox(
    "Slope of Peak Exercise ST Segment",
    options=[1, 2, 3],
    format_func=lambda x: {
        1: "Upsloping",
        2: "Flat",
        3: "Downsloping"
    }[x]
)

# Major vessels
ca = st.selectbox(
    "Number of Major Vessels (0–3)",
    options=[0, 1, 2, 3]
)

# Thalassemia
thal = st.selectbox(
    "Thalassemia",
    options=[3, 6, 7],
    format_func=lambda x: {
        3: "Normal",
        6: "Fixed Defect",
        7: "Reversible Defect"
    }[x]
)

# --------------------------------
# 4. Prediction button
# --------------------------------

predict_button = st.button("Predict Heart Disease")

if predict_button:

    # Collect all 13 patient inputs
    input_data = pd.DataFrame([[
        age, sex, cp, trestbps, chol,
        fbs, restecg, thalach, exang,
        oldpeak, slope, ca, thal
    ]], columns=[
        "age", "sex", "cp", "trestbps", "chol",
        "fbs", "restecg", "thalach", "exang",
        "oldpeak", "slope", "ca", "thal"
    ])

    # Display patient details
    st.subheader("Patient Details")
    st.write(input_data)

    # Scale patient input
    scaled_input = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(scaled_input)

    # Display prediction result
    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.error("Model Prediction: Heart Disease Detected")
    else:
        st.success("Model Prediction: No Heart Disease Detected")

    st.caption(
        "Educational project only. This prediction is not a medical diagnosis."
    )
