import streamlit as st
import pandas as pd
import joblib
import os

# load model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models/churn_model.pkl")
model = joblib.load(MODEL_PATH)

st.title("Customer Churn Prediction")

# User input
tenure = st.slider("Tenure", 0, 72, 12)

monthly_charges = st.slider("Monthly Charges", 0, 150, 70)

total_charges = st.slider("Total Charges", 0, 10000, 1000)

labels = {"Month-to-Month": 0,
          "1 Year": 1,
          "2 Year": 2
          }

contract = st.selectbox(
    "Contract Type",
    options = labels.keys()
)

# Example simplified input
input_data = pd.DataFrame({
    "gender": [1],
    "SeniorCitizen": [0],
    "Partner": [1],
    "Dependents": [0],
    "tenure": [tenure],
    "PhoneService": [1],
    "MultipleLines": [0],
    "InternetService": [1],
    "OnlineSecurity": [0],
    "OnlineBackup": [1],
    "DeviceProtection": [0],
    "TechSupport": [0],
    "StreamingTV": [1],
    "StreamingMovies": [1],
    "Contract": labels[contract],
    "PaperlessBilling": [1],
    "PaymentMethod": [2],
    "MonthlyCharges": [monthly_charges],
    "TotalCharges": [total_charges]
})

# Prediction
prediction = model.predict(input_data)

probability = model.predict_proba(input_data)

if prediction[0] == 1:
    st.error("Customer is likely to churn")
else:
    st.success("Customer is likely to stay")


st.write("Churn Probability:", probability[0][1])