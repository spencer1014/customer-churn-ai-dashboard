import joblib
import pandas as pd

# load trained model
model = joblib.load("models/churn_model.pkl")

# Example of customer data
sample_data = {
    "gender": [1],
    "SeniorCitizen": [0],
    "Partner": [1],
    "Dependents": [0],
    "tenure": [12],
    "PhoneService": [1],
    "MultipleLines": [0],
    "InternetService": [1],
    "OnlineSecurity": [0],
    "OnlineBackup": [1],
    "DeviceProtection": [0],
    "TechSupport": [0],
    "StreamingTV": [1],
    "StreamingMovies": [1],
    "Contract": [0],
    "PaperlessBilling": [1],
    "PaymentMethod": [2],
    "MonthlyCharges": [70.5],
    "TotalCharges": [850.2]
}

# Convert to dataframe
sample_df = pd.DataFrame(sample_data)

# Predict
prediction = model.predict(sample_df)

probability = model.predict_proba(sample_df)

# Output
print("Prediction", prediction)
print("Probability", probability)