# customer-churn-ai-dashboard

Project Overview:
This project predicts customer churn for a telecom company using machine learning.
The goal is to identify customers who are likely to stop using the service sot that proactive
retention strategies can be applied.

The Project includes:
1. Data exploration and cleaning.
2. Feature engineering.
3. Machine Learning Model Training.
4. Model Evaluation.
5. Interactive Dashboards for Predictions.

Dataset Used:
Kaggle, Telco Customer Churn.

Business Objectives:
1. Predict whether a customer will churn.
2. Identify key factors influencing churn.
3. Help business improve customer retention strategies.

Libraries used/ Tech Stack:
1. Python.
2. Pandas.
3. Matplotlib, Seaborn.
4. Scikit-learn.
5. Streamlit.
6. Joblib.

Project Workflow:
1. Business Understanding.
2. Data Exploratin (EDA).
3. Data Cleaning.
4. Feature Engineering.
5. Model Training (Logistic Regression).
6. Model Evaluation.
7. Prediction System.
8. Interactive Dashboard.

Machine Learning Model:
Algorithm: Logistic Regression.
Target: Customer Churn (Yes/No)
Evaluation Metrics:
1. Accuracy.
2. Confusion Matrix.

Key Insights:
1. Month-to-month contract customers are more likely to churn.
2. Longer tenure reduce churn probability.
3. Payment method and service type influence churn behavior.

Streamlit Dashboard:
1. Input customers details.
2. Predict churn probability.
3. View real-time results.

to run the app:
streamlit run app/app.py

How to run this project:
1. Clone the repository:
git clone https://github.com/spencer1014/customer-churn-ai-dashboard.git

2. Install dependencies:
pip install -r requirements.txt

3. Launch Dashboard:
streamlit run app/app.py

Project Structure:
customer-churn-ai-dashboard/
|
|-- data/
|-- notebooks/
|-- src/
|-- models/
|-- app/
|-- reports/
|-- requirements.txt
|-- README.md

IMPROVEMENTS THAT CAN BE MADE:
1. Using advance model i.e: Random Forest, XGBoost
2. Handle Class Imbalance, SMOTE.
3. Improve Dashboard UI
4. Deploy Online.

Project is tested for deployment using Streamlit Cloud.
bash:
streamlit run app/app.py

LIVE DEMO: STREAMLIT APP
https://customer-churn-ai-dashboard-ucnhm7cqbu5vzeyehtpqid.streamlit.app