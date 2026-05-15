import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import sys

# Load model with fallback
try:
    model = joblib.load("best_churn_pipeline.pkl")
except AttributeError as e:
    # Handle sklearn version mismatch
    if '_RemainderColsList' in str(e):
        st.error("""
        ⚠️ **Model Loading Error**
        
        The saved model is incompatible with the current scikit-learn version.
        This typically happens when sklearn version changes.
        
        **Solution**: The model needs to be retrained with the current sklearn version.
        """)
        st.stop()
    else:
        raise

# Constants from training data
MEDIAN_SALARY_BALANCE_RATIO = 1.25
BALANCE_75_PERCENTILE = 127644.24

# Title
st.title("Bank Customer Churn Prediction")

st.write("Predict whether a customer will churn or not.")

# User Inputs
credit_score = st.number_input(
    "Credit Score",
    min_value=350,
    max_value=850,
    value=650
)

country = st.selectbox(
    "Country",
    ["France", "Spain", "Germany"]
)

gender = st.radio(
    "Gender",
    ["Male", "Female"]
)

age = st.slider(
    "Age",
    min_value=18,
    max_value=92,
    value=40
)

tenure = st.slider(
    "Tenure",
    min_value=0,
    max_value=10,
    value=3
)

balance = st.number_input(
    "Balance",
    min_value=0.0,
    value=50000.0
)

products_number = st.slider(
    "Number of Products",
    min_value=1,
    max_value=4,
    value=2
)

credit_card = st.checkbox(
    "Has Credit Card",
    value=True
)

active_member = st.checkbox(
    "Is Active Member",
    value=True
)

estimated_salary = st.number_input(
    "Estimated Salary",
    min_value=0.0,
    value=60000.0
)

# Convert checkbox values
credit_card = 1 if credit_card else 0
active_member = 1 if active_member else 0

# Predict button
if st.button("Predict Churn"):

    # Raw input dataframe
    input_data = pd.DataFrame({
        'credit_score': [credit_score],
        'country': [country],
        'gender': [gender],
        'age': [age],
        'tenure': [tenure],
        'balance': [balance],
        'products_number': [products_number],
        'credit_card': [credit_card],
        'active_member': [active_member],
        'estimated_salary': [estimated_salary]
    })

    # Feature Engineering

    # balance_per_product
    input_data['balance_per_product'] = (
        input_data['balance'] /
        input_data['products_number'].replace(0, np.nan)
    )

    input_data['balance_per_product'] = (
        input_data['balance_per_product'].fillna(0)
    )

    # salary_balance_ratio
    input_data['salary_balance_ratio'] = (
        input_data['estimated_salary'] /
        input_data['balance'].replace(0, np.nan)
    )

    input_data['salary_balance_ratio'] = (
        input_data['salary_balance_ratio']
        .replace([np.inf, -np.inf], np.nan)
        .fillna(MEDIAN_SALARY_BALANCE_RATIO)
    )

    # age_group
    input_data['age_group'] = pd.cut(
        input_data['age'],
        bins=[0,25,35,45,55,65,100],
        labels=['<25','25-34','35-44','45-54','55-64','65+']
    )

    # tenure_bucket
    input_data['tenure_bucket'] = pd.cut(
        input_data['tenure'],
        bins=[-1,0,2,5,10,100],
        labels=['0','1-2','3-5','6-10','10+']
    )

    # high_balance
    input_data['high_balance'] = np.where(
        input_data['balance'] > BALANCE_75_PERCENTILE,
        1,
        0
    )

    # Prediction
    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0][1]

    # Output
    if prediction == 1:
        st.error("Customer will churn")
    else:
        st.success("Customer will not churn")

    st.write(f"Churn Probability: {probability*100:.2f}%")

    st.progress(float(probability))