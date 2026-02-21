import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("model.pkl")

# App title
st.title("Journey Time Prediction System")

st.write("This app predicts train journey duration based on distance and number of stops.")

# User inputs
distance = st.number_input("Enter total distance (km):", min_value=0.0)
stops = st.number_input("Enter number of stops:", min_value=0)

# Prediction
if st.button("Predict Journey Duration"):
    input_data = pd.DataFrame([[distance, stops]], columns=['Total_Distance', 'Number_of_Stops'])
    prediction = model.predict(input_data)[0]
    st.success(f"Estimated Journey Duration: {round(prediction, 2)} hours")

# Optional: Show dataset preview
if st.checkbox("Show sample dataset"):
    df = pd.read_csv("Dataset1.csv")
    st.dataframe(df.head())