import streamlit as st
import pickle
import numpy as np

# Load the trained model
with open("logistic_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Logistic Regression Prediction App")

# Example input fields (replace with your dataset features)
feature1 = st.number_input("Enter Feature 1")
feature2 = st.number_input("Enter Feature 2")

if st.button("Predict"):
    input_data = np.array([[feature1, feature2]])
    prediction = model.predict(input_data)
    st.write("Prediction:", prediction[0])
