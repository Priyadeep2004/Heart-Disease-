import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# Set page configuration for better aesthetics
st.set_page_config(
    page_title="Heart Disease Prediction App",
    page_icon="🧊", # Changed from ❤️ to a generic icon
    layout="centered",
    initial_sidebar_state="auto"
)

# --- 1. Load the trained model ---
# Ensure the model file is in the same directory as this Streamlit app,
# or provide the correct path.
model_path = 'heart_model.pkl'

if not os.path.exists(model_path):
    st.error(f"Error: Model file '{model_path}' not found. "
             "Please ensure 'heart_model.pkl' is in the same directory as this app.")
    st.stop() # Stop the app if the model is not found

try:
    model = joblib.load(model_path)
    st.sidebar.success("Model loaded successfully!")
except Exception as e:
    st.sidebar.error(f"Error loading model: {e}")
    st.stop() # Stop the app if model loading fails

# --- 2. Streamlit UI Elements ---
st.title("Heart Disease Prediction") # Removed ❤️ emoji
st.markdown("""
    This app predicts the likelihood of heart disease based on various physiological parameters.
    Please enter the patient's details below:
""")

# Define input ranges and defaults based on typical heart disease datasets
# (These match the features generated in the previous Python script's dummy data)
st.sidebar.header("Input Features")

age = st.sidebar.slider("Age", 20, 100, 50)
sex = st.sidebar.radio("Sex", ["Female (0)", "Male (1)"])
sex_val = 1 if sex == "Male (1)" else 0 # Convert to numerical value

chol = st.sidebar.slider("Cholesterol (mg/dl)", 100, 600, 200)
max_hr = st.sidebar.slider("Max Heart Rate (bpm)", 60, 220, 150)
st_dep = st.sidebar.slider("ST Depression (oldpeak)", 0.0, 6.0, 1.0, 0.1)
num_vessels = st.sidebar.slider("Number of Major Vessels Colored (0-3)", 0, 3, 0)

# Optional: Add more specific instructions or feature definitions if needed
st.sidebar.markdown("""
<small>
* **Age**: Age of the patient in years.
* **Sex**: Biological sex (0 = female, 1 = male).
* **Cholesterol**: Serum cholestoral in mg/dl.
* **Max Heart Rate**: Maximum heart rate achieved during exercise.
* **ST Depression**: ST depression induced by exercise relative to rest (oldpeak).
* **Number of Major Vessels**: Number of major vessels (0-3) colored by fluoroscopy.
</small>
""", unsafe_allow_html=True)


# Create a dictionary from user inputs
input_data = {
    'age': age,
    'sex': sex_val,
    'chol': chol,
    'max_hr': max_hr,
    'st_dep': st_dep,
    'num_vessels': num_vessels
}

# Convert input dictionary to a Pandas DataFrame
# The order of columns must match the order the model was trained on
input_df = pd.DataFrame([input_data])

st.subheader("Patient Details Submitted:")
st.write(input_df)

# --- 3. Prediction Button and Logic ---
if st.button("Predict"):
    try:
        # Make prediction
        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)

        st.subheader("Prediction Result:")
        if prediction[0] == 1:
            st.error(f"**Prediction: High likelihood of Heart Disease**")
            st.write(f"Confidence (Probability of Heart Disease): **{prediction_proba[0][1]*100:.2f}%**")
        else:
            st.success(f"**Prediction: Low likelihood of Heart Disease**")
            st.write(f"Confidence (Probability of No Heart Disease): **{prediction_proba[0][0]*100:.2f}%**")

        st.markdown("---")
        st.info("Disclaimer: This is a predictive model based on historical data and should not be used as a substitute for professional medical advice. Always consult with a qualified healthcare provider for any health concerns.")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.warning("Please check your input values and ensure the model is correctly loaded and trained on matching features.")

# Add a footer
st.markdown("---")
st.markdown("Developed by Gemini") # Removed ♊ emoji
