import streamlit as st
import pandas as pd
import joblib

# Load saved model and transformer
model = joblib.load("insurance_polynomial_model.pkl")
poly = joblib.load("poly_transformer.pkl")

st.set_page_config(page_title="Insurance Cost Predictor")

st.title("🏥 Insurance Cost Predictor")

st.write("Enter your details below.")

# Inputs
age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=30
)

bmi = st.number_input(
    "BMI",
    min_value=10.0,
    max_value=60.0,
    value=25.0
)

children = st.number_input(
    "Number of Children",
    min_value=0,
    max_value=10,
    value=0
)

sex = st.selectbox(
    "Sex",
    ["female", "male"]
)

smoker = st.selectbox(
    "Smoker",
    ["no", "yes"]
)

region = st.selectbox(
    "Region",
    [
        "northeast",
        "northwest",
        "southeast",
        "southwest"
    ]
)

if st.button("Predict Insurance Cost"):

    # Create dataframe manually
    input_data = pd.DataFrame({
        "age": [age],
        "bmi": [bmi],
        "children": [children],
        "sex_male": [1 if sex == "male" else 0],
        "smoker_yes": [1 if smoker == "yes" else 0],
        "region_northwest": [1 if region == "northwest" else 0],
        "region_southeast": [1 if region == "southeast" else 0],
        "region_southwest": [1 if region == "southwest" else 0]
    })

    # Apply polynomial transformation
    input_poly = poly.transform(input_data)

    # Prediction
    prediction = model.predict(input_poly)[0]

    st.metric(
        label="Estimated Insurance Cost",
        value=f"${prediction:,.2f}"
    )

    st.subheader("Encoded Input Used By Model")
    st.dataframe(input_data)
    st.sidebar.header("User Information")
    st.subheader("Model Performance")
    st.write("MAE: 2729")
    st.write("RMSE: 4551")
    st.write("R² Score: 0.867")
    