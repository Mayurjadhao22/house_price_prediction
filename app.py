import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="House Price Prediction Studio", layout="centered")

st.title("🏡 House Price Prediction Studio")
st.write("Enter the property details below to estimate the house value.")

# Load the Linear Regression model
@st.cache_resource
def load_model():
    with open("model (2).pkl", "rb") as file:
        model = pickle.load(file)
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model file: {e}")
    st.stop()

# Input Form matching the exact feature names seen at fit time
with st.form("house_price_form"):
    st.subheader("Property Characteristics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        square_footage = st.number_input("Square Footage (sq ft)", min_value=100, max_value=20000, value=2000, step=50)
        num_bedrooms = st.number_input("Number of Bedrooms", min_value=1, max_value=10, value=3, step=1)
        num_bathrooms = st.number_input("Number of Bathrooms", min_value=1.0, max_value=10.0, value=2.0, step=0.5)
        year_built = st.number_input("Year Built", min_value=1800, max_value=2026, value=2010, step=1)

    with col2:
        lot_size = st.number_input("Lot Size (sq ft)", min_value=500, max_value=100000, value=5000, step=100)
        garage_size = st.number_input("Garage Size (Car Capacity)", min_value=0, max_value=5, value=2, step=1)
        neighborhood_quality = st.slider("Neighborhood Quality (1-10)", min_value=1, max_value=10, value=7, step=1)

    submit_button = st.form_submit_button(label="Predict House Price")

if submit_button:
    # Pass features with the exact column names required by the trained model
    input_df = pd.DataFrame([{
        "Square_Footage": square_footage,
        "Num_Bedrooms": num_bedrooms,
        "Num_Bathrooms": num_bathrooms,
        "Year_Built": year_built,
        "Lot_Size": lot_size,
        "Garage_Size": garage_size,
        "Neighborhood_Quality": neighborhood_quality
    }])
    
    try:
        prediction = model.predict(input_df)[0]
        st.success(f"Estimated Price: **${prediction:,.2f}**")
    except Exception as e:
        st.error(f"Error executing prediction: {e}")
