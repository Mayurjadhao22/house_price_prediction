import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st

# Set page configuration
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="centered",
)

# Resolve directory paths dynamically
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "linear.pkl")


@st.cache_resource
def load_model_from_file(file_path):
    """Load trained model from local path."""
    with open(file_path, "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_model_from_bytes(uploaded_file):
    """Load model uploaded via Streamlit interface."""
    return pickle.load(uploaded_file)


def main():
    st.title("🏠 House Price Prediction App")
    st.write("Enter the property specifications below to estimate the market price.")

    model = None

    # Load local model.pkl or prompt for upload
    if os.path.exists(DEFAULT_MODEL_PATH):
        model = load_model_from_file(DEFAULT_MODEL_PATH)
    else:
        st.warning(f"`model.pkl` was not found at `{DEFAULT_MODEL_PATH}`.")
        uploaded_model = st.file_uploader(
            "Please upload your trained model file (`model.pkl`):",
            type=["pkl", "pickle"],
        )
        if uploaded_model is not None:
            model = load_model_from_bytes(uploaded_model)
            st.success("Model loaded successfully!")
        else:
            st.info("Upload a valid model file above to continue.")
            st.stop()

    st.subheader("Property Features")

    col1, col2 = st.columns(2)

    with col1:
        sqft = st.number_input("Square Footage (sq ft)", min_value=100, max_value=20000, value=1500, step=50)
        bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=3, step=1)
        bathrooms = st.number_input("Bathrooms", min_value=1.0, max_value=10.0, value=2.0, step=0.5)

    with col2:
        age = st.number_input("Property Age (Years)", min_value=0, max_value=150, value=10, step=1)
        location_score = st.slider("Location Rating (1-10)", min_value=1, max_value=10, value=7)
        garage_spaces = st.number_input("Garage Spaces", min_value=0, max_value=5, value=1, step=1)

    # Prepare DataFrame matching typical regression model inputs
    input_data = pd.DataFrame(
        [
            {
                "Square_Footage": sqft,
                "Bedrooms": bedrooms,
                "Bathrooms": bathrooms,
                "Property_Age": age,
                "Location_Score": location_score,
                "Garage_Spaces": garage_spaces,
            }
        ]
    )

    st.markdown("---")

    if st.button("Predict Price", type="primary"):
        try:
            # Generate numerical prediction
            predicted_price = model.predict(input_data)[0]
            
            st.subheader("Estimated Property Value")
            st.success(f"💰 **${predicted_price:,.2f}**")

        except Exception as e:
            st.error(f"Error executing prediction: {e}")


if __name__ == "__main__":
    main()
