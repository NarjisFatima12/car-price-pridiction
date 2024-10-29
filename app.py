# Import necessary libraries
import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

# Load Model
model = joblib.load('model1.keras')

# Define the Streamlit app
def main():
    st.title("Car Feature Prediction App")

    # Create user input fields for each feature
    brand = st.number_input("Brand", min_value=0, max_value=60)
    model = st.number_input("Model", min_value=0, max_value=1620)
    model_year = st.number_input("Model Year", min_value=1974, max_value=2024)
    milage = st.number_input("Milage", min_value=0)
    fuel_type = st.number_input("Fuel Type", min_value=0, max_value=6)
    engine = st.number_input("Engine", min_value=0, max_value=1000)
    transmission = st.number_input("Transmission", min_value=0, max_value=31)
    ext_col = st.number_input("Exterior Color Code", min_value=0, max_value=119)
    int_col = st.number_input("Interior Color Code", min_value=0, max_value=73)
    accident = st.selectbox("Accident History", [0, 1])

    # Create a dictionary with the input values
    input_data = {
        'brand': brand,
        'model': model,
        'model_year': model_year,
        'milage': milage,
        'fuel_type': fuel_type,
        'engine': engine,
        'transmission': transmission,
        'ext_col': ext_col,
        'int_col': int_col,
        'accident': accident
    }

    # Convert input data to a DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Display the DataFrame
    st.write("Input Data")
    st.write(input_df)

    # Make prediction when the button is pressed
    if st.button("Predict"):
        # Make a prediction using the model
        prediction = model.predict(input_df)
        st.write("Prediction:", prediction[0])

if __name__ == "__main__":
    main()
