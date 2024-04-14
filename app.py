import functools
import pickle
import pandas as pd
import streamlit as st
from src.Pipelines.prediction_pipeline import PredictPipeline
from src.Pipelines.prediction_pipeline import CustomData

# Load the trained model from the pickle file
@functools.lru_cache()
def load_model():
    with open(r'artifacts/model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model
def load_preprocessor():
    with open(r'artifacts/preprocessor.pkl', 'rb') as file:
        preprocessor = pickle.load(file)
    return preprocessor

# Define the Streamlit app
def main():
    # Set the title of the app
    st.title('Diamond Price Prediction App')

    # Add some description or instructions
    st.write('Enter the features of the diamond to predict its price.')


    # Load the trained model outside the form
    model = load_model()
    preprocessor = load_preprocessor()
    # Preprocess the features outside the form
    
    Carat = st.number_input('Carat', min_value=0.2, max_value=5.01, value=1.0, step=0.01, format='%.2f')
    Cut = st.selectbox('Cut', ['Ideal', 'Premium', 'Very Good', 'Good', 'Fair'])
    Color =  st.selectbox('Color', ['D', 'E', 'F', 'G', 'H', 'I', 'J'])
    Clarity = st.selectbox('Clarity', ['IF', 'VVS1', 'VVS2', 'VS1', 'VS2', 'SI1', 'SI2', 'I1'])
    Depth = st.number_input('Depth', min_value=50.0, max_value=75.0, value=60.0, step=0.1, format='%.1f')
    Table = st.number_input("table")
    X = st.number_input("x")
    Y = st.number_input("y")
    Z = st.number_input("z")
    custom_inputs = CustomData(Carat, Depth, Table, X, Y, Z, Cut, Color, Clarity)
    custom_df = custom_inputs.get_data_as_dataframe()
    # Make predictions when the user clicks the 'Predict' button
    if st.button("Predict"):   
        data_scaled=preprocessor.transform(custom_df)
        preds = model.predict(data_scaled)
        st.write(preds)

# Run the Streamlit app
if __name__ == '__main__':
    main()