
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from tensorflow import keras
import os

# Set Streamlit title and description
st.title("Medicine Recommendation System")
st.write("Enter the composition to get the top recommended medicines based on AI predictions.")

# Print working directory for debugging
# st.write("Current Directory:", os.getcwd())
# Load the trained model
try:
    model = keras.models.load_model("C:\\Users\\tmann\\snu\\Alternative_medicine_recommendation\\medicine_recommender.keras")

    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {str(e)}")

# Load the vectorizer
vectorizer_path = "C:\\Users\\tmann\\snu\\Alternative_medicine_recommendation\\vectorizer.pkl"
try:
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)  # Load inside 'with' block
    print("Vectorizer loaded successfully!")
except Exception as e:
    print(f"Error loading vectorizer: {e}")

# Load the dataset
dataset_path = "C:\\Users\\tmann\\snu\\Alternative_medicine_recommendation\\A_Z_medicines_dataset_of_India.csv"

try:
    df = pd.read_csv(dataset_path)
    print("Dataset loaded successfully!")
except Exception as e:
    print(f"Error loading dataset: {e}")

# Function to recommend medicines
def recommend_medicine(composition_query, top_n=5):
    try:
        # Convert input text into TF-IDF vector
        query_vec = vectorizer.transform([composition_query.lower()]).toarray()

        # Make predictions using the trained model
        predictions = model.predict(query_vec)[0]

        # Get indices of the top N predicted medicines
        top_indices = predictions.argsort()[-top_n:][::-1]

        # Fetch medicine names and prices
        recommendations = df.iloc[top_indices][["name", "price(₹)"]]
        return recommendations
    except Exception as e:
        return str(e)

# User Input for Composition Query
composition_query = st.text_input("Enter Medicine Composition:")

if st.button("Get Recommendations"):
    if composition_query:
        recommendations = recommend_medicine(composition_query)
        if isinstance(recommendations, str):
            st.error(f"Error: {recommendations}")
        else:
            st.write("### Recommended Medicines:")
            st.dataframe(recommendations)
    else:
        st.warning("Please enter a composition to get recommendations.")
