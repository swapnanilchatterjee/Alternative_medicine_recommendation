import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from tensorflow import keras
import os

# Set Streamlit title and description
st.title("💊 Medicine Recommendation System")
st.write("Enter a medicine composition to get AI-powered alternative medicine recommendations.")

# Define file paths dynamically
BASE_DIR = os.path.dirname(os.path.abspath("C:\\Users\\tmann\\snu\\Alternative_medicine_recommendation\\main.py"))
model_path = os.path.join(BASE_DIR, "medicine_recommender.keras")
vectorizer_path = os.path.join(BASE_DIR, "vectorizer.pkl")
dataset_path = os.path.join(BASE_DIR, "A_Z_medicines_dataset_of_India.csv")

# Load the trained model
try:
    model = keras.models.load_model(model_path)
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model: {str(e)}")

# Load the TF-IDF vectorizer
try:
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)
    st.success("✅ Vectorizer loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading vectorizer: {e}")

# Load the dataset
try:
    df = pd.read_csv(dataset_path)
    st.success("✅ Dataset loaded successfully!")
    
    # Ensure column names are trimmed (in case of extra spaces)
    df.columns = df.columns.str.strip()
except Exception as e:
    st.error(f"❌ Error loading dataset: {e}")

# Sidebar filters
st.sidebar.header("🔍 Filters")
price_range = st.sidebar.slider("Select Price Range (₹)", min_value=0, max_value=500, value=(0, 500))
sort_order = st.sidebar.radio("Sort by Price", ["Low to High", "High to Low"])

# Function to recommend medicines
def recommend_medicine(composition_query, top_n=5):
    try:
        # Ensure the composition is lowercase before transformation
        query_vec = vectorizer.transform([composition_query.lower()])
        
        # Predict medicine scores
        predictions = model.predict(query_vec)[0]

        # Get top N recommended medicines
        top_indices = predictions.argsort()[-top_n:][::-1]
        confidence_scores = predictions[top_indices]  # Get model confidence

        # Fetch medicine details
        recommendations = df.iloc[top_indices][["name", "price(₹)"]]
        recommendations["Confidence"] = (confidence_scores * 100).round(2)  # Convert to percentage

        # Apply price filtering
        recommendations = recommendations[
            (recommendations["price(₹)"] >= price_range[0]) & 
            (recommendations["price(₹)"] <= price_range[1])
        ]

        # Apply sorting
        if sort_order == "Low to High":
            recommendations = recommendations.sort_values(by="price(₹)", ascending=True)
        else:
            recommendations = recommendations.sort_values(by="price(₹)", ascending=False)

        return recommendations

    except ValueError as e:
        st.error("❌ The entered composition contains unseen words. Please try another or check spelling.")
        st.stop()
    except Exception as e:
        return str(e)

# User Input for Composition Query
composition_query = st.text_input("Enter Medicine Composition:")

if st.button("Get Recommendations"):
    if composition_query:
        recommendations = recommend_medicine(composition_query)
        if isinstance(recommendations, str):
            st.error(f"Error: {recommendations}")
        elif recommendations.empty:
            st.warning("⚠ No medicines found in the selected price range.")
        else:
            st.write("### 🔹 Recommended Medicines:")
            st.dataframe(recommendations)

            # Display medicine images (if dataset has an 'image_url' column)
            if "image_url" in df.columns:
                st.write("### 🖼 Medicine Images:")
                for _, row in recommendations.iterrows():
                    st.image(row["image_url"], caption=row["name"], width=150)
                    st.write(f"💰 Price: ₹{row['price(₹)']}")
                    st.write(f"📊 Confidence: {row['Confidence']}%")
                    st.markdown("---")  # Adds a separator

    else:
        st.warning("⚠ Please enter a composition to get recommendations.")
