import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from tensorflow import keras
from flask import Flask, request, jsonify
from flask_cors import CORS
import os

# Print working directory for debugging
print("Current Directory:", os.getcwd())

# Load the trained model
try:
    model = keras.models.load_model("medicine_recommender.keras")
    print("Model loaded successfully!")
except Exception as e:
    print("Error loading model:", str(e))

# Load the vectorizer
try:
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    print("Vectorizer loaded successfully!")
except Exception as e:
    print("Error loading vectorizer:", str(e))

# Load the dataset
try:
    df = pd.read_csv("A_Z_medicines_dataset_of_India.csv")
    print("Dataset loaded successfully!")
except Exception as e:
    print("Error loading dataset:", str(e))


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
        recommendations = df.iloc[top_indices][["name", "price(₹)"]].to_dict(orient="records")

        return recommendations
    except Exception as e:
        return {"error": str(e)}


# Flask API Setup
app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Home route to prevent 404 errors
@app.route("/", methods=["GET"])
def home():
    return "Medicine Recommendation API is running!"


@app.route("/recommend", methods=["POST"])
def get_recommendation():
    data = request.json
    composition_query = data.get("composition", "")

    if not composition_query:
        return jsonify({"error": "Composition is required"}), 400

    recommendations = recommend_medicine(composition_query)
    return jsonify({"recommendations": recommendations})


if __name__ == "__main__":
    app.run(port=5000, debug=True)
