from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
import uvicorn
import streamlit as st
import requests
from backend import FastAPI  # If needed

app = FastAPI()  # ✅ This should be present in main.py
@app.get("/")
def read_root():
    return {"message": "FastAPI is running!"}

# FastAPI endpoint (Ensure your API is running before using this)
API_URL = "http://127.0.0.1:8000/predict_price/"

# 🎨 Streamlit UI
st.title("Car Price Prediction Application")
st.write("Enter the car details you prefer to see your predicted price!")

# User Inputs
manufacturer = st.selectbox("Select Manufacturer", ["LEXUS", "FORD", "CHEVROLET", "HONDA"])
model = st.text_input("Car Model")
category = st.selectbox("Select Category", ["Jeep", "Hatchback", "Sedan"])
color = st.selectbox("Select Color", ["Silver", "Black", "White"])
wheel = st.selectbox("Wheel Drive", ["Left wheel", "Right wheel"])

year = st.number_input("Production Year", min_value=2000, max_value=2025, value=2015)
mileage = st.number_input("Mileage", min_value=0, max_value=300000, value=50000)
engine_volume = st.number_input("Engine Volume (L)", min_value=1.0, max_value=6.0, value=2.0, step=0.1)
cylinders = st.number_input("Cylinders", min_value=1, max_value=12, value=4)
doors = st.number_input("Number of Doors", min_value=2, max_value=5, value=4)
airbags = st.number_input("Number of Airbags", min_value=0, max_value=16, value=8)
leather_interior = st.selectbox("Leather Interior", ["Yes", "No"])
turbo = st.selectbox("Turbo", ["Yes", "No"])

# Convert categorical values
leather_interior = 1 if leather_interior == "Yes" else 0
turbo = 1 if turbo == "Yes" else 0

# 🔘 Button to Predict
if st.button("Predict Price"):
    input_data = {
        "manufacturer": manufacturer,
        "model": model,
        "category": category,
        "color": color,
        "wheel": wheel,
        "prod_year": year,
        "mileage": mileage,
        "engine_volume": engine_volume,
        "cylinders": cylinders,
        "doors": doors,
        "airbags": airbags,
        "leather_interior": leather_interior,
        "turbo": turbo
    }

    # Send data to FastAPI
    response = requests.post(API_URL, json=input_data)

    if response.status_code == 200:
        predicted_price = response.json()["predicted_price"]
        st.success(f"💰 Estimated Car Price: **${predicted_price:.2f}**")
    else:
        st.error("Error: Could not get prediction. Make sure API is running.")



