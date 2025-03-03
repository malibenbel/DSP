from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
import uvicorn
'''

# Carica il modello
with open("best_model.pkl", "rb") as file:
    model = pickle.load(file)

# Carica i LabelEncoder (solo se li usi davvero)
with open("label_encoders.pkl", "rb") as file:
    label_encoders = pickle.load(file)

app = FastAPI()

# Classe CarFeatures senza i vecchi campi fuel_type, gear_box_type, drive_wheels
class CarFeatures(BaseModel):
    # Numerici
    levy: float
    mileage: float
    engine_volume: float
    cylinders: float
    doors: int
    airbags: int
    prod_year: int
    leather_interior: int  # 0 o 1
    turbo: int  # 0 o 1

    # One-hot
    NewDrive_4x4: bool
    NewDrive_Front: bool
    NewDrive_Rear: bool
    NewGear_Automatic: bool
    NewGear_Manual: bool
    NewGear_Tiptronic: bool
    NewGear_Variator: bool
    NewFuel_CNG: bool
    NewFuel_Diesel: bool
    NewFuel_Hybrid: bool
    NewFuel_Hydrogen: bool
    NewFuel_LPG: bool
    NewFuel_Petrol: bool
    NewFuel_Plug_in_Hybrid: bool

    # Categoriali (label-encodate)
    manufacturer: str
    model: str
    category: str
    wheel: str
    color: str

@app.post("/predict_price/")
def predict_car_price(car: CarFeatures):
    try:
        # Trasforma i campi categorici con i LabelEncoder
        manufacturer_encoded = label_encoders["manufacturer"].transform([car.manufacturer])[0]
        model_encoded = label_encoders["model"].transform([car.model])[0]
        category_encoded = label_encoders["category"].transform([car.category])[0]
        wheel_encoded = label_encoders["wheel"].transform([car.wheel])[0]
        color_encoded = label_encoders["color"].transform([car.color])[0]
    except KeyError as e:
        # Se l'utente manda un valore fuori dal vocabolario
        raise ValueError(f"Valore sconosciuto per la variabile: {e}")

    # Metti insieme gli encodings
    encoded_categoricals = [
        manufacturer_encoded,
        model_encoded,
        category_encoded,
        wheel_encoded,
        color_encoded
    ]

    # Prepara i valori in un ordine coerente con quello del training
    input_data = np.array([[
        car.levy,
        car.mileage,
        car.engine_volume,
        car.cylinders,
        car.doors,
        car.airbags,
        car.prod_year,
        car.leather_interior,
        car.turbo,
        car.NewDrive_4x4,
        car.NewDrive_Front,
        car.NewDrive_Rear,
        car.NewGear_Automatic,
        car.NewGear_Manual,
        car.NewGear_Tiptronic,
        car.NewGear_Variator,
        car.NewFuel_CNG,
        car.NewFuel_Diesel,
        car.NewFuel_Hybrid,
        car.NewFuel_Hydrogen,
        car.NewFuel_LPG,
        car.NewFuel_Petrol,
        car.NewFuel_Plug_in_Hybrid
    ] + encoded_categoricals])

    # Predici sul log(price)
    predicted_log_price = model.predict(input_data)
    predicted_price = np.expm1(predicted_log_price)

    return {"predicted_price": float(predicted_price[0])}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

'''

import streamlit as st
import requests
import json
from backend import FastAPI  # If needed


# FastAPI endpoint (Ensure your API is running before using this)
API_URL = "http://127.0.0.1:8000/predict_price/"

# 🎨 Streamlit UI
st.title("🚗 Car Price Prediction App")
st.write("Enter car details to predict the estimated price.")

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



