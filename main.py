"""
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

# Create an instance of FastAPI
app = FastAPI()

# Load the best model and the encoders used for categorical features
with open("best_model.pkl", "rb") as file:
    model = pickle.load(file)

# Load the LabelEncoders from the file
with open("label_encoders.pkl", "rb") as file:
    label_encoders = pickle.load(file)


# Define a Pydantic model for request body validation
class CarFeatures(BaseModel):
    # Numerical Features
    mileage: float
    engine_volume: float
    cylinders: float
    doors: int
    airbags: int
    prod_year: int
    leather_interior: int
    turbo: int

    # Categorical Features (One-hot encoded)
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

    # Categorical Features (Label encoded)
    manufacturer: str
    model: str
    fuel_type: str
    gear_box_type: str
    drive_wheels: str
    wheel: str
    color: str


@app.post("/predict_price/")
def predict_car_price(car: CarFeatures):
    # Encode categorical features using the loaded label encoders
    encoded_features = []

    try:
        # Use the label encoders to transform the input values
        manufacturer_encoded = label_encoders["manufacturer"].transform(
            [car.manufacturer]
        )[0]
        model_encoded = label_encoders["model"].transform([car.model])[0]
        fuel_type_encoded = label_encoders["fuel_type"].transform([car.fuel_type])[0]
        gear_box_type_encoded = label_encoders["gear_box_type"].transform(
            [car.gear_box_type]
        )[0]
        drive_wheels_encoded = label_encoders["drive_wheels"].transform(
            [car.drive_wheels]
        )[0]
        wheel_encoded = label_encoders["wheel"].transform([car.wheel])[0]
        color_encoded = label_encoders["color"].transform([car.color])[0]
    except KeyError as e:
        raise ValueError(
            f"Unknown category: {e} in the input data. Please provide a valid value."
        )

    # Append the encoded features
    encoded_features = [
        manufacturer_encoded,
        model_encoded,
        fuel_type_encoded,
        gear_box_type_encoded,
        drive_wheels_encoded,
        wheel_encoded,
        color_encoded,
    ]

    # Prepare the input data for the model
    input_data = np.array(
        [
            [
                car.mileage,
                car.engine_volume,
                car.cylinders,
                car.doors,
                car.airbags,
                car.prod_year,
                car.leather_interior,
                car.turbo,
                # Add one-hot encoded features for Drive, Gear and Fuel
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
                car.NewFuel_Plug_in_Hybrid,
            ]
            + encoded_features  # Append the encoded categorical features
        ]
    )

    # Make prediction (Log_Price)
    predicted_log_price = model.predict(
        input_data
    )  # Prediction on log-transformed price
    predicted_price = np.expm1(predicted_log_price)  # Convert back to actual price

    return {"predicted_price": predicted_price[0]}  # Return the predicted price
"""
