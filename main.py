from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
import uvicorn


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






