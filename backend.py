from fastapi import FastAPI
import pickle
import pandas as pd

# Initialize FastAPI app
app = FastAPI()

# Load trained model
with open("best_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.get("/")
def home():
    return {"message": "Car Price Prediction API is running!"}


# Prediction Endpoint
@app.post("/predict")
def predict_price(data: dict):
    # Convert input dictionary to DataFrame
    input_data = pd.DataFrame([data])

    # Make prediction
    prediction = model.predict(input_data)[0]
    
    return {"predicted_price": round(prediction, 2)}