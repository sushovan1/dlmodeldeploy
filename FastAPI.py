# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 12:14:04 2025

@author: CSU5KOR
"""
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
import uvicorn

app = FastAPI()

#path=r"C:\Users\CSU5KOR\OneDrive - Bosch Group\GROW_PPGPL\Codes\Model"
# Load the Keras autoencoder model
model = tf.keras.models.load_model("best_model_v2.keras")

# Set anomaly threshold (adjust as needed)
anomaly_threshold = 10

# Pydantic model for request body
class InputData(BaseModel):
    AFT_TEMP: float
    FWD_TEMP: float
    RPM: float
    Temp_difference: float

@app.post("/predict")
async def predict_anomaly(input_data: InputData):
    try:
        # Convert input data to numpy array
        input_array = np.array([input_data.AFT_TEMP, input_data.FWD_TEMP, input_data.RPM, input_data.Temp_difference])
        print("Array construction done")
        # Predict using the loaded model
        reconstructed = model.predict(input_array.reshape(1,1,4))
        print("prediction done")
        # Calculate the error vector
        error_vector = input_array - reconstructed

        # Calculate the Mean Squared Error (MSE)
        mae = np.mean(np.abs(error_vector))

        # Flag as anomaly if MSE exceeds the threshold
        is_anomaly = mae > anomaly_threshold

        response = {
            "input": {
                "AFT_TEMP": input_data.AFT_TEMP,
                "FWD_TEMP": input_data.FWD_TEMP,
                "RPM": input_data.RPM,
                "Temp_difference": input_data.Temp_difference
            },
            "reconstructed": reconstructed[0].tolist(),
            "error_vector": error_vector[0].tolist(),
            "mse": mae,
            "is_anomaly": int(is_anomaly)
        }

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

# Run the app with uvicorn on host 0.0.0.0 and port 8000
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
