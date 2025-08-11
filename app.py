from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from typing import Dict, Any
import pandas as pd

import tensorflow as tf 

app=FastAPI()

interpreter = tf.lite.Interpreter(model_path="hand1_5_v3.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

class ModelInput(BaseModel):
     data: Dict[str, Any] 

@app.post("/predict")

def predict(input_data: ModelInput):
    row = input_data.data
    X = pd.DataFrame([row]).to_numpy().astype(np.float32)
    
    # Reshape if needed to match the model's expected input shape
    X = X.reshape(input_details[0]['shape'])
    
    interpreter.set_tensor(input_details[0]['index'], X)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])[0]
    predicted_class = int(prediction.argmax())
    confidence = float(max( prediction))

    return {
        "predicted_class": predicted_class,
        "confidence": confidence
    }
    