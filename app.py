from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from typing import Dict, Any
import pandas as pd

import tensorflow as tf 

app=FastAPI()
model = tf.keras.models.load_model("hand1_5_v3.h5")


class ModelInput(BaseModel):
     data: Dict[str, Any] 

@app.post("/predict")

def predict(input_data: ModelInput):
    row = input_data.data
    X = pd.DataFrame([row])
    
    bodylang_prob = model.predict(X)[0]  # e.g. [0.1, 0.7, 0.2]
    predicted_class = int(bodylang_prob.argmax())
    confidence = float(max(bodylang_prob))

    return {
        "predicted_class": predicted_class,
        "confidence": confidence
    }
    