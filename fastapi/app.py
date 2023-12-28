import uvicorn
import pandas as pd 
import json
from pydantic import BaseModel
from typing import Literal, List, Union
from fastapi import FastAPI, File, UploadFile
import joblib
from joblib import dump, load



description = """
###Bienvenue sur Getaround FastAPI. 
L'API de Get Around vous suggère le prix de location optimal par jour pour la voiture de votre choix.
"""


# initialise API object
app = FastAPI(
    title="GETAROUND API",
    description=description,
    version="1.0",
    openapi_tags= [
    {
        "name": "Home",
        "description": "Page d'accueil de l'API de Get Around."
    },
    {
        "name": "Predicts",
        "description": "API Get Around avec la méthode POST ou GET."
    }
]
)


tags_metadata = [
    {
        "name": "Introduction Endpoints",
        "description": "Des Endpoints simple a utiliser!",
    },
    {
        "name": "Prediction",
        "description": "Prédiction du prix de location basée sur un modèle de machine Learning"
    }
]


class PredictionFeatures(BaseModel):
    model_key: str = "Peugeot"
    mileage: int = 123886
    engine_power: int = 125
    fuel: str = "petrol"
    paint_color: str = "black"
    car_type: str = "convertible"
    private_parking_available: bool = True
    has_gps: bool = False
    has_air_conditioning: bool = False
    automatic_car: bool = False
    has_getaround_connect: bool = False
    has_speed_regulator: bool = False
    winter_tires: bool = True
    
    
@app.get("/", tags = ["Introduction Endpoint"])
async def index():
    message = "Hello! This `/` is the most simple and default endpoint for the API`"
    return message


@app.get("/preview", tags=["Preview"])
async def preview(rows: int):
    """ Give a preview of the dataset : Number of rows"""
    data = pd.read_csv("get_around_pricing_project.csv")
    preview = data.head(rows)
    return preview.to_dict()

@app.post("/predict", tags = ["Price prediction"])
async def predict(predictionFeatures: PredictionFeatures):
    # Read data 
    data = pd.DataFrame(dict(predictionFeatures), index=[0])
    # Load model
    loaded_model = load("model.joblib")
    # Prediction
    prediction = loaded_model.predict(data)
    # Format response
    response ={"predictions": prediction.tolist()[0]}
    return response

if __name__=="__main__":
    uvicorn.run(app, host="0.0.0.0", port=4000, debug=True, reload=True)


