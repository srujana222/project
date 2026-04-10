import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
model=joblib.load("git.pkl")
encoder = joblib.load("encoder.pkl")
#load saved model
app = FastAPI()
#input schema
class Github(BaseModel):
    primary_language:str
    forks:int
    open_issues:int
    watchers:int
    created_year:int
#predict Endpoint
@app.post("/predict")
def predict(data:Github):
    df=pd.DataFrame([data.dict()])
    df["primary_language"] = encoder.transform(df["primary_language"])
    pred=model.predict(df)[0]
    return {
    "predicted_stars": int(pred)
    }
