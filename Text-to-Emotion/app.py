from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.keras
import joblib

model_uri = "models:/TextEmotion/Production"
model = mlflow.keras.load_model(model_uri)

vectorizer = joblib.load("vectorizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")

app = FastAPI(title="MLflow Sklearn Model API")

class InputData(BaseModel):
    text: str

@app.post("/predict")
def predict(data: InputData):
    X = vectorizer.transform([data.text]).toarray()
    preds = model.predict(X)
    idx = preds.argmax(axis=1)[0]
    label = label_encoder.inverse_transform([idx])[0]
    return {"prediction": label}