
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from .model import SentimentModel

app = FastAPI()

sentiment_model = SentimentModel()

class TextInput(BaseModel):
    text: str

@app.get("/")
async def root():
    return {"message": "API Sentiment Analysis attiva"}

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(input: TextInput):
    try:
        predictions = sentiment_model.predict(texts=[input.text])
        return {"predictions": predictions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
