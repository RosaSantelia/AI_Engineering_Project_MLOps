# app/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

app = FastAPI()

# Caricamento modello e tokenizer all'avvio
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

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
        # Tokenizzazione
        inputs = tokenizer(input.text, return_tensors="pt", truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=1)
        scores = probs.tolist()[0]

        labels = ["negative", "neutral", "positive"]
        predictions = [{"label": label, "score": score} for label, score in zip(labels, scores)]

        return {"predictions": predictions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
