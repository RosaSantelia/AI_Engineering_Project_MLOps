# app/main.py
from fastapi import FastAPI
from pydantic import BaseModel
from app.model import SentimentModel
from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

# Modello singleton
model = SentimentModel()

app = FastAPI(title="Sentiment API")

class TextIn(BaseModel):
    text: str

# Metrics
REQUESTS = Counter("sentiment_requests_total", "Total sentiment requests")
ERRORS = Counter("sentiment_errors_total", "Total errors")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(payload: TextIn):
    REQUESTS.inc()
    try:
        res = model.predict([payload.text])
        return {"predictions": res}
    except Exception as e:
        ERRORS.inc()
        return {"error": str(e)}

@app.get("/metrics")
def metrics():
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)
