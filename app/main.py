# app/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from .model import SentimentModel

# Creazione dell'istanza dell'applicazione FastAPI
app = FastAPI()

# Inizializzazione del modello all'avvio dell'app.
# Questo garantisce che il modello venga caricato una sola volta in memoria.
sentiment_model = SentimentModel()

# Definizione del modello di dati per il corpo della richiesta POST
class TextInput(BaseModel):
    text: str

# Endpoint per la root dell'API
@app.get("/")
async def root():
    return {"message": "API Sentiment Analysis attiva"}

# Endpoint per il controllo dello stato di salute
@app.get("/health")
async def health():
    return {"status": "ok"}

# Endpoint per la predizione del sentiment
@app.post("/predict")
async def predict(input: TextInput):
    """
    Accetta un testo e restituisce il sentiment preditto.
    """
    try:
        # La predizione viene eseguita chiamando il metodo 'predict' della classe SentimentModel
        # Nota: il metodo predict accetta una lista di stringhe, quindi passiamo [input.text]
        predictions = sentiment_model.predict(texts=[input.text])
        
        # Il metodo predict restituisce una lista, quindi prendiamo il primo elemento
        # se si analizza un solo testo alla volta.
        return {"predictions": predictions[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
