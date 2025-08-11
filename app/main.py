from fastapi import FastAPI, Request, HTTPException, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from app.model import SentimentModel

app = FastAPI()
sentiment_model = SentimentModel()

templates = Jinja2Templates(directory="app/templates")  # crea questa cartella e ci metti l'html

@app.get("/predict", response_class=HTMLResponse)
async def get_predict_form(request: Request):
    # Mostra il form per inserire il tweet
    return templates.TemplateResponse("predict_form.html", {"request": request, "result": None})

@app.post("/predict", response_class=HTMLResponse)
async def post_predict_form(request: Request, text: str = Form(...)):
    try:
        prediction = sentiment_model.predict(texts=[text])[0]
    except Exception as e:
        prediction = f"Errore durante la predizione: {str(e)}"
    return templates.TemplateResponse("predict_form.html", {"request": request, "result": prediction, "text": text})