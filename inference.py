# inference.py
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import os

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
FINE_TUNED_MODEL_PATH = "./model_finetuned"

def load_model():
    if os.path.isdir(FINE_TUNED_MODEL_PATH):
        tokenizer = AutoTokenizer.from_pretrained(FINE_TUNED_MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(FINE_TUNED_MODEL_PATH)
    else:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

    nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    return nlp

def predict_sentiment(nlp, text: str):
    return nlp(text)

if __name__ == "__main__":
    nlp = load_model()
    test_text = "I love this project!"
    prediction = predict_sentiment(nlp, test_text)
    print(f"Testo: {test_text}")
    print(f"Predizione: {prediction}")
