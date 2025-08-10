# app/model.py
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"

class SentimentModel:
    def __init__(self, device=None):
        # device: 'cpu' o 'cuda'
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # tokenizer e modello
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        self.model.to(self.device)

    def predict(self, texts):
        # texts: lista di stringhe
        inputs = self.tokenizer(texts, truncation=True, padding=True, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self.model(**inputs)
            probs = torch.nn.functional.softmax(out.logits, dim=-1).cpu().numpy()
        labels = np.argmax(probs, axis=1)
        # mapping: 0 negative,1 neutral,2 positive (cardiffnlp convention)
        return [{"label_id": int(int(l)), "label": ["negative","neutral","positive"][int(l)], "scores": p.tolist()} for l, p in zip(labels, probs)]
