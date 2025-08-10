# app/model.py
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
import torch.nn.functional as F

# Nome del modello di Hugging Face
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"

class SentimentModel:
    """
    Una classe per incapsulare il modello di analisi del sentiment.
    Gestisce il caricamento del modello, il tokenizer e la predizione.
    """
    def __init__(self, device=None):
        # Imposta il dispositivo (CPU o GPU) per l'inferenza
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Inizializzazione di SentimentModel su dispositivo: {self.device}")
        
        # Carica il tokenizer e il modello pre-addestrato
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        
        # Sposta il modello sul dispositivo specificato (es. 'cuda')
        self.model.to(self.device)
        self.model.eval() # Imposta il modello in modalità valutazione
        
    def predict(self, texts):
        """
        Esegue la predizione del sentiment per una lista di testi.

        Args:
            texts (list[str]): Una lista di stringhe di testo da analizzare.
        
        Returns:
            list[dict]: Una lista di dizionari, ognuno contenente l'ID dell'etichetta,
                        l'etichetta testuale ('negative', 'neutral', 'positive') e i punteggi
                        di probabilità per ogni sentiment.
        """
        # Tokenizzazione dei testi
        inputs = self.tokenizer(texts, truncation=True, padding=True, return_tensors="pt")
        
        # Sposta i tensori di input sul dispositivo del modello
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Esegue l'inferenza senza calcolare i gradienti
        with torch.no_grad():
            out = self.model(**inputs)
            probs = F.softmax(out.logits, dim=-1).cpu().numpy()
            
        # Trova l'indice dell'etichetta con la probabilità più alta
        labels = np.argmax(probs, axis=1)
        
        # Mappa i risultati in un formato leggibile
        label_mapping = ["negative", "neutral", "positive"]
        results = []
        for l, p in zip(labels, probs):
            results.append({
                "label_id": int(l),
                "label": label_mapping[int(l)],
                "scores": p.tolist()
            })
            
        return results