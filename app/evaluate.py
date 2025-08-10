# app/evaluate.py
# Importazione delle librerie necessarie
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report
import numpy as np
import warnings
from tqdm import tqdm

# Sopprimi il warning UserWarning per chiarezza nell'output
warnings.filterwarnings("ignore", category=UserWarning, module='huggingface_hub.file_download')

# -----------------------------------------------------------------------------
# Passo 1: Implementazione del Modello RoBERTa per l'Analisi
# Caricamento del nome del modello
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
print("Caricamento del tokenizer e del modello per la valutazione...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, max_length=512)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
print("Modello e tokenizer caricati con successo.")

# Mappa le etichette numeriche del modello a etichette testuali
# 0 -> negativo, 1 -> neutro, 2 -> positivo (convenzione di cardiffnlp)
label_mapping = ["negative", "neutral", "positive"]

def predict_sentiment_batch(texts, model, tokenizer, batch_size=32):
    """
    Esegue la predizione del sentiment per una lista di testi in batch.
    
    Args:
        texts (list[str]): Una lista di stringhe di testo da analizzare.
        model (AutoModelForSequenceClassification): Il modello da utilizzare.
        tokenizer (AutoTokenizer): Il tokenizer da utilizzare.
        batch_size (int): La dimensione del batch per l'elaborazione.
        
    Returns:
        list[str]: Una lista con il sentiment predetto per ogni testo.
    """
    all_predictions = []
    # Itera sui testi in batch
    for i in tqdm(range(0, len(texts), batch_size), desc="Analisi in corso"):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, truncation=True, padding=True, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=1)
        
        predictions = torch.argmax(probs, dim=1).tolist()
        all_predictions.extend([label_mapping[p] for p in predictions])
    
    return all_predictions

# -----------------------------------------------------------------------------
# Passo 2: Utilizzo di un Dataset Pubblico per la Valutazione
# Caricamento del dataset "tweet_eval" da Hugging Face
print("\nCaricamento del dataset pubblico 'tweet_eval' (sotto-set 'sentiment')...")
try:
    # Carichiamo il sotto-set 'sentiment' che è più piccolo e focalizzato
    dataset = load_dataset("tweet_eval", "sentiment")
except Exception as e:
    print(f"Errore durante il caricamento del dataset: {e}")
    print("Assicurati di essere connesso a internet e di aver installato la libreria 'datasets'.")
    exit()

df = pd.DataFrame(dataset["test"])

# L'etichetta del sentiment in questo dataset è: 0 -> negativo, 1 -> neutro, 2 -> positivo
# Questa mappatura corrisponde già a quella del nostro modello
sentiment_map = {0: "negative", 1: "neutral", 2: "positive"}
df['label'] = df['label'].map(sentiment_map)
# Il testo è nella colonna 'text'
df.rename(columns={'label': 'sentiment'}, inplace=True)

texts_to_predict = df['text'].tolist()
true_sentiments = df['sentiment'].tolist()

print(f"Dataset caricato. Analizzo {len(texts_to_predict)} testi...")

# Predizione del sentiment per l'intero dataset di test usando il batch processing
predicted_sentiments = predict_sentiment_batch(texts_to_predict, model, tokenizer)

# Aggiunta delle predizioni al DataFrame
df['predicted_sentiment'] = predicted_sentiments

print("\n--- Valutazione delle Performance del Modello ---")
print("Confronto tra i sentiment reali e quelli predetti:")

# Stampa un report di classificazione per una valutazione dettagliata
# zero_division=0 evita un warning se non ci sono esempi per una classe
report = classification_report(true_sentiments, predicted_sentiments, zero_division=0)
print(report)

print("\n--- Esempi di Predizione del Modello ---")
# Mostra i primi 5 risultati per un controllo visivo
print(df[['text', 'sentiment', 'predicted_sentiment']].head())