# File: monitoring/monitoring.py
import os
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

# Mappa per le etichette del sentiment
LABELS = ["negative", "neutral", "positive"]

def predict_sentiment_batch(texts, tokenizer, model):
    """
    Esegue la predizione del sentiment per una lista di testi in batch.
    
    Args:
        texts (list): Lista di stringhe di testo da analizzare.
        tokenizer (AutoTokenizer): Il tokenizer per preparare l'input.
        model (AutoModelForSequenceClassification): Il modello da utilizzare per la predizione.
        
    Returns:
        list: Una lista di etichette di sentiment predette.
    """
    inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    
    probs = F.softmax(outputs.logits, dim=1)
    label_ids = torch.argmax(probs, dim=1).tolist()
    
    return [LABELS[label_id] for label_id in label_ids]

def log_sentiment(text, pred_label, true_label):
    """
    Logga i risultati dell'analisi del sentiment in un file CSV.
    
    Args:
        text (str): Il testo analizzato.
        pred_label (str): L'etichetta di sentiment predetta.
        true_label (str): L'etichetta di sentiment reale.
    """
    log_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'sentiment_log.csv')
    
    df_row = pd.DataFrame([{'text': text, 'predicted': pred_label, 'actual': true_label}])
    
    # Aggiungi un'intestazione solo se il file non esiste
    df_row.to_csv(log_file, mode='a', header=not os.path.exists(log_file), index=False)

def analyze_and_plot():
    """
    Analizza il file di log, calcola le metriche e genera un grafico.
    Viene salvato un report di classificazione e un'immagine della matrice di confusione.
    """
    log_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'sentiment_log.csv')
    
    if not os.path.exists(log_file):
        print("Nessun file di log da analizzare. Esecuzione interrotta.")
        return

    df = pd.read_csv(log_file)
    
    # Calcolo e stampa delle metriche
    print("\n--- Report di Classificazione del Modello ---")
    print(classification_report(df['actual'], df['predicted'], target_names=LABELS, zero_division=0))
    
    # Generazione e salvataggio della matrice di confusione
    cm = confusion_matrix(df['actual'], df['predicted'], labels=LABELS)
    
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Matrice di Confusione')
    plt.colorbar()
    tick_marks = np.arange(len(LABELS))
    plt.xticks(tick_marks, LABELS, rotation=45)
    plt.yticks(tick_marks, LABELS)
    plt.ylabel('Etichetta Reale')
    plt.xlabel('Etichetta Predetta')
    
    # Aggiungi i numeri nella matrice
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    report_dir = os.path.join(os.path.dirname(__file__), '..', 'reports')
    os.makedirs(report_dir, exist_ok=True)
    plt.savefig(os.path.join(report_dir, 'confusion_matrix.png'))
    print("Matrice di confusione salvata in reports/confusion_matrix.png")
    
def main():
    """
    Funzione principale del monitoring: carica il dataset, esegue la predizione
    su un sottoinsieme di dati, logga i risultati e analizza le performance.
    """
    # Caricamento del dataset di test
    dataset = load_dataset("tweet_eval", "sentiment")
    
    # Caricamento del modello pre-addestrato
    model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    # Pulisci il file di log per un nuovo test
    log_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'sentiment_log.csv')
    if os.path.exists(log_file):
        os.remove(log_file)

    # Seleziona un sottoinsieme di 10 esempi dal test set
    samples = dataset['test'].shuffle(seed=42).select(range(10))
    
    # Esegui la predizione per tutti i campioni
    texts = samples['text']
    predicted_labels = predict_sentiment_batch(texts, tokenizer, model)
    true_labels = [LABELS[label] for label in samples['label']]
    
    # Logga ogni singolo risultato
    for text, pred, true in zip(texts, predicted_labels, true_labels):
        log_sentiment(text, pred, true)
    
    # Analizza i risultati e genera il grafico
    analyze_and_plot()

if __name__ == "__main__":
    main()