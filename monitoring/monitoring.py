# File: monitoring/monitoring.py
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os
import pandas as pd
import matplotlib.pyplot as plt

# Definizione delle funzioni
def log_sentiment(text, pred_label, true_label):
    """Logga i risultati dell'analisi del sentiment in un file CSV."""
    log_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'sentiment_log.csv')
    df_row = pd.DataFrame([{'text': text, 'predicted': pred_label, 'actual': true_label}])
    df_row.to_csv(log_file, mode='a', header=not os.path.exists(log_file), index=False)
    print(f"Loggato: {text} -> Predetto: {pred_label}, Reale: {true_label}")

def analyze_and_plot():
    """Analizza il file di log e genera un grafico."""
    log_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'sentiment_log.csv')
    if not os.path.exists(log_file):
        print("Nessun file di log da analizzare.")
        return

    df = pd.read_csv(log_file)
    accuracy = (df['predicted'] == df['actual']).mean()
    
    print(f"\nReport di monitoraggio:")
    print(f"  Accuratezza: {accuracy:.2f}")

    # Genera un grafico semplice
    df['risultato'] = df.apply(lambda row: 'corretto' if row['predicted'] == row['actual'] else 'sbagliato', axis=1)
    df['risultato'].value_counts().plot(kind='bar', color=['green', 'red'])
    plt.title("Risultati delle predizioni di sentiment")
    plt.xlabel("Risultato")
    plt.ylabel("Numero di campioni")
    plt.savefig('monitoring_report.png')
    print("Grafico di monitoraggio generato: monitoring_report.png")

def predict_sentiment(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    label_id = torch.argmax(probs).item()
    labels = ["negative", "neutral", "positive"]
    return labels[label_id]

def main():
    dataset = load_dataset("tweet_eval", "sentiment")

    model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    # Pulisci file log per test pulito
    log_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'sentiment_log.csv')
    if os.path.exists(log_file):
        os.remove(log_file)

    # Prendi 20 esempi random dal train set
    samples = dataset['train'].shuffle(seed=42).select(range(20))

    for sample in samples:
        text = sample['text']
        true_label = ["negative", "neutral", "positive"][sample['label']]
        pred_label = predict_sentiment(text, tokenizer, model)
        log_sentiment(text, pred_label, true_label)

    analyze_and_plot()

if __name__ == "__main__":
    main()