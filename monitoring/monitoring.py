import os
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

LABELS = ["negative", "neutral", "positive"]

def predict_sentiment_batch(texts, tokenizer, model, device):
    inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = F.softmax(outputs.logits, dim=1)
    label_ids = torch.argmax(probs, dim=1).tolist()
    return [LABELS[label_id] for label_id in label_ids]

def log_sentiment(text, pred_label, true_label):
    log_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'sentiment_log.csv')
    df_row = pd.DataFrame([{'text': text, 'predicted': pred_label, 'actual': true_label}])
    df_row.to_csv(log_file, mode='a', header=not os.path.exists(log_file), index=False, encoding='utf-8')

def analyze_and_plot():
    log_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'sentiment_log.csv')
    if not os.path.exists(log_file):
        print("Nessun file di log da analizzare. Esecuzione interrotta.")
        return

    df = pd.read_csv(log_file)
    print("\n--- Report di Classificazione del Modello ---")
    print(classification_report(df['actual'], df['predicted'], target_names=LABELS, zero_division=0))

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

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    report_dir = os.path.join(os.path.dirname(__file__), 'reports')
    os.makedirs(report_dir, exist_ok=True)
    save_path = os.path.join(report_dir, 'confusion_matrix.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Matrice di confusione salvata in {os.path.abspath(save_path)}")

def generate_html_report():
    log_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'sentiment_log.csv')
    img_path = os.path.join(os.path.dirname(__file__), 'reports', 'confusion_matrix.png')
    report_html_path = os.path.join(os.path.dirname(__file__), 'reports', 'report.html')

    if not os.path.exists(log_file) or not os.path.exists(img_path):
        print("File necessari per il report non trovati.")
        return

    df = pd.read_csv(log_file)
    table_html = df.to_html(index=False, classes='table', border=1)

    html_content = f"""
    <html>
    <head>
      <title>Report Monitoring TweetEval</title>
      <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #333; }}
        .table {{ border-collapse: collapse; width: 100%; }}
        .table th, .table td {{ padding: 8px 12px; border: 1px solid #ccc; }}
        .table th {{ background-color: #f2f2f2; }}
        img {{ max-width: 600px; height: auto; margin-top: 20px; }}
      </style>
    </head>
    <body>
      <h1>Report Monitoring TweetEval</h1>
      <h2>Matrice di Confusione</h2>
      <img src="confusion_matrix.png" alt="Matrice di Confusione"/>
      <h2>Log Sentiment</h2>
      {table_html}
    </body>
    </html>
    """

    with open(report_html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"Report HTML salvato in {report_html_path}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando device: {device}")

    dataset = load_dataset("tweet_eval", "sentiment")

    model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)

    log_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'sentiment_log.csv')
    if os.path.exists(log_file):
        os.remove(log_file)

    samples = dataset['test'].shuffle(seed=42).select(range(10))

    texts = list(samples['text'])
    predicted_labels = predict_sentiment_batch(texts, tokenizer, model, device)
    true_labels = [LABELS[label] for label in samples['label']]

    for text, pred, true in zip(texts, predicted_labels, true_labels):
        log_sentiment(text, pred, true)

    analyze_and_plot()
    generate_html_report()

if __name__ == "__main__":
    main()

