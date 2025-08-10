from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from monitoring.monitoring import log_sentiment, analyze_and_plot
import os

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
