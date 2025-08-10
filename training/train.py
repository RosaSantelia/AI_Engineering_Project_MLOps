# training/train.py
"""
Script per fine-tuning veloce su TweetEval sentiment.
Esempio minimal: riduci num_train_epochs e train_size per test rapidi in CI.
"""

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
import evaluate
import argparse

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"

def preprocess(examples, tokenizer):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

def compute_metrics(eval_pred):
    metric_acc = evaluate.load("accuracy")
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = metric_acc.compute(predictions=preds, references=labels)
    return {"accuracy": acc["accuracy"]}

def main(sample=False, epochs=1):
    ds = load_dataset("cardiffnlp/tweet_eval", "sentiment")
    # per test rapido usa solo subset
    if sample:
        ds = ds["train"].train_test_split(test_size=0.95)  # tiene 5% per train veloce
        train_ds = ds["train"]
        eval_ds = ds["test"]
    else:
        train_ds = ds["train"]
        eval_ds = ds["validation"]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_ds = train_ds.map(lambda x: preprocess(x, tokenizer), batched=True)
    eval_ds = eval_ds.map(lambda x: preprocess(x, tokenizer), batched=True)
    train_ds.set_format(type="torch", columns=["input_ids","attention_mask","label"])
    eval_ds.set_format(type="torch", columns=["input_ids","attention_mask","label"])

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)

    args = TrainingArguments(
        output_dir="./runs",
        eval_strategy="epoch",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=epochs,
        save_strategy="epoch",
        logging_steps=10,
        load_best_model_at_end=True,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model("./model_finetuned")
    print("Training completo. Modello salvato in ./model_finetuned")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", action="store_true", help="usa piccolo subset per test rapido")
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()
    main(sample=args.sample, epochs=args.epochs)
