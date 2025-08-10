# training/train.py
"""
Script per fine-tuning veloce su TweetEval sentiment.
Esempio minimal: usa --sample per test rapidi in CI.
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

    if sample:
        # Usa solo 100 samples per train e 50 per eval
        train_ds = ds["train"].select(range(100))
        eval_ds = ds["validation"].select(range(50))
    else:
        train_ds = ds["train"]
        eval_ds = ds["validation"]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_ds = train_ds.map(lambda x: preprocess(x, tokenizer), batched=True)
    eval_ds = eval_ds.map(lambda x: preprocess(x, tokenizer), batched=True)

    train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    eval_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)

    training_args = TrainingArguments(
        output_dir="./runs",
        eval_strategy="epoch",
        per_device_train_batch_size=2,   # batch size piccolo per risorse limitate
        per_device_eval_batch_size=4,
        num_train_epochs=epochs,
        save_strategy=no,
        logging_steps=2, 
        load_best_model_at_end=False,
        report_to=[],  # disabilita logging esterni
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model("./model_finetuned")
    print("Training completato. Modello salvato in ./model_finetuned")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", action="store_true", help="usa subset per test rapido")
    parser.add_argument("--epochs", type=int, default=1, help="numero di epoche")
    args = parser.parse_args()
    main(sample=args.sample, epochs=args.epochs)
