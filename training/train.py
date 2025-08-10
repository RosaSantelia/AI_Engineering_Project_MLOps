# training/train.py
import os
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import numpy as np

# Definizione del nome del modello pre-addestrato
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
MODEL_PATH = "./my_sentiment_model"

def compute_metrics(eval_pred):
    """
    Funzione per calcolare le metriche di valutazione durante il training.
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": (predictions == labels).mean()}

def main():
    """
    Funzione principale per l'addestramento del modello.
    """
    print("Inizio il training del modello.")

    # Caricamento del dataset "tweet_eval" e del tokenizer
    # Il dataset sentiment di tweet_eval usa 0, 1, 2 per le etichette
    # che corrispondono alle etichette del modello (negative, neutral, positive)
    dataset = load_dataset("tweet_eval", "sentiment")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    
    # Rinomina la colonna delle etichette per essere compatibile con il Trainer
    dataset = dataset.rename_column("label", "labels")

    # Iperparametri per il training
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=1,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
    )
    
    # Tokenizzazione dei dati
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
    # Suddividi i dati in training e validation set, usando un sottoinsieme
    # per velocizzare il training nel contesto di un test CI/CD.
    train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(500))
    eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(100))
    
    # Inizializza il Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )
    
    # Addestra il modello
    trainer.train()
    
    # Salva il modello e il tokenizer nella cartella specificata
    model.save_pretrained(MODEL_PATH)
    tokenizer.save_pretrained(MODEL_PATH)
    
    print(f"Training completato e modello salvato in {MODEL_PATH}.")

if __name__ == "__main__":
    main()