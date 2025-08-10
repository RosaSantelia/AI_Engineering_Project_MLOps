# training/push_to_hub.py
import os
from huggingface_hub import HfApi, HfFolder
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def main():
    """
    Funzione per l'autenticazione e il push del modello.
    """
    # Recupera il token dall'ambiente (impostato da GitHub Actions)
    HF_TOKEN = os.getenv("HF_HUB_TOKEN")
    if not HF_TOKEN:
        raise ValueError("HF_HUB_TOKEN non trovato nelle variabili d'ambiente.")

    # Salva il token per l'autenticazione
    HfFolder.save_token(HF_TOKEN)

    # Inizializza l'API di Hugging Face
    api = HfApi()

    # Nome del repository su Hugging Face Hub.
    # Sostituisci "tuo-username" con il tuo username di Hugging Face.
    repo_name = "tuo-username/sentiment-analysis-roberta"

    # Carica il modello e il tokenizer salvati
    try:
        model = AutoModelForSequenceClassification.from_pretrained("./my_sentiment_model")
        tokenizer = AutoTokenizer.from_pretrained("./my_sentiment_model")
    except Exception as e:
        print(f"Errore nel caricare il modello o il tokenizer: {e}")
        return

    # Esegui il push del modello e del tokenizer
    print(f"Eseguo il push del modello e del tokenizer su {repo_name}.")
    api.upload_folder(
        folder_path="./my_sentiment_model",
        repo_id=repo_name,
        repo_type="model"
    )
    
    print(f"Modello e tokenizer sono stati pubblicati con successo su {repo_name}!")

if __name__ == "__main__":
    main()
